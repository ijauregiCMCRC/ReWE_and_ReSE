"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
#This two libraries were added for max-margin loss
#################################
from torch.autograd import Variable
from torch.nn import Parameter
#################################
import onmt
import onmt.io
import tensorflow as tf
import tensorflow_hub as hub


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)

        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .long().sum()
        return onmt.Statistics(loss.data.cpu().numpy(),
                               non_padding.long().sum().data.cpu().numpy(),
                               num_correct.data.cpu().numpy())

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0, ReWE=False, ReSE=False, generator_ReWE=None, generator_ReSE=None, dec_embeddings=None,
                 alpha_loss_ReWE=None, alpha_loss_ReSE=None, tgt_emb_size=None):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing
        self.ReWE=ReWE
        self.ReSE=ReSE
        if ReWE:
            self.generator_ReWE = generator_ReWE
            self.dec_embeddings = dec_embeddings
            self.tgt_emb_size = tgt_emb_size
            self.alpha_loss_ReWE = alpha_loss_ReWE
            self.criterion_ReWE = nn.CosineEmbeddingLoss(size_average=False)
        if ReSE:
            self.generator_ReSE = generator_ReSE
            g = tf.Graph()
            with g.as_default():
                self.messages = tf.placeholder(dtype=tf.string, shape=[None])
                self.embed = hub.Module("../ReSE/universal_sentence_encoder")
                self.embeddings = self.embed(self.messages)
                init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
            g.finalize()
            self.sess = tf.Session(graph=g)
            self.sess.run(init_op)
            self.dec_embeddings = dec_embeddings
            self.tgt_emb_size = tgt_emb_size
            self.alpha_loss_ReSE = alpha_loss_ReSE
            # self.criterion_continous = nn.MSELoss(size_average=False)
            self.criterion_ReSE = nn.CosineEmbeddingLoss(size_average=False)


    def _make_shard_state(self, batch, output, range_, attns=None):

        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def idSEQ2strSEQ(self,target_IDs):
        list_batch=[]
        sen_len, batch_len = target_IDs.size()
        for i in range(batch_len):
            list_sent = []
            for j in range(sen_len):
                list_sent.append(self.tgt_vocab.itos[target_IDs[j,i]])
            list_sent = " ".join(list_sent)
            list_batch.append(list_sent)
        return list_batch

    def _compute_loss(self, batch, output, target):
        scores = self.generator(self._bottle(output))
        if self.ReWE:
            scores_reg = self.generator_ReWE(self._bottle(output))

        # print (target.size())

        if self.ReWE:
            target_dim=target.unsqueeze(2)
            # print (target_dim.size())
            with torch.no_grad():
                embeddings_tgt=self.dec_embeddings(target_dim)
            # print (embeddings_tgt)
            # print (embeddings_tgt.size())

            gtruth_emb = embeddings_tgt.view(-1, self.tgt_emb_size)
            #print (gtruth_emb.requires_grad)

        gtruth = target.view(-1)

        if self.ReSE:
            sen_vecs = self.generator_ReSE(output.transpose(0,1))
            target_dim = target.unsqueeze(2)
            #print (target_dim.size())
            # with torch.no_grad():
            #     embeddings_tgt = self.dec_embeddings(target_dim)
            # sen_gt_embs = self.ReSE_gt(embeddings_tgt.transpose(0,1))
            sentString = self.idSEQ2strSEQ(target)
            list_embs = []
            for sen in sentString:
                new_embedding = self.sess.run(self.embeddings, feed_dict={self.messages: [sen]})
                new_embedding = torch.from_numpy(new_embedding)
                list_embs.append(new_embedding)
            sen_gt_embs = torch.stack(list_embs).squeeze().cuda()
            #print (sen_gt_embs.size())

        #Convert IDs to embeddings


        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.numel() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_

        if self.ReWE==True and self.ReSE==False:
            #Loss 1
            loss_NLL= self.criterion(scores, gtruth)
            #Loss 2
            loss_ReWE = self.criterion_ReWE(scores_reg, gtruth_emb,torch.tensor(1, dtype=torch.float,device='cuda:0')) #Cosine loss
            #Combined loss
            loss = loss_NLL + self.alpha_loss_ReWE*loss_ReWE
        elif self.ReWE==False and self.ReSE==True:
            # Loss 1
            loss_NLL = self.criterion(scores, gtruth)
            # Loss 2
            loss_ReSE = self.criterion_ReSE(sen_vecs, sen_gt_embs, torch.tensor(1, dtype=torch.float,device='cuda:0'))  # Cosine loss
            #Combined loss
            loss = loss_NLL + self.alpha_loss_ReSE * loss_ReSE
        elif self.ReWE==True and self.ReSE==True:
            #Loss 1
            loss_NLL = self.criterion(scores, gtruth)
            #Loss 2
            loss_ReWE = self.criterion_ReWE(scores_reg, gtruth_emb,torch.tensor(1, dtype=torch.float, device='cuda:0'))  # Cosine loss
            #Loss 3
            loss_ReSE = self.criterion_ReSE(sen_vecs, sen_gt_embs,torch.tensor(1, dtype=torch.float, device='cuda:0'))  # Cosine loss
            #Combined loss
            loss = loss_NLL + self.alpha_loss_ReWE*loss_ReWE + self.alpha_loss_ReSE * loss_ReSE
        else:
            loss = self.criterion(scores, gtruth)


        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


class max_margin_loss(nn.Module):
    def __init__(self, num_classes,decoder_embeddings, margin=0.1, weights=None):
        """
        :param num_classes: An int. The number of possible classes.
        :param embed_size: An int. EmbeddingLockup size
        :param num_sampled: An int. The number of sampled from noise examples
        :param weights: A list of non negative floats. Class weights. None if
            using uniform sampling. The weights are calculated prior to
            estimation and can be of any form, e.g equation (5) in [1]
        """

        super(max_margin_loss, self).__init__()

        self.vocab_size = num_classes
        self.decoder_embeddings= decoder_embeddings
        self.margin=margin


        self.weights = weights
        if self.weights is not None:
            assert min(self.weights) >= 0, "Each weight should be >= 0"

            self.weights = Variable(torch.from_numpy(weights)).float()

    def sample(self, num_sample):
        """
        draws a sample from classes based on weights
        """
        return torch.multinomial(self.weights, num_sample, True)

    def forward(self, pred_embs, ground_truth_embs, num_sampled):
        """
        :param pred_embs: Tensor with shape of [predictions_num x emb_length]
        :param ground_truth_embs: Tensor with shape of [predictions_num x emb_length]
        :param num_sampled: An int. The number of sampled from noise examples
        :return: Loss estimation with shape of [1]
        """

        [predictions_num, emb_length] = ground_truth_embs.size()



        # if self.weights is not None:
        #     noise_sample_count = batch_size * window_size * num_sampled
        #     draw = self.sample(noise_sample_count)
        #     noise = draw.view(batch_size * window_size, num_sampled)
        # else:
        noise = Variable(torch.Tensor(num_sampled,predictions_num,1).uniform_(0, self.vocab_size - 1).long()) #Size: [predictions_num x num_sampled ]
        noise=noise.cuda()
        #print (noise.size())

        with torch.no_grad():
            sampled_embeddings = self.decoder_embeddings(noise)
        #sampled_embeddings = noise.view(-1, emb_length)  #Size: [predictions_num x num_sampled x embedding_size]

        #Compute all the cosines
        cos_truth=nn.functional.cosine_similarity(pred_embs,ground_truth_embs) #Not sure if it is well defined
        cos_neg_samples= torch.zeros(predictions_num, dtype=torch.float,device='cuda:0')#ALERT! it needs to be a scalar an it is a vector!
        for i in range(num_sampled):
            cos_neg_samples+=nn.functional.cosine_similarity(pred_embs,sampled_embeddings[i,:,:])
        loss_almost=cos_neg_samples - cos_truth  #ALERT!!! margin is a scalar and needs to be a vector!!!!!!! Size: [predictions_num]
        loss=torch.add(loss_almost,self.margin)

        #Hinge loss
        #Select the maximum between zero and the margin
        vector_zeros=torch.zeros(predictions_num, dtype=torch.float,device='cuda:0')  #ALERT!!! margin is a scalar and needs to be a vector!!!!!!!  Size: [predictions_num]
        concatenated=torch.stack((vector_zeros,loss),dim=1) #Size: [predictions_num x 2]
        final_loss=torch.max(concatenated,dim=1)[0] #Size: [predictions_num]

        #Sum all the loss
        final_loss_sum=final_loss.sum()  #Size: ESCALAR

        return final_loss_sum


        # #OLD CODE
        # log_target = (pred_embs * output).sum(1).squeeze().sigmoid().log()
        #
        # ''' ∑[batch_size * window_size, num_sampled, embed_size] * [batch_size * window_size, embed_size, 1] ->
        #     ∑[batch_size, num_sampled, 1] -> [batch_size] '''
        # sum_log_sampled = torch.bmm(sampled_embeddings, pred_embs.unsqueeze(2)).sigmoid().log().sum(1).squeeze()
        #
        # loss = log_target + sum_log_sampled
        #
        # return -loss.sum() / batch_size

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()
