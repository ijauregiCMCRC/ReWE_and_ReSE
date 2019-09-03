from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import os
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules

from onmt.Utils import use_gpu
import glob
import copy




def show_optimizer_state_mine(optim):
    print("optim.optimizer.state_dict()['state'] keys: ")
    for key in optim.optimizer.state_dict()['state'].keys():
        print("optim.optimizer.state_dict()['state'] key: " + str(key))

    print("optim.optimizer.state_dict()['param_groups'] elements: ")
    for element in optim.optimizer.state_dict()['param_groups']:
        print("optim.optimizer.state_dict()['param_groups'] element: " + str(
            element))

def lazily_load_dataset_mine(corpus_type,logger,opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def make_dataset_iter_mine(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        # In token batching scheme, the number of sequences is limited
        # such that the total number of src/tgt tokens (including padding)
        # in a batch <= batch_size
        def batch_size_fn(new, count, sofar):
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)

    device = opt.gpuid[0] if opt.gpuid else -1

    return DatasetLazyIter_mine(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)

class DatasetLazyIter_mine(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        return self.loss / self.n_words

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.

        Returns:
           msg (str): log message.
        """
        t = self.elapsed_time()
        msg = (("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; xent: " +
                "%6.2f;  %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s " +
                "elapsed") %
               (epoch, batch,  n_batches,
                self.accuracy(),
                self.ppl(),
                self.xent(),
                self.n_src_words / (t + 1e-5),
                self.n_words / (t + 1e-5),
                time.time() - start))
        return msg

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, step):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper",  self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", lr, step)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.progress_step = 0

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:  #HERE IS WHERE THE MODEL IS PROPAGATED!
                self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                        epoch, idx, num_batches,
                        self.progress_step,
                        total_stats.start_time, self.optim.lr,
                        report_stats)
                    self.progress_step += 1

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                true_batchs, total_stats,
                report_stats, normalization)
            true_batchs = []

        return total_stats

    def train_validate(self, train_iter, epoch, report_func=None,
                       batch_number=0,validate_number=500,logger=None, no_impr_ppl_num=0,
                       saved_models=0,need_to_save=1,model_opt=None,fields=None,
                       string_saved_model=None,opt=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            batch_number+=1
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:  #HERE IS WHERE THE MODEL IS PROPAGATED!
                self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                        epoch, idx, num_batches,
                        self.progress_step,
                        total_stats.start_time, self.optim.lr,
                        report_stats)
                    self.progress_step += 1

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

            #Check if we need to validate, batch_number is multiple of valid number
            if batch_number % validate_number == 0:
                valid_iter = make_dataset_iter_mine(lazily_load_dataset_mine("valid",logger,opt),
                                               fields, opt, is_train=False)
                valid_stats=self.validate(valid_iter)
                logger.info('Validation perplexity: %g' % valid_stats.ppl())
                logger.info('Validation accuracy: %g' % valid_stats.accuracy())

                # Check perplexity and update the learning rate
                decay, best_true, lr = self.epoch_step_mine(valid_stats.ppl(),no_impr_ppl_num,need_to_save)

                if best_true:
                    no_impr_ppl_num = 0
                    #Save best model with the index I would say
                    print ("Saving best model in validation so far!")
                    if saved_models < need_to_save:
                        print ('Save new '+str(need_to_save))
                        #Drop a checkpoint here
                        saved_models += 1
                        aux=copy.deepcopy(fields)
                        string_saved_model= self.drop_checkpoint_mine(model_opt, epoch, aux,
                                                                       valid_stats,saved_models)



                    elif saved_models == need_to_save:
                        print('Overwrite ' + str(need_to_save))
                        #Delete old
                        os.remove(string_saved_model)
                        aux = copy.deepcopy(fields)
                        #Save new
                        string_saved_model= self.drop_checkpoint_mine(model_opt, epoch, aux,
                                                                       valid_stats,saved_models)


                if decay:
                    if saved_models<need_to_save:
                        print('Save new ' + str(need_to_save))
                        # Drop a checkpoint here
                        saved_models += 1
                        aux = copy.deepcopy(fields)
                        string_not_good = self.drop_checkpoint_mine(model_opt, epoch, aux,
                                                                       valid_stats, saved_models)
                    aux = copy.deepcopy(fields)
                    logger.info("Decaying learning rate to %g" % self.optim.lr)
                    no_impr_ppl_num=0
                    need_to_save+=1
                    #Load the parameters of the previous best model
                    logger.info('Loading checkpoint from %s' % string_saved_model)
                    checkpoint = torch.load(string_saved_model,map_location=lambda storage, loc: storage)
                    self.model=self.build_model_mine(model_opt,opt,aux,checkpoint,logger)
                    self.optim=self.build_optim_mine(self.model,checkpoint,logger,opt)
                    self.optim.lr=lr
                    self.optim.optimizer.param_groups[0]['lr'] = lr
                    logger.info("Confirm new learning rate to %g" % self.optim.lr)

                if decay==False and best_true==False:
                    no_impr_ppl_num+=1


                if no_impr_ppl_num == 20:
                    if saved_models<need_to_save:
                        print('Save new ' + str(need_to_save))
                        # Drop a checkpoint here
                        saved_models += 1
                        aux = copy.deepcopy(fields)
                        string_saved_model = self.drop_checkpoint_mine(model_opt, epoch, aux,
                                                                       valid_stats, saved_models)
                    # Finish the training
                    break

            # if batch_number == 110:
            #     no_impr_ppl_num=20
            #     break

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                true_batchs, total_stats,
                report_stats, normalization)
            true_batchs = []

        return total_stats, no_impr_ppl_num,saved_models,need_to_save,string_saved_model,batch_number,epoch

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def epoch_step_mine(self, ppl,no_impr_ppl_num,need_to_save):
        return self.optim.update_learning_rate_mine(ppl,no_impr_ppl_num,need_to_save)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)
        if opt.ReWE:
            real_generator_ReWE = (real_model.generator_ReWE.module
                              if isinstance(real_model.generator_ReWE, nn.DataParallel)
                              else real_model.generator_ReWE)
        if opt.ReSE:
            real_generator_ReSE = (real_model.generator_ReSE.module
                                  if isinstance(real_model.generator_ReSE, nn.DataParallel)
                                  else real_model.generator_ReSE)

        model_state_dict = real_model.state_dict()

        if opt.ReWE==True and opt.ReSE==False:
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if ('generator' or 'generator_ReWE') not in k}
            generator_state_dict = real_generator.state_dict()
            generator_ReWE_state_dict = real_generator_ReWE.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'generator_ReWE': generator_ReWE_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
        elif opt.ReWE==False and opt.ReSE==True:
            # print ('ReSE save')
            # for k, v in model_state_dict.items():
            #     print(k)
            #     if 'generator' not in k and 'ReSE_pred' not in k and 'ReSE_gt' not in k:
            #         print ('yes')
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k and 'generator_ReSE' not in k}
            generator_state_dict = real_generator.state_dict()
            generator_ReSE_state_dict = real_generator_ReSE.state_dict()
            #ReSE_gt_state_dict = real_ReSE_gt.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'generator_ReSE': generator_ReSE_state_dict,
                #'ReSE_gt': ReSE_gt_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
        elif opt.ReWE==True and opt.ReSE==True:
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k and 'generator_ReWE' not in k and 'generator_ReSE' not in k}
            generator_state_dict = real_generator.state_dict()
            generator_ReWE_state_dict = real_generator_ReWE.state_dict()
            generator_ReSE_state_dict = real_generator_ReSE.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'generator_ReWE': generator_ReWE_state_dict,
                'generator_ReSE': generator_ReSE_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
        else:
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k}
            generator_state_dict = real_generator.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def drop_checkpoint_mine(self, opt, epoch, fields, valid_stats,saved_model):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)
        if opt.ReWE:
            real_generator_ReWE = (real_model.generator_ReWE.module
                              if isinstance(real_model.generator_ReWE, nn.DataParallel)
                              else real_model.generator_ReWE)
        if opt.ReSE:
            real_generator_ReSE = (real_model.generator_ReSE.module
                                  if isinstance(real_model.generator_ReSE, nn.DataParallel)
                                  else real_model.generator_ReSE)

        model_state_dict = real_model.state_dict()

        if opt.ReWE==True and opt.ReSE==False:
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if ('generator' or 'generator_ReWE') not in k}
            generator_state_dict = real_generator.state_dict()
            generator_ReWE_state_dict = real_generator_ReWE.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'generator_ReWE': generator_ReWE_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
        elif opt.ReWE==False and opt.ReSE==True:
            # print ('ReSE save')
            # for k, v in model_state_dict.items():
            #     print(k)
            #     if 'generator' not in k and 'ReSE_pred' not in k and 'ReSE_gt' not in k:
            #         print ('yes')
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k and 'generator_ReSE' not in k}
            generator_state_dict = real_generator.state_dict()
            generator_ReSE_state_dict = real_generator_ReSE.state_dict()
            #ReSE_gt_state_dict = real_ReSE_gt.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'generator_ReSE': generator_ReSE_state_dict,
                #'ReSE_gt': ReSE_gt_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
        elif opt.ReWE==True and opt.ReSE==True:
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k and 'generator_ReWE' not in k and 'generator_ReSE' not in k}
            generator_state_dict = real_generator.state_dict()
            generator_ReWE_state_dict = real_generator_ReWE.state_dict()
            generator_ReSE_state_dict = real_generator_ReSE.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'generator_ReWE': generator_ReWE_state_dict,
                'generator_ReSE': generator_ReSE_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
        else:
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k}
            generator_state_dict = real_generator.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d_num%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch, saved_model))

        string_saved_model='%s_acc_%.2f_ppl_%.2f_e%d_num%d.pt'% (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch, saved_model)
        return string_saved_model

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')

            for j in range(0, target_size - 1, trunc_size):  #By default it only goed to the for loop once

                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state = self.model(src, tgt, src_lengths, dec_state)
                #print (outputs.size())

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None and j+trunc_size < target_size-1:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()

    def build_model_mine(self,model_opt, opt, fields, checkpoint,logger):
        logger.info('Building model...')
        model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                      use_gpu(opt), checkpoint)
        if len(opt.gpuid) > 1:
            logger.info('Multi gpu training: ', opt.gpuid)
            model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
        logger.info(model)

        return model

    def build_optim_mine(self,model, checkpoint, logger, opt):
        saved_optimizer_state_dict = None


        logger.info('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        # We need to save a copy of optim.optimizer.state_dict() for setting
        # the, optimizer state later on in Stage 2 in this method, since
        # the method optim.set_parameters(model.parameters()) will overwrite
        # optim.optimizer, and with ith the values stored in
        # optim.optimizer.state_dict()
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        # print (len(saved_optimizer_state_dict))


        # Stage 1:
        # Essentially optim.set_parameters (re-)creates and optimizer using
        # model.paramters() as parameters that will be stored in the
        # optim.optimizer.param_groups field of the torch optimizer class.
        # Importantly, this method does not yet load the optimizer state, as
        # essentially it builds a new optimizer with empty optimizer state and
        # parameters from the model.
        optim.set_parameters(model.named_parameters())

        print(
            "Stage 1: Keys after executing optim.set_parameters" +
            "(model.parameters())")
        show_optimizer_state_mine(optim)

        # print ("New optimizer: "+str(len(optim.optimizer.param_groups)))
        # print("Old optimizer: " + str(len(saved_optimizer_state_dict['param_groups'])))
        #
        # param_lens = (len(g['params']) for g in optim.optimizer.param_groups)
        # saved_lens = (len(g['params']) for g in saved_optimizer_state_dict['param_groups'])
        #
        # for p_len, s_len in zip(param_lens, saved_lens):
        #     print (p_len)
        #     print (s_len)


        # Stage 2: In this stage, which is only performed when loading an
        # optimizer from a checkpoint, we load the saved_optimizer_state_dict
        # into the re-created optimizer, to set the optim.optimizer.state
        # field, which was previously empty. For this, we use the optimizer
        # state saved in the "saved_optimizer_state_dict" variable for
        # this purpose.
        # See also: https://github.com/pytorch/pytorch/issues/2830
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        # Convert back the state values to cuda type if applicable
        if use_gpu(opt):
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        print(
            "Stage 2: Keys after executing  optim.optimizer.load_state_dict" +
            "(saved_optimizer_state_dict)")
        show_optimizer_state_mine(optim)

        # We want to make sure that indeed we have a non-empty optimizer state
        # when we loaded an existing model. This should be at least the case
        # for Adam, which saves "exp_avg" and "exp_avg_sq" state
        # (Exponential moving average of gradient and squared gradient values)
        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

        return optim
