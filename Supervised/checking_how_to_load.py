import torch
import onmt
import torch.nn as nn
from onmt.Utils import use_gpu
import glob


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

def build_model_mine(self, model_opt, opt, fields, checkpoint, logger):
    logger.info('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        logger.info('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    logger.info(model)

    return model


def build_optim_mine(self, model, checkpoint, logger, opt):
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


string_saved_model="IWSLT_2016/english_french/models/LAMBDA_0.1_embs_512_rnn_1024_FIX_BOTH_EMBEDDINGS_COSINE_LOSS_neubig_style_training/models/EPOCH_acc_56.82_ppl_11.63_e4_num1.pt"

checkpoint = torch.load(string_saved_model,map_location=lambda storage, loc: storage)

model=build_model_mine(model_opt,opt,aux,checkpoint,logger)
optim=build_optim_mine(self.model,checkpoint,logger,opt)