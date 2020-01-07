
import os

ENV_VARIABLE = {
    'DIR_PRETRAINED_MODELS': './pretrained_models',
    'DIR_DATASETS': './datasets',
    'DIR_CHECKPOINTS': './checkpoints',
}


def get_available_models():
    d = ENV_VARIABLE['DIR_PRETRAINED_MODELS']
    return [
        folder
        for folder in os.listdir(d)
        if os.path.isdir(os.path.join(d, folder))
    ]


def get_available_datasets(dataset_type):
    d = os.path.join(ENV_VARIABLE['DIR_DATASETS'], dataset_type)
    return [
        folder
        for folder in os.listdir(d)
        if os.path.isdir(os.path.join(d, folder))
    ]


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            # s = self.grouped["{}".format(self.n_sent)]
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None


def prune_examples(_examples, ratio=None):
    if ratio is None:
        return _examples
    else:
        num_examples_new = int(ratio*float(len(_examples)))
        print(f'use {num_examples_new} of {len(_examples)} examples')
        return _examples[:num_examples_new]


