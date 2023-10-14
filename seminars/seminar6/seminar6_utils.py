import sys

import matplotlib
import numpy as np
import torch
import torch.autograd as autograd
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

sys.path.append('../../homeworks')
from dgm_utils.visualize import (
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TICKS_FONT_SIZE,
    TITLE_FONT_SIZE,
)


def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)


class DataLoaderWrapper:
    '''
    Helpful class for using the
    torch.distribution in torch's
    DataLoader manner
    '''

    class DatasetEmulator:
        def __init__(self, len_):
            self.len_ = len_
            pass

        def __len__(self):
            return self.len_

    class FiniteRepeatDSIterator:
        def __init__(self, sampler, batch_size, n_batches):
            dataset = sampler.sample((batch_size * n_batches,))
            assert len(dataset.shape) >= 2
            new_size = (n_batches, batch_size) + dataset.shape[1:]
            self.dataset = dataset.view(new_size)
            self.batch_size = batch_size
            self.n_batches = n_batches

        def __iter__(self):
            for i in range(self.n_batches):
                yield self.dataset[i]

    class FiniteUpdDSIterator:
        def __init__(self, sampler, batch_size, n_batches):
            self.sampler = sampler
            self.batch_size = batch_size
            self.n_batches = n_batches

        def __iter__(self):
            for i in range(self.n_batches):
                yield self.sampler.sample((self.batch_size,))

    class InfiniteDsIterator:
        def __init__(self, sampler, batch_size):
            self.sampler = sampler
            self.batch_size = batch_size

        def __iter__(self):
            return self

        def __next__(self):
            return self.sampler.sample((self.batch_size,))

    def __init__(self, sampler, batch_size, n_batches=None, store_dataset=False):
        '''
        n_batches : count of batches before stop_iterations, if None, the dataset is infinite
        store_datset : if n_batches is not None and store_dataset is True,
        during the first passage through the dataset the data will be stored,
        and all other epochs will use the same dataset, stored during the first pass
        '''
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.store_dataset = store_dataset
        self.sampler = sampler

        if self.n_batches is None:
            self.ds_iter = DataLoaderWrapper.InfiniteDsIterator(
                sampler, self.batch_size
            )
            return

        self.dataset = DataLoaderWrapper.DatasetEmulator(
            self.batch_size * self.n_batches
        )
        if not self.store_dataset:
            self.ds_iter = DataLoaderWrapper.FiniteUpdDSIterator(
                sampler, self.batch_size, self.n_batches
            )
            return

        self.ds_iter = DataLoaderWrapper.FiniteRepeatDSIterator(
            sampler, self.batch_size, self.n_batches
        )

    def __iter__(self):
        return iter(self.ds_iter)
