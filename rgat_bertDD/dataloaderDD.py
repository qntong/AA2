from datasetDD import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from transformers import BertTokenizer


def get_train_valid_sampler(trainset):
    size = len(trainset)     # 11118   # 1000
    idx = list(range(size))
    return SubsetRandomSampler(idx)



def load_vocab(dataset_name):
    # speaker_vocab={'stoi': {'A': 0, 'B': 1}, 'itos': ['A', 'B']}
    speaker_vocab = pickle.load(open('../data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    #{'stoi': {'none': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6},
    # 'itos': ['none', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']}
    label_vocab = pickle.load(open('../data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    person_vec = None

    return speaker_vocab, label_vocab, person_vec


def get_IEMOCAP_loaders(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    # -->dataloader.py:17
    print('building datasets..')
    # -->dataset.py:14
    trainset = IEMOCAPDataset(dataset_name, 'train',  speaker_vocab, label_vocab, args)       # 11118
    devset = IEMOCAPDataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)            # 1000
    # -->dataloader.py:11
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(dataset_name, 'test',  speaker_vocab, label_vocab, args)     # 1000
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec