import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import json
import random

class IEMOCAPDataset(Dataset):
    def __init__(self, dataset_name='DailyDialog', split='train', speaker_vocab=None, label_vocab=None, args=None, tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data= self.read(dataset_name, split, tokenizer)  # 获取数据和索引
        print(len(self.data))
        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('../data/%s/processed_%s_data1.pkl' % (dataset_name, split), 'rb') as f:
            raw_data = pickle.load(f)

        dialogs = []
        # indices = []  # 创建一个用于存储索引的列表

        for idx, d in enumerate(raw_data):  # 添加一个枚举，这样我们可以得到当前对话的索引
            utterances = []
            labels = []
            speakers = []
            features = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['features'])

            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features,
                # 'index': idx  # 在每个对话中存储索引
            })
            # indices.append(idx)  # 将索引添加到索引列表中


        # random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):

    # 话语的特征 标签 说话人 标签的长度 话语原文

        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

        # return torch.stack(self.data[index]['features']), \
        #       torch.LongTensor(self.data[index]['speakers']), \
        #      torch.LongTensor([0 if label == 0 else 1 for label in self.data[index]['labels']]), \
        #       torch.LongTensor(self.data[index]['labels']), \
        #       self.data[index]['utterances']  # 返回对话的索引

    def __len__(self):
            return self.len

    def collate_fn(self, data):

        max_dialog_len = max([d[3] for d in data])
        features = pad_sequence([torch.FloatTensor(d[0]) for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([torch.LongTensor(d[1]) for d in data], batch_first=True, padding_value=-1)
        # labels = pad_sequence([d[1] for d in data])
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]
        return max_dialog_len, features, labels, lengths, speakers, utterances

        # d = pd.DataFrame(data)
        # return [pad_sequence(d[i]) if i < 2 else pad_sequence(d[i], True) if i < 4 else d[i].tolist() for i in d]


