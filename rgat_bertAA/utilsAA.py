import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasetAA import IEMOCAPDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import pickle
import parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_train_valid_sampler(trainset, valid=0.0):
    size = len(trainset)  # 11118   # 1000
    idx = list(range(size))
    return SubsetRandomSampler(idx)


def load_vocab(dataset_name):
    # speaker_vocab={'stoi': {'A': 0, 'B': 1}, 'itos': ['A', 'B']}
    speaker_vocab = pickle.load(open('../data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    #{'stoi': {'none': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6},
    # 'itos': ['none', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']}
    label_vocab = pickle.load(open('../data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    # person_vec_dir = './data/%s/person_vect.pkl' % (dataset_name)
    # if os.path.exists(person_vec_dir):
    #     print('Load person vec from ' + person_vec_dir)
    #     person_vec = pickle.load(open(person_vec_dir, 'rb'))
    # else:
    #     print('Creating personality vectors')
    #     person_vec = np.random.randn(len(speaker_vocab['itos']), 100)a
    #     print('Saving personality vectors to' + person_vec_dir)
    #     with open(person_vec_dir,'wb') as f:
    #         pickle.dump(person_vec, f, -1)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec

def get_IEMOCAP_loaders(dataset_name = 'DailyDialog', batch_size=32, num_workers=0, pin_memory=False, args = None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    # -->dataloader.py:17
    print('building datasets..')
    # -->dataset.py:14
    trainset = IEMOCAPDataset(dataset_name, 'train', speaker_vocab, label_vocab, args)  # 11118
    devset = IEMOCAPDataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)  # 1000
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

    testset = IEMOCAPDataset(dataset_name, 'test', speaker_vocab, label_vocab, args)  # 1000
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec


def get_attention_mask(t, device):
    mask = torch.ones(t.size())
    for i in range(t.size()[0]):
        for j in range(t.size()[1]):
            if t[i][j] == 0:
                mask[i][j] = 0
    if device == "cuda":
        mask = mask.cuda()
    return mask


def edge_perms(seq, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    l -> sequence length
    """

    all_perms = set()
    array = np.arange(seq)
    for j in range(seq):
        perms = set()
        # select utterances in the window
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(seq, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(seq, j + window_future + 1)]
        # add edges(verticle1, verticle2) in perms
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)

    return list(all_perms)


def batch_graphify(features, qmask, lengths, window_past, window_future, dim, edge_type_mapping, device):
    """
    features -> seq*batch, dim
    qmask -> seq, batch, party
    """

    node_features, edge_index, edge_type = [], [], []
    batch_size = qmask.size(1)
    features = features.reshape(-1, batch_size, dim)
    length_sum = 0
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
        perms1 = edge_perms(lengths[j], window_past, window_future)  # local
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]  # global
        length_sum += lengths[j]

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            speaker0 = int(qmask[j][item1[0]])
            speaker1 = int(qmask[j][item1[1]])
            # speaker0 = (qmask[item1[0], j, :] == 1).nonzero(as_tuple=False)[0][0].tolist()
            # speaker1 = (qmask[item1[1], j, :] == 1).nonzero(as_tuple=False)[0][0].tolist()
            if item1[0] < item1[1]:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_type = torch.tensor(edge_type)

    if device == "cuda":
        node_features, edge_index, edge_type = node_features.cuda(), edge_index.cuda(), edge_type.cuda()

    return node_features, edge_index, edge_type


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, device, optimizer, train):
    losses, preds, labels = [], [], []
    assert not train or optimizer is not None

    seed_everything(seed=1234)

    if train:
        print("\t* Training epoch {}:".format(epoch))
        model.train()
        for data in tqdm(dataloader):
            optimizer.zero_grad()
            ids = torch.stack([d.cuda() for d in data[1] if device == "cuda"])
            qmask = torch.stack([d.cuda() for d in data[4] if device == "cuda"])
            # umask = [d.cuda() for d in data[1] if device == "cuda"]
            label = torch.stack([d.cuda() for d in data[2] if device == "cuda"])
            # ids, qmask, umask, label = [d.cuda() for d in data[:-1]] if device == "cuda" else data[:-1]

            lengths = []
            lengths.append(data[3].tolist())
            # print(ids.dtype,qmask.dtype)
            # print(ids.long())
            # print(ids.long().dtype)
            # exit()
            ids = ids.long()  # 原本ids类型为float32, 需要转成int64才能输入模型
            logits = model(ids, qmask, lengths)
            label = label[0]
            loss = loss_function(logits, label)
            preds.append(torch.argmax(logits, 1).cpu().numpy())
            labels.append(label.cpu().numpy())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
    else:
        print("\t* Validating epoch {}:".format(epoch))
        model.eval()
        with torch.no_grad():
            for data in tqdm(dataloader):
                ids, qmask, umask, label = [d.cuda() for d in data[:-1]] if device == "cuda" else data[:-1]
                lengths = []
                for i in umask:
                    lengths.append(int(i.sum().tolist()))
                logits = model(ids, qmask, lengths)
                label = label[0]
                loss = loss_function(logits, label)
                preds.append(torch.argmax(logits, 1).cpu().numpy())
                labels.append(label.cpu().numpy())
                losses.append(loss.item())

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    print("\t-> avg_loss: {:.4f}, avg_accuracy: {:.4f}, avg_fscore: {:.4f}".format(avg_loss, avg_accuracy, avg_fscore))
    return avg_loss, avg_accuracy, avg_fscore
