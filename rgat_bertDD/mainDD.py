import torch
# import torch.nn as nn
# import torch.optim as optim
from dataloaderDD import get_IEMOCAP_loaders
import argparse




device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument('--bert_model_dir', type=str, default='')
parser.add_argument('--bert_tokenizer_dir', type=str, default='')

parser.add_argument('--bert_dim', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
parser.add_argument('--gnn_layers', type=int, default=3, help='Number of gnn layers.')
parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')

parser.add_argument('--dataset_name', default='DailyDialog', type=str,
                    help='dataset name, IEMOCAP or MELD or DailyDialog')

parser.add_argument('--windowp', type=int, default=1,
                    help='context window size for constructing edges in graph model for past utterances')

parser.add_argument('--windowf', type=int, default=0,
                    help='context window size for constructing edges in graph model for future utterances')

parser.add_argument('--lr', type=float, default=2e-5, metavar='LR', help='learning rate')

parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')

parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')

parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')


args = parser.parse_args()

encoding = "multi"

if encoding is not None:
    print('Positional Encoding: ' + encoding)
else:
    print('Positional Encoding is None')

if device == "cuda":
    print('Running on GPU')
else:
    print('Running on CPU')

print(20 * '=', "Preprocessing data", 20 * '=')
print("\t* building iterators")
n_epochs = args.epochs
batch_size = args.batch_size
train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec = get_IEMOCAP_loaders(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args = args)
