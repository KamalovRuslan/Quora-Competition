import pandas as pd
import numpy as np
import torch

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import SimpleBiLSTMBaseline
from loader import BatchWrapper

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

TRAIN_DIR = '../data/train'
TEST_CSV = '../data/test/test.csv'

import argparse
parser = argparse.ArgumentParser(description='PyTroch Quora Competition')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                    help='batch size (default: 16)')
parser.add_argument('--emb_dim', type=int, default=100, metavar='ED',
                    help='embedding dimension (default: 100)')
parser.add_argument('--hdim', type=int, default=500, metavar='HD',
                    help='hidden unit dimension (default: 500)')
parser.add_argument('--num_layers', type=int, default=3, metavar='NL',
                    help='number of model tail layers (default: 3)')
parser.add_argument('--vectors', type=str, default="glove.6B.100d", metavar='PV',
                    help='pretrained vectors model (default: glove.6B.100d)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)

tv_datafields = [("qid", None),
                 ("question_text", TEXT),
                 ("target", LABEL)]

trn, vld = TabularDataset.splits(
        path=TRAIN_DIR,
        train='train.csv', validation="val.csv",
        format='csv',
        skip_header=True,
        fields=tv_datafields)

tst_datafields = [("qid", None),
                 ("question_text", TEXT)
]

tst = TabularDataset(
        path=TEST_CSV,
        format='csv',
        skip_header=True,
        fields=tst_datafields)

TEXT.build_vocab(trn, vectors=args.vectors)

train_iter, val_iter = BucketIterator.splits(
        (trn, vld),
        batch_sizes=(args.batch_size, args.batch_size),
        device='cuda:0' if args.cuda else -1,
        sort_key=lambda x: len(x.question_text),
        sort_within_batch=False,
        repeat=False
)
test_iter = Iterator(tst, batch_size=64, device='cuda:0' if args.cuda else -1,
                    sort=False, sort_within_batch=False, repeat=False)

train_dl = BatchWrapper(train_iter, "question_text", "target")
valid_dl = BatchWrapper(val_iter, "question_text", "target")
test_dl = BatchWrapper(test_iter, "question_text", None)

print("Build model...")
model = SimpleBiLSTMBaseline(hidden_dim=args.hdim,
                             emb_dim=args.emb_dim, emb_len=len(TEXT.vocab),
                             num_linear=args.num_layers)
criterion = nn.BCELoss()
if args.cuda:
    model.cuda()
    criterion.cuda()
    print("Model on GPU!")
else:
    print("Model on CPU!")

opt = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    running_loss = 0.0
    running_corrects = 0
    model.train()
    for x, y in tqdm(train_dl, total=len(train_dl)):
        opt.zero_grad()

        preds = model(x)
        loss = criterion(preds, y)

        loss.backward()
        opt.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(trn)

    val_loss = 0.0
    model.eval()
    for x, y in valid_dl:
        preds = model(x)
        loss = criterion(preds, y)
        val_loss += loss.item() * x.size(0)

    val_loss /= len(vld)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print("Saving model...")
    torch.save(model.state_dict(), os.path.join('models', 'model.pth'))
