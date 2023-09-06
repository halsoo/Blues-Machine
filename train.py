import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Encoder, Decoder, Seq2Seq
from dataset import BluesPairSet, make_collate_fn
from trainer import Trainer


dataset = BluesPairSet()

DEVICE = 'cuda'

EMB_LEN = dataset.vocab_size
EMB_DIM = 64
HID_DIM = 256
NUM_LAYER = 3
PAD_IDX = dataset.token2idx['<pad>']

encoder = Encoder(EMB_LEN, EMB_DIM, HID_DIM, num_layers=NUM_LAYER, device=DEVICE) # input size(embed len), embed size, hidden size
decoder = Decoder(EMB_LEN, EMB_DIM, HID_DIM, num_layers=NUM_LAYER, device=DEVICE) # output size(embed len), embed size, hidden size

model = Seq2Seq(encoder, decoder, device=DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

collate = make_collate_fn(DEVICE)

trainset, validset = torch.utils.data.random_split(dataset, [42, 10])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, collate_fn=collate)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=10, shuffle=False, collate_fn=collate)

trainer = Trainer(model, optimizer, loss_fn, train_loader, valid_loader, device=DEVICE)