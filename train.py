# config
import sys, getopt
from omegaconf import OmegaConf

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# models
from model import Encoder, Decoder, Seq2Seq
from dataset import BluesPairSet, make_collate_fn
from trainer import Trainer

# wandb
import wandb

def getConfs(argv):
  args = argv[1:]
  
  try:
    opt_list, _ = getopt.getopt(args, 'f:n:p:e:')
    
  except getopt.GetoptError:
    print('somethings gone wrong')
    return None

  opt_dict = {}
  
  for opt in opt_list:
    name, value = opt 
    name = name.replace('-', '')
    
    if value:
      opt_dict[name] = value
    else:
      opt_dict[name] = name
    
  conf = OmegaConf.load(opt_dict['f'])
  
  if opt_dict.get('e'):
    num_epoch_arg = int(opt_dict['e'])
    conf.num_epoch = num_epoch_arg
    
  if opt_dict.get('n'):
    name_arg = opt_dict['n']
    conf.run_name = name_arg
    
  if opt_dict.get('p'):
    project_arg = opt_dict['p']
    conf.project_name = project_arg
  
  return conf

def train(argv):
  
  conf = getConfs(argv)
  
  dataset = BluesPairSet()

  # Device conf
  DEVICE = conf.device
  
  # RNN conf
  EMB_LEN = dataset.vocab_size
  EMB_DIM = conf.rnn.emb_dim
  HID_DIM = conf.rnn.hid_dim
  NUM_LAYER = conf.rnn.num_layer
  
  # Data conf
  PAD_IDX = dataset.token2idx[conf.data.pad_token]
  TRAIN_SPLIT = conf.data.train.split
  TRAIN_BATCH_SIZE = conf.data.train.batch_size
  VALID_SPLIT = conf.data.valid.split
  VALID_BATCH_SIZE = conf.data.valid.batch_size
  
  # Train conf
  LR = conf.lr
  NUM_EPOCH = conf.num_epoch
  
  hyperparams = { 
    'rnn_emb_len': EMB_LEN,
    'rnn_emb_dim': EMB_DIM, 
    'rnn_hid_dim': HID_DIM,
    'rnn_num_layer': NUM_LAYER, 
    'train_set_len': TRAIN_SPLIT, 
    'train_batch_size': TRAIN_BATCH_SIZE, 
    'valid_set_len': VALID_SPLIT, 
    'valid_batch_size': VALID_BATCH_SIZE,
    'epoch': conf.num_epoch,
    'learning_rate': LR
  }
  
  wandb_run = wandb.init(
    project=conf.project_name,
    name=conf.run_name,
    config=hyperparams
  )

  encoder = Encoder(EMB_LEN, EMB_DIM, HID_DIM, num_layers=NUM_LAYER, device=DEVICE) # input size(embed len), embed size, hidden size
  decoder = Decoder(EMB_LEN, EMB_DIM, HID_DIM, num_layers=NUM_LAYER, device=DEVICE) # output size(embed len), embed size, hidden size

  model = Seq2Seq(encoder, decoder, device=DEVICE)

  optimizer = optim.Adam(model.parameters(), lr=LR)
  loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  
  collate_fn = make_collate_fn(DEVICE)

  trainset, validset = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, VALID_SPLIT])
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
  valid_loader = torch.utils.data.DataLoader(validset, batch_size=VALID_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

  trainer = Trainer(model, optimizer, loss_fn, train_loader, valid_loader, device=DEVICE, wandb=wandb_run)
  
  trainer.train_by_num_epoch(NUM_EPOCH, do_validate=True)

if __name__ == '__main__':
  train(sys.argv)
  