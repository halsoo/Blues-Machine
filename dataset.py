from pathlib import Path

import torch

import muspy
import pypianoroll as pypir

from tqdm import tqdm

from utils import pianoroll2binaryroll, binaryroll2pitchlist, pitchlist2shortpitchlist
from consts import PATH_PREFIX, PATH_PREFIX_Q

class BluesPairSet:
  def __init__(self, data_dir_path=PATH_PREFIX_Q):
    midi_path_list = sorted(list(Path(data_dir_path).rglob('*.mid')))
    self.data_list = [self._get_short_pitch_list(midi_path) for midi_path in tqdm(midi_path_list)]
    self.pair_list = self._get_pair_list(self.data_list)
    
    self.max_part_len_list = self._get_max_part_len(self.pair_list)
    
    self.idx2token, self.token2idx = self._get_vocab()
    self.vocab_size = len(self.idx2token)

  def _get_short_pitch_list(self, midi_path):
    song = muspy.read(f'{midi_path}')

    pianoroll_obj = muspy.to_pypianoroll(song)
    pianoroll = pianoroll_obj.tracks[0].pianoroll

    pianoroll_binaryroll = pianoroll2binaryroll(pianoroll)
    pianoroll_pitch_list = binaryroll2pitchlist(pianoroll_binaryroll)

    short_pitch_list = pitchlist2shortpitchlist(pianoroll_pitch_list)
    
    return short_pitch_list

  def _get_pair_list(self, data_list):
    assert len(data_list) % 2 == 0, "num of call and response not matched"
    
    pair_list = []
    
    for i in range(0, len(data_list), 2):
      data_pair = data_list[i:i+2]
      pair_list.append(data_pair)
        
    return pair_list
  
  def _get_max_part_len(self, pair_list):
    # 0: list of calls, 1: list of reses
    pair_part_list = [[pair[0] for pair in pair_list], [pair[1] for pair in pair_list]] 
    
    pair_part_len_list = [[len(part) for part in part_list] for part_list in pair_part_list]
    
    # 0: call max len, 1: res max len
    max_len_list = [max(part_len_list) for part_len_list in pair_part_len_list]
    
    return max_len_list
    
  def _get_vocab(self):
    vocab = [i for i in range(21, 109)]
    vocab = ['<pad>', '<start>', '<end>', 0] + vocab
    
    token2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return vocab, token2idx
    
  def _add_pad(self, idx, part):
    part_max_len = self.max_part_len_list[idx]
    part_len = len(part)
    num_pad_to_add = part_max_len - part_len
    
    if idx == 1:
      num_pad_to_add += 2
    
    return part + (['<pad>'] * num_pad_to_add)

  def __len__(self):
    return len(self.pair_list)
    
  def __getitem__(self, idx):
    [call, res] = self.pair_list[idx]

    res_token_attached = ['<start>'] + res + ['<end>']

    pair_padded = [self._add_pad(idx, part) for idx, part in enumerate([call, res_token_attached])]

    pair_in_idx = [ [self.token2idx[token] for token in token_list] for token_list in pair_padded ]

    return torch.tensor(pair_in_idx[0]), torch.tensor(pair_in_idx[1])
    

def make_collate_fn(device):
    
  def collate_fn(raw_batch):
    call_list = []
    res_list = []
    
    for pair in raw_batch:
        call_list.append(pair[0])
        res_list.append(pair[1])
        
    call_list = [item.to(device) for item in call_list]
    res_list = [item.to(device) for item in res_list]
    
    return torch.stack(call_list), torch.stack(res_list)
  
  return collate_fn