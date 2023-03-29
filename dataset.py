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
        
        self.max_data_len = self._get_max_data_len(self.data_list)
        
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
    
    def _get_max_data_len(self, data_list):
        data_len_list = [len(data) for data in data_list]
        max_data_len = max(data_len_list)
        
        return max_data_len
    
    def _get_vocab(self):
        vocab = [i for i in range(21, 109)]
        vocab = ['<pad>', '<start>', '<end>', 0] + vocab
        
        token2idx = {tok: idx for idx, tok in enumerate(vocab)}
        return vocab, token2idx
    
    def _add_pad_short_part_of_pair(self, pair):
        [call, res] = pair
        
        len_diff = len(call) - len(res)
        num_pad_to_add = abs(len_diff)
        
        if len_diff == 0:
            return pair
        
        if len_diff < 0:
            call += ['<pad>'] * num_pad_to_add
        else:
            res += ['<pad>'] * num_pad_to_add
            
        return [call, res]
    
    def _add_pad_accord_max_data_len(self, pair):
        assert len(pair[0]) == len(pair[1]), 'pair len not matched'
        
        pair_len = len(pair[0])
        num_pad_to_add = (self.max_data_len + 2) - pair_len # 2 is num of special tokens(start, end)
        
        return [part + (['<pad>'] * num_pad_to_add) for part in pair ]
     
    def __len__(self):
        return len(self.pair_list)
    
    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        pair_token_attached = [['<start>'] + pitch_list + ['<end>'] for pitch_list in pair]
        pair_token_padded = self._add_pad_short_part_of_pair(pair_token_attached)
        pair_token_padded = self._add_pad_accord_max_data_len(pair_token_padded)
        
        pair_in_idx = [ [self.token2idx[token] for token in token_list] for token_list in pair_token_padded ]
        
        pair_in_idx_tensor = torch.tensor(pair_in_idx)
        
        return pair_in_idx_tensor[0, :], pair_in_idx_tensor[1, :]