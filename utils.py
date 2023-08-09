import subprocess
from collections import Counter

import torch
import numpy as np

import muspy
import pypianoroll as pypi

UNIT_AS_TICK = 120 # 16th note as tick
BEAT_AS_TICK = 4 * UNIT_AS_TICK # 4th note as tick

# UNIT_AS_TICK = 60 # 32th note as tick
# BEAT_AS_TICK = 8 * UNIT_AS_TICK # 4th note as tick

BAR_AS_TICK = 4 * BEAT_AS_TICK

relative_semitones_major = {
    'F#': -6,
    'Gb': -6,
    'G': -5,
    'G#': -4,
    'Ab': -4,
    'A': -3,
    'A#': -2,
    'Bb': -2,
    'B': -1,
    'C': 0,
    'C#': 1,
    'Db': 1,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'F': 5
}

relative_semitones_minor = {
    'D#': -6,
    'Eb': -6,
    'E': 5,
    'F': 4,
    'F#': -3,
    'Gb': -3,
    'G': -2,
    'G#': -1,
    'Ab': -1,
    'A': 0,
    'A#': 1,
    'Bb': 1,
    'B': 2,
    'C': 3,
    'C#': 4,
    'Db': 4,
    'D': 5,
}

def pianoroll2binaryroll(pianoroll):
    new_pianoroll = []

    for timeframe in pianoroll:
        new_timeframe = []
        
        for velocity in timeframe:
            if velocity > 0 :
                new_timeframe.append(1)
            else:
                new_timeframe.append(0)   
                
        new_pianoroll.append(new_timeframe)
    
    return new_pianoroll

def binaryroll2pitchlist(binaryroll): # also remove note overwrapping
    pitch_list = []

    for timeframe in binaryroll:
        prev_pitch = 0
        
        sum_timeframe = sum(timeframe)
        
        if sum(timeframe) == 0:
            pitch_list.append(0)
            continue
        
        for pitch, is_on in enumerate(timeframe):
            if sum_timeframe == 1 and is_on == 1:
                pitch_list.append(pitch + 1)
                prev_pitch = pitch + 1
                break
            
            if sum_timeframe > 1 and is_on == 1 and pitch + 1 != prev_pitch:
                pitch_list.append(pitch + 1)
                prev_pitch = pitch + 1
                break
    
    return pitch_list

def pitchlist2pianoroll(pitch_list):
    new_pianoroll = []

    for pitch in pitch_list:
        new_timeframe = [0 for i in range(128)]
        if pitch > 0:
            new_timeframe[pitch - 1] = 127
        new_pianoroll.append(new_timeframe)
    
    return np.array(new_pianoroll)

def shortpitchlist2pitchlist(short_pitch_list):
    pitch_list = []
    
    for pitch in short_pitch_list:
        for _ in range(UNIT_AS_TICK):
            pitch_list.append(pitch)
            
    return pitch_list

def pitchlist2shortpitchlist(pitch_list):
    short_pitch_list = []
    
    for i in range(0, len(pitch_list), UNIT_AS_TICK):
        sliced = pitch_list[i : i + UNIT_AS_TICK]
        short_pitch_list.append(sliced[0])
        
    return short_pitch_list

def add_padding_bar(pitch_list):
    ticks_in_pianoroll = len(pitch_list)

    num_bars_in_pianoroll = ticks_in_pianoroll // BAR_AS_TICK
    left_over_ticks_in_pianoroll = ticks_in_pianoroll % BAR_AS_TICK

    num_ticks_to_add = BAR_AS_TICK - left_over_ticks_in_pianoroll

    ticks_to_add_array = np.array([0 for _ in range(num_ticks_to_add)])

    pitch_list = np.concatenate((pitch_list, ticks_to_add_array), axis=0)
    
    return pitch_list

def midi2audio(midi_file, audio_file, sr=44100):
    subprocess.call(['/usr/bin/fluidsynth', midi_file, '-F', audio_file, '-r', str(sr)])
    
    
def prepare_inference_res(token2idx):
    start_only = ["<start>"]
    start_only = [token2idx[tk] for tk in start_only]
    return torch.tensor(start_only)
    

def prepare_inference_midi(midi_path, token2idx):
    song = muspy.read(f'{midi_path}')

    song_attrs = midi_path.name.split('.')[0].split('-')

    # clean up bpm
    bpm = float(song_attrs[3])
    new_tempos = np.array([muspy.Tempo(0, bpm)])
    
    song.tempos = new_tempos

    # transpose to C major / A minor 
    key_and_mode = song_attrs[2]

    semitone_dict = relative_semitones_minor if 'm' in key_and_mode else relative_semitones_major

    orig_key = key_and_mode[:-1] if len(key_and_mode) > 1 else key_and_mode

    song.transpose(-1 * semitone_dict[orig_key])
    
    # quantize process
    pianoroll_obj = muspy.to_pypianoroll(song)

    quantized_pianoroll_obj = pianoroll_obj.copy()
    before_quantized_pianoroll = quantized_pianoroll_obj.tracks[0].pianoroll
    before_quantized_pianoroll_binaryroll = pianoroll2binaryroll(before_quantized_pianoroll)
    before_quantized_pianoroll_pitch_list = binaryroll2pitchlist(before_quantized_pianoroll_binaryroll)

    len_pitch_list = len(before_quantized_pianoroll_pitch_list)

    quantized_pianoroll_short_pitch_list = []

    for i in range(0, len_pitch_list, UNIT_AS_TICK):
        sliced = before_quantized_pianoroll_pitch_list[i : i + UNIT_AS_TICK]

        pitch_cnt_dict = dict(Counter(sliced))
        
        max_pitch_cnt_value = max(pitch_cnt_dict.values())
        max_pitch_cnt_pitch = [key for key, val in pitch_cnt_dict.items() if val == max_pitch_cnt_value][0]
        
        quantized_pianoroll_short_pitch_list.append(max_pitch_cnt_pitch)
    
    # attach token
    call_token_attached = quantized_pianoroll_short_pitch_list
    call_ready = [token2idx[token] for token in call_token_attached]
    
    quantized_pianoroll = pitchlist2pianoroll(shortpitchlist2pitchlist(quantized_pianoroll_short_pitch_list))
    quantized_pianoroll_obj.tracks[0].pianoroll = quantized_pianoroll

    return [pianoroll_obj, quantized_pianoroll_obj], torch.tensor(call_ready) # return original pianoroll also for convenience

def model_output_to_pianoroll(output, idx2token, original_pianoroll_obj):
    token_list = [idx2token[idx] for idx in output.tolist()]
    short_pitch_list = [token for token in token_list if token != '<pad>' and token != '<start>' and token != '<end>']

    pitch_list = shortpitchlist2pitchlist(short_pitch_list)
    pianoroll = pitchlist2pianoroll(pitch_list)
    
    result_pianoroll_obj = make_empty_pianoroll(original_pianoroll_obj)
    
    result_pianoroll_obj.tracks[0].pianoroll = pianoroll
    
    return result_pianoroll_obj

def pianoroll2muspy(pianoroll_obj):
    song = muspy.from_pypianoroll(pianoroll_obj)
    return song

def prepare_compare_res(midi_path, token2idx, idx2token):
    [original_pianoroll_obj, quantized_pianoroll_obj], res = prepare_inference_midi(midi_path, token2idx)
    compare_res = model_output_to_pianoroll(res, idx2token, quantized_pianoroll_obj)
    
    return compare_res

def make_empty_pianoroll(original_obj):
    resolution = original_obj.resolution
    tempo = original_obj.tempo
    tracks = [pypi.Track()]
    
    return pypi.Multitrack(resolution=resolution, tempo=tempo, tracks=tracks)