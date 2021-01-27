import os
import numpy as np
import librosa
import torch
import torchaudio
from torchaudio.transforms import Resample

from midi_xt import load_midi

config = {
    "dataset_path": '../Dataset/sk_multi_singer',
    "singer_list": ['saebyul', 'female_style1', 'female_style2', 'male_style1', 'male_style2', 'younha', 'swja', 'gwangseok'],
    "save_path": '../Dataset/sk_multi_singer_split',
    "sample_rate": 44100,

    # Split audio by amp
    "fft_size": 1024,
    "hop_size": 256,
    "amp_threshold": 0.15,
    "min_length_amp": 3.0, # Min audio length in second
    "low_length": 60, # Find low rmse range after min length
    "low_ratio": 1.0, # Low energy ratio within min length

    # Split audio by MIDI
    "min_length_midi": 2.5, # Min audio length in second
    "max_length_midi": 15.0, # Max audio length in second
    "min_silence": 0.3, # Min silence length in second
    "max_silence": 1.0, # Max silence length in second
    "offset_threshold": 0.2, 
    "mono": True
}

class Range(object):
    def __init__(self, start=None, duration=None, sample_rate=44100):
        if start is None:
            self.start = 0
        else:
            self.start = int(sample_rate*start)

        if duration is None:
            self.end = start
            self.duration = 0
        else:
            self.end = int(sample_rate*(start + duration))
            self.duration = self.end - self.start
    
    def to_arange(self):
        return torch.arange(self.start, self.end)


def load(filename, sample_rate):
    y, source_rate = torchaudio.load(filename)
    if source_rate != sample_rate:
        resample = Resample(source_rate, sample_rate)
        y = resample(y)

    return y


def get_rmse(filename, sample_rate=22050, fft_size=1024, hop_size=256):
    y, sr = librosa.load(filename, sr=sample_rate)
    S, phase = librosa.magphase(librosa.stft(y, n_fft=fft_size, hop_length=hop_size))
    rmse = np.sqrt(np.mean(np.abs(S)**2, axis=0, keepdims=True))[0]

    return y, rmse


def split_audio_by_amp(filename, set_name='train'):
    y, rmse = get_rmse(filename, config["sample_rate"], config["fft_size"], config["hop_size"])
    min_length = int(config["sample_rate"]*config["min_length_amp"])

    y_split = y[:config["hop_size"]]
    pre_lows = 0
    post_lows = 0
    index = 0
    for i in range(1, rmse.shape[0]):
        y_next = y[i*config["hop_size"]:(i+1)*config["hop_size"]]

        if y_split.shape[0] < min_length: # Mininum audio length
            if rmse[i-1] < config["amp_threshold"] and rmse[i] < config["amp_threshold"]:
                pre_lows += 1

            y_split = np.append(y_split, y_next)

        elif y_split.shape[0] > min_length and pre_lows < config["low_ratio"]*min_length/config["hop_size"]: # When audio is longer then mininum length
            if rmse[i-1] < config["amp_threshold"] and rmse[i] < config["amp_threshold"]: # Check low energy
                post_lows += 1
            
            if post_lows < config["low_length"]:
                y_split = np.append(y_split, y_next)
            else:
                y_split = torch.from_numpy(y_split).unsqueeze(0)

                singer = filename.split('/')[-3]
                basename = os.path.basename(filename).replace('.wav', '_' + str(index) + '.wav')
                savename = os.path.join(config["save_path"], set_name, singer, basename)
                torchaudio.save(savename, y_split, config["sample_rate"])

                y_split = y_next
                pre_lows = 0
                post_lows = 0
                index += 1


def split_audio_by_midi(filename, midi, set_name='train'):
    y = load(filename, config["sample_rate"])
    y_range = Range()

    offset_threshold = int(config["sample_rate"]*config["offset_threshold"])
    min_silence = int(config["sample_rate"]*config["min_silence"])
    max_silence = int(config["sample_rate"]*config["max_silence"])
    min_length = int(config["sample_rate"]*config["min_length_midi"])
    max_length = int(config["sample_rate"]*config["max_length_midi"])

    index = 0
    for i in range(len(midi)):
        # Remove MIDI overlap
        if i < len(midi) - 1:
            if midi[i][0] + midi[i][2] > midi[i+1][0]:
                midi[i][2] = midi[i+1][0] - midi[i][0]

        prev_range = Range(0, 0, config["sample_rate"])
        curr_range = Range(midi[i][0], midi[i][2], config["sample_rate"])
        next_range = Range(midi[-1][0] + midi[-1][2], None, config["sample_rate"])
        if i > 0:
            prev_range = Range(midi[i-1][0], midi[i-1][2], config["sample_rate"])
        if i < len(midi) - 1:
            next_range = Range(midi[i+1][0], midi[i+1][2], config["sample_rate"])

        if i == 0 and curr_range.start > max_silence:
            y_range.start = curr_range.start - max_silence
        
        split = False
        # Conditions to split audio
        if next_range.end - y_range.start > max_length:
            split = True
        elif curr_range.end - y_range.start < min_length:
            if next_range.start - curr_range.end > max_silence:
                split = True
            else:
                split = False
        else:
            if next_range.start - curr_range.end < min_silence:
                split = False
            else:
                split = True

        if i == len(midi) - 1:
            split = True

        if split:
            y_range.end = curr_range.end
            if next_range.start - curr_range.end > offset_threshold:
                y_range.end = curr_range.end + offset_threshold

            # Save audio
            y_split = y[...,y_range.start:y_range.end]
            y_split = y_split[0].unsqueeze(0)

            singer = filename.split('/')[-3]
            basename = os.path.basename(filename).replace('.wav', '_' + str(index) + '.wav')
            savename = os.path.join(config["save_path"], set_name, singer, basename)
            torchaudio.save(savename, y_split, config["sample_rate"])
            index += 1

            # Initialize next audio range
            y_range.start = curr_range.end
            if next_range.start - curr_range.end > offset_threshold:
                y_range.start = curr_range.end + offset_threshold
            if next_range.start - curr_range.end > max_silence:
                y_range.start = next_range.start - max_silence


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def read_file_list(filename):
    with open(filename) as f:
        file_list = f.read().splitlines()

    return file_list


def main():
    create_path(config["save_path"])
    create_path(os.path.join(config["save_path"], 'train'))
    create_path(os.path.join(config["save_path"], 'valid'))
    for singer in config["singer_list"]:
        singer_path = os.path.join(config["dataset_path"], singer)
        train_list_file = os.path.join(singer_path, 'train_list.txt')
        valid_list_file = os.path.join(singer_path, 'valid_list.txt')

        train_list = read_file_list(train_list_file)
        valid_list = read_file_list(valid_list_file)

        create_path(os.path.join(config["save_path"], 'train', singer))
        create_path(os.path.join(config["save_path"], 'valid', singer))

        for basename in train_list:
            wav_file = os.path.join(singer_path, 'wav', basename + '.wav')
            mid_file = os.path.join(singer_path, 'mid', basename + '.mid')
            if os.path.exists(mid_file):
                midi = load_midi(mid_file)
                split_audio_by_midi(wav_file, midi, set_name='train')
            else:
                split_audio_by_amp(wav_file, set_name='train')

            print(basename)

        for basename in valid_list:
            wav_file = os.path.join(singer_path, 'wav', basename + '.wav')
            mid_file = os.path.join(singer_path, 'mid', basename + '.mid')
            if os.path.exists(mid_file):
                midi = load_midi(mid_file)
                split_audio_by_midi(wav_file, midi, set_name='valid')
            else:
                split_audio_by_amp(wav_file, set_name='valid')

            print(basename)

    print("Splitted audio saved to %s." % (config["save_path"]))

if __name__ == '__main__':
    main()