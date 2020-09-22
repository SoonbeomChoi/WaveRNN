import math
import numpy as np
import librosa
from utils import hparams as hp
from scipy.signal import lfilter


def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def load_wav(path):
    return librosa.load(path, sr=hp.sample_rate)[0]


def save_wav(x, path):
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.sample_rate)


def split_signal(x):
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2**15


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)

'''
def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)
'''

def normalize(S, max_s):
    S = np.power(S/max_s, hp.preemphasis)
    S = np.clip(S, 0, 1)

    return S


def denormalize(S):
    max_s = np.load(hp.max_s_path)
    S = np.power(S, hp.deemphasis/hp.preemphasis)
    S = max_s*S

    return S


def db_normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)


def db_denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y):
    D = stft(y)
    if hp.norm_type == 'db':
        S = amp_to_db(np.abs(D)) - hp.ref_level_db
        S = db_normalize(S)
    elif hp.norm_type == 'linear':
        max_s = db_to_amp(hp.ref_level_db)
        S = np.abs(D)
        S = normalize(S, max_s)
    elif hp.norm_type == 'max':
        max_s = np.load(hp.max_s_path)
        S = np.abs(D)
        S = normalize(S, max_s)

    return S


def melspectrogram(y):
    D = stft(y)
    if hp.norm_type == 'db':
        S = amp_to_db(linear_to_mel(np.abs(D)))
        S = db_normalize(S)
    elif hp.norm_type == 'linear':
        S = linear_to_mel(np.abs(D))
        S = normalize(S, 1.0)
    elif hp.norm_type == 'max':
        max_s = np.load(hp.max_s_path)
        S = linear_to_mel(np.abs(D))
        S = normalize(S, max_s)

    return S


def stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)


def pre_emphasis(x):
    return lfilter([1, -hp.preemphasis], [1], x)


def de_emphasis(x):
    return lfilter([1], [1, -hp.preemphasis], x)


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels: y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

def reconstruct_waveform(mel, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    denormalized = denormalize(mel)
    amp_mel = db_to_amp(denormalized)
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel, power=1, sr=hp.sample_rate,
        n_fft=hp.n_fft, fmin=hp.fmin)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hp.hop_length, win_length=hp.win_length)
    return wav

