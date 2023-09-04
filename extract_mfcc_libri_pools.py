import multiprocessing
import random
import librosa
import os
import glob
import numpy as np


def helper_read_libri(path_to_Libri_data: str) -> dict:
    """Read paths to audio and create dict, where key is a speaker label

    :param path_to_Libri_data:
    :return:
    """
    path = path_to_Libri_data
    spk_to_utts = dict()

    spk_list = os.listdir(path)

    for idx in spk_list:
        temp_path = os.path.join(path, idx)
        temp_path2 = os.listdir(temp_path)
        temp_path4 = []
        for dir2 in temp_path2:
            temp_path3 = os.path.join(temp_path, dir2)
            temp_path4 = temp_path4 + (glob.glob(temp_path3 + '//*.flac'))
            # temp_path5.append(temp_path4)
        spk_to_utts[idx] = temp_path4

    return spk_to_utts


def mfcc_extractor(audio_file):
    """Extract MFCC from an PCM file, shape=(TIME_FRAMES, MFCC_FEATURES).
    :param audio_file: path to audio (WAV/FLAC)
    :return:
    """
    pcm, sample_rate = librosa.load(audio_file[0])

    # Convert to mono-channel.
    if len(pcm.shape) == 2:
        pcm = librosa.to_mono(pcm.transpose())

    # Convert to 16kHz.
    if sample_rate != 16000:
        pcm = librosa.resample(pcm, orig_sr=sample_rate, target_sr=16000)
    features = librosa.feature.mfcc(y=pcm, sr=sample_rate, n_mfcc=40)

    return features.transpose()


def rand_spk_path_wav(spk_uttr: dict):
    """chose one PCM path"""
    spk = random.sample(list(spk_uttr.keys()), 1)
    spk_path_wav = random.sample(spk_uttr[spk[0]], 1)
    return spk_path_wav

def trim_mfcc_mat(mfcc_mat, SEQ_LEN):
    new_mfcc_mat = mfcc_mat[:SEQ_LEN, :]
    return new_mfcc_mat

def get_mfcc_spk(spk_uttr: dict, SEQ_LEN):
    spk = rand_spk_path_wav(spk_uttr)
    mfcc_mat = mfcc_extractor(spk)
    while(mfcc_mat.shape[0]<SEQ_LEN):
        spk = rand_spk_path_wav(spk_uttr)
        mfcc_mat = mfcc_extractor(spk)
    mfcc_trimed_mat = trim_mfcc_mat(mfcc_mat, SEQ_LEN)
    return mfcc_trimed_mat


class FetchMFCC:
    """Class for multiprocessing pool.map

    """
    def __init__(self, spk_uttr: dict, SEQ_LEN: int):
        self.spk_uttr = spk_uttr
        self.seq_len = SEQ_LEN

    def __call__(self, _):
        mfcc = get_mfcc_spk(self.spk_uttr, self.seq_len)
        return mfcc


spk_uttr = helper_read_libri(
    os.path.expanduser("~") + "/data/LibriSpeech/train-clean-100")
n_CPU = multiprocessing.cpu_count()
BACH_SIZE = 4
SEQ_LEN = 200

"""Example of use
"""
with multiprocessing.Pool(n_CPU) as pool:
    fecthMFCC = FetchMFCC(spk_uttr, SEQ_LEN)
    mfcc_arrays = pool.map(fecthMFCC, range(BACH_SIZE))
    mfcc_data = np.array(mfcc_arrays)

stop = 0
