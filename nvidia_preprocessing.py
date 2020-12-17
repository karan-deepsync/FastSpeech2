import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.util import read_wav_np
from utils.hparams import HParam
from dataset.audio.pitch_mod import Dio
from utils.util import str_to_int_list
from dataset.audio.energy import Energy



def preprocess(data_path, hp, file):
    stft = TacotronSTFT(
        filter_length=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mels,
        sampling_rate=hp.audio.sample_rate,
        mel_fmin=hp.audio.fmin,
        mel_fmax=hp.audio.fmax,
    )
    pitch = Dio(J = hp.audio.cwt_bins)
    energy = Energy(J = hp.audio.e_cwt_bins)


    wav_files = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    mel_path = os.path.join(hp.data.data_dir, "mels")
    energy_path = os.path.join(hp.data.data_dir, "energy")
    pitch_path = os.path.join(hp.data.data_dir, "pitch")
    pitch_avg_path = os.path.join(hp.data.data_dir, "p_avg")
    pitch_std_path = os.path.join(hp.data.data_dir, "p_std")
    pitch_cwt_coefs = os.path.join(hp.data.data_dir, "p_cwt_coef")
    energy_avg_path = os.path.join(hp.data.data_dir, "e_avg")
    energy_std_path = os.path.join(hp.data.data_dir, "e_std")
    energy_cwt_coefs = os.path.join(hp.data.data_dir, "e_cwt_coef")
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(energy_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)
    os.makedirs(pitch_avg_path, exist_ok=True)
    os.makedirs(pitch_std_path, exist_ok=True)
    os.makedirs(pitch_cwt_coefs, exist_ok=True)
    os.makedirs(energy_avg_path, exist_ok=True)
    os.makedirs(energy_std_path, exist_ok=True)
    os.makedirs(energy_cwt_coefs, exist_ok=True)

    print("Sample Rate : ", hp.audio.sample_rate)

    with open("{}".format(file), encoding="utf-8") as f:
        _metadata = [line.strip().split("|") for line in f]


    for metadata in tqdm.tqdm(_metadata, desc="preprocess wav to mel"):
        wavpath = os.path.join(data_path, metadata[4])
        sr, wav = read_wav_np(wavpath, hp.audio.sample_rate)
        input_wav = torch.from_numpy(wav)

        dur = str_to_int_list(metadata[2])
        dur = torch.from_numpy(np.array(dur))

        p, avg, std, p_coef = pitch.forward(input_wav, durations = dur)  # shape in order - (T,) (no of utternace, ), (no of utternace, ), (10, T)
        #print(p.shape, avg.shape, std.shape, p_coef.shape)

        wav = torch.from_numpy(wav).unsqueeze(0)
        mel, mag = stft.mel_spectrogram(wav)  # mel [1, 80, T]  mag [1, num_mag, T]
        mel = mel.squeeze(0)  # [num_mel, T]
        mag = mag.squeeze(0)  # [num_mag, T]
        e = torch.norm(mag, dim=0)  # [T, ]
        e, e_avg, e_std, e_coef = energy.forward(e)
        #print(e.shape, e_avg.shape, e_std.shape, e_coef.shape)


        id = os.path.basename(wavpath).split(".")[0]

        assert(e.shape == p.shape)

        np.save("{}/{}.npy".format(mel_path, id), mel.numpy(), allow_pickle=False)


        np.save("{}/{}.npy".format(pitch_path, id), p, allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_avg_path, id), avg, allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_std_path, id), std, allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_cwt_coefs, id), p_coef.reshape(-1, hp.audio.cwt_bins), allow_pickle=False)

        np.save("{}/{}.npy".format(energy_path, id), e, allow_pickle=False)
        np.save("{}/{}.npy".format(energy_avg_path, id), e_avg, allow_pickle=False)
        np.save("{}/{}.npy".format(energy_std_path, id), e_std, allow_pickle=False)
        np.save("{}/{}.npy".format(energy_cwt_coefs, id), e_coef.reshape(-1, hp.audio.e_cwt_bins), allow_pickle=False)



def main(args, hp):
    print("Preprocess Training dataset :")
    preprocess(args.data_path, hp, hp.data.train_filelist)
    print("Preprocess Validation dataset :")
    preprocess(args.data_path, hp, hp.data.valid_filelist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="root directory of wav files"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
