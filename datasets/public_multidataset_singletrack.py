import torch.nn.functional as F
import pandas as pd
import librosa
import pyloudnorm as pyln
import soundfile as sf
import torch
import numpy as np
import os
import glob
import yaml
from pathlib import Path
import random

from utils.data_utils import read_wav_segment, get_audio_length
import torch.distributed as dist

import pickle

from tqdm import tqdm

from utils.data_utils import efficient_roll
from utils.common_audioeffects import (
    AugmentationChain,
    ConvolutionalReverb,
    Compressor,
    Equaliser,
    Panner,
    Haas,
    Gain,
)


def load_audio(file, start=None, end=None, stereo=True, target_fs=44100):
    file = str(file)

    x, fs = read_wav_segment(file, start, end)
    if fs != target_fs:
        x, fs = read_wav_segment(
            file, start, start + int((end - start) * 48000 / 44100)
        )
        # Resample x_dry from fs to self.fs using lib
        x = librosa.resample(x, orig_sr=fs, target_sr=target_fs, axis=0)
        x = torch.from_numpy(x)
        l = end - start
        x = x[:l]

    if stereo:
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
            x = np.concatenate((x, x), axis=-1)
        elif len(x.shape) == 2 and x.shape[-1] == 1:
            x = np.concatenate((x, x), axis=-1)

    x = torch.from_numpy(x).permute(1, 0)

    return x, fs


class MultiDatasetDry(torch.utils.data.IterableDataset):

    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        stereo=True,
        num_examples=4,
        RMS_threshold_dB=-40,
        seed=42,
        tracks="all",
        random_polarity=False,
        random_shift=1024,  # shift in samples to apply to the audio, if random_shift is None, no shift is applied
        mode=None,
        datasets=None,
        probabilities=None,
        datasets_IR=None,
        probabilities_IR=None,
        path_ReverbFx=None,
        path_csv_ReverbFx="/scratch/elec/t412-asp/ReverbFx/meta/train.csv",
        path_QMUL=None,
        path_csv_QMUL=None,
        path_Arni=None,
        path_csv_Arni=None,
        path_ASH=None,
        path_csv_ASH=None,
        path_medleydb=None,
        path_csv_medleydb=None,
        path_IDMT_SMT_DRUMS=None,
        path_csv_IDMT_SMT_DRUMS=None,
        path_GuitarSet=None,
        path_csv_GuitarSet=None,
        path_OpenSinger=None,
        path_csv_OpenSinger=None,
        path_IDMT_SMT_GUITAR=None,
        path_csv_IDMT_SMT_GUITAR=None,
        path_IDMT_SMT_BASS=None,
        path_csv_IDMT_SMT_BASS=None,
        path_Aalto_anechoic=None,
        path_csv_Aalto_anechoic=None,
        apply_effects_on_dry=False,  # if True, the dry signal will be augmented with effects
        p_effects_dry=0.1,
        p_effects_wet=0.5,
    ):

        super().__init__()

        self.apply_effects_on_dry = apply_effects_on_dry
        self.random_polarity = random_polarity
        self.random_shift = random_shift

        self.mode = mode

        self.segment_length = segment_length
        self.fs = fs

        self.stereo = stereo

        self.RMS_threshold_dB = RMS_threshold_dB

        self.get_RMS = lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x**2, dim=-1)))

        assert datasets is not None
        self.datasets = datasets

        assert probabilities is not None, "probabilities must be provided"
        self.probabilities = probabilities
        self.probabilities = [
            self.probabilities[j] / sum(self.probabilities)
            for j in range(len(self.probabilities))
        ]
        self.probabilities = np.array(self.probabilities)
        self.probabilities_cumsum = np.cumsum(self.probabilities)

        self.tracks = tracks
        assert self.tracks == "all", "only all model is implemented for now"

        self.path_medleydb = path_medleydb
        self.path_csv_MDB = path_csv_medleydb
        assert self.path_csv_MDB is not None, "path_csv must be provided"

        self.df_MDB = pd.read_csv(self.path_csv_MDB)
        assert (
            len(self.df_MDB) > 0
        ), f"DataFrame loaded from {self.path_csv_MDB} is empty"

        self.path_IDMT_SMT_DRUMS = path_IDMT_SMT_DRUMS
        self.path_csv_IDMT_SMT_DRUMS = path_csv_IDMT_SMT_DRUMS
        assert (
            self.path_csv_IDMT_SMT_DRUMS is not None
        ), "path_csv_IDMT_SMT_DRUMS must be provided"
        self.df_IDMT_SMT_DRUMS = pd.read_csv(self.path_csv_IDMT_SMT_DRUMS)
        assert (
            len(self.df_IDMT_SMT_DRUMS) > 0
        ), f"DataFrame loaded from {self.path_csv_IDMT_SMT_DRUMS} is empty"

        self.path_GuitarSet = path_GuitarSet
        self.path_csv_GuitarSet = path_csv_GuitarSet
        assert self.path_csv_GuitarSet is not None
        self.df_GuitarSet = pd.read_csv(self.path_csv_GuitarSet)
        assert (
            len(self.df_GuitarSet) > 0
        ), f"DataFrame loaded from {self.path_csv_GuitarSet} is empty"

        self.path_OpenSinger = path_OpenSinger
        self.path_csv_OpenSinger = path_csv_OpenSinger
        assert self.path_csv_OpenSinger is not None
        self.df_OpenSinger = pd.read_csv(self.path_csv_OpenSinger)
        assert (
            len(self.df_OpenSinger) > 0
        ), f"DataFrame loaded from {self.path_csv_OpenSinger} is empty"

        self.path_IDMT_SMT_GUITAR = path_IDMT_SMT_GUITAR
        self.path_csv_IDMT_SMT_GUITAR = path_csv_IDMT_SMT_GUITAR
        assert self.path_csv_IDMT_SMT_GUITAR is not None
        self.df_IDMT_SMT_GUITAR = pd.read_csv(self.path_csv_IDMT_SMT_GUITAR)
        assert (
            len(self.df_IDMT_SMT_GUITAR) > 0
        ), f"DataFrame loaded from {self.path_csv_IDMT_SMT_GUITAR} is empty"

        self.path_IDMT_SMT_BASS = path_IDMT_SMT_BASS
        self.path_csv_IDMT_SMT_BASS = path_csv_IDMT_SMT_BASS
        assert self.path_csv_IDMT_SMT_BASS is not None
        self.df_IDMT_SMT_BASS = pd.read_csv(self.path_csv_IDMT_SMT_BASS)
        assert (
            len(self.df_IDMT_SMT_BASS) > 0
        ), f"DataFrame loaded from {self.path_csv_IDMT_SMT_BASS} is empty"

        self.path_Aalto_anechoic = path_Aalto_anechoic
        self.path_csv_Aalto_anechoic = path_csv_Aalto_anechoic
        assert self.path_csv_Aalto_anechoic is not None
        self.df_Aalto_anechoic = pd.read_csv(self.path_csv_Aalto_anechoic)
        assert (
            len(self.df_Aalto_anechoic) > 0
        ), f"DataFrame loaded from {self.path_csv_Aalto_anechoic} is empty"

        self.audio_length_cache = {}
        self.path_files_cache = {}

        self.datasets_IR = datasets_IR

        RIR_files = []

        for dset_IR in self.datasets_IR:
            if dset_IR == "ReverbFx":
                df = pd.read_csv(path_csv_ReverbFx)
                # list of RIR files
                RIR_list = df["RIR"].tolist()

                assert path_ReverbFx is not None
                RIR_list = [os.path.join(path_ReverbFx, f) for f in RIR_list]
                RIR_files.extend(RIR_list)
            elif dset_IR == "ASH":
                df = pd.read_csv(path_csv_ASH)
                RIR_list = df["rir"].tolist()
                assert path_ASH is not None, "path_ASH must be provided"
                RIR_list = [os.path.join(path_ASH, f) for f in RIR_list]
                RIR_files.extend(RIR_list)
            elif dset_IR == "QMUL":
                df = pd.read_csv(path_csv_QMUL)
                RIR_list = df["rir"].tolist()
                assert path_QMUL is not None, "path_QMUL must be provided"
                RIR_list = [os.path.join(path_QMUL, f) for f in RIR_list]
                RIR_files.extend(RIR_list)
            elif dset_IR == "Arni":
                df = pd.read_csv(path_csv_Arni)
                RIR_list = df["rir"].tolist()
                assert path_Arni is not None, "path_Arni must be provided"
                RIR_list = [os.path.join(path_Arni, f) for f in RIR_list]
                RIR_files.extend(RIR_list)

        acceoted_sampling_rates = [44100]

        dataset_rir = pd.DataFrame(columns=["impulse_response"])
        for i, file in enumerate(RIR_files):
            # load the RIR file
            x, fs = sf.read(file)
            if fs != self.fs:
                x = librosa.resample(x, orig_sr=fs, target_sr=self.fs, axis=0)
            n_samples = x.shape[0]
            if len(x.shape) == 1:
                x = x[:, None]
            my_tuple = (int(n_samples), x)
            # add my_tuple to the dataset_rir DataFrame (column 'impulse_response')
            dataset_rir.loc[i, "impulse_response"] = my_tuple

        self.augment_chain_wet = AugmentationChain(
            [
                (
                    ConvolutionalReverb(
                        impulse_responses=dataset_rir,
                        sample_rates=acceoted_sampling_rates,
                    ),
                    p_effects_wet,
                ),
                (Haas(sample_rates=acceoted_sampling_rates), p_effects_wet),
                (Gain(), p_effects_wet),
                (Panner(sample_rates=acceoted_sampling_rates), p_effects_wet),
                (Compressor(sample_rates=acceoted_sampling_rates), p_effects_wet),
                (
                    Equaliser(n_channels=2, sample_rates=acceoted_sampling_rates),
                    p_effects_wet,
                ),
            ],
            shuffle=True,
            apply_to="target",
        )

        self.augment_chain_dry = AugmentationChain(
            [
                (
                    ConvolutionalReverb(
                        impulse_responses=dataset_rir,
                        sample_rates=acceoted_sampling_rates,
                    ),
                    p_effects_dry,
                ),
                (Haas(sample_rates=acceoted_sampling_rates), p_effects_dry),
                (Gain(), p_effects_dry),
                (Panner(sample_rates=acceoted_sampling_rates), p_effects_dry),
                (Compressor(sample_rates=acceoted_sampling_rates), p_effects_dry),
                (
                    Equaliser(n_channels=2, sample_rates=acceoted_sampling_rates),
                    p_effects_dry,
                ),
            ],
            shuffle=True,
            apply_to="target",
        )

    def __iter__(self):

        while True:
                #try:
                randnum = np.random.uniform(0, 1 - 1e-5)
                # search for the last element in priorities_cumsum that is smaller than randnum
                if randnum < self.probabilities_cumsum[0]:
                    idx = 0
                else:
                    idx = int(np.where(self.probabilities_cumsum < randnum)[0][-1] + 1)

                dataset = self.datasets[idx]

                skip_iteration = False

                if dataset == "MedleyDB":

                    num = np.random.randint(0, len(self.df_MDB) - 1)

                    row = self.df_MDB.iloc[num]

                    path = os.path.join(self.path_medleydb, row["track_name"])

                    # seach for a directory under "path" that ends with "_RAW"
                    dry_path = glob.glob(os.path.join(path, "*_RAW"))[0]

                    files = glob.glob(os.path.join(dry_path, "*.wav"))

                    if len(files) == 0:
                        print("no dry files found in", dry_path)
                        continue
                elif dataset == "IDMT_SMT_DRUMS":
                    num = np.random.randint(0, len(self.df_IDMT_SMT_DRUMS) - 1)
                    row = self.df_IDMT_SMT_DRUMS.iloc[num]
                    files = [os.path.join(self.path_IDMT_SMT_DRUMS, row["track_name"])]

                elif dataset == "GuitarSet":
                    num = np.random.randint(0, len(self.df_GuitarSet) - 1)
                    row = self.df_GuitarSet.iloc[num]
                    files = [os.path.join(self.path_GuitarSet, row["track_name"])]

                elif dataset == "OpenSinger":
                    num = np.random.randint(0, len(self.df_OpenSinger) - 1)
                    row = self.df_OpenSinger.iloc[num]
                    files = [row["track_name"]]
                elif dataset == "IDMT_SMT_GUITAR":
                    num = np.random.randint(0, len(self.df_IDMT_SMT_GUITAR) - 1)
                    row = self.df_IDMT_SMT_GUITAR.iloc[num]
                    files = [os.path.join(self.path_IDMT_SMT_GUITAR, row["track_name"])]
                elif dataset == "IDMT_SMT_BASS":
                    num = np.random.randint(0, len(self.df_IDMT_SMT_BASS) - 1)
                    row = self.df_IDMT_SMT_BASS.iloc[num]
                    files = [os.path.join(self.path_IDMT_SMT_BASS, row["track_name"])]
                elif dataset == "Aalto-anechoic":
                    num = np.random.randint(0, len(self.df_Aalto_anechoic) - 1)
                    row = self.df_Aalto_anechoic.iloc[num]
                    files = [os.path.join(self.path_Aalto_anechoic, row["track_name"])]
                else:
                    raise ValueError()

                if self.tracks == "all":
                    dry_files = []
                    for i in range(len(files)):
                        dry_files.append(files[i])
                else:
                    raise NotImplementedError("all mode is only implemented for now")

                if skip_iteration:
                    continue

                assert len(dry_files) > 0, "no dry files found in {}".format(dry_path)

                total_frames = 999999999999
                dry_frames = []

                for i in range(len(dry_files)):
                    try:
                        dry_duration, dry_total_frames, dry_samplerate = (
                            get_audio_length(str(dry_files[i]))
                        )
                    except:
                        print("error in", dry_files[i])
                        raise

                    total_frames = min(total_frames, dry_total_frames)
                    dry_frames.append(dry_total_frames)

                order = np.random.permutation(len(dry_files))
                dry_files = [dry_files[i] for i in order]

                found = False
                number_tries = 3

                for i, dry_file in enumerate(dry_files):

                    for n in range(number_tries):

                        if total_frames >= self.segment_length:

                            start_dry = np.random.randint(
                                0, total_frames - self.segment_length
                            )
                            end_dry = start_dry + self.segment_length

                            out = load_audio(
                                str(dry_file),
                                start_dry,
                                end_dry,
                                stereo=self.stereo,
                                target_fs=self.fs,
                            )
                        else:
                            # --- Short file: load entire file, loop, shift, and crop
                            out = load_audio(
                                str(dry_file),
                                0,
                                total_frames,
                                stereo=self.stereo,
                                target_fs=self.fs,
                            )

                            if out is not None:
                                x_dry, fs = out
                                if x_dry.ndim == 1:
                                    x_dry = x_dry[None, :]  # shape: (1, L)

                                repeats = (self.segment_length // total_frames) + 1
                                x_dry = x_dry.repeat(1, repeats)  # repeat in time
                                shift = torch.randint(0, x_dry.shape[1], (1,)).item()
                                x_dry = torch.roll(x_dry, shifts=shift, dims=-1)
                                x_dry = x_dry[:, : self.segment_length]
                                fs = fs  # sampling rate unchanged
                                out = (x_dry, fs)

                        if out is None:
                            skip_iteration = True
                            print("Could not load dry audio file: {}".format(dry_file))
                            continue

                        x_dry, fs = out

                        assert (
                            x_dry.shape[-1] == self.segment_length
                        ), "x_dry must have the same length as segment_length, got {}".format(
                            x_dry.shape[-1]
                        )
                        assert (
                            x_dry.shape[0] == 2
                        ), "x_dry must have 2 channels, got {}".format(x_dry.shape[0])

                        # convert stereo to mono for dry
                        x_dry = x_dry.mean(0, keepdim=True)

                        RMS_dB = self.get_RMS(x_dry.mean(0))
                        if RMS_dB < self.RMS_threshold_dB:
                            continue

                        found = True
                        break

                if not found:
                    print("No valid dry file found in", dry_files)
                    continue

                x_np = x_dry.cpu().numpy().T  # (L, 1)
                if x_np.ndim == 1:
                    x_np = x_np[:, None]
                if x_np.shape[1] == 1:
                    x_np = np.repeat(x_np, 2, axis=1)  # (L, 2)

                if self.apply_effects_on_dry:
                    _, x_dry_aug = self.augment_chain_dry(x_np, x_np)
                    x_dry_aug = (
                        torch.from_numpy(x_dry_aug.T).float().to(x_dry_aug.device)
                    )
                    # convert to monon if necessary
                    if x_dry_aug.shape[0] == 2:
                        x_dry_aug = x_dry_aug.mean(0, keepdim=True)

                    assert (
                        x_dry.shape == x_dry_aug.shape
                    ), "augment_target shape must match x_wet shape, got {} and {}".format(
                        x_dry_aug.shape, x_dry.shape
                    )
                else:
                    x_dry_aug = x_dry

                _, y_wet_aug = self.augment_chain_wet(x_np, x_np)
                y_wet_aug = torch.from_numpy(y_wet_aug.T).float().to(y_wet_aug.device)

                assert (
                    x_dry.shape[-1] == y_wet_aug.shape[-1]
                ), "augment_target shape must match x_wet shape, got {} and {}".format(
                    y_wet_aug.shape, x_dry.shape
                )

                yield x_dry_aug, y_wet_aug

                #except Exception as e:
                #print("Error in public dataset iteration:", e)
                #continue


class MultiDatasetDry_Test(torch.utils.data.Dataset):

    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        num_tracks=-1,
        mode="dry-wet",
        stereo=True,
        num_examples=4,
        RMS_threshold_dB=-40,
        x_as_mono=False,
        seed=42,
        tracks="all",
        path_medleydb=None,
        path_csv_medleydb=None,
        path_ReverbFx=None,
        RIR_path_csv="/scratch/elec/t412-asp/ReverbFx/meta/train.csv",
        p_effects_wet=0.8,
    ):

        super().__init__()
        print(num_examples, "num_examples")

        np.random.seed(seed)
        random.seed(seed)

        self.segment_length = segment_length
        self.fs = fs

        self.stereo = stereo

        self.num_tracks = num_tracks

        self.x_as_mono = x_as_mono

        self.RMS_threshold_dB = RMS_threshold_dB

        self.get_RMS = lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x**2, dim=-1)))

        self.test_samples = []

        counter = 0

        self.tracks = tracks
        assert self.tracks == "all", "only all model is implemented for now"

        self.path_medleydb = path_medleydb
        self.path_csv = path_csv_medleydb
        assert self.path_csv is not None, "path_csv must be provided"

        assert os.path.exists(self.path_csv)
        self.df = pd.read_csv(self.path_csv)

        if self.num_tracks != -1:
            self.df = self.df.sample(n=self.num_tracks, random_state=seed)
        else:
            self.df = self.df.sample(frac=1, random_state=seed)

        num_skips = 0

        df = pd.read_csv(RIR_path_csv)
        # list of RIR files
        RIR_files = df["RIR"].tolist()
        assert path_ReverbFx is not None
        RIR_files = [os.path.join(path_ReverbFx, f) for f in RIR_files]

        acceoted_sampling_rates = [44100]

        dataset_rir = pd.DataFrame(columns=["impulse_response"])
        for i, file in enumerate(RIR_files):

            # load the RIR file
            x, fs = sf.read(file)
            n_samples = x.shape[0]
            if len(x.shape) == 1:
                x = x[:, None]
            my_tuple = (int(n_samples), x)
            # add my_tuple to the dataset_rir DataFrame (column 'impulse_response')
            dataset_rir.loc[i, "impulse_response"] = my_tuple

        self.augment_chain_wet = AugmentationChain(
            [
                (
                    ConvolutionalReverb(
                        impulse_responses=dataset_rir,
                        sample_rates=acceoted_sampling_rates,
                    ),
                    p_effects_wet,
                ),
                (Haas(sample_rates=acceoted_sampling_rates), p_effects_wet),
                (Gain(), p_effects_wet),
                (Panner(sample_rates=acceoted_sampling_rates), p_effects_wet),
                (Compressor(sample_rates=acceoted_sampling_rates), p_effects_wet),
                (
                    Equaliser(n_channels=2, sample_rates=acceoted_sampling_rates),
                    p_effects_wet,
                ),
            ],
            shuffle=True,
            apply_to="target",
        )

        for idx, row in tqdm(self.df.iterrows()):

            path = os.path.join(self.path_medleydb, row["track_name"])

            # seach for a directory under "path" that ends with "_RAW"
            dry_path = glob.glob(os.path.join(path, "*_RAW"))[0]

            files = glob.glob(os.path.join(dry_path, "*.wav"))

            if len(files) == 0:
                print("no dry files found in", dry_path)
                continue

            skip_iteration = False

            if len(files) == 0:
                print("no dry files found in", dry_path)
                num_skips += 1
                continue

            if self.tracks == "all":
                # filter files by tracks
                dry_files = []
                for i in range(len(files)):
                    dry_files.append(files[i])
            else:
                raise NotImplementedError("all mode is only implemented for now")

            if skip_iteration:
                num_skips += 1
                continue

            assert len(dry_files) > 0, "no dry files found in {}".format(dry_path)

            for i, dry_file in enumerate(dry_files):

                out = load_audio(str(dry_file), stereo=self.stereo, target_fs=self.fs)
                if out is None:
                    skip_iteration = True
                    print("Could not load dry audio file: {}".format(dry_file))
                    continue

                x_dry_long, fs = out
                assert fs == self.fs, "wrong sampling rate: {}".format(fs)

                x_dry_long = x_dry_long.mean(0, keepdim=True)

                if i == 0:
                    x_all = torch.zeros(
                        (len(dry_files), 2, x_dry_long.shape[-1]), dtype=torch.float32
                    )
                    x_all[i] = x_dry_long

                else:
                    if x_dry_long.shape[-1] < x_all.shape[-1]:
                        padding = x_all.shape[-1] - x_dry_long.shape[-1]
                        x_dry_long = torch.nn.functional.pad(x_dry_long, (0, padding))
                    elif x_dry_long.shape[-1] > x_all.shape[-1]:
                        # pas
                        x_dry_long = x_dry_long[..., : x_all.shape[-1]]

                    x_all[i] = x_dry_long

            max_length = x_all.size(-1)
            max_length = (
                (max_length + self.segment_length - 1)
                // self.segment_length
                * self.segment_length
            )

            x_all = torch.nn.functional.pad(
                x_all, (0, max_length - x_all.size(-1)), mode="constant", value=0
            )

            assert (
                x_all.size(-1) % self.segment_length == 0
            ), "x_dry_long must be a multiple of segment_length, got {}".format(
                x_all.size(-1)
            )

            for i in range(0, x_all.size(-1), self.segment_length):
                x_all_i = x_all[..., i : i + self.segment_length]

                skip_segment = False
                selected_tracks = torch.ones((x_all_i.shape[0],), dtype=torch.bool)

                for i in range(x_all_i.shape[0]):
                    RMS_dB = self.get_RMS(x_all_i[i].mean(0))
                    if RMS_dB < self.RMS_threshold_dB:
                        # remove track if it is below the threshold
                        selected_tracks[i] = False

                if selected_tracks.sum() == 0:
                    skip_segment = True

                if skip_segment:
                    continue

                # now remove the tracks that are not selected
                x_all_i = x_all_i[selected_tracks]

                for i in range(torch.sum(selected_tracks)):
                    _, y_wet = self.augment_chain_wet(
                        x_all_i[i].cpu().numpy().T, x_all_i[i].cpu().numpy().T
                    )
                    y_wet = torch.from_numpy(y_wet.T).float().to(x_all_i.device)
                    assert (
                        x_all_i[i].shape == y_wet.shape
                    ), "augment_target shape must match x_wet shape, got {} and {}".format(
                        x_all_i[i].shape, y_wet.shape
                    )

                    self.test_samples.append((x_all_i[i], y_wet))

                counter += 1

            if skip_iteration:
                num_skips += 1
                continue

        random.shuffle(self.test_samples)

        if num_examples != -1:
            self.test_samples = self.test_samples[:num_examples]

        print(
            "test_samples",
            len(self.test_samples),
            "num_examples",
            num_examples,
            "num_skips",
            num_skips,
        )

    def __getitem__(self, idx):
        return self.test_samples[idx]

    def __len__(self):
        return len(self.test_samples)
