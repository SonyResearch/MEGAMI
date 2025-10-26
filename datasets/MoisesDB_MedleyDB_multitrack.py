import torch.nn.functional as F
import random
import pandas as pd
import torch
import numpy as np
import os
import glob
from utils.data_utils import read_wav_segment, get_audio_length
import yaml
from tqdm import tqdm

from utils.feature_extractors.dsp_features import compute_log_rms_gated_max
from collections import defaultdict


def load_audio(file, start=None, end=None, stereo=True):
    x, fs = read_wav_segment(file, start, end)
    if stereo:
        if len(x.shape) == 1:
            # print( "dry not stereo , doubling channels", x_dry.shape)
            x = x[:, np.newaxis]
            x = np.concatenate((x, x), axis=-1)
        elif len(x.shape) == 2 and x.shape[-1] == 1:
            # print( "dry not stereo , doubling channels", x_dry.shape)
            x = np.concatenate((x, x), axis=-1)

    x = torch.from_numpy(x).permute(1, 0)

    return x, fs


class MoisesDB_MedleyDB_Multitrack_Dataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        normalize_params=None,
        stereo=True,
        mode="dry-wet",
        num_examples=4,
        RMS_threshold_dB=-40,
        seed=42,
        tracks="all",
        MoisesDB_path=None,
        MedleyDB_path=None,
        path_csv=None,
        random_order=False,
        random_polarity=False,
        only_dry=False,  # if True, only dry files are used, if False, both dry and wet files are used
        mixing_file_MedleyDB="conf/data/mixing_coefficients_version2.yaml",
        max_n_tracks=14,  # maximum number of tracks in a multitrack file, used to limit the number of tracks in the dataset
        use_mixing_augmentation=True,
        normalize_mixture=True,  # if True,
        RMS_mixture_norm=-8,  # RMS of the mixture in dB, used to normalize the mixture and the tracks accordingly
    ):

        super().__init__()

        self.random_order = random_order

        self.random_polarity = random_polarity

        self.only_dry = only_dry

        self.mode = mode

        self.segment_length = segment_length
        self.segment_length = (
            int(segment_length * 48000 / 44100) + 1
        )  # most of the data is samples at 48kHz, so we convert the segment length to 48kHz samples. will be resampled and cut later
        self.fs = fs

        self.normalize_mode = normalize_params.normalize_mode

        self.stereo = stereo

        self.RMS_threshold_dB = RMS_threshold_dB

        self.get_RMS = lambda x: 20 * torch.log10(torch.sqrt(torch.mean(x**2, dim=-1)))

        self.tracks = tracks
        assert self.tracks == "all", "only all model is implemented for now"

        self.MoisesDB_path = MoisesDB_path
        self.MedleyDB_path = MedleyDB_path
        self.path_csv = path_csv
        assert self.path_csv is not None, "path_csv must be provided"

        self.df = pd.read_csv(self.path_csv)

        self.mixing_file_MedleyDB = mixing_file_MedleyDB
        # opem the mixing coefficients file
        self.mxing_coefs = yaml.safe_load(open(self.mixing_file_MedleyDB, "r"))

        self.normalize_mixture = normalize_mixture
        self.RMS_mixture_norm = RMS_mixture_norm  # RMS of the mixture in dB, used to normalize the mixture and the tracks accordingly

        self.use_mixing_augmentation = use_mixing_augmentation

        self.max_n_tracks = max_n_tracks

    def normalize_wrt_mixture(self, x_wet):

        x_sum = x_wet.sum(dim=0)

        mixture_logRMS = compute_log_rms_gated_max(x_sum.unsqueeze(0)).mean()

        gain_dB = self.RMS_mixture_norm - mixture_logRMS

        gain_linear = 10 ** (gain_dB / 20 + 1e-6)

        x_wet = x_wet * gain_linear

        x_sum_norm = x_wet.sum(dim=0)

        mixture_logRMS_norm = compute_log_rms_gated_max(x_sum_norm.unsqueeze(0)).mean()

        return x_wet

    def MedleyDB_get_mixing_coefficients(self, stem_idx, coef_dict):
        """Get best availalbe mixing coefficient for a stem.

        Parameters
        ----------
        stem_idx : int
            Stem index

        Returns
        -------
        mixing_coefficient : float
            Stem's mixing coefficient

        """

        # get stem idx from each of wet files (last two characters of the filename, excuding the extension .wav)

        assert coef_dict is not None

        mixing_coefficients = []

        use_manual = "manual" in coef_dict.keys() and coef_dict["manual"] != {}

        for i in stem_idx:

            if use_manual:
                coef = coef_dict["manual"][i]
            else:
                coef = (coef_dict["audio"][i] + coef_dict["stft"][i]) * 0.5

            mixing_coefficients.append(coef)

        assert len(mixing_coefficients) == len(stem_idx)

        # print("mixing coefficients for stem {}: {}".format(stem_idx, mixing_coefficients))

        return mixing_coefficients

    def MedleyDB_get_taxonomies(self, stem_idx, metadata_dict):
        stems_in_metadata = metadata_dict["stems"].keys()

        assert len(stem_idx) == len(
            set(stem_idx)
        ), "stem_idx must be unique, got {}".format(stem_idx)

        # make sure we have the same number of stems in the metadata and in the stem_idx
        assert len(stem_idx) == len(
            stems_in_metadata
        ), "stem_idx and metadata must have the same number of stems, got {} and {}".format(
            len(stem_idx), len(stems_in_metadata)
        )

        taxonomies = []

        for idx in stem_idx:
            # idx string should be in format S01, S02, etc.
            idx_str = "S{:02d}".format(idx)
            taxonomy = metadata_dict["stems"][idx_str]["instrument"]

            taxonomies.append(taxonomy)

        return taxonomies

    def apply_mixing_augmentation(
        self, selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
    ):
        n_tracks_init = len(selected_tracks_wet)
        assert n_tracks_init > 0, "no tracks found to apply mixing augmentation"
        assert n_tracks_init == len(selected_taxonomies)

        track_indices = list(range(len(selected_tracks_wet)))
        taxonomy_groups = defaultdict(list)
        for idx, tax in zip(track_indices, selected_taxonomies):
            taxonomy_groups[tax].append(idx)

        grouped_tracks = []
        grouped_taxonomies = []

        for tax, tracks in taxonomy_groups.items():
            n_tracks = len(tracks)

            if n_tracks == 1:
                grouped_tracks.append(selected_tracks_wet[tracks[0]])
                grouped_taxonomies.append(tax)
                continue

            # Shuffle track order to get random combinations
            random.shuffle(tracks)

            remaining_tracks = tracks.copy()

            while remaining_tracks:
                # Randomly decide size of this subset (1..remaining)
                subset_size = random.randint(1, len(remaining_tracks))
                subset = [remaining_tracks.pop(0) for _ in range(subset_size)]


                if len(subset) == 1:
                    # Just add the single track
                    grouped_tracks.append(selected_tracks_wet[subset[0]].clone())
                else:
                    # Mix the subset
                    subset_mixed = torch.stack(
                        [selected_tracks_wet[t].clone() for t in subset], dim=0
                    ).sum(dim=0)

                    grouped_tracks.append(subset_mixed)

                grouped_taxonomies.append(tax)


        return grouped_tracks, grouped_taxonomies

    def iter_MoisesDB(self, path):

        try:
            song_path = os.path.join(self.MoisesDB_path, path)
            wet_files = glob.glob(os.path.join(song_path, "*/*.wav"))

            assert len(wet_files) > 0, "no wet files found in {}".format(song_path)

            # check the subdir to retreieve taxonomy
            taxonomies = []
            for wet_file in wet_files:
                taxonomy = os.path.basename(os.path.dirname(wet_file))
                taxonomies.append(taxonomy)

            total_frames = self.get_total_frames(wet_files)

            start = np.random.randint(0, total_frames - self.segment_length)
            end = start + self.segment_length

            selected_tracks_wet = []
            selected_taxonomies = []

            x_sum = None

            for i, wet_file in enumerate(wet_files):

                try:
                    out = load_audio(str(wet_file), start, end, stereo=self.stereo)
                except:
                    continue

                if out is None:
                    raise Exception(
                        f"Error loading wet file {wet_file} at segment {start}-{end}"
                    )

                x_wet, fs = out

                assert (
                    x_wet.shape[-1] == self.segment_length
                ), "x_wet_long must have the same length as segment_length, got {}".format(
                    x_wet.shape[-1]
                )
                assert (
                    x_wet.shape[0] == 2
                ), "x_wet_long must have 2 channels, got {}".format(x_wet.shape[0])

                if self.random_polarity:
                    if np.random.rand() > 0.5:
                        x_wet = -x_wet

                RMS_dB_wet = self.get_RMS(x_wet.mean(0))
                if RMS_dB_wet < self.RMS_threshold_dB:
                    continue

                selected_tracks_wet.append(x_wet)
                selected_taxonomies.append(taxonomies[i])

            assert (
                len(selected_tracks_wet) > 0
            ), "no wet tracks found after filtering by RMS threshold"


            if self.use_mixing_augmentation:
                # will use selected_taxonomies
                if np.random.rand() < 0.5:
                    selected_tracks_wet, selected_taxonomies = (
                        self.apply_mixing_augmentation(
                            selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
                        )
                    )

                if len(selected_tracks_wet) > self.max_n_tracks:
                    selected_tracks_wet, selected_taxonomies = (
                        self.apply_mixing_augmentation(
                            selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
                        )
                    )
                if len(selected_tracks_wet) > self.max_n_tracks:
                    selected_tracks_wet, selected_taxonomies = (
                        self.apply_mixing_augmentation(
                            selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
                        )
                    )
                if len(selected_tracks_wet) > self.max_n_tracks:
                    selected_tracks_wet, selected_taxonomies = (
                        self.apply_mixing_augmentation(
                            selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
                        )
                    )
                else:
                    return None

            else:
                print("not using mixing augmentation, using only the original tracks")


            if len(selected_tracks_wet) > self.max_n_tracks:
                print(
                    "warning: more than {} tracks found, selecting only {} tracks".format(
                        self.max_n_tracks, self.max_n_tracks
                    )
                )

            selected_tracks_wet = torch.stack(selected_tracks_wet, dim=0)

            if self.normalize_mixture:
                # use x_sum to normalize the mixture
                selected_tracks_wet = self.normalize_wrt_mixture(selected_tracks_wet)

            if self.random_order:
                # randomly shuffle the tracks
                indices = torch.randperm(selected_tracks_wet.shape[0])
                selected_tracks_wet = selected_tracks_wet[indices]
            else:
                raise NotImplementedError("random_order must be used")

            return selected_tracks_wet, path, fs

        except Exception as e:
            print("Error processing MoisesDB file {}: {}".format(path, e))
            return None

    def get_total_frames(self, wet_files):
        total_frames = 999999999999
        for i in range(len(wet_files)):
            wet_duration, wet_total_frames, wet_samplerate = get_audio_length(
                str(wet_files[i])
            )
            total_frames = min(total_frames, wet_total_frames)
        return total_frames

    def iter_MedleyDB(self, path):

        try:
            metadata_file = os.path.join(
                self.MedleyDB_path, path, path + "_METADATA.yaml"
            )
            metadata_dict = yaml.safe_load(open(metadata_file, "r"))

            wet_path = os.path.join(self.MedleyDB_path, path, path + "_STEMS")
            wet_files = glob.glob(os.path.join(wet_path, "*.wav"))

            assert len(wet_files) > 0, "no wet files found"

            stem_idx = [int(os.path.basename(f)[-6:-4]) for f in wet_files]

            mixing_coefficients = self.MedleyDB_get_mixing_coefficients(
                stem_idx, self.mxing_coefs[path]
            )

            taxonomies = self.MedleyDB_get_taxonomies(stem_idx, metadata_dict)

            total_frames = self.get_total_frames(wet_files)

            start = np.random.randint(0, total_frames - self.segment_length)
            end = start + self.segment_length

            selected_tracks_wet = []
            selected_taxonomies = []

            x_sum = None

            for i, wet_file in enumerate(wet_files):

                try:
                    out = load_audio(str(wet_file), start, end, stereo=self.stereo)
                except:
                    continue

                if out is None:
                    raise Exception(
                        f"Error loading wet file {wet_file} at segment {start}-{end}"
                    )

                x_wet, fs = out

                x_wet *= mixing_coefficients[i]

                assert (
                    x_wet.shape[-1] == self.segment_length
                ), "x_wet_long must have the same length as segment_length, got {}".format(
                    x_wet.shape[-1]
                )
                assert (
                    x_wet.shape[0] == 2
                ), "x_wet_long must have 2 channels, got {}".format(x_wet.shape[0])

                if self.random_polarity:
                    if np.random.rand() > 0.5:
                        x_wet = -x_wet

                RMS_dB_wet = self.get_RMS(x_wet.mean(0))
                if RMS_dB_wet < self.RMS_threshold_dB:
                    continue

                selected_tracks_wet.append(x_wet)

                selected_taxonomies.append(taxonomies[i])

            assert (
                len(selected_tracks_wet) > 0
            ), "no wet tracks found after filtering by RMS threshold file: {}".format(
                path
            )


            if self.use_mixing_augmentation:
                # do it with a probability of 0.5
                if np.random.rand() < 0.5:
                    selected_tracks_wet, selected_taxonomies = (
                        self.apply_mixing_augmentation(
                            selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
                        )
                    )

                if len(selected_tracks_wet) > self.max_n_tracks:
                    selected_tracks_wet, selected_taxonomies = (
                        self.apply_mixing_augmentation(
                            selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
                        )
                    )
                if len(selected_tracks_wet) > self.max_n_tracks:
                    selected_tracks_wet, selected_taxonomies = (
                        self.apply_mixing_augmentation(
                            selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
                        )
                    )
                if len(selected_tracks_wet) > self.max_n_tracks:
                    selected_tracks_wet, selected_taxonomies = (
                        self.apply_mixing_augmentation(
                            selected_tracks_wet, selected_taxonomies, dset="MoisesDB"
                        )
                    )
                if len(selected_tracks_wet) > self.max_n_tracks:
                    print(
                        "warning: more than {} tracks found, selecting only {} tracks".format(
                            self.max_n_tracks, self.max_n_tracks
                        )
                    )
                    return None
            else:
                print("not using mixing augmentation, using only the original tracks")

            if len(selected_tracks_wet) > self.max_n_tracks:
                print(
                    "warning: more than {} tracks found, selecting only {} tracks".format(
                        self.max_n_tracks, self.max_n_tracks
                    )
                )

            selected_tracks_wet = torch.stack(selected_tracks_wet, dim=0)

            if self.normalize_mixture:
                # use x_sum to normalize the mixture
                selected_tracks_wet = self.normalize_wrt_mixture(selected_tracks_wet)

            if self.random_order:
                # randomly shuffle the tracks
                indices = torch.randperm(selected_tracks_wet.shape[0])
                selected_tracks_wet = selected_tracks_wet[indices]
            else:
                raise NotImplementedError("random_order must be used")

            return selected_tracks_wet, path, fs
        except Exception as e:
            print("Error processing MedleyDB file {}: {}".format(path, e))
            return None

    def __iter__(self):

        while True:
            num = np.random.randint(0, len(self.df) - 1)

            row = self.df.iloc[num]

            subdir = row["subdir"]

            path = row["path"]

            if subdir == "MoisesDB":
                out = self.iter_MoisesDB(path)
                if out is None:
                    continue
                yield out

            elif subdir == "MedleyDB":
                out = self.iter_MedleyDB(path)
                if out is None:
                    continue
                yield out

            else:
                raise NotImplementedError(
                    "subdir must be either MoisesDB or MedleyDB, got {}".format(subdir)
                )

