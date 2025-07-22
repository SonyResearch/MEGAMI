

import torch
import torch.nn.functional as F
import torchaudio

def collate_multitrack_sim(batch, max_tracks=None):

        x= [ data_i[0] for data_i in batch ]  # x is a list of tensors, each tensor is a track
        clusters=[ torch.tensor(data_i[1]) for data_i in batch ]  # cluster is a list of tensors, each tensor is a cluster
        taxonomies=[ data_i[2] for data_i in batch ]  # taxonomy is a

        if max_tracks is None:
            max_tracks = max(track.shape[0] for track in x)  # Find the maximum number of tracks in the batch
        else:
            if max_tracks > max(track.shape[0] for track in x):
                print(f"Warning: max_tracks is set to {max_tracks}, but the maximum number of tracks in the batch is {max(track.shape[0] for track in x)}. I dont know what will happen, consider increasing max_tracks," )

        # Pad each examples with zeros to the maximum length (dimension 0)

        padded_x = []
        padded_taxonomies = []
        masks = torch.zeros((len(x), max_tracks), dtype=torch.bool)  # Create a mask tensor

        for i in range(len(x)):
            current_x = x[i]
            current_taxonomies = taxonomies[i]
            
            # Get current number of tracks
            current_tracks = current_x.shape[0]

            masks[i, :current_tracks] = 1  # Set mask for current tracks
            
            if current_tracks < max_tracks:
                # Pad dimension N (first dimension)
                # For a tensor of shape [N, C, L], we need to pad the first dimension
                # F.pad expects padding as (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
                # To pad the first dimension, we need to specify padding for the third-from-last dimension
                pad_size = (0, 0,  # last dim (L): no padding
                            0, 0,  # second-to-last dim (C): no padding
                            0, max_tracks - current_tracks)  # third-to-last dim (N): pad at the end

                padded_x.append(F.pad(current_x, pad_size))
            
                # Pad taxonomies
                padded_taxonomies.append(current_taxonomies + [None] * (max_tracks - len(current_taxonomies)))
            elif current_tracks > max_tracks :
                raise ValueError(f"Number of tracks {current_tracks} exceeds maximum allowed {max_tracks}. that is impossible")
            else:
                padded_x.append(current_x)
                padded_taxonomies.append(current_taxonomies[:max_tracks])
            


        x_stacked = torch.stack(padded_x, dim=0)  # Shape: [B, max_tracks, C, L]

        clusters_stacked = torch.stack(clusters, dim=0)  # Shape: [B, max_tracks]

        return {
            'x': x_stacked,
            'clusters': clusters_stacked,
            'taxonomies': padded_taxonomies,
            "masks": masks
        }

def collate_multitrack_train(batch, max_tracks=None, sample_rate=None, segment_length=None, device=None):

        x= [ data_i[0] for data_i in batch ]  # x is a list of tensors, each tensor is a track
        paths=[ data_i[1] for data_i in batch ]  # paths is a list of paths to the audio files, each path is a string
        fs= [ data_i[2] for data_i in batch ]  # fs is a list of sample rates, each sample rate is an integer

        if max_tracks is None:
            max_tracks = max(track.shape[0] for track in x)  # Find the maximum number of tracks in the batch
        else:
            if max_tracks < max(track.shape[0] for track in x):
                print(f"Warning: max_tracks is set to {max_tracks}, but the maximum number of tracks in the batch is {max(track.shape[0] for track in x)}. I will crop the last tracks.  I hope the order is random..." )

                for i in range(len(x)):
                    if x[i].shape[0] > max_tracks:
                        print("cropping x[i] to max_tracks")
                        x[i] = x[i][:max_tracks]

        is_x_none= any(x_i is None for x_i in x)

        assert not (is_x_none ), "Either x or y should be None, but not both. This is a bug in the collator"

        #now resample audio to 44100 Hz and cut to segment length
        for i in range(len(x)):
            x[i] = x[i].to(device)  # Move to device
            if fs[i] != sample_rate:
                if fs[i] == 48000 and sample_rate == 44100:
                    x[i]=torchaudio.functional.resample(x[i], orig_freq=160, new_freq=147)
                else:
                    print(f"Resampling audio from {fs[i]} Hz to {sample_rate} Hz")
                    x[i]=torchaudio.functional.resample(x[i], fs[i], sample_rate)

            assert segment_length is not None, "segment_length should be set to the length of the audio segment in samples"
            if x[i].shape[-1] > segment_length:
                x[i] = x[i][..., :segment_length]
            elif x[i].shape[-1] < segment_length:
                raise ValueError(f"Audio length {x[i].shape[-1]} is less than segment length {segment_length}. Please check your data.")

        # Pad each examples with zeros to the maximum length (dimension 0)
        padded_x = []
        masks = torch.zeros((len(x), max_tracks), dtype=torch.bool)  # Create a mask tensor

        for i in range(len(x)):
            if not is_x_none:
                current_x = x[i]
            
            # Get current number of tracks
            current_tracks = current_x.shape[0]

            masks[i, :current_tracks] = 1  # Set mask for current tracks
            
            if current_tracks < max_tracks:
                # Pad dimension N (first dimension)
                # For a tensor of shape [N, C, L], we need to pad the first dimension
                # F.pad expects padding as (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
                # To pad the first dimension, we need to specify padding for the third-from-last dimension
                pad_size = (0, 0,  # last dim (L): no padding
                            0, 0,  # second-to-last dim (C): no padding
                            0, max_tracks - current_tracks)  # third-to-last dim (N): pad at the end


                if not is_x_none:
                    padded_x.append(F.pad(current_x, pad_size))
            
            elif current_tracks > max_tracks :
                raise ValueError(f"Number of tracks {current_tracks} exceeds maximum allowed {max_tracks}. that is impossible")
            else:
                if not is_x_none:
                    padded_x.append(current_x)


        if not is_x_none:
            x_stacked = torch.stack(padded_x, dim=0)  # Shape: [B, max_tracks, C, L]


        return {
            'y': x_stacked.to(device),  # Shape: [B, max_tracks, C, L]
            "masks": masks.to(device),  # Shape: [B, max_tracks]
            "paths": paths,
            "fs": fs
        }

def collate_multitrack_paired(batch, max_tracks=None):

        x= [ data_i[0] for data_i in batch ]  # x is a list of tensors, each tensor is a track
        y= [ data_i[1] for data_i in batch ]  # x is a list of tensors, each tensor is a track

        taxonomies=[ data_i[2] for data_i in batch ]  # taxonomy is a

        paths=[ data_i[3] for data_i in batch ]  # paths is a list of paths to the audio files, each path is a string

        if max_tracks is None:
            max_tracks = max(track.shape[0] for track in x)  # Find the maximum number of tracks in the batch
        else:
            if max_tracks < max(track.shape[0] for track in x):
                print(f"Warning: max_tracks is set to {max_tracks}, but the maximum number of tracks in the batch is {max(track.shape[0] for track in x)}. I will crop the last tracks.  I hope the order is random..." )

                for i in range(len(x)):
                    if x[i].shape[0] > max_tracks:
                        print("cropping x[i] to max_tracks")
                        x[i] = x[i][:max_tracks]
                    if y[i].shape[0] > max_tracks:
                        print("cropping y[i] to max_tracks")
                        y[i] = y[i][:max_tracks]
                    if len(taxonomies[i]) > max_tracks:
                        print("cropping taxonomies[i] to max_tracks")
                        taxonomies[i] = taxonomies[i][:max_tracks]



        is_x_none= any(x_i is None for x_i in x)
        is_y_none= any(y_i is None for y_i in y)

        assert not (is_x_none and is_y_none), "Either x or y should be None, but not both. This is a bug in the collator"

        # Pad each examples with zeros to the maximum length (dimension 0)
        padded_x = []
        padded_y = []
        padded_taxonomies = []
        masks = torch.zeros((len(x), max_tracks), dtype=torch.bool)  # Create a mask tensor

        for i in range(len(x)):
            
            if not is_x_none:
                current_x = x[i]
            if not is_y_none:
                current_y = y[i]    
            current_taxonomies = taxonomies[i]
            
            # Get current number of tracks
            if is_x_none:
                current_tracks = current_y.shape[0]
            else:
                current_tracks = current_x.shape[0]
                if not is_y_none:
                    assert current_tracks == current_y.shape[0], f"Number of tracks in x ({current_tracks}) does not match number of tracks in y ({current_y.shape[0]})"

            masks[i, :current_tracks] = 1  # Set mask for current tracks
            
            if current_tracks < max_tracks:
                # Pad dimension N (first dimension)
                # For a tensor of shape [N, C, L], we need to pad the first dimension
                # F.pad expects padding as (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
                # To pad the first dimension, we need to specify padding for the third-from-last dimension
                pad_size = (0, 0,  # last dim (L): no padding
                            0, 0,  # second-to-last dim (C): no padding
                            0, max_tracks - current_tracks)  # third-to-last dim (N): pad at the end


                if not is_x_none:
                    padded_x.append(F.pad(current_x, pad_size))
                if not is_y_none:
                    padded_y.append(F.pad(current_y, pad_size))
            
                # Pad taxonomies
                padded_taxonomies.append(current_taxonomies + [None] * (max_tracks - len(current_taxonomies)))
            elif current_tracks > max_tracks :
                raise ValueError(f"Number of tracks {current_tracks} exceeds maximum allowed {max_tracks}. that is impossible")
            else:
                if not is_x_none:
                    padded_x.append(current_x)
                if not is_y_none:
                    padded_y.append(current_y)

                padded_taxonomies.append(current_taxonomies[:max_tracks])
            

        if not is_x_none:
            x_stacked = torch.stack(padded_x, dim=0)  # Shape: [B, max_tracks, C, L]

        if not is_y_none:
            y_stacked = torch.stack(padded_y, dim=0)  # Shape: [B, max_tracks, C, L]

        if is_x_none:
            return {
                'y': y_stacked,
                'taxonomies': padded_taxonomies,
                "masks": masks,
                "paths": paths
            }
        if is_y_none:
            return {
                'x': x_stacked,
                'taxonomies': padded_taxonomies,
                "masks": masks,
                "paths": paths
            }

        return {
            'x': x_stacked,
            'y': y_stacked,
            'taxonomies': padded_taxonomies,
            "masks": masks,
            "paths": paths
        }