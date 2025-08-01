
import laion_clap
import librosa
import torch
from glob import glob
import os
import numpy as np
import torch.nn as nn
import time
import julius
from tqdm import tqdm
import sys
import torchaudio

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from audio_features import compute_audio_features
# from tency_dataloader import chunks, load_tency_chunks



def convert_audio(wav: torch.Tensor, from_rate: float, to_rate: float, ensure_stereo=True) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    if from_rate!=to_rate:
        wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    if ensure_stereo:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] == 1:
            wav = torch.cat([wav, wav], dim=0)
    return wav


def extract_clap_features(target_dir, 
                            output_dir, 
                            device="cuda", 
                            audio_extension='wav', 
                            sample_rate=48000, 
                            window_sec=1, 
                            drop_last=False, 
                            save_mean=False,
                            data_type='custom',
                            save_ext=None,
                            overwrite=False):

    clap_model = laion_clap.CLAP_Module(enable_fusion=False).to(device)
    clap_model.load_ckpt() # download the default pretrained checkpoint.

    if data_type=='custom':
        target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)
    elif data_type=='tency':
        target_file_paths = chunks
    start_time = time.time()
    for cur_file_idx, target_file_path in enumerate(tqdm(target_file_paths)):
        # set output path
        output_ext = '.npy' if save_ext==None else f'_{save_ext}.npy'
        if data_type=='custom':
            cur_output_path = target_file_path.replace(target_dir, output_dir).replace(f'.{audio_extension}', output_ext)
        elif data_type=='tency':
            # set the directory number for every 1000 chunks
            cur_output_path = os.path.join(output_dir, f'chunk_{cur_file_idx//1000:03d}', f'{cur_file_idx}{output_ext}')
        # check if the output file already exists
        if os.path.exists(cur_output_path) and not overwrite:
            continue
        try:
            # load audio and resample
            if data_type=='custom':
                cur_audio, input_sr = librosa.load(target_file_path, mono=True, sr=None)
            else:
                cur_audio, input_sr = load_tency_chunks(cur_file_idx)
            cur_audio = convert_audio(wav=torch.from_numpy(cur_audio), from_rate=input_sr, to_rate=sample_rate, ensure_stereo=False).numpy()
            cur_audio = cur_audio.reshape(1, -1) # Make it (1,T) or (N,T)
            # trim audio upto 120 seconds
            if cur_audio.shape[1] > 120 * sample_rate:
                cur_audio = cur_audio[:, :120 * sample_rate]
            # chunk audio into 10 seconds with <window_sec> second windowing
            window_size = 10 * sample_rate
            stride = window_sec * sample_rate
            if cur_audio.shape[1] >= window_size:
                cur_audio_chunks = []
                for i in range(0, cur_audio.shape[1] - window_size + 1, stride):
                    cur_audio_chunks.append(cur_audio[:, i:i + window_size])
                # Handle the last chunk if needed
                if not drop_last:
                    last_start = cur_audio.shape[1] - window_size
                    if last_start % stride != 0:
                        last_chunk = cur_audio[:, last_start:]
                        padded_chunk = np.pad(last_chunk, ((0, 0), (0, window_size - last_chunk.shape[1])), 'constant')
                        cur_audio_chunks.append(padded_chunk)
            else:
                # If audio is shorter than window, pad it to window size
                padded_audio = np.pad(cur_audio, ((0, 0), (0, window_size - cur_audio.shape[1])), 'constant')
                cur_audio_chunks = [padded_audio]
            cur_audio = np.vstack(cur_audio_chunks)
            # model inference
            audio_embed = clap_model.get_audio_embedding_from_data(x = cur_audio, use_tensor=False)
            if save_mean:
                audio_embed = audio_embed.mean(axis=0)
            # save outputs
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, audio_embed)
        except Exception as e:
            print(f"Error processing {target_file_path}: {e}")
            continue
    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")



def extract_vggish_features(target_dir, 
                                output_dir, 
                                device="cuda", 
                                audio_extension='wav', 
                                sample_rate=16000,
                                data_type='custom',
                                save_ext=None,
                                overwrite=False):

    model = torch.hub.load(repo_or_dir='harritaylor/torchvggish', model='vggish')
    model.postprocess = False
    model.embeddings = nn.Sequential(*list(model.embeddings.children())[:-1])
    model.eval()
    
    if data_type=='custom':
        target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)
    elif data_type=='tency':
        target_file_paths = chunks
    start_time = time.time()
    for cur_file_idx, target_file_path in enumerate(tqdm(target_file_paths)):
        # set output path
        output_ext = '.npy' if save_ext==None else f'_{save_ext}.npy'
        if data_type=='custom':
            cur_output_path = target_file_path.replace(target_dir, output_dir).replace(f'.{audio_extension}', output_ext)
        elif data_type=='tency':
            # set the directory number for every 1000 chunks
            cur_output_path = os.path.join(output_dir, f'chunk_{cur_file_idx//1000:03d}', f'{cur_file_idx}{output_ext}')
        # check if the output file already exists
        if os.path.exists(cur_output_path) and not overwrite:
            continue
        try:
            # load audio and resample
            if data_type=='custom':
                cur_audio, input_sr = librosa.load(target_file_path, mono=True, sr=None)
            else:
                cur_audio, input_sr = load_tency_chunks(cur_file_idx)
            cur_audio = convert_audio(wav=torch.from_numpy(cur_audio), from_rate=input_sr, to_rate=sample_rate, ensure_stereo=False).numpy()
            # trim audio upto 120 seconds
            if len(cur_audio) > 120 * sample_rate:
                cur_audio = cur_audio[:120 * sample_rate]
            # model inference
            audio_embed = model.forward(cur_audio, 16000)
            audio_embed = audio_embed.mean(axis=0).cpu().detach().numpy()
            # save outputs
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, audio_embed)
        except Exception as e:
            print(f"Error processing {target_file_path}: {e}")
            continue
    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")



def extract_pann_features(target_dir, 
                            output_dir, 
                            ckpt_dir, 
                            device="cuda", 
                            audio_extension='wav', 
                            sample_rate=32000,
                            data_type='custom',
                            save_ext=None,
                            overwrite=False):
    from pann import Cnn14

    model_path = os.path.join(ckpt_dir, "Cnn14_mAP%3D0.431.pth")
    if not(os.path.exists(model_path)):
        torch.hub.download_url_to_file(
            url='https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth', 
            dst=model_path
        )
    model = Cnn14(
        sample_rate=32000, 
        window_size=1024, 
        hop_size=320, 
        mel_bins=64, 
        fmin=50, 
        fmax=16000, 
        classes_num=527
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    if data_type=='custom':
        target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)
    elif data_type=='tency':
        target_file_paths = chunks
    start_time = time.time()
    for cur_file_idx, target_file_path in enumerate(tqdm(target_file_paths)):
        # set output path
        output_ext = '.npy' if save_ext==None else f'_{save_ext}.npy'
        if data_type=='custom':
            cur_output_path = target_file_path.replace(target_dir, output_dir).replace(f'.{audio_extension}', output_ext)
        elif data_type=='tency':
            # set the directory number for every 1000 chunks
            cur_output_path = os.path.join(output_dir, f'chunk_{cur_file_idx//1000:03d}', f'{cur_file_idx}{output_ext}')
        # check if the output file already exists
        if os.path.exists(cur_output_path) and not overwrite:
            continue
        try:
            # load audio and resample
            if data_type=='custom':
                cur_audio, input_sr = librosa.load(target_file_path, mono=True, sr=None)
            else:
                cur_audio, input_sr = load_tency_chunks(cur_file_idx)
                cur_audio = cur_audio.mean(axis=0)
            cur_audio = convert_audio(wav=torch.from_numpy(cur_audio), from_rate=input_sr, to_rate=sample_rate, ensure_stereo=False).numpy()
            # trim audio upto 120 seconds
            if len(cur_audio) > 120 * sample_rate:
                cur_audio = cur_audio[:120 * sample_rate]
            # model inference
            with torch.no_grad():
                out = model(torch.tensor(cur_audio).float().unsqueeze(0), None)
                embd = out['embedding'].data[0]
            if device == torch.device('cuda'):
                embd = embd.cpu()
            audio_embed = embd.detach().numpy()
            # save outputs
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, audio_embed)
        except Exception as e:
            print(f"Error processing {target_file_path}: {e}")
            continue
    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")


def extract_mert_features(target_dir,
                            output_dir,
                            device="cuda", 
                            audio_extension='wav', 
                            data_type='custom',
                            save_ext=None,
                            overwrite=False):
    from transformers import Wav2Vec2FeatureExtractor
    from transformers import AutoModel
    import torchaudio.transforms as T

    # loading our model weights
    # model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
    # loading the corresponding preprocessor config
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
    sample_rate = processor.sampling_rate

    if data_type=='custom':
        target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)
    elif data_type=='tency':
        target_file_paths = chunks
    start_time = time.time()
    for cur_file_idx, target_file_path in enumerate(tqdm(target_file_paths)):
        # set output path
        output_ext = '.npy' if save_ext==None else f'_{save_ext}.npy'
        if data_type=='custom':
            cur_output_path = target_file_path.replace(target_dir, output_dir).replace(f'.{audio_extension}', output_ext)
        elif data_type=='tency':
            # set the directory number for every 1000 chunks
            cur_output_path = os.path.join(output_dir, f'chunk_{cur_file_idx//1000:03d}', f'{cur_file_idx}{output_ext}')
        # check if the output file already exists
        if os.path.exists(cur_output_path) and not overwrite:
            continue
        try:
            # load audio and resample
            if data_type=='custom':
                cur_audio, input_sr = librosa.load(target_file_path, mono=True, sr=None)
            else:
                cur_audio, input_sr = load_tency_chunks(cur_file_idx)
            cur_audio = convert_audio(wav=torch.from_numpy(cur_audio), from_rate=input_sr, to_rate=sample_rate, ensure_stereo=False).numpy()

            # trim audio upto 120 seconds
            if len(cur_audio) > 120 * sample_rate:
                cur_audio = cur_audio[:120 * sample_rate]
            # model inference
            inputs = processor(cur_audio, sampling_rate=sample_rate, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()  # [13 layer, Time steps, 768 feature_dim]
            time_reduced_hidden_states = all_layer_hidden_states.mean(-2) # [13, 768]
            if device == torch.device('cuda'):
                time_reduced_hidden_states = time_reduced_hidden_states.cpu()
            audio_embed = time_reduced_hidden_states.cpu().detach().numpy()
            # save outputs
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, audio_embed)
        except Exception as e:
            print(f"Error processing {target_file_path}: {e}")
            continue
    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")


def extract_dsp_features(target_dir, 
                            output_dir, 
                            audio_extension='wav', 
                            sample_rate=44100,
                            data_type='custom',
                            save_ext=None,
                            overwrite=False):
    target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)
    start_time = time.time()
    for target_file_path in tqdm(target_file_paths):
        # set output path
        output_ext = '.npy' if save_ext==None else f'_{save_ext}.npy'
        cur_output_path = target_file_path.replace(target_dir, output_dir).replace(f'.{audio_extension}', output_ext)
        # check if the output file already exists
        if os.path.exists(cur_output_path) and not overwrite:
            continue
        try:
            # load audio and resample
            cur_audio, input_sr = librosa.load(target_file_path, mono=False, sr=None)
            cur_audio = convert_audio(wav=torch.from_numpy(cur_audio), from_rate=input_sr, to_rate=sample_rate).numpy()
            # extract DSP feauteres
            dsp_features = compute_audio_features(cur_audio, sample_rate)
            # save outputs
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, dsp_features)
        except Exception as e:
            print(f"Error processing {target_file_path}: {e}")
            continue
    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")



def extract_hpcp_features(target_dir,
                            output_dir, 
                            audio_extension='wav', 
                            sample_rate=44100,
                            data_type='custom',
                            save_ext=None,
                            overwrite=False):
    import essentia.standard as estd
    from essentia.pytools.spectral import hpcpgram
    target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)
    start_time = time.time()
    for target_file_path in tqdm(target_file_paths):
        # set output path
        output_ext = '.npy' if save_ext==None else f'_{save_ext}.npy'
        cur_output_path = target_file_path.replace(target_dir, output_dir).replace(f'.{audio_extension}', output_ext)
        
        # check if the output file already exists
        if os.path.exists(cur_output_path) and not overwrite:
            continue
        try:
            # load audio and resample
            if data_type=='custom':
                cur_audio, input_sr = librosa.load(target_file_path, mono=True, sr=None)
            else:
                cur_audio, input_sr = load_tency_chunks(cur_file_idx)
                cur_audio = cur_audio.mean(axis=0)
            cur_audio = convert_audio(wav=torch.from_numpy(cur_audio), from_rate=input_sr, to_rate=sample_rate, ensure_stereo=False).numpy()
            # trim audio upto 120 seconds
            if len(cur_audio) > 120 * sample_rate:
                cur_audio = cur_audio[:120 * sample_rate]
            # extract HPCP features
            hpcp_features = hpcpgram(cur_audio, sampleRate=sample_rate)
            # save outputs
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, hpcp_features)
        except Exception as e:
            print(f"Error processing {target_file_path}: {e}")
            continue
    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")



def extract_fxenc_features(target_dir, 
                            output_dir, 
                            ckpt_dir, 
                            ckpt_name, 
                            device="cuda", 
                            audio_extension='wav', 
                            sample_rate=44100,
                            data_type='custom',
                            save_ext=None,
                            overwrite=False):
    from fx_encoder import load_effects_encoder
    # load model
    model = load_effects_encoder(os.path.join(ckpt_dir, f"fxenc_{ckpt_name}.pt"), device=device)

    # extract features
    target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)
    start_time = time.time()
    for target_file_path in tqdm(target_file_paths):
        # set output path
        output_ext = '.npy' if save_ext==None else f'_{save_ext}_{ckpt_name}.npy'
        cur_output_path = target_file_path.replace(target_dir, output_dir).replace(f'.{audio_extension}', output_ext)
        # check if the output file already exists
        if os.path.exists(cur_output_path) and not overwrite:
            continue
        try:
            # load audio and resample
            cur_audio, input_sr = librosa.load(target_file_path, mono=False, sr=None)
            cur_audio = convert_audio(wav=torch.from_numpy(cur_audio), from_rate=input_sr, to_rate=sample_rate).unsqueeze(0).to(device)
            # model inference
            with torch.no_grad():
                embedding = model(cur_audio)
            audio_embed = embedding.squeeze().detach().cpu().numpy()
            # save outputs
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, audio_embed)
        except Exception as e:
            print(f"Error processing {target_file_path}: {e}")
            continue
    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")



def extract_stito_features(target_dir, 
                            output_dir, 
                            ckpt_dir, 
                            device="cuda", 
                            audio_extension='wav', 
                            sample_rate=48000,
                            save_ext=None,
                            overwrite=False):
    from stito_encoder import load_param_model_custom, get_param_embeds
    # load model
    model = load_param_model_custom(os.path.join(ckpt_dir, f"afx-rep.ckpt"), use_gpu=True)
    model = model.eval()

    # extract features
    target_file_paths = glob(os.path.join(target_dir, '**', f'*.{audio_extension}'), recursive=True)
    start_time = time.time()
    for target_file_path in tqdm(target_file_paths):
        # set output path
        output_ext = '.npy' if save_ext==None else f'_{save_ext}.npy'
        cur_output_path = target_file_path.replace(target_dir, output_dir).replace(f'.{audio_extension}', output_ext)
        # check if the output file already exists
        if os.path.exists(cur_output_path) and not overwrite:
            continue
        try:
            # load audio and resample
            cur_audio, input_sr = librosa.load(target_file_path, mono=False, sr=None)
            cur_audio = convert_audio(wav=torch.from_numpy(cur_audio), from_rate=input_sr, to_rate=sample_rate).unsqueeze(0).to(device)
            
            # model inference
            with torch.no_grad():
                embed_dict = get_param_embeds(cur_audio, model, sample_rate) 
                mid = embed_dict['mid']
                side = embed_dict['side']
                embedding = torch.cat([mid, side], dim=-1)
            audio_embed = embedding.squeeze().detach().cpu().numpy()
            # save outputs
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, audio_embed)
        except Exception as e:
            print(f"Error processing {target_file_path}: {e}")
            continue
    print(f"Extraction finished. Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")



if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    ''' Configurations for inferencing music effects encoder '''
    currentdir = os.path.dirname(os.path.realpath(__file__))

    import argparse
    import yaml
    parser = argparse.ArgumentParser()

    directory_args = parser.add_argument_group('Directory args')
    directory_args.add_argument('--target_dir', type=str, default='./samples/')
    directory_args.add_argument('--output_dir', type=str, default=None, help='if no output_dir is specified (None), the results will be saved inside the target_dir')
    directory_args.add_argument('--ckpt_dir', type=str, default='/home/tony/features/ckpt/', help='path to checkpoint weights')
    directory_args.add_argument('--data_type', type=str, default='custom', choices=['custom', 'tency'], help='extracting data type')
    directory_args.add_argument('--save_ext', type=str, default=None, help='extension name to be added to the output file name. If None, the original filename will be used')

    inference_args = parser.add_argument_group('Inference args')
    inference_args.add_argument('--using_gpu', type=str, default='0')
    inference_args.add_argument('--inference_device', type=str, default='cuda', help="if this option is not set to 'cpu', inference will happen on gpu only if there is a detected one")
    inference_args.add_argument('--audio_extension', type=str, default='wav')
    inference_args.add_argument('--feature_type', type=str, default='vggish', choices=['clap', 'vggish', 'pann', 'fxenc', 'stito', 'mert', 'dsp', 'hpcp'])
    inference_args.add_argument('--fxenc_ckpt_name', type=str, default='default', choices=['default', 'master_blackbox_tuned', 'master_whitebox_tuned'])
    inference_args.add_argument('--save_mean', type=str2bool, default=True)
    inference_args.add_argument('--overwrite', type=str2bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.using_gpu

    # Extract features
    if args.feature_type=='clap':
        extract_clap_features(target_dir=args.target_dir, \
                                    output_dir=args.output_dir, \
                                    device=args.inference_device, \
                                    audio_extension=args.audio_extension, \
                                    save_mean=args.save_mean, \
                                    data_type=args.data_type, \
                                    save_ext=args.save_ext, \
                                    overwrite=args.overwrite)
    elif args.feature_type=='vggish':
        extract_vggish_features(target_dir=args.target_dir, \
                                    output_dir=args.output_dir, \
                                    device=args.inference_device, \
                                    audio_extension=args.audio_extension, \
                                    data_type=args.data_type, \
                                    save_ext=args.save_ext, \
                                    overwrite=args.overwrite)
    elif args.feature_type=='pann':
        extract_pann_features(target_dir=args.target_dir, \
                                    output_dir=args.output_dir, \
                                    ckpt_dir=args.ckpt_dir, \
                                    device=args.inference_device, \
                                    audio_extension=args.audio_extension, \
                                    data_type=args.data_type, \
                                    save_ext=args.save_ext, \
                                    overwrite=args.overwrite)
    elif args.feature_type=='mert':
        extract_mert_features(target_dir=args.target_dir, \
                                    output_dir=args.output_dir, \
                                    device=args.inference_device, \
                                    audio_extension=args.audio_extension, \
                                    data_type=args.data_type, \
                                    save_ext=args.save_ext, \
                                    overwrite=args.overwrite)
    elif args.feature_type=='fxenc':
        extract_fxenc_features(target_dir=args.target_dir, \
                                    output_dir=args.output_dir, \
                                    ckpt_dir=args.ckpt_dir, \
                                    ckpt_name=args.fxenc_ckpt_name, \
                                    device=args.inference_device, \
                                    audio_extension=args.audio_extension, \
                                    save_ext=args.save_ext, \
                                    overwrite=args.overwrite)
    elif args.feature_type=='stito':
        extract_stito_features(target_dir=args.target_dir, \
                                    output_dir=args.output_dir, \
                                    ckpt_dir=args.ckpt_dir, \
                                    device=args.inference_device, \
                                    audio_extension=args.audio_extension, \
                                    save_ext=args.save_ext, \
                                    overwrite=args.overwrite)
    elif args.feature_type=='dsp':
        extract_dsp_features(target_dir=args.target_dir, \
                                    output_dir=args.output_dir, \
                                    audio_extension=args.audio_extension, \
                                    data_type=args.data_type, \
                                    save_ext=args.save_ext, \
                                    overwrite=args.overwrite)
    elif args.feature_type=='hpcp':
        extract_hpcp_features(target_dir=args.target_dir, \
                                    output_dir=args.output_dir, \
                                    audio_extension=args.audio_extension, \
                                    data_type=args.data_type, \
                                    save_ext=args.save_ext, \
                                    overwrite=args.overwrite)

