""" 
    Implementation of neural networks used in the task 'Music Mixing Style Transfer'
        - 'Effects Encoder'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from ITOMater_network_utils import ConvBlock, Res_ConvBlock, FiLM
sys.path.append(os.path.dirname(currentdir))
#from mixing_manipulator import Multiband_Compressor, Distortion, Limiter

import omegaconf


# compute receptive field
def compute_receptive_field(kernels, strides, dilations):
    rf = 0
    for i in range(len(kernels)):
        rf += rf * strides[i] + (kernels[i]-strides[i]) * dilations[i]
    return rf


# Encoder of music effects for contrastive learning of music effects
class Effects_Encoder(nn.Module):
    def __init__(self, config):
        super(Effects_Encoder, self).__init__()
        # input is stereo channeled audio
        config["channels"].insert(0, 2)

        # encoder layers
        encoder = []
        for i in range(len(config["kernels"])):
            if config["conv_block"]=='res':
                encoder.append(Res_ConvBlock(dimension=1, \
                                                in_channels=config["channels"][i], \
                                                out_channels=config["channels"][i+1], \
                                                kernel_size=config["kernels"][i], \
                                                stride=config["strides"][i], \
                                                padding="SAME", \
                                                dilation=config["dilation"][i], \
                                                norm=config["norm"], \
                                                activation=config["activation"], \
                                                last_activation=config["activation"]))
            elif config["conv_block"]=='conv':
                encoder.append(ConvBlock(dimension=1, \
                                            layer_num=1, \
                                            in_channels=config["channels"][i], \
                                            out_channels=config["channels"][i+1], \
                                            kernel_size=config["kernels"][i], \
                                            stride=config["strides"][i], \
                                            padding="VALID", \
                                            dilation=config["dilation"][i], \
                                            norm=config["norm"], \
                                            activation=config["activation"], \
                                            last_activation=config["activation"], \
                                            mode='conv'))
        self.encoder = nn.Sequential(*encoder)

        # pooling method
        self.glob_pool = nn.AdaptiveAvgPool1d(1)

    # network forward operation
    def forward(self, input):
        enc_output = self.encoder(input)
        glob_pooled = self.glob_pool(enc_output).squeeze(-1)

        # outputs c feature
        return glob_pooled


class TCNBlock(torch.nn.Module):
    def __init__(self, 
                in_ch, 
                out_ch, 
                kernel_size=3, 
                stride=1, 
                dilation=1, 
                cond_dim=2048, 
                grouped=False, 
                causal=False,
                conditional=False, 
                **kwargs):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal
        self.conditional = conditional

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        self.pad_length = ((kernel_size-1)*dilation) if self.causal else ((kernel_size-1)*dilation)//2
        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.pad_length,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=False)
        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        if conditional:
            self.film = FiLM(cond_dim, out_ch)
        self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.LeakyReLU()
        self.res = torch.nn.Conv1d(in_ch, 
                                   out_ch, 
                                   kernel_size=1,
                                   stride=stride,
                                   groups=in_ch,
                                   bias=False)

    def forward(self, x, p):
        x_in = x

        x = self.relu(self.bn(self.conv1(x)))
        x = self.film(x, p)

        x_res = self.res(x_in)

        if self.causal:
            x = x[..., :-self.pad_length]
        x += x_res

        return x



class TCNModel(nn.Module):
    """ Temporal convolutional network with conditioning module.
        Args:
            nparams (int): Number of conditioning parameters.
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 3
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
            causal (bool): Causal TCN configuration does not consider future input values. Default: False
            skip_connections (bool): Skip connections from each block to the output. Default: False
            num_examples (int): Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self, 
                 ninputs=1,
                 noutputs=2,
                 nblocks=14, 
                 kernel_size=15, 
                 stride=1,
                 dilation_growth=2, 
                 channel_growth=1, 
                 channel_width=128, 
                 stack_size=15,
                 cond_dim=2048,
                 grouped=False,
                 causal=False,
                 skip_connections=False,
                 num_examples=4,
                 save_dir=None,
                 **kwargs):
        super(TCNModel, self).__init__()

        self.hparams = {
            "ninputs": ninputs,
            "noutputs": noutputs,
            "nblocks": nblocks,
            "kernel_size": kernel_size,
            "stride": stride,
            "dilation_growth": dilation_growth,
            "channel_growth": channel_growth,
            "channel_width": channel_width,
            "stack_size": stack_size,
            "cond_dim": cond_dim,
            "grouped": grouped,
            "causal": causal,
            "skip_connections": skip_connections,
            "num_examples": num_examples,
        }
        self.hparams= omegaconf.OmegaConf.create(self.hparams)

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            
            if self.hparams.channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth 
            else:
                out_ch = self.hparams.channel_width

            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            cur_stride = stride[n] if isinstance(stride, list) else stride
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=self.hparams.kernel_size, 
                                        stride=cur_stride, 
                                        dilation=dilation,
                                        padding="same" if self.hparams.causal else "valid",
                                        causal=self.hparams.causal,
                                        cond_dim=cond_dim, 
                                        grouped=self.hparams.grouped,
                                        conditional=True ))

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x, cond):
        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            # for SeFa
            if isinstance(cond, list):
                x = block(x, cond[idx])
            else:
                x = block(x, cond)
            skips = 0

        # out = torch.tanh(self.output(x + skips))
        out = torch.clamp(self.output(x + skips), min=-1, max=1)

        return out

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
        return rf




# differentiable models
import dasp_pytorch
EPS = 1e-6

# Module Mastering Style Transfer using differentiable audio effects processors
class Dasp_Mastering_Style_Transfer(nn.Module):
    def __init__(self, num_features, sample_rate, \
                    tgt_fx_names = ['eq', 'comp', 'imager', 'gain'], \
                    model_type='2mlp', \
                    config=None, \
                    batch_size=4):
        super(Dasp_Mastering_Style_Transfer, self).__init__()
        raise NotImplementedError("Dasp_Mastering_Style_Transfer is not implemented yet, please use DiffVoxParamEstimator instead")
        self.sample_rate = sample_rate
        self.tgt_fx_names = tgt_fx_names

        self.fx_processors = {}
        self.last_predicted_params = None
        for cur_fx in tgt_fx_names:
            if cur_fx=='eq':
                cur_fx_module = dasp_pytorch.ParametricEQ(sample_rate=sample_rate, \
                                                            min_gain_db = -20.0, \
                                                            max_gain_db = 20.0, \
                                                            min_q_factor = 0.1, \
                                                            max_q_factor=5.0)
            elif cur_fx=='distortion':
                # cur_fx_module = Distortion(sample_rate=sample_rate)
                cur_fx_module = Distortion(sample_rate=sample_rate, 
                                            min_gain_db = 0.0,
                                            max_gain_db = 8.0)
            elif cur_fx=='comp':
                cur_fx_module = dasp_pytorch.Compressor(sample_rate=sample_rate)
            elif cur_fx=='multiband_comp':
                cur_fx_module = Multiband_Compressor(sample_rate=sample_rate)
            elif cur_fx=='gain':
                cur_fx_module = dasp_pytorch.Gain(sample_rate=sample_rate)
            elif cur_fx=='imager':
                continue
            elif cur_fx=='limiter':
                cur_fx_module = Limiter(sample_rate=sample_rate)
            else:
                raise AssertionError(f"current fx name ({cur_fx}) not found")
            self.fx_processors[cur_fx] = cur_fx_module
        total_num_param = sum([self.fx_processors[cur_fx].num_params for cur_fx in self.fx_processors])
        if 'imager' in tgt_fx_names:
            total_num_param += 1
            
        ''' model architecture '''
        self.model_type = model_type
        if self.model_type=='2mlp':
            self.layer_1 = nn.Linear(num_features, 128)
            self.layer_out = nn.Linear(128, total_num_param)
        elif self.model_type=='3mlp':
            self.layer_1 = nn.Linear(num_features, 512)
            self.layer_2 = nn.Linear(512, 128)
            self.layer_out = nn.Linear(128, total_num_param)
        elif self.model_type.lower()=='tcn':
            self.network = TCNModel(nparams=config["condition_dimension"], ninputs=2, \
                                    noutputs=total_num_param, \
                                    nblocks=config["nblocks"], \
                                    dilation_growth=config["dilation_growth"], \
                                    kernel_size=config["kernel_size"], \
                                    stride=config['stride'], \
                                    channel_width=config["channel_width"], \
                                    stack_size=config["stack_size"], \
                                    cond_dim=config["condition_dimension"], \
                                    causal=config["causal"])
        elif self.model_type.lower()=='ito':
            self.params = torch.nn.Parameter(torch.ones((batch_size,total_num_param))*0.5)

    # network forward operation
    def forward(self, x, embedding):
        # embedding mapper
        if self.model_type=='2mlp':
            est_param = F.relu(self.layer_1(embedding))
            est_param = self.layer_out(est_param)
        elif self.model_type=='3mlp':
            est_param = F.relu(self.layer_1(embedding))
            est_param = F.relu(self.layer_2(est_param))
            est_param = self.layer_out(est_param)
        elif self.model_type.lower()=='tcn':
            est_param = self.network(x, embedding)
            est_param = est_param.mean(axis=-1)
        elif self.model_type.lower()=='ito':
            est_param = self.params
            est_param = torch.clamp(est_param, min=0.0, max=1.0)
        
        if self.model_type.lower()!='ito':
            est_param = F.sigmoid(est_param)

        self.last_predicted_params = est_param

        # dafx chain
        cur_param_idx = 0
        for cur_fx in self.tgt_fx_names:
            if cur_fx=='imager':
                cur_param_count = 1
                x = dasp_pytorch.functional.stereo_widener(x, \
                                                            sample_rate=self.sample_rate, \
                                                            width=est_param[:,cur_param_idx:cur_param_idx+1])
            else:
                cur_param_count = self.fx_processors[cur_fx].num_params
                cur_input_param = est_param[:, cur_param_idx:cur_param_idx+cur_param_count]
                x = self.fx_processors[cur_fx].process_normalized(x, cur_input_param)
            # update param index
            cur_param_idx += cur_param_count

        return x


    def reset_fx_chain(self, ):
        self.fx_processors = {}
        for cur_fx in self.tgt_fx_names:
            if cur_fx=='eq':
                cur_fx_module = dasp_pytorch.ParametricEQ(sample_rate=self.sample_rate, \
                                                            min_gain_db = -20.0, \
                                                            max_gain_db = 20.0, \
                                                            min_q_factor = 0.1, \
                                                            max_q_factor=5.0)
            elif cur_fx=='distortion':
                cur_fx_module = Distortion(sample_rate=self.sample_rate, 
                                            min_gain_db = 0.0,
                                            max_gain_db = 8.0)
            elif cur_fx=='comp':
                cur_fx_module = dasp_pytorch.Compressor(sample_rate=self.sample_rate)
            elif cur_fx=='multiband_comp':
                cur_fx_module = Multiband_Compressor(sample_rate=self.sample_rate)
            elif cur_fx=='gain':
                cur_fx_module = dasp_pytorch.Gain(sample_rate=self.sample_rate)
            elif cur_fx=='imager':
                continue
            elif cur_fx=='limiter':
                cur_fx_module = Limiter(sample_rate=self.sample_rate)
            else:
                raise AssertionError(f"current fx name ({cur_fx}) not found")
            self.fx_processors[cur_fx] = cur_fx_module

    def get_last_predicted_params(self):
        if self.last_predicted_params is None:
            return None

        params_dict = {}
        cur_param_idx = 0

        for cur_fx in self.tgt_fx_names:
            if cur_fx == 'imager':
                cur_param_count = 1
                normalized_param = self.last_predicted_params[:, cur_param_idx:cur_param_idx+1]
                original_param = self.denormalize_param(normalized_param, 0, 1)  # Assuming imager width range is 0 to 1
                params_dict[cur_fx] = original_param
            else:
                cur_param_count = self.fx_processors[cur_fx].num_params
                normalized_params = self.last_predicted_params[:, cur_param_idx:cur_param_idx+cur_param_count]
                original_params = self.denormalize_params(cur_fx, normalized_params)
                params_dict[cur_fx] = original_params

            cur_param_idx += cur_param_count

        return params_dict

    def denormalize_params(self, fx_name, normalized_params):
        fx_processor = self.fx_processors[fx_name]
        original_params = {}

        for i, (param_name, (min_val, max_val)) in enumerate(fx_processor.param_ranges.items()):
            original_param = self.denormalize_param(normalized_params[:, i:i+1], min_val, max_val)
            original_params[param_name] = original_param

        return original_params

    @staticmethod
    def denormalize_param(normalized_param, min_val, max_val):
        return normalized_param * (max_val - min_val) + min_val


class DiffVoxParamEstimator(nn.Module):
    def __init__(self, num_features, sample_rate, \
                    tgt_fx_names = ['eq', 'comp', 'imager', 'gain'], \
                    model_type='2mlp', \
                    config=None, \
                    batch_size=4):

        super(DiffVoxParamEstimator, self).__init__()

        raise NotImplementedError("DiffVoxParamEstimator is not implemented yet")
        self.sample_rate = sample_rate
        self.tgt_fx_names = tgt_fx_names

        self.fx_processors = {}
        self.last_predicted_params = None
        for cur_fx in tgt_fx_names:
            if cur_fx=='eq':
                cur_fx_module = dasp_pytorch.ParametricEQ(sample_rate=sample_rate, \
                                                            min_gain_db = -20.0, \
                                                            max_gain_db = 20.0, \
                                                            min_q_factor = 0.1, \
                                                            max_q_factor=5.0)
            elif cur_fx=='distortion':
                # cur_fx_module = Distortion(sample_rate=sample_rate)
                cur_fx_module = Distortion(sample_rate=sample_rate, 
                                            min_gain_db = 0.0,
                                            max_gain_db = 8.0)
            elif cur_fx=='comp':
                cur_fx_module = dasp_pytorch.Compressor(sample_rate=sample_rate)
            elif cur_fx=='multiband_comp':
                cur_fx_module = Multiband_Compressor(sample_rate=sample_rate)
            elif cur_fx=='gain':
                cur_fx_module = dasp_pytorch.Gain(sample_rate=sample_rate)
            elif cur_fx=='imager':
                continue
            elif cur_fx=='limiter':
                cur_fx_module = Limiter(sample_rate=sample_rate)
            else:
                raise AssertionError(f"current fx name ({cur_fx}) not found")
            self.fx_processors[cur_fx] = cur_fx_module
        total_num_param = sum([self.fx_processors[cur_fx].num_params for cur_fx in self.fx_processors])
        if 'imager' in tgt_fx_names:
            total_num_param += 1
            
        ''' model architecture '''
        self.model_type = model_type
        if self.model_type=='2mlp':
            self.layer_1 = nn.Linear(num_features, 128)
            self.layer_out = nn.Linear(128, total_num_param)
        elif self.model_type=='3mlp':
            self.layer_1 = nn.Linear(num_features, 512)
            self.layer_2 = nn.Linear(512, 128)
            self.layer_out = nn.Linear(128, total_num_param)
        elif self.model_type.lower()=='tcn':
            self.network = TCNModel(nparams=config["condition_dimension"], ninputs=2, \
                                    noutputs=total_num_param, \
                                    nblocks=config["nblocks"], \
                                    dilation_growth=config["dilation_growth"], \
                                    kernel_size=config["kernel_size"], \
                                    stride=config['stride'], \
                                    channel_width=config["channel_width"], \
                                    stack_size=config["stack_size"], \
                                    cond_dim=config["condition_dimension"], \
                                    causal=config["causal"])
        elif self.model_type.lower()=='ito':
            self.params = torch.nn.Parameter(torch.ones((batch_size,total_num_param))*0.5)

    # network forward operation
    def forward(self, x, embedding):
        # embedding mapper
        if self.model_type=='2mlp':
            est_param = F.relu(self.layer_1(embedding))
            est_param = self.layer_out(est_param)
        elif self.model_type=='3mlp':
            est_param = F.relu(self.layer_1(embedding))
            est_param = F.relu(self.layer_2(est_param))
            est_param = self.layer_out(est_param)
        elif self.model_type.lower()=='tcn':
            est_param = self.network(x, embedding)
            est_param = est_param.mean(axis=-1)
        elif self.model_type.lower()=='ito':
            est_param = self.params
            est_param = torch.clamp(est_param, min=0.0, max=1.0)
        
        if self.model_type.lower()!='ito':
            est_param = F.sigmoid(est_param)

        self.last_predicted_params = est_param

        # dafx chain
        cur_param_idx = 0
        for cur_fx in self.tgt_fx_names:
            if cur_fx=='imager':
                cur_param_count = 1
                x = dasp_pytorch.functional.stereo_widener(x, \
                                                            sample_rate=self.sample_rate, \
                                                            width=est_param[:,cur_param_idx:cur_param_idx+1])
            else:
                cur_param_count = self.fx_processors[cur_fx].num_params
                cur_input_param = est_param[:, cur_param_idx:cur_param_idx+cur_param_count]
                x = self.fx_processors[cur_fx].process_normalized(x, cur_input_param)
            # update param index
            cur_param_idx += cur_param_count

        return x


    def reset_fx_chain(self, ):
        self.fx_processors = {}
        for cur_fx in self.tgt_fx_names:
            if cur_fx=='eq':
                cur_fx_module = dasp_pytorch.ParametricEQ(sample_rate=self.sample_rate, \
                                                            min_gain_db = -20.0, \
                                                            max_gain_db = 20.0, \
                                                            min_q_factor = 0.1, \
                                                            max_q_factor=5.0)
            elif cur_fx=='distortion':
                cur_fx_module = Distortion(sample_rate=self.sample_rate, 
                                            min_gain_db = 0.0,
                                            max_gain_db = 8.0)
            elif cur_fx=='comp':
                cur_fx_module = dasp_pytorch.Compressor(sample_rate=self.sample_rate)
            elif cur_fx=='multiband_comp':
                cur_fx_module = Multiband_Compressor(sample_rate=self.sample_rate)
            elif cur_fx=='gain':
                cur_fx_module = dasp_pytorch.Gain(sample_rate=self.sample_rate)
            elif cur_fx=='imager':
                continue
            elif cur_fx=='limiter':
                cur_fx_module = Limiter(sample_rate=self.sample_rate)
            else:
                raise AssertionError(f"current fx name ({cur_fx}) not found")
            self.fx_processors[cur_fx] = cur_fx_module

    def get_last_predicted_params(self):
        if self.last_predicted_params is None:
            return None

        params_dict = {}
        cur_param_idx = 0

        for cur_fx in self.tgt_fx_names:
            if cur_fx == 'imager':
                cur_param_count = 1
                normalized_param = self.last_predicted_params[:, cur_param_idx:cur_param_idx+1]
                original_param = self.denormalize_param(normalized_param, 0, 1)  # Assuming imager width range is 0 to 1
                params_dict[cur_fx] = original_param
            else:
                cur_param_count = self.fx_processors[cur_fx].num_params
                normalized_params = self.last_predicted_params[:, cur_param_idx:cur_param_idx+cur_param_count]
                original_params = self.denormalize_params(cur_fx, normalized_params)
                params_dict[cur_fx] = original_params

            cur_param_idx += cur_param_count

        return params_dict

    def denormalize_params(self, fx_name, normalized_params):
        fx_processor = self.fx_processors[fx_name]
        original_params = {}

        for i, (param_name, (min_val, max_val)) in enumerate(fx_processor.param_ranges.items()):
            original_param = self.denormalize_param(normalized_params[:, i:i+1], min_val, max_val)
            original_params[param_name] = original_param

        return original_params

    @staticmethod
    def denormalize_param(normalized_param, min_val, max_val):
        return normalized_param * (max_val - min_val) + min_val

