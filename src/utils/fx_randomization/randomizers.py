import torch
import torch.nn as nn

import grafx

from utils.fx_randomization.processors.STFT_diffused_reverb import STFTMaskedNoiseReverb_filterbank
from grafx.processors import Compressor


class ReverbRandomizer:
    """
    GRAFx model class for managing and processing graphical data.
    """

    def __init__(self, 
        num_filters=50,
        range_coloration=2, #+- dB for coloration randomizatio
        min_T60=0.5,
        max_T60=1.5,
        sample_rate=44100,
        ir_len=44100,
        drywet_ratio_min=0.0,
        drywet_ratio_max=1.0,
        batch_size=8,
        ):
        """
        Initialize the GRAFx model with the provided data.

        :param data: The data to be processed by the GRAFx model.
        """

        assert num_filters==50, "num_filters must be 50, overrides not implemented yet"


        config = grafx.data.NodeConfigs(["reverb"])

        G = grafx.data.GRAFX(config=config)

        self.batch_size = batch_size

        for i in range(self.batch_size):
            chain = ["in", "reverb"]
            start_id, end_id = G.add_serial_chain(chain)
            out_id= G.add("out")
            G.connect(end_id, out_id)

        G_t=grafx.data.convert_to_tensor(G)


        self.processors = {
            "reverb": STFTMaskedNoiseReverb_filterbank(processor_channel="mono", fixed_noise=False, ir_len=ir_len)
        }

        from grafx.utils import create_empty_parameters
        self.parameters = create_empty_parameters(self.processors, G)


        type_sequence, render_order = grafx.render.compute_render_order(G_t, method="beam")
        G_t = grafx.render.reorder_for_fast_render(G_t, method="beam")

        self.render_data = grafx.render.prepare_render(G_t)

        self.range_coloration = range_coloration
        self.min_T60 = min_T60  
        self.max_T60 = max_T60

        self.sample_rate=sample_rate

        self.drywet_ratio_min = drywet_ratio_min
        self.drywet_ratio_max = drywet_ratio_max

        self.drywet_ratio= torch.rand(self.batch_size) * (self.drywet_ratio_max - self.drywet_ratio_min) + self.drywet_ratio_min

    def sample_random_parameters(self):

        log_magnitude_params=torch.zeros_like(self.parameters["reverb"]["init_log_magnitude"])

        log_magnitude_params.uniform_(-self.range_coloration, self.range_coloration)  # Uniformly sample between -5 and 5 dB

        self.parameters["reverb"]["init_log_magnitude"]=log_magnitude_params


        t60_values=torch.zeros_like(self.parameters["reverb"]["delta_log_magnitude"])
        t60_values.uniform_(self.min_T60, self.max_T60)

        delta_log_magnitude_params = 6.908/ (t60_values*(self.sample_rate/self.processors["reverb"].hop_length))  # Convert T60 to delta log magnitude

        self.parameters["reverb"]["delta_log_magnitude"] = delta_log_magnitude_params


    def forward(self, x, resample=True):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        """

        input=x

        orig_shape = x.shape

        if orig_shape[0] != self.batch_size:
            if orig_shape[0] < self.batch_size:
                x=torch.cat([x, torch.zeros(self.batch_size - orig_shape[0], orig_shape[1], orig_shape[2], device=x.device)], dim=0)


        assert x.shape[0] == self.batch_size, f"Input batch size {x.shape[0]} does not match model batch size {self.batch_size}"

        to_mono = False
        if x.shape[1] == 2:
            # Convert stereo to mono by averaging the two channels
            to_mono = True
            x = x.mean(dim=1, keepdim=True)
            #print("Converting stereo to mono for reverb processing")

        device_x= x.device

        if device_x != self.parameters["reverb"]["init_log_magnitude"].device:
            for processor in self.processors.values():
                processor.to(device_x)

            for k, v in self.parameters.items():
                self.parameters[k] = v.to(device_x)
            #for key in self.parameters:
            #    for param in self.parameters[key]:
            #        self.parameters[key][param] = self.parameters[key][param].to(device_x)
            

        if resample:
            self.sample_random_parameters()
            self.drywet_ratio= torch.rand(self.batch_size, device=device_x) * (self.drywet_ratio_max - self.drywet_ratio_min) + self.drywet_ratio_min

        output, intermediates, signal_bufer = grafx.render.render_grafx( self.processors, x, self.parameters, self.render_data, parameters_grad=False, input_signal_grad=False)

        if to_mono:
            # Convert the output back to stereo by duplicating the mono channel
            output = output.repeat(1, 2, 1)
        
        if orig_shape[0] < self.batch_size:
            output = output[:orig_shape[0], :, :]
            self.drywet_ratio = self.drywet_ratio[:orig_shape[0]]

        output= input*(self.drywet_ratio).view(-1, 1, 1) + output *(1- self.drywet_ratio.view(-1, 1, 1))

        return output



class CompRandomizer:
    """
    GRAFx model class for managing and processing graphical data.
    """

    def __init__(self, 
        batch_size=8,
        smoother_mode_energy="iir",  # Options: "iir", "ballistics"
        smoother_mode_gain="iir",  # Options: "iir", "ballistics"
        range_log_threshold=[-30, 0],  # dB range for log threshold randomization
        range_log_ratio=[-3, 1],  # dB range for log ratio randomization
        range_log_knee=[-4.0, 2.0],  # Knee range for compressor randomization
        range_z_alpha_pre=[-1.0, 4.0], # Z-alpha range for compressor randomization
        range_z_alpha_post=[-1.0, 4.0], # Z-alpha range for compressor randomization
        RMS_mean=-25,  # Mean RMS level for randomization
        RMS_std=5,  # Standard deviation for RMS level randomization
        ):
        """
        Initialize the GRAFx model with the provided data.

        :param data: The data to be processed by the GRAFx model.
        """

        config = grafx.data.NodeConfigs(["compressor"])

        G = grafx.data.GRAFX(config=config)

        self.batch_size = batch_size

        for i in range(self.batch_size):
            chain = ["in", "compressor"]
            start_id, end_id = G.add_serial_chain(chain)
            out_id= G.add("out")
            G.connect(end_id, out_id)

        G_t=grafx.data.convert_to_tensor(G)


        self.processors = {
            "compressor": Compressor(energy_smoother=smoother_mode_energy, gain_smoother=smoother_mode_gain, gain_smooth_in_log=False, knee="quadratic", iir_len=16384, flashfftconv=False, max_input_len=2**17,)
        }

        from grafx.utils import create_empty_parameters
        self.parameters = create_empty_parameters(self.processors, G)

        type_sequence, render_order = grafx.render.compute_render_order(G_t, method="beam")
        G_t = grafx.render.reorder_for_fast_render(G_t, method="beam")

        self.render_data = grafx.render.prepare_render(G_t)

        self.range_log_threshold = range_log_threshold
        self.range_log_ratio = range_log_ratio
        self.range_log_knee = range_log_knee
        self.range_z_alpha_pre = range_z_alpha_pre
        self.range_z_alpha_post = range_z_alpha_post


        self.RMS_mean = RMS_mean
        self.RMS_std = RMS_std
        #self.range_coloration = range_coloration
        #self.min_T60 = min_T60  
        #self.max_T60 = max_T60

        #self.sample_rate=sample_rate

        #self.drywet_ratio_min = drywet_ratio_min
        #self.drywet_ratio_max = drywet_ratio_max
        self.RMS= torch.randn(self.batch_size) * self.RMS_std + self.RMS_mean
        self.RMS= self.RMS.view(-1, 1, 1)  # Ensure RMS is broadcastable

        #self.drywet_ratio= torch.rand(self.batch_size) * (self.drywet_ratio_max - self.drywet_ratio_min) + self.drywet_ratio_min

    def sample_random_parameters(self):

        for k in self.parameters["compressor"].keys():
            if k == "log_threshold":
                log_threshold_params = torch.zeros_like(self.parameters["compressor"]["log_threshold"])
                log_threshold_params.uniform_(self.range_log_threshold[0], self.range_log_threshold[1])
                self.parameters["compressor"]["log_threshold"] = log_threshold_params
            elif k == "log_ratio":
                log_ratio_params = torch.zeros_like(self.parameters["compressor"]["log_ratio"])
                log_ratio_params.uniform_(self.range_log_ratio[0], self.range_log_ratio[1])
                self.parameters["compressor"]["log_ratio"] = log_ratio_params
            elif k == "log_knee":
                log_knee_params = torch.zeros_like(self.parameters["compressor"]["log_knee"])
                log_knee_params.uniform_(self.range_log_knee[0], self.range_log_knee[1])
                self.parameters["compressor"]["log_knee"] = log_knee_params
            elif k == "z_alpha_pre":
                z_alpha_pre_params = torch.zeros_like(self.parameters["compressor"]["z_alpha_pre"])
                z_alpha_pre_params.uniform_(self.range_z_alpha_pre[0], self.range_z_alpha_pre[1])
                self.parameters["compressor"]["z_alpha_pre"] = z_alpha_pre_params
            elif k == "z_alpha_post":
                z_alpha_post_params = torch.zeros_like(self.parameters["compressor"]["z_alpha_post"])
                z_alpha_post_params.uniform_(self.range_z_alpha_post[0], self.range_z_alpha_post[1])
                self.parameters["compressor"]["z_alpha_post"] = z_alpha_post_params


    def forward(self, x, resample=True):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        """

        input=x


        orig_shape = x.shape

        if orig_shape[0] != self.batch_size:
            if orig_shape[0] < self.batch_size:
                x=torch.cat([x, torch.zeros(self.batch_size - orig_shape[0], orig_shape[1], orig_shape[2], device=x.device)], dim=0)

        assert x.shape[0] == self.batch_size, f"Input batch size {x.shape[0]} does not match model batch size {self.batch_size}"

        to_mono = False
        if x.shape[1] == 2:
            # Convert stereo to mono by averaging the two channels
            to_mono = True
            x = x.mean(dim=1, keepdim=True)
            #print("Converting stereo to mono for reverb processing")

        input_rms = 20*torch.log10(torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)))
        gain= self.RMS_mean - input_rms
        gain_linear = 10 ** (gain / 20)
        #print("gain_linear", gain_linear.shape, gain_linear.device, x.shape, x.device)
        x = x * gain_linear.view(-1, 1, 1)

        device_x= x.device

        if device_x != self.parameters["compressor"]["log_threshold"].device:
            for processor in self.processors.values():
                processor.to(device_x)

            for k, v in self.parameters.items():
                self.parameters[k] = v.to(device_x)

        if resample:
            self.sample_random_parameters()
            self.RMS= torch.randn(self.batch_size, device=device_x) * self.RMS_std + self.RMS_mean
            self.RMS= self.RMS.view(-1, 1, 1)  # Ensure RMS is broadcastable
            #self.drywet_ratio= torch.rand(self.batch_size, device=device_x) * (self.drywet_ratio_max - self.drywet_ratio_min) + self.drywet_ratio_min

        output, intermediates, signal_bufer = grafx.render.render_grafx( self.processors, x, self.parameters, self.render_data, parameters_grad=False, input_signal_grad=False)

        output_rms = 20*torch.log10(torch.sqrt(torch.mean(output**2, dim=-1, keepdim=True)))
        gain= self.RMS - output_rms
        gain_linear = 10 ** (gain / 20)
        #print("gain_linear", gain_linear.shape, gain_linear.device, output.shape, output.device)
        output = output * gain_linear.view(-1, 1, 1)

        if to_mono:
            # Convert the output back to stereo by duplicating the mono channel
            output = output.repeat(1, 2, 1)
        
        if orig_shape[0] < self.batch_size:
            output = output[:orig_shape[0], :, :]
            #self.RMS = self.RMS[:orig_shape[0]]
            #self.drywet_ratio = self.drywet_ratio[:orig_shape[0]]
        
        # Apply RMS normalization
        #get RMS of the output


        #output= input*(self.drywet_ratio).view(-1, 1, 1) + output *(1- self.drywet_ratio.view(-1, 1, 1))

        return output



