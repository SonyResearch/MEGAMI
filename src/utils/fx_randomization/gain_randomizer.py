
import torch
import torch.nn as nn


class GainRandomizer:

    def __init__(self, 
        min_gain_dB=-5,  # Minimum RMS gain in dB
        max_gain_dB=5, # Maximum RMS gain in dB
        ):
        """
        Initialize the GRAFx model with the provided data.

        :param data: The data to be processed by the GRAFx model.
        """

        self.min_gain_dB = min_gain_dB
        self.max_gain_dB = max_gain_dB

        self.gain=None



    def forward(self, x, resample=True):
        """
        Process the data using GRAFx-specific algorithms.
        x: shape (batch_size, num_channels, num_samples)
        """

        if self.gain is None or resample:
            self.gain = torch.rand(x.shape[0], device=x.device) * (self.max_rms - self.min_rms) + self.min_rms

        #convert to a linear scale
        gain_linear = 10 ** (self.gain / 20)

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