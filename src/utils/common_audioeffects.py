"""
Audio effects for data augmentation.

Several audio effects can be combined into an augmentation chain.

Important note: We assume that the parallelization during training is done using
                multi-processing and not multi-threading. Hence, we do not need the
                `@sox.sox_context()` decorators as discussed in this
                [thread](https://github.com/pseeth/soxbindings/issues/4).

Section 2, TL21
AI Speech and Sound Group, SL1
"""

from itertools import permutations
import os
import io
import functools
import lameenc
import logging
import numpy as np
import pymixconsole as pymc
from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList
from pymixconsole.processor import Processor
from random import shuffle
from scipy.signal import oaconvolve
import soundfile as sf
#import soxbindings as sox
from typing import List, Optional, Tuple, Union, Dict
from numba import jit

#from common_dataprocessing import sample_data

# prevent pysox from logging warnings regarding non-opimal timestretch factors
logging.getLogger('sox').setLevel(logging.ERROR)

# set maximum peak value if we pass a signal through SOX
MAX_SOX_PROCESSING_PEAK = 0.707


def convert_audio2data(x):
    """
    Convert audio data from the format it was stored in to float32.

    Args:
        x (Numpy array): input with `x.dtype` either `np.int16`, `np.int32`, `np.float32` or `np.float64`.

    Returns:
        Numpy array: output with values in [-1., 1.) where `dtype` is `np.float32`.
    """
    if x.dtype in [np.float32, np.float64]:
        return x.astype(dtype=np.float32)
    else:
        return (x.astype(dtype=np.float64) / (1. + np.iinfo(x.dtype).max)).astype(np.float32)



def sample_data(data: Tuple[int, Union[np.ndarray, functools.partial]],
                start: int = 0, length: Optional[int] = None) -> np.ndarray:
    """
    Load one stem specified by `data`.

    Returns the audio beginning from `start` either up to the end (if `length` is None)
    or until the provided `length`. For the case that `start + length > n_samples`, we do a wrap-around and
    load the remaining samples from the beginning of `data`.

    Args:
        data: Data with shape (n_samples, data).
        start: Start index.
        length: Length of sample. If `length` is not None, `length` samples are returned (possibly with a wrap-around).
            Otherwise, everything until the end of `data` is returned.

    Returns:
        samples: data with shape `n_samples x n_channels`
    """
    n_samples, audio = data

    # determine whether we have to load the audio or whether it was already loaded
    is_loaded = True if type(audio) is np.ndarray else False

    # if `length` is not None, then only select subset
    do_wrap_around = False
    if length is not None:
        if start + length > n_samples:
            # we need to wrap around and concatenate `start:` and `:stop`
            do_wrap_around = True
            stop = length - (n_samples - start)
        else:
            # no wrap around as it is inside the array/file boundaries
            stop = start + length
    else:
        stop = None

    if is_loaded:
        if not do_wrap_around:
            samples = convert_audio2data(audio[start:stop])
        else:
            samples = np.vstack((convert_audio2data(audio[start:]),
                                 convert_audio2data(audio[:stop])))
    else:
        if not do_wrap_around:
            samples = convert_audio2data(audio(start=start, stop=stop)[0])
        else:
            samples = np.vstack((convert_audio2data(audio(start=start)[0]),
                                 convert_audio2data(audio(stop=stop)[0])))

    return samples


# Monkey-Patch `Processor` for convenience
# (a) Allow `None` as blocksize if processor can work on variable-length audio
def new_init(self, name, parameters, block_size, sample_rate=None, normalize=None, dtype='float32'):
    """
    Initialize processor.

    Args:
        self: Reference to object
        name (str): Name of processor.
        parameters (parameter_list): Parameters for this processor.
        block_size (int): Size of blocks for blockwise processing.
            Can also be `None` if full audio can be processed at once.
        sample_rate (int): Sample rate of input audio. Use `None` if effect is independent of this value.
        normalize (str): Defines, whether the processed signal is normalized.
            Possible values are `'rms'`, `'max'` and `None`.
        dtype (str): data type of samples

    Raises:
        ValueError: If `normalize` is not equal to `'rms'`, `'max'` or `False`.
    """
    self.name = name
    self.parameters = parameters
    self.block_size = block_size
    self.sample_rate = sample_rate
    self.dtype = dtype
    if normalize not in [None, 'rms', 'max']:
        raise ValueError(f'Unknown value {normalize} for `normalize`. Must be either `rms`, `max` or `None`')
    self.normalize = normalize


# (b) make code simpler
def new_update(self, parameter_name):
    """
    Update processor after randomization of parameters.

    Args:
        self: Reference to object.
        parameter_name (str): Parameter whose value has changed.
    """
    pass


# (c) representation for nice print
def new_repr(self):
    """
    Create human-readable representation.

    Args:
        self: Reference to object.

    Returns:
        string representation of object.
    """
    return f'Processor(name={self.name!r}, parameters={self.parameters!r}'


Processor.__init__ = new_init
Processor.__repr__ = new_repr
Processor.update = new_update


class AugmentationChain:
    """Basic audio Fx chain which is used for data augmentation."""

    def __init__(self,
                 fxs: Optional[List[Tuple[Union[Processor, 'AugmentationChain'], float]]] = [],
                 shuffle: Optional[bool] = False,
                 apply_to: Optional[str] = 'both'):
        """
        Create augmentation chain from the dictionary `fxs`.

        Args:
            fxs (list of tuples): Each tuple has three elements:
                First tuple element is an instance of `pymc.processor` or `AugmentationChain` that
                we want to use for data augmentation.
                Second element gives probability that effect should be applied.
            shuffle (bool): If `True` then order of Fx are changed whenever chain is applied.
            apply_to (str): Apply the chain to both input and target or one of them only.
                Possible values are `'both'`, `'input'` and `target`.

        Raises:
            ValueError: If `apply_to` is not equal to `both`, `input` or `target`.
        """
        self.fxs = fxs
        self.shuffle = shuffle
        if apply_to not in ['both', 'input', 'target']:
            raise ValueError(f'Unknown value {apply_to} for `apply_to`. Must be `both`, `input` or `target`')
        else:
            self.apply_to = apply_to

    def apply_processor(self, x, processor: Processor):
        """
        Pass audio in `x` through `processor` and output the respective processed audio.

        Args:
            x (Numpy array): Input audio of shape `n_samples` x `n_channels`.
            processor (Processor): Audio effect that we want to apply.

        Returns:
            Numpy array: Processed audio of shape `n_samples` x `n_channels` (same size as `x')
        """
        n_samples_input = x.shape[0]

        if processor.block_size is None:
            y = processor.process(x)
        else:
            # make sure that n_samples is a multiple of `processor.block_size`
            if x.shape[0] % processor.block_size != 0:
                n_pad = processor.block_size - x.shape[0] % processor.block_size
                x = np.pad(x, ((0, n_pad), (0, 0)), mode='reflective')

            y = np.zeros_like(x)
            for idx in range(0, x.shape[0], processor.block_size):
                y[idx:idx+processor.block_size, :] = processor.process(x[idx:idx+processor.block_size, :])

        if processor.normalize is not None:
            if processor.normalize == 'rms':  # normalize output energy such that it is the same as the input energy
                scale = np.sqrt(np.mean(np.square(x)) / np.maximum(1e-7, np.mean(np.square(y))))
            elif processor.normalize == 'max':  # normalize output signal by its max. amplitude
                scale = (1 + 1e-7)/(np.max(np.abs(y)) + 1e-7)
            y *= scale

        # return audio of same length as x
        return y[:n_samples_input, :]

    def __call__(self, input_x, target_x):
        """
        Apply augmentation chain to audio in `input_x` and `target_x`.

        Args:
            input_x (Numpy array): Audio samples of shape `n_samples` x `n_channels`.
            target_x (Numpy array): Audio samples of shape `n_samples` x `n_channels`.

        Returns:
            input_y (Numpy array): Processed audio of same shape as `input_x` where effects have been applied.
            target_y (Numpy array): Processed audio of same shape as `target_x` where effects have been applied.
        """
        # randomly shuffle effect order if `self.shuffle` is True
        if self.shuffle:
            shuffle(self.fxs)

        input_y = input_x
        target_y = target_x

        # check whether we only need to process once later
        if self.apply_to == 'both' and np.allclose(input_y, target_y):
            is_input_equal_target = True
        else:
            is_input_equal_target = False

        # apply effects with probabilities given in `self.fxs`
        for fx, p in self.fxs:
            if np.random.rand() < p:
                if isinstance(fx, Processor):
                    # randomize all effect parameters (also calls `update()` for each processor)
                    fx.randomize()
                    # apply processor dependent on `apply_to`
                    if self.apply_to == 'both':
                        input_y = self.apply_processor(input_y, fx)
                        if not is_input_equal_target:
                            target_y = self.apply_processor(target_y, fx)
                    elif self.apply_to == 'input':
                        input_y = self.apply_processor(input_y, fx)
                    elif self.apply_to == 'target':
                        target_y = self.apply_processor(target_y, fx)
                else:
                    # apply effect chain
                    if is_input_equal_target:
                        target_y = input_y
                    input_y, target_y = fx(input_y, target_y)
                    # check whether input and target are still the same
                    if not fx.apply_to == 'both':
                        is_input_equal_target = False

        if is_input_equal_target:
            target_y = input_y

        return input_y, target_y

    def __repr__(self):
        """
        Human-readable representation.

        Returns:
            string representation of object.
        """
        return f'AugmentationChain(fxs={self.fxs!r}, shuffle={self.shuffle!r})'


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DISTORTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def hard_clip(x, threshold_dB, drive):
    """
    Hard clip distortion.

    Args:
        x: input audio
        threshold_dB: threshold
        drive: drive

    Returns:
        (Numpy array): distorted audio
    """
    drive_linear = np.power(10., drive / 20.).astype(np.float32)
    threshold_linear = 10. ** (threshold_dB / 20.)
    return np.clip(x * drive_linear, -threshold_linear, threshold_linear)


def overdrive(x, drive, colour, sample_rate):
    """
    Overdrive distortion.

    Args:
        x: input audio
        drive: Controls the amount of distortion (dB).
        colour: Controls the amount of even harmonic content in the output(dB)
        sample_rate: sampling rate

    Returns:
        (Numpy array): distorted audio
    """
    scale = np.max(np.abs(x))
    if scale > MAX_SOX_PROCESSING_PEAK:
        clips = True
        x = x * (MAX_SOX_PROCESSING_PEAK / scale)
    else:
        clips = False

    tfm = sox.Transformer()
    tfm.overdrive(gain_db=drive, colour=colour)
    y = tfm.build_array(input_array=x, sample_rate_in=sample_rate).astype(np.float32)

    if clips:
        y *= scale / MAX_SOX_PROCESSING_PEAK  # rescale output to original scale
    return y


def hyperbolic_tangent(x, drive):
    """
    Hyperbolic Tanh distortion.

    Args:
        x: input audio
        drive: drive

    Returns:
        (Numpy array): distorted audio
    """
    drive_linear = np.power(10., drive / 20.).astype(np.float32)
    return np.tanh(2. * x * drive_linear)


def soft_sine(x, drive):
    """
    Soft sine distortion.

    Args:
        x: input audio
        drive: drive

    Returns:
        (Numpy array): distorted audio
    """
    drive_linear = np.power(10., drive / 20.).astype(np.float32)
    y = np.clip(x * drive_linear, -np.pi/4.0, np.pi/4.0)
    return np.sin(2. * y)


def bit_crusher(x, bits):
    """
    Bit crusher distortion.

    Args:
        x: input audio
        bits: bits

    Returns:
        (Numpy array): distorted audio
    """
    return np.rint(x * (2 ** bits)) / (2 ** bits)


class Distortion(Processor):
    """
    Distortion processor.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        mode (str): Currently supports the following five modes: hard_clip, waveshaper, soft_sine, tanh, bit_crusher.
            Each mode has different parameters such as threshold, factor, or bits.
        threshold (float): threshold
        drive (float): drive
        factor (float): factor
        limit_range (float): limit range
        bits (int): bits
    """

    def __init__(self, sample_rates, name='Distortion', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): sample rates of audio.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name, None, block_size=None, **kwargs)
        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('mode', 'hard_clip', 'string',
                                          options=['hard_clip',
                                                   'overdrive',
                                                   'soft_sine',
                                                   'tanh',
                                                   'bit_crusher']))
            self.parameters.add(Parameter('threshold', 0.0, 'float',
                                          units='dB', maximum=0.0, minimum=-20.0))
            self.parameters.add(Parameter('drive', 0.0, 'float',
                                          units='dB', maximum=20.0, minimum=0.0))
            self.parameters.add(Parameter('colour', 20.0, 'float',
                                          maximum=100.0, minimum=0.0))
            self.parameters.add(Parameter('bits', 12, 'int',
                                          maximum=12, minimum=8))
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): distorted audio of size `n_samples x n_channels`.
        """
        if self.parameters.mode.value == 'hard_clip':
            y = hard_clip(x, self.parameters.threshold.value, self.parameters.drive.value)
        elif self.parameters.mode.value == 'overdrive':
            y = overdrive(x, self.parameters.drive.value,
                          self.parameters.colour.value, self.parameters.sample_rate.value)
        elif self.parameters.mode.value == 'soft_sine':
            y = soft_sine(x, self.parameters.drive.value)
        elif self.parameters.mode.value == 'tanh':
            y = hyperbolic_tangent(x, self.parameters.drive.value)
        elif self.parameters.mode.value == 'bit_crusher':
            y = bit_crusher(x, self.parameters.bits.value)

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EQUALISER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Equaliser(Processor):
    """
    Five band parametric equaliser (two shelves and three central bands).

    All gains are set in dB values and range from `MIN_GAIN` dB to `MAX_GAIN` dB.
    This processor is implemented as cascade of five biquad IIR filters
    that are implemented using the infamous cookbook formulae from RBJ.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        low_shelf_gain (float), low_shelf_freq (float)
        first_band_gain (float), first_band_freq (float), first_band_q (float)
        second_band_gain (float), second_band_freq (float), second_band_q (float)
        third_band_gain (float), third_band_freq (float), third_band_q (float)

    original from https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/equaliser.py
    """

    def __init__(self, n_channels, sample_rates, gain_range=(-15.0, 15.0), q_range=(0.1, 2.0), hard_clip=False,
                 name='Equaliser', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of audio channels.
            sample_rates (list of ints): Sample rates of audio.
            gain_range (tuple of floats): minimum and maximum gain that can be used.
            q_range (tuple of floats): minimum and maximum q value.
            hard_clip (bool): Whether we clip to [-1, 1.] after processing.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name, parameters=parameters, block_size=None, **kwargs)

        self.n_channels = n_channels

        MIN_GAIN, MAX_GAIN = gain_range
        MIN_Q, MAX_Q = q_range

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            # low shelf parameters -------
            self.parameters.add(Parameter('low_shelf_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('low_shelf_freq', 80.0, 'float', minimum=30.0, maximum=200.0))
            # first band parameters ------
            self.parameters.add(Parameter('first_band_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('first_band_freq', 400.0, 'float', minimum=200.0, maximum=1000.0))
            self.parameters.add(Parameter('first_band_q', 0.7, 'float', minimum=MIN_Q, maximum=MAX_Q))
            # second band parameters -----
            self.parameters.add(Parameter('second_band_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('second_band_freq', 2000.0, 'float', minimum=1000.0, maximum=3000.0))
            self.parameters.add(Parameter('second_band_q', 0.7, 'float', minimum=MIN_Q, maximum=MAX_Q))
            # third band parameters ------
            self.parameters.add(Parameter('third_band_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('third_band_freq', 4000.0, 'float', minimum=3000.0, maximum=8000.0))
            self.parameters.add(Parameter('third_band_q', 0.7, 'float', minimum=MIN_Q, maximum=MAX_Q))
            # high shelf parameters ------
            self.parameters.add(Parameter('high_shelf_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('high_shelf_freq', 8000.0, 'float', minimum=5000.0, maximum=10000.0))
        else:
            self.parameters = parameters

        self.bands = ['low_shelf', 'first_band', 'second_band', 'third_band', 'high_shelf']
        self.filters = self.setup_filters()
        self.hard_clip = hard_clip

    def setup_filters(self):
        """
        Create IIR filters.

        Returns:
            IIR filters
        """
        filters = {}

        for band in self.bands:

            G = getattr(self.parameters, band + '_gain').value
            fc = getattr(self.parameters, band + '_freq').value
            rate = self.parameters.sample_rate.value

            if band in ['low_shelf', 'high_shelf']:
                Q = 0.707
                filter_type = band
            else:
                Q = getattr(self.parameters, band + '_q').value
                filter_type = 'peaking'

            filters[band] = pymc.components.iirfilter.IIRfilter(G, Q, fc, rate, filter_type, n_channels=self.n_channels)

        return filters

    def update_filter(self, band):
        """
        Update filters.

        Args:
            band (str): Band that should be updated.
        """
        self.filters[band].G = getattr(self.parameters, band + '_gain').value
        self.filters[band].fc = getattr(self.parameters, band + '_freq').value
        self.filters[band].rate = self.parameters.sample_rate.value

        if band in ['first_band', 'second_band', 'third_band']:
            self.filters[band].Q = getattr(self.parameters, band + '_q').value

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        if parameter_name is not None:
            bands = ['_'.join(parameter_name.split('_')[:2])]
        else:
            bands = self.bands

        for band in bands:
            self.update_filter(band)

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): equalized audio of size `n_samples x n_channels`.
        """
        for _band, iirfilter in self.filters.items():
            iirfilter.reset_state()
            x = iirfilter.apply_filter(x)

        if self.hard_clip:
            x = np.clip(x, -1.0, 1.0)

        # make sure that we have float32 as IIR filtering returns float64
        x = x.astype(np.float32)

        # make sure that we have two dimensions (if `n_channels == 1`)
        if x.ndim == 1:
            x = x[:, np.newaxis]

        return x


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPRESSOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@jit(nopython=True)
def compressor_process(x, threshold, attack_time, release_time, ratio, makeup_gain, sample_rate):
    """
    Apply compressor.

    Args:
        x (Numpy array): audio data.
        threshold: threshold in dB.
        attack_time: attack_time in ms.
        release_time: release_time in ms.
        ratio: ratio.
        makeup_gain: makeup_gain.
        sample_rate: sample rate.

    Returns:
        compressed audio.
    """
    M = x.shape[0]
    x_g = np.zeros(M)
    x_l = np.zeros(M)
    y_g = np.zeros(M)
    y_l = np.zeros(M)
    c = np.zeros(M)
    yL_prev = 0.

    alpha_attack = np.exp(-1/(0.001 * sample_rate * attack_time))
    alpha_release = np.exp(-1/(0.001 * sample_rate * release_time))

    for i in np.arange(M):
        if np.abs(x[i]) < 0.000001:
            x_g[i] = -120.0
        else:
            x_g[i] = 20 * np.log10(np.abs(x[i]))

        if x_g[i] >= threshold:
            y_g[i] = threshold + (x_g[i] - threshold) / ratio
        else:
            y_g[i] = x_g[i]

        x_l[i] = x_g[i] - y_g[i]

        if x_l[i] > yL_prev:
            y_l[i] = alpha_attack * yL_prev + (1 - alpha_attack) * x_l[i]
        else:
            y_l[i] = alpha_release * yL_prev + (1 - alpha_release) * x_l[i]

        c[i] = np.power(10.0, (makeup_gain - y_l[i]) / 20.0)
        yL_prev = y_l[i]

    y = x * c

    return y


class Compressor(Processor):
    """
    Single band stereo dynamic range compressor.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        threshold (float)
        attack_time (float)
        release_time (float)
        ratio (float)
        makeup_gain (float)
    """

    def __init__(self, sample_rates, name='Compressor', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('threshold', 0.0, 'float', units='dB', minimum=-40.0, maximum=0.0))
            self.parameters.add(Parameter('attack_time', 2.0, 'float', units='ms', minimum=0.03, maximum=30.0))
            self.parameters.add(Parameter('release_time', 50.0, 'float', units='ms', minimum=50.0, maximum=100.0))
            self.parameters.add(Parameter('ratio', 2.0, 'float', minimum=2.0, maximum=10.0))
            self.parameters.add(Parameter('makeup_gain', 0.0, 'float', units='dB', minimum=-3.0, maximum=6.0))
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): compressed audio of size `n_samples x n_channels`.
        """
        if not self.parameters.threshold.value == 0.0:
            y = np.zeros_like(x)

            for ch in range(x.shape[1]):
                y[:, ch] = compressor_process(x[:, ch],
                                              self.parameters.threshold.value,
                                              self.parameters.attack_time.value,
                                              self.parameters.release_time.value,
                                              self.parameters.ratio.value,
                                              self.parameters.makeup_gain.value,
                                              self.parameters.sample_rate.value)
        else:
            y = x

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%% CONVOLUTIONAL REVERB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ConvolutionalReverb(Processor):
    """
    Convolutional Reverb.

    Important: Due to convolving the audio sequence with some impulse response, we should ignore the
               first/last samples of the augmented audio sequence using `config['AUGMENTER_PADDING']`.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        wet_dry (float): Wet/dry ratio.
        decay (float): Applies a fade out to the impulse response.
        pre_delay (float): Value in ms. Shifts the IR in time and allows.
            A positive value produces a traditional delay between the dry signal and the wet.
            A negative delay is, in reality, zero delay, but effectively trims off the start of IR,
            so the reverb response begins at a point further in.
    """

    def __init__(self, impulse_responses, sample_rates, name='ConvolutionalReverb', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            impulse_responses (list): List with impulse responses created by `common_dataprocessing.create_dataset`
            sample_rates (list of ints): Sample rates that we should assume (used for fade-out computation)
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.

        Raises:
            ValueError: if no impulse responses are provided.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if impulse_responses is None:
            raise ValueError('List of impulse responses must be provided for ConvolutionalReverb processor.')
        self.impulse_responses = impulse_responses

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('index', 0, 'int', minimum=0, maximum=len(impulse_responses)))
            self.parameters.add(Parameter('wet_dry', 1.0, 'float', minimum=0.1, maximum=1.0))
            self.parameters.add(Parameter('decay', 1.0, 'float', minimum=0.1, maximum=1.0))
            self.parameters.add(Parameter('pre_delay', 0, 'int', units='ms', minimum=-100, maximum=100))
        else:
            self.parameters = parameters

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        # copy IR from current index (to avoid modifying it in-place)
        self.h = np.copy(sample_data(self.impulse_responses.loc[self.parameters.index.value]['impulse_response']))

        # fade out the impulse based on the decay setting (starting from peak value) - constant 20ms fade-out
        if self.parameters.decay.value < 1.:
            idx_peak = np.argmax(np.max(np.abs(self.h), axis=1), axis=0)
            fstart = np.minimum(self.h.shape[0],
                                idx_peak + int(self.parameters.decay.value * (self.h.shape[0] - idx_peak)))
            fstop = np.minimum(self.h.shape[0], fstart + int(0.020*self.parameters.sample_rate.value))
            flen = fstop - fstart

            fade = np.arange(1, flen+1, dtype=self.dtype)/flen
            fade = np.power(0.1, fade * 5)
            self.h[fstart:fstop, :] *= fade[:, np.newaxis]
            self.h = self.h[:fstop]

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): reverbed audio of size `n_samples x n_channels`.
        """
        # reshape IR to the correct size
        n_channels = x.shape[1]
        if self.h.shape[1] == 1 and n_channels > 1:
            self.h = np.hstack([self.h] * n_channels)  # repeat mono IR for multi-channel input
        if self.h.shape[1] > 1 and n_channels == 1:
            self.h = self.h[:, np.random.randint(self.h.shape[1]), np.newaxis]  # randomly choose one IR channel

        if self.parameters.wet_dry.value == 0.0:
            return x
        else:
            # perform convolution to get wet signal
            y = oaconvolve(x, self.h, mode='full', axes=0)

            # cut out wet signal (compensating for the delay that the IR is introducing + predelay)
            idx = np.argmax(np.max(np.abs(self.h), axis=1), axis=0)
            idx += int(0.001 * self.parameters.pre_delay.value * self.parameters.sample_rate.value)

            idx = np.clip(idx, 0, self.h.shape[0]-1)

            y = y[idx:idx+x.shape[0], :]

            # return weighted sum of dry and wet signal
            return (1.0 - self.parameters.wet_dry.value) * x + self.parameters.wet_dry.value * y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%% HAAS EFFECT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def haas_process(x, delay, feedback, wet_channel):
    """
    Add Haas effect to audio.

    Args:
        x (Numpy array): input audio.
        delay: Delay that we apply to one of the channels (in samples).
        feedback: Feedback value.
        wet_channel: Which channel we process (`left` or `right`).

    Returns:
        (Numpy array): Audio with Haas effect.
    """
    y = np.copy(x)
    if wet_channel == 'left':
        y[:, 0] += feedback * np.roll(x[:, 0], delay)
    elif wet_channel == 'right':
        y[:, 1] += feedback * np.roll(x[:, 1], delay)

    return y


class Haas(Processor):
    """
    Haas Effect Processor.

    Randomly selects one channel and applies a short delay to it.

    Important: This audio effect uses `np.roll` to perform the shift of one channel. Hence, you should use
               `config['AUGMENTER_PADDING']` to ignore the first samples of the augmented audio sequence.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        delay (float)
        feedback (float)
        wet_channel (string)
    """

    def __init__(self, sample_rates, delay_range=(-0.040, 0.040), name='Haas', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            delay_range (tuple of floats): minimum/maximum delay in milliseconds for Haas effect.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('delay', delay_range[1], 'float', units='ms',
                                          minimum=delay_range[0], maximum=delay_range[1]))
            self.parameters.add(Parameter('feedback', 0.35, 'float', minimum=0.33, maximum=0.66))
            self.parameters.add(Parameter('wet_channel', 'left', 'string', options=['left', 'right']))
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): audio with Haas effect of size `n_samples x n_channels`.
        """
        assert x.shape[1] == 1 or x.shape[1] == 2, 'Haas effect only works with monaural or stereo audio.'

        if x.shape[1] < 2:
            x = np.repeat(x, 2, axis=1)

        y = haas_process(x, int(self.parameters.delay.value * self.parameters.sample_rate.value),
                         self.parameters.feedback.value, self.parameters.wet_channel.value)

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PANNER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Panner(Processor):
    """
    Simple stereo panner (adjusting amplitude and delay).

    If input is mono, output is stereo.
    Original edited from https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/panner.py

    Important: This audio effect uses `np.roll` to perform the shift of one channel. Hence, you should use
               `config['AUGMENTER_PADDING']` to ignore the first samples of the augmented audio sequence.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        pan (float): Panning angle. Can take values in [0, 1] where `0` corresponds to fully panned to left
            and `1` corresponds to fully panned to the right.
        pan_law (str): Pan law to be used for amplitude panning. Can be '-4.5dB', 'linear' or 'constant_power'.
        pan_mode (str): Scheme that is used for panning. Can be 'amplitude', 'delay' or 'both'.
        pan_maxdelay (float): Maximum delay if we pan a source fully to the left/right.
            For example `2. / 343.` is the maximum delay if the microphones are 2 meters apart.
    """

    def __init__(self, sample_rates, maxdelay_range=(0, 2. / 343.0), name='Panner', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            maxdelay_range (tuple of floats): minimum/maximum delay for panning effect. `2. / 343.` corresponds to
                the maximum delay if we have a microphone array where the microphones are 2 meters apart.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        # default processor class constructor
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('pan', 0.5, 'float', minimum=0.1, maximum=0.9))
            self.parameters.add(Parameter('pan_law', '-4.5dB', 'string',
                                          options=['-4.5dB', 'linear', 'constant_power']))
            self.parameters.add(Parameter('pan_mode', 'amplitude', 'string',
                                          options=['amplitude', 'delay', 'both']))
            self.parameters.add(Parameter('pan_maxdelay', (maxdelay_range[0] + maxdelay_range[1]) / 2., 'float',
                                          units='ms', minimum=maxdelay_range[0], maximum=maxdelay_range[1]))
        else:
            self.parameters = parameters

        # setup the coefficents based on default params
        self.update()

    def _calculate_pan_coefficents(self):
        """
        Calculate panning coefficients from the chosen pan law.

        Based on the set pan law determine the gain value
        to apply for the left and right channel to achieve panning effect.
        This operates on the assumption that the input channel is mono.
        The output data will be stereo at the moment, but could be expanded
        to a higher channel count format.
        The panning value is in the range [0, 1], where
        0 means the signal is panned completely to the left, and
        1 means the signal is apanned copletely to the right.

        Raises:
            ValueError: `self.parameters.pan_law` is not supported.
        """
        self.gains = np.zeros(2, dtype=self.dtype)

        # first scale the linear [0, 1] to [0, pi/2]
        theta = self.parameters.pan.value * (np.pi/2)

        if self.parameters.pan_law.value == 'linear':
            self.gains[0] = ((np.pi/2) - theta) * (2/np.pi)
            self.gains[1] = theta * (2/np.pi)
        elif self.parameters.pan_law.value == 'constant_power':
            self.gains[0] = np.cos(theta)
            self.gains[1] = np.sin(theta)
        elif self.parameters.pan_law.value == '-4.5dB':
            self.gains[0] = np.sqrt(((np.pi/2) - theta) * (2/np.pi) * np.cos(theta))
            self.gains[1] = np.sqrt(theta * (2/np.pi) * np.sin(theta))
        else:
            raise ValueError(f'Invalid pan_law {self.parameters.pan_law.value}.')

    def _calculate_pan_delay(self):
        """Calculate delay for the chosen pan angle."""
        self.shifts = np.zeros(2, dtype=np.int32)

        # compute overall shift that we need between the two channels
        pan_maxdelay_samples = self.parameters.pan_maxdelay.value * self.parameters.sample_rate.value
        shift = 2 * int(pan_maxdelay_samples * np.abs(self.parameters.pan.value - 0.5))

        if self.parameters.pan.value < 0.5:
            # panning to the left
            self.shifts[0] = -shift // 2
            self.shifts[1] = shift - shift // 2
        else:
            self.shifts[0] = shift - shift // 2
            self.shifts[1] = -shift // 2

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): panned audio of size `n_samples x n_channels`.
        """
        assert x.shape[1] == 1 or x.shape[1] == 2, 'Panner only works with monaural or stereo audio.'

        # convert to stereo if signal is monaural
        if x.shape[1] < 2:
            x = np.repeat(x, 2, axis=1)

        y = np.copy(x)
        if self.parameters.pan_mode.value in ['delay', 'both']:
            y[:, 0] = np.roll(y[:, 0], self.shifts[0])
            y[:, 1] = np.roll(y[:, 1], self.shifts[1])

        if self.parameters.pan_mode.value in ['amplitude', 'both']:
            y *= self.gains

        return y

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        self._calculate_pan_coefficents()
        self._calculate_pan_delay()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Gain(Processor):
    """
    Gain Processor.

    Applies gain in dB and optionally inverts polarity.

    Processor parameters:
        gain (float): Gain that should be applied (dB scale).
        invert (bool): If True, then we also invert the waveform (all channels jointly).
    """

    def __init__(self, name='Gain', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('gain', 1.0, 'float', units='dB', minimum=-6.0, maximum=6.0))
            self.parameters.add(Parameter('invert', False, 'bool'))
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): gain-augmented audio of size `n_samples x n_channels`.
        """
        gain = 10 ** (self.parameters.gain.value / 20.)
        if self.parameters.invert.value:
            gain = -gain
        return gain * x


# %%%%%%%%%%%%%%%%%%%%%%% SIMPLE CHANNEL SWAP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class SwapChannels(Processor):
    """
    Swap channels in multi-channel audio.

    Processor parameters:
        index (int) Selects the permutation that we are using.
            Please note that "no permutation" is one of the permutations in `self.permutations` at index `0`.
            Hence, this effect should be applied with probability "1.".
    """

    def __init__(self, n_channels, name='SwapChannels', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of channels in audio that we want to process.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        self.permutations = tuple(permutations(range(n_channels), n_channels))

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('index', 0, 'int', minimum=0, maximum=len(self.permutations)))
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): channel-swapped audio of size `n_samples x n_channels`.
        """
        return x[:, self.permutations[self.parameters.index.value]]


# %%%%%%%%%%%%%%%%%%%%%%% Monauralize %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Monauralize(Processor):
    """
    Monauralizes audio (i.e., removes spatial information).

    Process parameters:
        seed_channel (int): channel that we use for overwriting the others.
            `-1` refers to using the mean over all channels.
    """

    def __init__(self, n_channels, name='Monauralize', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of channels in audio that we want to process.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('seed_channel', 0, 'int', minimum=-1, maximum=n_channels))
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): monauralized audio of size `n_samples x n_channels`.
        """
        tile_reps = (1, x.shape[1])
        if self.parameters.seed_channel.value == -1:
            # use average as seed
            return np.tile(np.mean(x, axis=1, keepdims=True), tile_reps)
        else:
            # use one channel as seed
            return np.tile(x[:, [self.parameters.seed_channel.value]], tile_reps)


# %%%%%%%%%%%%%%%%%%%%%%% ChunkShuffle %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ChunkShuffle(Processor):
    """
    Split audio into chunks and randomly re-arrange it.

    Process parameters:
        n_chunks (int): number of chunks into which we split
    """

    def __init__(self, name='ChunkShuffle', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('n_chunks', 2, 'int', minimum=2, maximum=6))
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): chunk-shuffled audio of size `n_samples x n_channels`.
        """
        # determine random chunk boundaries
        boundaries = np.sort(np.random.randint(x.shape[0], size=self.parameters.n_chunks.value-1))

        # create chunks
        chunks = np.split(x, boundaries)

        # shuffle them
        np.random.shuffle(chunks)

        # return chunks in new random order
        return np.concatenate(chunks)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PITCH SHIFT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class PitchShift(Processor):
    """
    Simple pitch shifter using SoX and soxbindings (https://github.com/pseeth/soxbindings).

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        steps (float): Pitch shift as positive/negative semitones
        quick (bool): If True, this effect will run faster but with lower sound quality.
    """

    def __init__(self, sample_rates, fix_length=True, name='PitchShift', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('steps', 0.0, 'float', minimum=-6., maximum=6.))
            self.parameters.add(Parameter('quick', False, 'bool'))
        else:
            self.parameters = parameters

        self.fix_length = fix_length
        self.clips = False

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): pitch-shifted audio of size `n_samples x n_channels`.
        """
        if self.parameters.steps.value == 0.0:
            y = x
        else:
            scale = np.max(np.abs(x))
            if scale > MAX_SOX_PROCESSING_PEAK:
                clips = True
                x = x * (MAX_SOX_PROCESSING_PEAK / scale)
            else:
                clips = False

            tfm = sox.Transformer()
            tfm.pitch(self.parameters.steps.value, quick=bool(self.parameters.quick.value))
            y = tfm.build_array(input_array=x, sample_rate_in=self.parameters.sample_rate.value).astype(np.float32)

            if clips:
                y *= scale / MAX_SOX_PROCESSING_PEAK  # rescale output to original scale

        if self.fix_length:
            n_samples_input = x.shape[0]
            n_samples_output = y.shape[0]
            if n_samples_input < n_samples_output:
                idx1 = (n_samples_output - n_samples_input) // 2
                idx2 = idx1 + n_samples_input
                y = y[idx1:idx2]
            elif n_samples_input > n_samples_output:
                n_pad = n_samples_input - n_samples_output
                y = np.pad(y, ((n_pad//2, n_pad - n_pad//2), (0, 0)))

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TIME STRETCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class TimeStretch(Processor):
    """
    Simple time stretcher using SoX and soxbindings (https://github.com/pseeth/soxbindings).

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        factor (float): Time stretch factor.
        quick (bool): If True, this effect will run faster but with lower sound quality.
        stretch_type (str): Algorithm used for stretching (`tempo` or `stretch`).
        audio_type (str): Sets which time segments are most optmial when finding
            the best overlapping points for time stretching.
    """

    def __init__(self, sample_rates, fix_length=True, name='TimeStretch', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('factor', 1.0, 'float', minimum=1/1.33, maximum=1.33))
            self.parameters.add(Parameter('quick', False, 'bool'))
            self.parameters.add(Parameter('stretch_type', 'tempo', 'string', options=['tempo', 'stretch']))
            self.parameters.add(Parameter('audio_type', 'l', 'string', options=['m', 's', 'l']))
        else:
            self.parameters = parameters

        self.fix_length = fix_length

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): time-stretched audio of size `n_samples x n_channels`.
        """
        if self.parameters.factor.value == 1.0:
            y = x
        else:
            scale = np.max(np.abs(x))
            if scale > MAX_SOX_PROCESSING_PEAK:
                clips = True
                x = x * (MAX_SOX_PROCESSING_PEAK / scale)
            else:
                clips = False

            tfm = sox.Transformer()
            if self.parameters.stretch_type.value == 'stretch':
                tfm.stretch(self.parameters.factor.value)
            elif self.parameters.stretch_type.value == 'tempo':
                tfm.tempo(1/self.parameters.factor.value,
                          audio_type=self.parameters.audio_type.value,
                          quick=bool(self.parameters.quick.value))
            y = tfm.build_array(input_array=x, sample_rate_in=self.parameters.sample_rate.value).astype(np.float32)

            if clips:
                y *= scale / MAX_SOX_PROCESSING_PEAK  # rescale output to original scale

        if self.fix_length:
            n_samples_input = x.shape[0]
            n_samples_output = y.shape[0]
            if n_samples_input < n_samples_output:
                idx1 = (n_samples_output - n_samples_input) // 2
                idx2 = idx1 + n_samples_input
                y = y[idx1:idx2]
            elif n_samples_input > n_samples_output:
                n_pad = n_samples_input - n_samples_output
                y = np.pad(y, ((n_pad//2, n_pad - n_pad//2), (0, 0)))

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLAYBACK SPEED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class PlaybackSpeed(Processor):
    """
    Simple playback speed effect using SoX and soxbindings (https://github.com/pseeth/soxbindings).

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        factor (float): Playback speed factor.
    """

    def __init__(self, sample_rates, fix_length=True, name='PlaybackSpeed', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('factor', 1.0, 'float', minimum=1./1.33, maximum=1.33))
        else:
            self.parameters = parameters

        self.fix_length = fix_length

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): resampled audio of size `n_samples x n_channels`.
        """
        if self.parameters.factor.value == 1.0:
            y = x
        else:
            scale = np.max(np.abs(x))
            if scale > MAX_SOX_PROCESSING_PEAK:
                clips = True
                x = x * (MAX_SOX_PROCESSING_PEAK / scale)
            else:
                clips = False

            tfm = sox.Transformer()
            tfm.speed(self.parameters.factor.value)
            y = tfm.build_array(input_array=x, sample_rate_in=self.parameters.sample_rate.value).astype(np.float32)

            if clips:
                y *= scale / MAX_SOX_PROCESSING_PEAK  # rescale output to original scale

        if self.fix_length:
            n_samples_input = x.shape[0]
            n_samples_output = y.shape[0]
            if n_samples_input < n_samples_output:
                idx1 = (n_samples_output - n_samples_input) // 2
                idx2 = idx1 + n_samples_input
                y = y[idx1:idx2]
            elif n_samples_input > n_samples_output:
                n_pad = n_samples_input - n_samples_output
                y = np.pad(y, ((n_pad//2, n_pad - n_pad//2), (0, 0)))

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEND %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Bend(Processor):
    """
    Simple bend effect using SoX and soxbindings (https://github.com/pseeth/soxbindings).

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        n_bends (int): Number of segments or intervals to pitch shift
    """

    def __init__(self, sample_rates, pitch_range=(-600, 600), fix_length=True, name='Bend', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            pitch_range (tuple of ints): min and max pitch bending ranges in cents
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('n_bends', 2, 'int', minimum=2, maximum=10))
        else:
            self.parameters = parameters
        self.pitch_range_min, self.pitch_range_max = pitch_range

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): pitch-bended audio of size `n_samples x n_channels`.
        """
        n_bends = self.parameters.n_bends.value
        max_length = x.shape[0] / self.parameters.sample_rate.value

        # Generates random non-overlapping segments
        delta = 1. / self.parameters.sample_rate.value
        boundaries = np.sort(delta + np.random.rand(n_bends-1) * (max_length - delta))

        start, end = np.zeros(n_bends), np.zeros(n_bends)
        start[0] = delta
        for i, b in enumerate(boundaries):
            end[i] = b
            start[i+1] = b
        end[-1] = max_length

        # randomly sample pitch-shifts in cents
        cents = np.random.randint(self.pitch_range_min, self.pitch_range_max+1, n_bends)

        # remove segment if cent value is zero or start == end (as SoX does not allow such values)
        idx_keep = np.logical_and(cents != 0, start != end)
        n_bends, start, end, cents = sum(idx_keep), start[idx_keep], end[idx_keep], cents[idx_keep]

        scale = np.max(np.abs(x))
        if scale > MAX_SOX_PROCESSING_PEAK:
            clips = True
            x = x * (MAX_SOX_PROCESSING_PEAK / scale)
        else:
            clips = False

        tfm = sox.Transformer()
        tfm.bend(n_bends=int(n_bends), start_times=list(start), end_times=list(end), cents=list(cents))
        y = tfm.build_array(input_array=x, sample_rate_in=self.parameters.sample_rate.value).astype(np.float32)

        if clips:
            y *= scale / MAX_SOX_PROCESSING_PEAK  # rescale output to original scale

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MASTERING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Mastering(Processor):
    """
    Mastering Processor.

    Models different mastering effects, i.e., ways how we deal with sample values outside [-1, 1].

    Processor parameters:
        method (str): Method that should be applied. Can take the following values:
            `none`: Do nothing. E.g., useful if training a network for VST plugin where DAW can also handle peaks
                larger/smaller than `+1`/`-1`.
            `scale`: Scale to max-abs peak to avoid clipping.
            `hardclip`: Apply hardclipping to [-1, 1].
            `softclip`: Apply softclipping to [-1, 1] using tanh.
    """

    def __init__(self, name='Mastering', parameters=None, maximum_amplitude=1.0, **kwargs):
        """
        Initialize processor.

        Args:
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            maximum_amplitude (float): Maximum amplitude after applying the mastering processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('method', 'none', 'string',
                                          options=['none', 'scale', 'hardclip', 'softclip']))
        else:
            self.parameters = parameters

        self.maximum_amplitude = maximum_amplitude

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): mastered audio of size `n_samples x n_channels`.
        """
        y = np.copy(x)
        if self.parameters.method.value == 'none':
            return y
        elif self.parameters.method.value == 'scale':
            maxabs = np.maximum(self.maximum_amplitude, np.max(np.abs(y)))
            return y / (maxabs / self.maximum_amplitude)
        elif self.parameters.method.value == 'hardclip':
            return np.clip(y, -self.maximum_amplitude, self.maximum_amplitude)
        elif self.parameters.method.value == 'softclip':
            return self.maximum_amplitude * np.tanh(y)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MP3 COMPRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class MP3Compression(Processor):
    """
    Models the effect of an mp3 compression using LAME.

    LAME adds some delay at the beginning - we therefore truncate the first samples

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        quality (int): in the range from 2 ... 7 (2 = highest, 7 = fastest)
        bitrate (int): supported bitrates are
            [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 320]
    """

    def __init__(self, sample_rates, fix_length=True, name='MP3Compression', parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            supported_bitrates = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 320]
            self.parameters = ParameterList()
            self.parameters.add(Parameter('sample_rate', 44100, 'string', options=sample_rates))
            self.parameters.add(Parameter('quality', 2, 'string', options=[2, 3, 4, 5, 6, 7]))
            self.parameters.add(Parameter('bitrate', 96, 'string', options=supported_bitrates))
        else:
            self.parameters = parameters

        self.fix_length = fix_length

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): MP3 compressed audio of size `n_samples x n_channels`.
        """
        # scale if max-abs peak is larger than `MAX_SOX_PROCESSING_PEAK`
        scale = np.max(np.abs(x))
        if scale > MAX_SOX_PROCESSING_PEAK:
            clips = True
            x = x * (MAX_SOX_PROCESSING_PEAK / scale)
        else:
            clips = False

        # convert to int16
        x_int = (x * np.iinfo(np.int16).max).astype(np.int16)

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self.parameters.bitrate.value)
        encoder.set_in_sample_rate(self.parameters.sample_rate.value)
        encoder.set_channels(x.shape[1])
        encoder.set_quality(self.parameters.quality.value)
        encoder.silence()

        mp3_data = encoder.encode(x_int.tobytes())
        mp3_data += encoder.flush()

        memory_file = io.BytesIO(initial_bytes=mp3_data)
        y, fs = sf.read(memory_file, always_2d=True, dtype=np.float32)

        # resample to original sampling rate if LAME changed it
        if fs != self.parameters.sample_rate.value:
            tfm = sox.Transformer()
            tfm.speed(fs / self.parameters.sample_rate.value)
            y = tfm.build_array(input_array=y, sample_rate_in=fs).astype(np.float32)

        if clips:
            y *= scale / MAX_SOX_PROCESSING_PEAK  # rescale output to original scale

        if self.fix_length:
            # LAME adds some samples at the beginning - remove them here
            n_samples_input = x.shape[0]
            n_samples_output = y.shape[0]
            if n_samples_input < n_samples_output:
                y = y[-n_samples_input:, :]

        return y


def __main__():
    """
    Main function for testing the audio effects.

    examples:

    config['AUGMENTER_CHAIN'] = AugmentationChain([(ConvolutionalReverb(impulse_responses=impulse_responses,
                                                                    sample_rates=config['ACCEPTED_SAMPLING_RATES']), 0.5),
                                               (Gain(), 0.5),
                                               (Haas(sample_rates=config['ACCEPTED_SAMPLING_RATES']), 0.5),
                                               (Panner(sample_rates=config['ACCEPTED_SAMPLING_RATES']), 0.5),
                                               (SwapChannels(n_channels=config['N_CHANNELS']), 1.),
                                               (Monauralize(n_channels=config['N_CHANNELS']), 0.5)],
                                              shuffle=True)

    config['AUGMENTER_CHAIN'] = AugmentationChain([(Gain(), 1.),
                                               (SwapChannels(n_channels=config['N_CHANNELS']), 1.),
                                               (Monauralize(n_channels=config['N_CHANNELS']), 0.5)], shuffle=True)
    """
    pass

