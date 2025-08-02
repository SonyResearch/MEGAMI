import os
import torch 
import time
import numpy as np
#import torchaudio
import plotly.express as px
import soundfile as sf
#import plotly.graph_objects as go
import pandas as pd
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#from notebooks.shared_log_filter import frequencies

"""
Logging related functions that I wrote for my own use
This is quite a mess, but I'm too lazy to clean it up
"""
#from src.CQT_nsgt import CQT_cpx

def do_stft(noisy, clean=None, win_size=2048, hop_size=512, device="cpu", DC=True):
    """
        applies the stft, this an ugly old function, but I'm using it for logging and I'm to lazy to modify it
    """
    
    #window_fn = tf.signal.hamming_window

    #win_size=args.stft.win_size
    #hop_size=args.stft.hop_size
    window=torch.hamming_window(window_length=win_size)
    window=window.to(noisy.device)
    noisy=torch.cat((noisy, torch.zeros(noisy.shape[0],win_size).to(noisy.device)),1)
    stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
    stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)
    #stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
    #stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)
    
    if clean!=None:

       # stft_signal_clean=tf.signal.stft(clean,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
        clean=torch.cat((clean, torch.zeros(clean.shape[0],win_size).to(device)),1)
        stft_signal_clean=torch.stft(clean, win_size, hop_length=hop_size,window=window, center=False,return_complex=False)
        stft_signal_clean=stft_signal_clean.permute(0,3,2,1)
        #stft_clean_stacked=tf.stack( values=[tf.math.real(stft_signal_clean), tf.math.imag(stft_signal_clean)], axis=-1)


        if DC:
            return stft_signal_noisy, stft_signal_clean
        else:
            return stft_signal_noisy[...,1:], stft_signal_clean[...,1:]
    else:

        if DC:
            return stft_signal_noisy
        else:
            return stft_signal_noisy[...,1:]

def plot_norms(path, normsscores, normsguides, t, name):
    values=t.cpu().numpy()
     
    df=pd.DataFrame.from_dict(
                {"sigma": values[0:-1], "score": normsscores.cpu().numpy(), "guidance": normsguides.cpu().numpy()}
                )
    fig= px.line(df, x="sigma", y=["score", "guidance"],log_x=True,  log_y=True, markers=True)

    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)
    return fig
    




# Create and style traces

def plot_loss_by_sigma(sigma_means, sigma_stds, sigma_bins, log_scale=True):
    df=pd.DataFrame.from_dict(
                {"sigma": sigma_bins, "loss": sigma_means, "std": sigma_stds
                }
                )

    fig= error_line('bar', data_frame=df, x="sigma", y="loss", error_y="std", log_x=log_scale,  markers=True, range_y=[0, 2])
    
    return fig


def plot_cpxspectrogram(X):
    X=X.squeeze(1)
    X=X.cpu().numpy()
    fig=px.imshow(X, facet_col=3, animation_frame=0)
    fig.update_layout(coloraxis_showscale=False)
    return fig
def print_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reservedk
    print("memrylog",t,r,a,f)



def write_audio_file(x, sr, string: str, path='tmp', stereo=False, normalize=False):
    if normalize:
        x=x/torch.max(torch.abs(x))
    if not(os.path.exists(path)): 
        os.makedirs(path)
      
    path=os.path.join(path,string+".wav")
    if stereo:
        '''
        x has shape (B,2,T)
        '''
        x=x.permute(0,2,1) #B,T,2
        x=x.flatten(0,1) #B*T,2
        x=x.cpu().numpy()
        #if np.abs(np.max(x))>=1:
        #    #normalize to avoid clipping
        #    x=x/np.abs(np.max(x))
    else:
        x=x.flatten()
        x=x.unsqueeze(1)
        x=x.cpu().numpy()
        #if np.abs(np.max(x))>=1:
        #    #normalize to avoid clipping
        #    x=x/np.abs(np.max(x))
    sf.write(path,x,sr)
    return path


def error_line(error_y_mode='band', **kwargs):
    print("hello error line", error_y_mode)
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        print("Using error bars", kwargs)
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig


def plot_filters( x, freqs):
    '''
    This function plots a batch of lines using plotly
    args:
        x: (B, F)
    '''
    fig = px.line(x=freqs, y=x[0,:], log_x=True)

    for i in range(1,x.shape[0]):
        fig.add_trace(go.Scatter(x=freqs, y=x[i,:]))
    return fig

def plot_batch_of_lines( x, freqs, log_x=True):
    '''
    This function plots a batch of lines using plotly
    args:
        x: (B, F)
    '''
    fig = px.line(x=freqs, y=x[0,:], log_x=log_x)

    for i in range(1,x.shape[0]):
        fig.add_trace(go.Scatter(x=freqs, y=x[i,:]))
    return fig




def plot_STFT(x, operator, type='mag_dB', min_freq=100):

    # Ensure that shape is (B, F, T)
    fs = operator.sample_rate
    nfft = operator.n_fft
    hop_size = operator.hop_length

    print(x.device, operator.device)
    X = operator.apply_stft(x.to(operator.device))

    print(X.shape, x.shape, X.device)

    # , X.device)
    if type == 'mag':
        x_spec = X.abs()
    if type == 'mag_dB':
        x_spec = 20*torch.log10(X.abs()+1e-7)
    if type == 'phase':
        x_spec = X.angle()
    # if type == 'phase_unwrap':
    #     x_spec = X.angle().grad()


    if "mag" in type:
        z_min=-100
        z_max= 10
    else:
        z_min=-np.pi
        z_max= np.pi




    frequencies = torch.fft.rfftfreq(nfft) * fs
    timestamps = torch.arange(X.shape[-1]) * (hop_size / fs)
    fig = go.Figure(data=go.Heatmap(
        z=x_spec.squeeze(0).cpu().numpy(),
        x=timestamps.numpy(),
        y=frequencies.numpy(),
        colorscale='Magma',
        zmax=z_max,
        zmin=z_min
    ))

    fig.update_layout(
        yaxis=dict(
            type='log',
            title='Frequency(Hz)',
            range= [np.log10(min_freq), np.log10(frequencies[-1].item())],
        ),
        xaxis_title='Time(s)',
        title='Magnitude'
    )

    #define min and max for frequency axis

    return fig

def lineplot(x=None, dict=None, xaxis="linear", yaxis="log"):
    assert dict is not None

    y=dict.items()
    y=torch.stack([v for k,v in y])

    labels = [k for k,v in dict.items()]



    # Ensure y is a NumPy array for compatibility with Plotly
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if x is None:
        x = np.arange(y.shape[-1])  # Ensure x is NumPy array too
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Validate labels
    num_series = y.shape[0]
    if labels is None:
        labels = [f"Series {i}" for i in range(num_series)]
    elif len(labels) != num_series:
        raise ValueError(f"Expected {num_series} labels, but got {len(labels)}.")

    # Convert to DataFrame for Plotly
    df = pd.DataFrame(y.T, index=x, columns=labels)  # Use labels as column names
    df.index.name = "x"
    df = df.reset_index()

    # Melt DataFrame for Plotly (long format)
    df_melted = df.melt(id_vars=["x"], var_name="series", value_name="y")

    fig = px.line(df_melted, x="x", y="y", color="series", labels={"y": "Value", "x": "t", "series": "Legend"})

    # Set y-axis to log scale
    if yaxis=="log":
        fig.update_yaxes(type="log")
        #fix range from 1e-4 to 10
        fig.update_yaxes(range=[-4, 1])
    if xaxis=="log":
        fig.update_xaxes(type="log")


    return fig

def make_PCA_figure(data_dict, num_bins=20, title="PCA"):

    fig = plt.figure(figsize=(6, 5), dpi=200)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
    )

    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    xymax=0
    xymin=999999
    markers= ["o", "x", "D", "^", "v", "x", "p", "*", "h", "+", "s"]
    colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


    for i,(k, v) in enumerate(data_dict.items()):
        if v is None or len(v) == 0:
            continue  # Skip empty or None values

        # the scatter plot:
        ax.scatter(
            v[:, 0],
            v[:, 1],
            marker=markers[i % len(markers)],
            c=colors[i % len(colors)],
            linewidths=0.8,
            s=24,
            label=k,
            alpha=0.7,
        )

        try:
            xymax=max(xymax, np.max(v[:, :2]))
            xymin=min(xymin, np.min(v[:, :2]))
        except Exception as e:
            print(f"Error processing data for {k}: {e}")
            print("data:",v)
            print(f"Data shape: {v.shape}")
            continue

    ax.legend(loc="upper left")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    # now determine nice limits by hand:
    data_range = xymax - xymin
    binwidth = data_range / num_bins  # Calculate bin width based on the range and number of bins

    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)

    style = { "edgecolor": "black", "linewidth": 0.8, "alpha": 0.5 }



    #linestyles=[":", "-", "--", "-.", ":", "-"]
    facecolors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    hatches = ['/', '\\', '.', '+', '*', 'o', 'O', '.', '-', '|', 'x']

    for i, (k, v) in enumerate(data_dict.items()):

        hist_style = style.copy()
        hist_style["hatch"] = hatches[i % len(hatches)]

        ax_histx.hist(v[:, 0], bins=bins, density=False,  label=k, facecolor=facecolors[i], **hist_style)

        ax_histy.hist(
            v[:, 1],
            bins=bins,
            orientation="horizontal",
            density=False,
            facecolor=facecolors[i],
            **hist_style
        )

    ax_histx.set_ylabel("Count")
    ax_histy.set_xlabel("Count")

    #ax_histx.legend(loc="upper right")

    #add title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    return fig

def make_histogram_figure(data_dict, num_bins=20):

    fig = plt.figure(figsize=(5, 3), dpi=100)

    # Create the Axes.
    #ax = fig.add_subplot(gs[1, 0])
    #ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    #ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    # no labels

    #get ax from the figure
    ax = fig.add_subplot(111)

    ax.tick_params(axis="x", labelbottom=True)


    xymax=0
    xymin=999999

    for i,(k, v) in enumerate(data_dict.items()):

        xymax=max(xymax, np.max(v))
        xymin=min(xymin, np.min(v))


    data_range = xymax - xymin  
    binwidth = data_range / num_bins  # Calculate bin width based on the range and number of bins

    lim_up = (int(xymax / binwidth) + 1) * binwidth+binwidth
    lim_down= (int(xymin / binwidth) - 1) * binwidth-binwidth

    bins = np.arange(lim_down, lim_up, binwidth)

    style = { "edgecolor": "black", "linewidth": 0.8, "alpha": 0.5 }

    #linestyles=[":", "-", "--", "-.", ":", "-"]
    facecolors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    hatches = ['/', '\\', '.', '+', '*', 'o', 'O', '.', '-', '|', 'x']

    for i, (k, v) in enumerate(data_dict.items()):

        hist_style = style.copy()
        hist_style["hatch"] = hatches[i % len(hatches)]

        ax.hist(v[:], bins=bins, density=False,  label=k, facecolor=facecolors[i], **hist_style)

    ax.set_ylabel("Count")

    ax.legend(loc="upper right")


    return fig

