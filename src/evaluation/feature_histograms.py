

from numpy.lib.scimath import sqrt as scisqrt
from scipy import linalg
import numpy as np  
import torchaudio
import os
from importlib import import_module
import torch



from utils.feature_extractors.audio_features import compute_audio_features

from utils.log import make_histogram_figure




class HistogramFeatures:
    """
    Class for computing the pairwise spectral metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise spectral metric.
    """
    def __init__(self,
        type=None,
        sample_rate=44100,
        histogram_feature_args=None,
        *args, **kwargs):
        """
        Initialize the PairwiseSpectral instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.type = type
        self.sample_rate = sample_rate

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.histogram_feature_args = histogram_feature_args

        #assert self.histogram_feature_args is not None, "FAD_args must be provided"

        #if self.FAD_args.do_PCA_figure:
        #    self.pca = None

        if self.type == "audio_features":

            self.feat_extractor = lambda x: compute_audio_features(x, sample_rate)

            
        #super().__init__(*args, **kwargs)
    

    def do_histogram_figure(self, dict_features_y, dict_features_y_hat, dict_features_x=None):
        """
        Perform PCA on the features and create a figure.
        
        Args:
            dict_features_y (dict): Dictionary containing dictionaries of features for the first set.
            dict_features_y_hat (dict): Dictionary containing dictionaries of features for the second set.
            
        Returns:
            fig: The created figure.
        """

        #first get the list of features keys

        features_keys = list(dict_features_y.values())[0].keys()

        figs_dict = {}

        for key in features_keys:

            print(f"Computing histogram for feature {key}...")

            y_values= []
            y_hat_values = []

            for k2 in dict_features_y.keys():
                #y_values.append(torch.tensor(dict_features_y[k2][key]))
                #y_hat_values.append(torch.tensor(dict_features_y_hat[k2][key]))
                y_val=torch.tensor(dict_features_y[k2][key])
                y_hat_val=torch.tensor(dict_features_y_hat[k2][key])

                if y_val.dim() == 0:
                    y_val = y_val.reshape(1)
                if y_hat_val.dim() == 0:
                    y_hat_val = y_hat_val.reshape(1)
    
                y_values.append(y_val)
                y_hat_values.append(y_hat_val)

            #concatenate all values
            y_values = torch.cat(y_values, dim=0)
            y_values= y_values.cpu().numpy()
            #y_values=np.ar
            #y_values=torch.tensor(y_values, device=self.device)
            y_hat_values = torch.cat(y_hat_values, dim=0)
            y_hat_values= y_hat_values.cpu().numpy()
            #y_hat_values= np.concatenate(y_hat_values, axis=0)
            #y_hat_values=torch.tensor(y_hat_values, device=self.device)
            data_dict = {
                "y": y_values,
                "y_hat": y_hat_values,
            }


            if dict_features_x is not None:
                x_values = []
                for k2 in dict_features_x.keys():
                    x_val=torch.tensor(dict_features_x[k2][key])
                    if x_val.dim() == 0:
                        x_val = x_val.reshape(1)
                    
                    x_values.append(x_val)


                #x_values = np.concatenate(x_values, axis=0)
                x_values = torch.cat(x_values, dim=0)
                #x_values = torch.tensor(x_values, device=self.device)
                x_values = x_values.cpu().numpy()

                data_dict["x"] = x_values   



            fig= make_histogram_figure(data_dict)

            figs_dict[key] = fig


        return figs_dict





    def compute(self, dict_y, dict_y_hat, dict_x,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """

        print("Computing histogram features...")



        dict_features_y={}
        dict_features_y_hat={}
        dict_features_x={}


        for key in dict_y.keys():

            y= dict_y[key]
            y_hat= dict_y_hat[key]
            #x= dict_x[key]

            assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"

            #if x.shape[-2]==1:
            #    x=x.repeat(2,1)

            #assert x.shape== y.shape, f"Shape mismatch for key {key}: {x.shape} vs {y.shape}"

            c, d=y.shape


            assert c==2, f"Expected 2 channels, got {c} for key {key}"



            #feat_x=self.feat_extractor(x)
            feat_y= self.feat_extractor(y)
            feat_y_hat= self.feat_extractor(y_hat)

            #feat.. are dictionaries of features

            dict_features_y[key] = feat_y
            dict_features_y_hat[key] = feat_y_hat
            #dict_features_x[key] = feat_x


        figs_dict=self.do_histogram_figure(dict_features_y, dict_features_y_hat )



        #modify the keys of figs_dict to include the type

        figs_dict = {"histogram_figure_"+k: v for k, v in figs_dict.items()}


        return  None, figs_dict


def metric_factory(metric_name, sample_rate, *args, **kwargs):
    """
    Factory function to create a metric function based on the metric name.
    
    Args:
        metric_name (str): The name of the metric to create.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        
    Returns:
        An instance of a class that implements the metric function.
    """
    if metric_name == "histogram-features":
        return HistogramFeatures(*args, **kwargs, type="audio_features", sample_rate=sample_rate )
    else:
        raise ValueError(f"Unknown metric: {metric_name}")



