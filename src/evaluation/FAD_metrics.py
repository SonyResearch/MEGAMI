
from numpy.lib.scimath import sqrt as scisqrt
from scipy import linalg
import numpy as np  
import torchaudio
import os
from importlib import import_module
import torch


from evaluation.pairwise_metrics import load_AFxRep, load_fx_encoder

from utils.log import make_PCA_figure


class FADMetric:
    """
    Base class for pairwise metrics.
    
    This class should be subclassed to implement specific pairwise metrics.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the PairwiseMetric instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")



class FADFeatures(FADMetric):
    """
    Class for computing the pairwise spectral metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise spectral metric.
    """
    def __init__(self,
        type=None,
        sample_rate=44100,
        FAD_args=None,
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


        self.FAD_args = FAD_args

        assert self.FAD_args is not None, "FAD_args must be provided"

        if self.FAD_args.do_PCA_figure:
            self.pca = None

        if self.type == "fx_encoder":
            self.model_args= kwargs.get("fx_encoder_args", None)

            assert self.model_args is not None, "model_args must be provided for fx_encoder type"

            self.distance_type=self.model_args.distance_type

            self.feat_extractor = load_fx_encoder(self.model_args, self.device)


            #self.feat_extractor = load_effects_encoder(ckpt_path=ckpt_path).to(self.device)
        
        elif self.type== "AFxRep-mid" or self.type== "AFxRep-side" or self.type== "AFxRep":

            self.model_args= kwargs.get("AFxRep_args", None)

            assert self.model_args is not None, "model_args must be provided for AFxRep type"

            self.distance_type=self.model_args.distance_type

            feat_extractor = load_AFxRep(self.model_args, self.device)

            if self.type == "AFxRep-mid":
                def feat_extractor_mid(x):

                    features= feat_extractor(x)

                    #print("features shape:", features.shape)
                    #divide by 2 to get mid and side features

                    feat_mid, feat_side = features.chunk(2, dim=-1)

                    return feat_mid

                self.feat_extractor = feat_extractor_mid
            
            elif self.type == "AFxRep-side":
                def feat_extractor_side(x):

                    features= feat_extractor(x)

                    #divide by 2 to get mid and side features

                    feat_mid, feat_side = features.chunk(2, dim=-1)

                    return feat_side

                self.feat_extractor = feat_extractor_side
            else:
                self.feat_extractor = feat_extractor


        super().__init__(*args, **kwargs)
    
    def extract_features(self, y, y_hat, x=None):

        y=torch.tensor(y).permute(1,0).unsqueeze(0).to(self.device)
        y_hat=torch.tensor(y_hat).permute(1,0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_y= self.feat_extractor(y)
            feat_y_hat= self.feat_extractor(y_hat)

        if x is not None:
            x=torch.tensor(x).permute(1,0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat_x= self.feat_extractor(x)
        else:
            feat_x=None


        return feat_y, feat_y_hat, feat_x


    def calculate_emb_statistics(self, features_dicto):
        """
        Calculate the mean and standard deviation of the features.
        
        Args:
            features_dicto (dict): Dictionary containing features for each key.
            
        Returns:
            mean_features (torch.Tensor): Mean of the features.
            std_features (torch.Tensor): Standard deviation of the features.
        """

        all_features = torch.cat(list(features_dicto.values()), dim=0)

        print("all_features shape:", all_features.shape)

        #mean
        mean_features = all_features.mean(dim=0)

        #cov
        cov_features = torch.cov(all_features.T)


        return mean_features, cov_features
    
    def calculate_FAD_distance(self, mean_y, cov_y, mean_y_hat, cov_y_hat, eps=1e-6, do_check=False):
        """
        Calculate the FAD distance between two sets of features.

        #adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        # and https://github.com/microsoft/fadtk/blob/main/fadtk/fad.py
        
        Args:
            mean_y (torch.Tensor): Mean of the first set of features.
            cov_y (torch.Tensor): Covariance of the first set of features.
            mean_y_hat (torch.Tensor): Mean of the second set of features.
            cov_y_hat (torch.Tensor): Covariance of the second set of features.
            
        Returns:
            float: The FAD distance.
        """
        # Compute the squared difference between means
        mean_diff = mean_y_hat - mean_y
        mean_distance = mean_diff.dot(mean_diff)


        trace_cov_y_hat = torch.trace(cov_y_hat)
        trace_cov_y= torch.trace(cov_y)


        cov_y_hat=cov_y_hat.cpu().numpy()
        cov_y=cov_y.cpu().numpy()


        cov_dot_product = cov_y_hat.dot(cov_y)


        covmean, _ = linalg.sqrtm(cov_dot_product, disp=False)



        if not np.isfinite(covmean).all():
            #print('fid calculation produces singular product; '
            #    'adding %s to diagonal of cov estimates') % eps

            print("fid calculation produces singular product; adding eps to diagonal of cov estimates, eps:", eps)

            offset = np.eye(cov_y_hat.shape[0]) * eps
            covmean = linalg.sqrtm((cov_y_hat + offset).dot(cov_y + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        covmean_torch=torch.from_numpy(covmean).to(self.device)
        tr_covmean=torch.trace(covmean_torch)


        cov_distance= trace_cov_y_hat + trace_cov_y  - 2 * tr_covmean


        # Combine both distances
        FAD_distance = mean_distance + cov_distance


        if do_check:
            # eigenvalue method
            D, V = linalg.eig(cov_dot_product)
            covmean = (V * scisqrt(D)) @ linalg.inv(V)

            tr_covmean_eigen = np.trace(covmean)

            delt= np.abs(tr_covmean - tr_covmean_eigen)
            if delt > 1e-3:
                print("Warning: FAD distance calculation is not stable, difference between trace of covmean and eigenvalue method:", delt)


        return FAD_distance


    def do_PCA_figure(self, dict_features_y, dict_features_y_hat, dict_features_x=None):
        """
        Perform PCA on the features and create a figure.
        
        Args:
            dict_features_y (dict): Dictionary containing features for the first set.
            dict_features_y_hat (dict): Dictionary containing features for the second set.
            
        Returns:
            fig: The created figure.
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        y_values = list(dict_features_y.values())
        y_values = torch.cat(y_values, dim=0)

        print("y_values shape:", y_values.shape)


        if self.pca is None:
            self.pca = PCA(n_components=2)
            pca_result = self.pca.fit_transform(y_values.cpu().numpy())
        else:
            pca_result = self.pca.transform(y_values.cpu().numpy())


        y_hat_values = list(dict_features_y_hat.values())
        y_hat_values = torch.cat(y_hat_values, dim=0)

        #project y_hat values into the same PCA space
        pca_result_hat = self.pca.transform(y_hat_values.cpu().numpy())

        data_dict = {
            "y": pca_result,
            "y_hat": pca_result_hat
        }

        if dict_features_x is not None:
            x_values = list(dict_features_x.values())
            x_values = torch.cat(x_values, dim=0)

            pca_result_x = self.pca.transform(x_values.cpu().numpy())

            data_dict["x"] = pca_result_x



        fig= make_PCA_figure(data_dict)


        return fig





    def compute(self, dict_y, dict_y_hat, dict_x,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """

        print("Computing FAD distance...")



        dict_features_y={}
        dict_features_y_hat={}
        dict_features_x={}


        for key in dict_y.keys():
            y= dict_y[key]
            y_hat= dict_y_hat[key]
            x= dict_x[key]

            if x.shape[-2] == 1:
                x = x.repeat( 2, 1)

            assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"
            assert x.shape == y.shape, f"Shape mismatch for key {key}: {x.shape} vs {y.shape}"

            c, d=y.shape
            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            y=y.T
            y_hat=y_hat.T
            x=x.T


            #if self.type=="fx_encoder":
            feat_y, feat_y_hat, feat_x = self.extract_features(y, y_hat, x)
            dict_features_y[key] = feat_y
            dict_features_y_hat[key] = feat_y_hat
            dict_features_x[key] = feat_x
            #elif self.type=="AFxRep":
            #    feat_y, feat_y_hat = self.extract_features(y, y_hat)
            #    dict_features_y[key] = feat_y
            #    dict_features_y_hat[key] = feat_y_hat
            #else:
            #    raise ValueError(f"Unknown type: {self.type}")
            
        if self.FAD_args.do_PCA_figure:
            fig=self.do_PCA_figure(dict_features_y, dict_features_y_hat, dict_features_x)
            key= self.type+ "_PCA_figure"
            dict_output = {key: fig}

        # Compute mean features

        y_mean, y_cov = self.calculate_emb_statistics(dict_features_y)

        y_hat_mean, y_hat_cov = self.calculate_emb_statistics(dict_features_y_hat)

        # Compute the distance between the mean features
        FAD_distance = self.calculate_FAD_distance(y_mean, y_cov, y_hat_mean, y_hat_cov)


        return  FAD_distance, dict_output


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
    if metric_name == "fad-fx_encoder":
        return FADFeatures(*args, **kwargs, type="fx_encoder", sample_rate=sample_rate, model_args=kwargs.get('fx_encoder_args', None) )
    elif metric_name == "fad-AFxRep":
        return FADFeatures(*args, **kwargs, type="AFxRep", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None) )
    elif metric_name == "fad-AFxRep-mid":
        return FADFeatures(*args, **kwargs, type="AFxRep-side", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None) )
    elif metric_name == "fad-AFxRep-side":
        return FADFeatures(*args, **kwargs, type="AFxRep-mid", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None) )
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

# Example usage:
#metric_instance = metric_factory("pairwise-spectral")
#```
