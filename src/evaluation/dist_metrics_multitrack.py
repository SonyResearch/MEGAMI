
from numpy.lib.scimath import sqrt as scisqrt
from scipy import linalg
import numpy as np  
import torchaudio
import os
from importlib import import_module
import torch


from evaluation.feature_extractors import load_AFxRep, load_fx_encoder, load_fx_encoder_plusplus, load_MERT, load_CLAP

from utils.log import make_PCA_figure

SCALE_FACTOR = 100

def calc_kernel_audio_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    bandwidth=None,
    kernel='gaussian',
    precision=torch.float32,
    eps=1e-8
) -> torch.Tensor:
    """
    Compute the Kernel Audio Distance (KAD) between two samples using PyTorch.

    Args:
        x: The first set of embeddings of shape (m, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).
        cache_dirs: Directories to cache kernel statistics.
        bandwidth: The bandwidth value for the Gaussian RBF kernel.
        kernel: Kernel function to use ('gaussian', 'iq', 'imq').
        precision: Type setting for matrix calculation precision.
        eps: Small value to prevent division by zero.

    Returns:
        The KAD between x and y embedding sets.
    """
    # Ensure x and y are of the correct precision
    x = x.to(dtype=precision, device=device)
    y = y.to(dtype=precision, device=device)

    assert bandwidth is not None, "Bandwidth must be provided for KAD calculation"

    m, n = x.shape[0], y.shape[0]
    
    # Define kernel functions
    gamma = 1 / (2 * bandwidth**2 + eps)
    if kernel == 'gaussian':    # Gaussian Kernel
        kernel = lambda a: torch.exp(-gamma * a)
    elif kernel == 'iq':        # Inverse Quadratic Kernel
        kernel = lambda a: 1 / (1 + gamma * a)
    elif kernel == 'imq':       # Inverse Multiquadric Kernel
        kernel = lambda a: 1 / torch.sqrt(1 + gamma * a)
    else:
        raise ValueError("Invalid kernel type. Valid kernels: 'gaussian', 'iq', 'imq'")
    
    # Load x kernel statistics
    xx = x @ x.T
    x_sqnorms = torch.diagonal(xx)
    d2_xx = x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx # shape (m, m)
            
    k_xx = kernel(d2_xx)
    k_xx = k_xx - torch.diag(torch.diagonal(k_xx))
    k_xx_mean = k_xx.sum() / (m * (m - 1))
    
    # Load y kernel statistics
    yy = y @ y.T
    y_sqnorms = torch.diagonal(yy)
    d2_yy = y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy # shape (n, n)

    k_yy = kernel(d2_yy)
    k_yy = k_yy - torch.diag(torch.diagonal(k_yy))
    k_yy_mean = k_yy.sum() / (n * (n - 1))
    
    # Compute kernel statistics for xy
    xy = x @ y.T
    d2_xy = x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy # shape (m, n)
    k_xy = kernel(d2_xy)
    k_xy_mean = k_xy.mean()
    
    # Compute MMD
    result = k_xx_mean + k_yy_mean - 2 * k_xy_mean

    return result * SCALE_FACTOR


def median_pairwise_distance(x, subsample=None):
    """
    Compute the median pairwise distance of an embedding set.
    
    Args:
    x: torch.Tensor of shape (n_samples, embedding_dim)
    subsample: int, number of random pairs to consider (optional)
    
    Returns:
    The median pairwise distance between points in x.
    """
    x = torch.tensor(x, dtype=torch.float32)
    n_samples = x.shape[0]
    
    if subsample is not None and subsample < n_samples * (n_samples - 1) / 2:
        # Randomly select pairs of indices
        idx1 = torch.randint(0, n_samples, (subsample,))
        idx2 = torch.randint(0, n_samples, (subsample,))
        
        # Ensure idx1 != idx2
        mask = idx1 == idx2
        idx2[mask] = (idx2[mask] + 1) % n_samples
        
        # Compute distances for selected pairs
        distances = torch.sqrt(torch.sum((x[idx1] - x[idx2])**2, dim=1))
    else:
        # Compute all pairwise distances
        distances = torch.pdist(x)
        
    return torch.median(distances).item()

class DistMetric:
    """
    Base class for pairwise metrics.
    
    This class should be subclassed to implement specific pairwise metrics.
    """
    def __init__(self,type, sample_rate,*args, **kwargs):
        """
        Initialize the PairwiseMetric instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.type=type
        self.sample_rate=sample_rate

        self.taxonomy_ref= kwargs.get("taxonomy_ref", None)

        if self.type == "fx_encoder":
            self.model_args= kwargs.get("fx_encoder_args", None)

            assert self.model_args is not None, "model_args must be provided for fx_encoder type"

            self.distance_type=self.model_args.distance_type

            self.feat_extractor = load_fx_encoder(self.model_args, self.device)


            #self.feat_extractor = load_effects_encoder(ckpt_path=ckpt_path).to(self.device)
        elif self.type == "fx_encoder_++":
            self.model_args= kwargs.get("fx_encoder_plusplus_args", None)

            assert self.model_args is not None, "model_args must be provided for fx_encoder_plusplus type"

            self.distance_type=self.model_args.distance_type

            self.feat_extractor = load_fx_encoder_plusplus(self.model_args, self.device)

            #self.feat_extractor = load_effects_encoder(ckpt_path=ckpt_path).to(self.device)
        
        elif self.type== "AFxRep-mid" or self.type== "AFxRep-side" or self.type== "AFxRep":

            self.model_args= kwargs.get("AFxRep_args", None)

            assert self.model_args is not None, "model_args must be provided for AFxRep type"

            self.distance_type=self.model_args.distance_type

            feat_extractor = load_AFxRep(self.model_args, self.device, sample_rate=self.sample_rate)

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
        elif self.type == "MERT":
            self.model_args= kwargs.get("MERT_args", None)
            assert self.model_args is not None, "model_args must be provided for MERT type"

            self.feat_extractor = load_MERT(self.model_args, self.device)

        elif self.type == "CLAP":
            self.model_args= kwargs.get("CLAP_args", None)
            assert self.model_args is not None, "model_args must be provided for CLAP type"

            self.feat_extractor = load_CLAP(self.model_args, self.device)


        else:
            raise ValueError(f"Unknown type: {self.type}. Supported types: fx_encoder, fx_encoder_plusplus, AFxRep-mid, AFxRep-side, AFxRep")


    def compute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

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


    def do_TSNE_figure(self, dict_features_y, dict_features_y_hat, dict_features_x=None, fit_mode="target", dict_cluster=None, dict_taxonomy=None):
        """
        Perform PCA on the features and create a figure.
        
        Args:
            dict_features_y (dict): Dictionary containing features for the first set.
            dict_features_y_hat (dict): Dictionary containing features for the second set.
            
        Returns:
            fig: The created figure.
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE   

        print("cluster", dict_cluster)

        y_values = {}
        y_hat_values = {}
        clusters={}
        clusters_hat= {}
        for key, track_name in self.taxonomy_ref.items():

            print("searching for key:", key, "track_name:", track_name)

            y_values[track_name] = []
            clusters[track_name] = []
            for k in range(len(dict_features_y.keys())):
                index = [i for i, tax_id in enumerate(dict_taxonomy[k]) if tax_id == key]
                if index:  # If any matches were found
                    # Select the corresponding rows from dict_features_y[k]
                    selected_features = dict_features_y[k][index]
                    y_values[track_name].append(selected_features)

                    clusters[track_name].append(dict_cluster[k].unsqueeze(0))



            y_values[track_name]=torch.cat(y_values[track_name], dim=0)
            y_values[track_name]=torch.nan_to_num(y_values[track_name], nan=0)

            #print(clusters[track_name])
            clusters[track_name]=torch.cat(clusters[track_name], dim=0)

            y_hat_values[track_name] = []
            clusters_hat[track_name] = []
            for k in range(len(dict_features_y_hat.keys())):
                index = [i for i, tax_id in enumerate(dict_taxonomy[k]) if tax_id == key]
                if index:  # If any matches were found
                    # Select the corresponding rows from dict_features_y[k]
                    selected_features = dict_features_y_hat[k][index]
                    y_hat_values[track_name].append(selected_features)
                    clusters_hat[track_name].append(dict_cluster[k].unsqueeze(0))
                #else:
                #    y_hat_values[track_name].append(None)  # Append zeros if no match found


            y_hat_values[track_name]=torch.cat(y_hat_values[track_name], dim=0)
            y_hat_values[track_name]=torch.nan_to_num(y_hat_values[track_name], nan=0)

            #print(clusters_hat[track_name])
            clusters_hat[track_name]=torch.cat(clusters_hat[track_name], dim=0)
            
        if self.pca is None:
            self.pca= {}
        

        tsne_result = {}
        tsne_result_hat = {}
        data_dict = {}
        figs= {}

        for k in y_values.keys():
            print("Processing track:", k)
            self.tsne =TSNE(n_components=2, perplexity=30)
            combined_data = torch.cat([y_values[k], y_hat_values[k]], dim=0).cpu().numpy()
            combined_result = self.tsne.fit_transform(combined_data)

            n_y = y_values[k].shape[0]
            tsne_result[k] = combined_result[:n_y]
            tsne_result_hat[k] = combined_result[n_y:]


            if dict_cluster is not None:
               data_dict[k] = {
                   "y_C0": tsne_result[k][clusters[k] == 0],
                   "y_hat_C0": tsne_result_hat[k][clusters_hat[k] == 0],
                   "y_C1": tsne_result[k][clusters[k] == 1],
                   "y_hat_C1": tsne_result_hat[k][clusters_hat[k] == 1]
               }
            else:
               data_dict[k] = {
                   "y": tsne_result[k],
                   "y_hat": tsne_result_hat[k] } 

            figs[k]= make_PCA_figure(data_dict[k], title=self.type + " TSNE; track: "+k)


        return figs

    def do_PCA_figure(self, dict_features_y, dict_features_y_hat, dict_features_x=None, fit_mode="target", dict_cluster=None, dict_taxonomy=None):
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

        print("cluster", dict_cluster)

        y_values = {}
        y_hat_values = {}
        clusters={}
        clusters_hat= {}
        for key, track_name in self.taxonomy_ref.items():

            print("searching for key:", key, "track_name:", track_name)

            y_values[track_name] = []
            clusters[track_name] = []
            for k in range(len(dict_features_y.keys())):
                index = [i for i, tax_id in enumerate(dict_taxonomy[k]) if tax_id == key]
                if index:  # If any matches were found
                    # Select the corresponding rows from dict_features_y[k]
                    selected_features = dict_features_y[k][index]
                    y_values[track_name].append(selected_features)

                    clusters[track_name].append(dict_cluster[k].unsqueeze(0))

                #else:
                #    y_values[track_name].append(None)  # Append zeros if no match found
            #for i in range(len(y_values[track_name])):
            #    if y_values[track_name][i] is None:
            #        print(f"Warning: No features found for track {track_name} at index {i}")
            #    print("y_values[track_name][i] shape:", y_values[track_name][i].shape)

            y_values[track_name]=torch.cat(y_values[track_name], dim=0)
            y_values[track_name]=torch.nan_to_num(y_values[track_name], nan=0)

            #print(clusters[track_name])
            clusters[track_name]=torch.cat(clusters[track_name], dim=0)

            y_hat_values[track_name] = []
            clusters_hat[track_name] = []
            for k in range(len(dict_features_y_hat.keys())):
                index = [i for i, tax_id in enumerate(dict_taxonomy[k]) if tax_id == key]
                if index:  # If any matches were found
                    # Select the corresponding rows from dict_features_y[k]
                    selected_features = dict_features_y_hat[k][index]
                    y_hat_values[track_name].append(selected_features)
                    clusters_hat[track_name].append(dict_cluster[k].unsqueeze(0))
                #else:
                #    y_hat_values[track_name].append(None)  # Append zeros if no match found


            y_hat_values[track_name]=torch.cat(y_hat_values[track_name], dim=0)
            y_hat_values[track_name]=torch.nan_to_num(y_hat_values[track_name], nan=0)

            #print(clusters_hat[track_name])
            clusters_hat[track_name]=torch.cat(clusters_hat[track_name], dim=0)
            

        #if dict_cluster is not None:
        #    clusters= list(dict_cluster.values())
        #    clusters = [c.unsqueeze(0) if c.dim() == 0 else c for c in clusters]
        #    clusters = torch.cat(clusters, dim=0)

        #    #print(clusters)
        #    #print("Clusters shape:", clusters.shape)
            #print("Number of unique clusters:", len(torch.unique(clusters)), "clusters",torch.unique(clusters))

        #    #check different clusters (0,1,2,3...)
        #    assert len(torch.unique(clusters)) <= 2, "Only two clusters are supported for PCA visualization, no more no less"
        #    C0= clusters == 0
        #    C1= clusters == 1

        if self.pca is None:
            self.pca= {}
        

        pca_result = {}
        pca_result_hat = {}
        data_dict = {}
        figs= {}

        for k in y_values.keys():
            print("Processing track:", k)
            #check if pca already exists for this key
            if k not in self.pca.keys():
                if fit_mode == "target":
                    print("Fitting PCA for target values only track:", k)
                    self.pca[k] = PCA(n_components=2).fit(y_values[k].cpu().numpy())
                elif fit_mode == "all":
                    print("Fitting PCA for all values track:", k)
                    self.pca[k] = PCA(n_components=2).fit(torch.cat([y_values[k], y_hat_values[k]], dim=0).cpu().numpy())
                else:
                    raise ValueError(f"Unknown fit_mode: {fit_mode}. Supported modes: target, all")

            pca_result[k]=self.pca[k].transform(y_values[k].cpu().numpy())
            pca_result_hat[k] = self.pca[k].transform(y_hat_values[k].cpu().numpy())

            print(pca_result[k].shape, pca_result_hat[k].shape)


            if dict_cluster is not None:
               data_dict[k] = {
                   "y_C0": pca_result[k][clusters[k] == 0],
                   "y_hat_C0": pca_result_hat[k][clusters_hat[k] == 0],
                   "y_C1": pca_result[k][clusters[k] == 1],
                   "y_hat_C1": pca_result_hat[k][clusters_hat[k] == 1]
               }
            else:
               data_dict[k] = {
                   "y": pca_result[k],
                   "y_hat": pca_result_hat[k]
               }

            if dict_features_x is not None:
                raise NotImplementedError("PCA for x features is not implemented yet")
                #x_values = list(dict_features_x.values())
                #x_values = torch.cat(x_values, dim=0)
    
                #pca_result_x = self.pca.transform(x_values.cpu().numpy())
    
                #data_dict["x"] = pca_result_x


            print("data dict", data_dict[k])

            figs[k]= make_PCA_figure(data_dict[k], title=self.type + " PCA; track: "+k)



        return figs




class KADFeatures(DistMetric):
    """
    Class for computing the pairwise spectral metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise spectral metric.
    """
    def __init__(self,
        type=None,
        sample_rate=44100,
        KAD_args=None,
        classwise=False,
                  *args, **kwargs):
        """
        Initialize the PairwiseSpectral instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.type = type

        self.classwise = classwise

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.KAD_args = KAD_args

        assert self.KAD_args is not None, "FAD_args must be provided"

        if self.KAD_args.do_PCA_figure:
            self.pca = None

        self.bandwidth = None

        super().__init__(self.type, sample_rate, *args, **kwargs)
    
    def calculate_KAD_distance_classwise(self, dict_features_y, dict_features_y_hat, dict_cluster, dict_taxonomy=None):

        #if dict_cluster is not None:
        #    clusters= list(dict_cluster.values())
        #    clusters = [c.unsqueeze(0) if c.dim() == 0 else c for c in clusters]
        #    clusters = torch.cat(clusters, dim=0)

        #    #check different clusters (0,1,2,3...)
        #    assert len(torch.unique(clusters)) == 2, "Only two clusters are supported for PCA visualization"
        #    C0= clusters == 0
        #    C1= clusters == 1

        y_values = {}
        y_hat_values = {}
        clusters={}
        clusters_hat= {}
        for key, track_name in self.taxonomy_ref.items():
            y_values[track_name] = []
            clusters[track_name] = []
            for k in range(len(dict_features_y.keys())):
                index = [i for i, tax_id in enumerate(dict_taxonomy[k]) if tax_id == key]
                if index:  # If any matches were found
                    # Select the corresponding rows from dict_features_y[k]
                    selected_features = dict_features_y[k][index]
                    y_values[track_name].append(selected_features)
                    clusters[track_name].append(dict_cluster[k].unsqueeze(0))
                #else:
                #    y_values[track_name].append(None)  # Append zeros if no match found

            y_values[track_name]=torch.cat(y_values[track_name], dim=0)
            y_values[track_name]=torch.nan_to_num(y_values[track_name], nan=0)
            clusters[track_name]=torch.cat(clusters[track_name], dim=0)

            y_hat_values[track_name] = []
            clusters_hat[track_name] = []
            for k in range(len(dict_features_y_hat.keys())):
                index = [i for i, tax_id in enumerate(dict_taxonomy[k]) if tax_id == key]
                if index:  # If any matches were found
                    # Select the corresponding rows from dict_features_y[k]
                    selected_features = dict_features_y_hat[k][index]
                    y_hat_values[track_name].append(selected_features)
                    clusters_hat[track_name].append(dict_cluster[k].unsqueeze(0))
                #else:
                #    y_hat_values[track_name].append(None)  # Append zeros if no match found

            y_hat_values[track_name]=torch.cat(y_hat_values[track_name], dim=0)
            y_hat_values[track_name]=torch.nan_to_num(y_hat_values[track_name], nan=0)

            clusters_hat[track_name]=torch.cat(clusters_hat[track_name], dim=0)

        KAD_C0={}
        weight_C0={}
        KAD_C1={}
        weight_C1={}

        total_examples= 0
        for key in y_values.keys():
            y= y_values[key]
            y_hat= y_hat_values[key]
            y_C0= y[clusters[key] == 0]
            y_hat_C0= y_hat[clusters_hat[key] == 0]

            with torch.no_grad():
                KAD_C0[key]=calc_kernel_audio_distance(y_C0, y_hat_C0, device=self.device, bandwidth=self.bandwidth, kernel=self.KAD_args.kernel, precision=torch.float32)

            weight_C0[key]= y.shape[0]
            total_examples += weight_C0[key]

            y_C1= y[clusters[key] == 1]
            y_hat_C1= y_hat[clusters_hat[key] == 1]

            with torch.no_grad():
                KAD_C1[key]=calc_kernel_audio_distance(y_C1, y_hat_C1, device=self.device, bandwidth=self.bandwidth, kernel=self.KAD_args.kernel, precision=torch.float32)
            
            weight_C1[key]= y.shape[0]
            total_examples += weight_C1[key]

        
        KAD= sum([KAD_C0[k] * weight_C0[k] for k in KAD_C0.keys()]) / total_examples
        KAD+= sum([KAD_C1[k] * weight_C1[k] for k in KAD_C1.keys()]) / total_examples

        return KAD.cpu().item()

    def calculate_KAD_distance(self, dict_features_y, dict_features_y_hat, dict_taxonomy=None):

        y_values = {}
        y_hat_values = {}
        for key, track_name in self.taxonomy_ref.items():
            y_values[track_name] = []
            for k in range(len(dict_features_y.keys())):
                index = [i for i, tax_id in enumerate(dict_taxonomy[k]) if tax_id == key]
                if index:  # If any matches were found
                    # Select the corresponding rows from dict_features_y[k]
                    selected_features = dict_features_y[k][index]
                    y_values[track_name].append(selected_features)
                #else:
                #    y_values[track_name].append(None)  # Append zeros if no match found

            y_values[track_name]=torch.cat(y_values[track_name], dim=0)
            y_values[track_name]=torch.nan_to_num(y_values[track_name], nan=0)

            y_hat_values[track_name] = []
            for k in range(len(dict_features_y_hat.keys())):
                index = [i for i, tax_id in enumerate(dict_taxonomy[k]) if tax_id == key]
                if index:  # If any matches were found
                    # Select the corresponding rows from dict_features_y[k]
                    selected_features = dict_features_y_hat[k][index]
                    y_hat_values[track_name].append(selected_features)
                #else:
                #    y_hat_values[track_name].append(None)  # Append zeros if no match found

            y_hat_values[track_name]=torch.cat(y_hat_values[track_name], dim=0)
            y_hat_values[track_name]=torch.nan_to_num(y_hat_values[track_name], nan=0)
            

        KAD={}
        weight={}
        total_examples= 0
        for key in y_values.keys():
            y= y_values[key]
            y_hat= y_hat_values[key]
            with torch.no_grad():
                KAD[key]=calc_kernel_audio_distance(y, y_hat, device=self.device, bandwidth=self.bandwidth, kernel=self.KAD_args.kernel, precision=torch.float32)

            weight[key]= y.shape[0]
            total_examples += weight[key]

        KAD= sum([KAD[k] * weight[k] for k in KAD.keys()]) / total_examples

        return KAD.cpu().item()

    def calculate_bandwidth(self, dict_features_y):

        print("Calculating bandwidth...")
        y= torch.cat(list(dict_features_y.values()), dim=0)
        print("y shape:", y.shape)
        self.bandwidth = median_pairwise_distance(y)
        print("Bandwidth:", self.bandwidth)



    def compute(self, dict_y, dict_y_hat, dict_x, dict_taxonomy=None, dict_p_hat=None, dict_cluster=None, *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """

        print("Computing KAD distance...")


        if self.classwise:
            assert dict_cluster is not None, "dict_cluster must be provided if classwise is True"

        dict_features_y={}
        dict_features_y_hat={}
        #dict_features_x={}

        if dict_y_hat is None:
            print("computing KAD with style embeddings")
            assert dict_p_hat is not None, "dict_p_hat must be provided if dict_y_hat is None"

            for key in dict_y.keys():
                y= dict_y[key]

                embed= dict_p_hat[key]
                embed=torch.tensor(embed).to(self.device)

                embed_mid, embed_side = torch.chunk(embed, 2, dim=-1)

                if self.type== "AFxRep-mid":
                    p_hat= embed_mid
                elif self.type== "AFxRep-side":
                    p_hat= embed_side
                elif self.type== "AFxRep":
                    p_hat= embed
    
    
                n, c, d=y.shape
    
                assert c==2, f"Expected 2 channels, got {c} for key {key}"
    

                y=torch.tensor(y).to(self.device)

                with torch.no_grad():
                    feat_y= self.feat_extractor(y)
                    #feat_x= self.feat_extractor(x)
    
                assert p_hat.shape == feat_y.shape, f"Shape mismatch for key {key}: {p_hat.shape} vs {feat_y.shape}"
                #assert p_hat.shape == feat_x.shape, f"Shape mismatch for key {key}: {p_hat.shape} vs {feat_x.shape}"

                dict_features_y[key] = feat_y
                #dict_features_x[key] = feat_x
                dict_features_y_hat[key] = p_hat

        else:
            for key in dict_y.keys():
                y= dict_y[key]
                #x= dict_x[key]
                y_hat= dict_y_hat[key]
    
                #if x.shape[-2] == 1:
                #    x = x.repeat( 2, 1)
                
    
                assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"
                #assert x.shape == y.shape, f"Shape mismatch for key {key}: {x.shape} vs {y.shape}"
    
                n, c, d=y.shape
                #assert b==1, f"Expected batch size of 1, got {b} for key {key}"
                if dict_taxonomy is not None:
                    taxonomy= dict_taxonomy[key]
                    assert len(taxonomy) == n, f"Taxonomy length mismatch for key {key}: {len(taxonomy)} vs {n}"
    
                assert c==2, f"Expected 2 channels, got {c} for key {key}"
    
                #y=y.T
                #y_hat=y_hat.T
                #x=x.T

                y=torch.tensor(y).to(self.device)
                y_hat=torch.tensor(y_hat).to(self.device)
    
                #y=torch.tensor(y).permute(1,0).unsqueeze(0).to(self.device)
                #y_hat=torch.tensor(y_hat).permute(1,0).unsqueeze(0).to(self.device)
                #x=torch.tensor(x).permute(1,0).unsqueeze(0).to(self.device)
        
                with torch.no_grad():
                    feat_y= self.feat_extractor(y)
                    feat_y_hat= self.feat_extractor(y_hat)
                    #feat_x= self.feat_extractor(x)

                dict_features_y[key] = feat_y
                dict_features_y_hat[key] = feat_y_hat
                #dict_features_x[key] = feat_x
            
        dict_output = {}
        if self.KAD_args.do_PCA_figure:
            dict_figs=self.do_PCA_figure(dict_features_y, dict_features_y_hat,  dict_cluster=dict_cluster, dict_taxonomy=dict_taxonomy, fit_mode=self.KAD_args.PCA_fit_mode)
            #returns a dictionary with figures for each taxonomy key
            #modify the key to include the type
            #key= self.type+ "_PCA_figure"
            #dict_output = {key: fig}

            for k, v in dict_figs.items():
                #modify the key to include the type
                key= self.type+ "_PCA_figure_" + k
                dict_output[key] = v

        if self.KAD_args.do_TSNE_figure:
            dict_figs=self.do_TSNE_figure(dict_features_y, dict_features_y_hat,dict_cluster=dict_cluster, dict_taxonomy=dict_taxonomy)
            for  k, fig in dict_figs.items():
                key= self.type+ "_TSNE_figure"+k
                dict_output[key] = fig
        

        # Compute mean features

        #y_mean, y_cov = self.calculate_emb_statistics(dict_features_y)

        #y_hat_mean, y_hat_cov = self.calculate_emb_statistics(dict_features_y_hat)


        # Compute the distance between the mean features
        #FAD_distance = self.calculate_FAD_distance(y_mean, y_cov, y_hat_mean, y_hat_cov)
        if self.bandwidth is None:
            self.calculate_bandwidth(dict_features_y)
        
        if self.classwise:
            #calculate KAD distance for each class
            assert dict_cluster is not None, "dict_cluster must be provided if classwise is True"
            KAD_distance = self.calculate_KAD_distance_classwise(dict_features_y, dict_features_y_hat, dict_cluster, dict_taxonomy)
        else:

            KAD_distance = self.calculate_KAD_distance(dict_features_y, dict_features_y_hat, dict_taxonomy)


        return  KAD_distance, dict_output



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
    if metric_name == "kad-fx_encoder-multitrack":
        return KADFeatures(*args, **kwargs, type="fx_encoder", sample_rate=sample_rate)
    elif metric_name == "kad-AFxRep-multitrack":
        return KADFeatures(*args, **kwargs, type="AFxRep", sample_rate=sample_rate)
    elif metric_name == "kad-AFxRep-mid-multitrack":
        return KADFeatures(*args, **kwargs, type="AFxRep-side", sample_rate=sample_rate)
    elif metric_name == "kad-AFxRep-side-multitrack":
        return KADFeatures(*args, **kwargs, type="AFxRep-mid", sample_rate=sample_rate)
    elif metric_name == "kad-class-fx_encoder-multitrack":
        return KADFeatures(*args, **kwargs, type="fx_encoder", sample_rate=sample_rate, classwise=True)
    elif metric_name == "kad-class-AFxRep-multitrack":
        return KADFeatures(*args, **kwargs, type="AFxRep", sample_rate=sample_rate, classwise=True)
    elif metric_name == "kad-class-AFxRep-mid-multitrack":
        return KADFeatures(*args, **kwargs, type="AFxRep-side", sample_rate=sample_rate, classwise=True)
    elif metric_name == "kad-class-AFxRep-side-multitrack":
        return KADFeatures(*args, **kwargs, type="AFxRep-mid", sample_rate=sample_rate, classwise=True)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

# Example usage:
#metric_instance = metric_factory("pairwise-spectral")
#```
