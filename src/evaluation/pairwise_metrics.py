
import torchaudio
import os
from importlib import import_module
import yaml
import torch
from evaluation.feature_extractors import load_AFxRep, load_fx_encoder, load_fx_encoder_plusplus

class PairwiseMetric:
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

class PairwiseFeatures(PairwiseMetric):
    """
    Class for computing the pairwise spectral metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise spectral metric.
    """
    def __init__(self,
        type=None,
        sample_rate=44100,
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
    
    def compute_feature_distance(self, y, y_hat, sample_rate, type):    

        y=torch.tensor(y).permute(1,0).unsqueeze(0).to(self.device)
        y_hat=torch.tensor(y_hat).permute(1,0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_y= self.feat_extractor(y)
            feat_y_hat= self.feat_extractor(y_hat)

        if self.distance_type == "cosine":
            cos_dist= 1- torch.cosine_similarity(feat_y_hat, feat_y, dim=1)


            return {"distance": cos_dist.mean().item()}



    def compute(self, dict_y, dict_y_hat, dict_x,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """



        dict_features={}

        for key in dict_y.keys():
            y= dict_y[key]
            y_hat= dict_y_hat[key]

            assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"

            c, d=y.shape
            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            y=y.T
            y_hat=y_hat.T


            if self.type == "spectral":
                from evaluation.automix_evaluation import compute_spectral_features 
                dict_features_out= compute_spectral_features(y_hat, y ,self.sample_rate)
                dict_features[key] = dict_features_out['mean_mape_spectral']
            elif self.type=="panning":
                from evaluation.automix_evaluation import compute_panning_features 
                dict_features_out = compute_panning_features(y_hat, y, self.sample_rate)
                dict_features[key] = dict_features_out['mean_mape_panning']
            elif self.type=="loudness":
                from evaluation.automix_evaluation import compute_loudness_features 
                dict_features_out = compute_loudness_features(y_hat, y, self.sample_rate)
                dict_features[key] = dict_features_out['mean_mape_loudness']
            elif self.type=="dynamic":
                from evaluation.automix_evaluation import compute_dynamic_features 
                dict_features_out = compute_dynamic_features(y_hat, y, self.sample_rate)
                dict_features[key] = dict_features_out['mean_mape_dynamic']
            elif self.type=="fx_encoder":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="AFxRep":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="AFxRep-mid":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="AFxRep-side":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            

        # Compute the mean of the features across all keys

        mean_features = sum(dict_features.values()) / len(dict_features)

        return  mean_features, {}

class PairwiseStyleFeatures(PairwiseMetric):
    """
    Class for computing the pairwise spectral metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise spectral metric.
    """
    def __init__(self,
        type=None,
        sample_rate=44100,
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

        if self.type == "fx_encoder":
            raise NotImplementedError("Style features for fx_encoder not implemented yet")
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
    
    def compute_feature_distance(self, y, p_hat, sample_rate, type):    

        y=torch.tensor(y).permute(1,0).unsqueeze(0).to(self.device)
        #y_hat=torch.tensor(y_hat).permute(1,0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_y= self.feat_extractor(y)
            #feat_y_hat= self.feat_extractor(y_hat)
        
        assert p_hat.shape == feat_y.shape, f"Shape mismatch: p_hat {p_hat.shape} vs feat_y {feat_y.shape}"

        if self.distance_type == "cosine":
            cos_dist= 1- torch.cosine_similarity(p_hat, feat_y, dim=1)

            return {"distance": cos_dist.mean().item()}



    def compute(self, dict_y, dict_y_hat, dict_x, dict_p_hat=None,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """

        dict_features={}

        for key in dict_y.keys():
            y= dict_y[key]

            embed= dict_p_hat[key]
            embed=torch.tensor(embed).to(self.device).unsqueeze(0)

            print(f"Processing key: {key}, embed shape: {embed.shape}")
            embed_mid, embed_side = torch.chunk(embed, 2, dim=-1)

            if self.type== "AFxRep-mid":
                    p_hat= embed_mid
            elif self.type== "AFxRep-side":
                    p_hat= embed_side
            elif self.type== "AFxRep":
                    p_hat= embed


            #assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"

            c, d=y.shape
            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            y=y.T
            #y_hat=y_hat.T

            if self.type=="fx_encoder":
                raise NotImplementedError("Style features for fx_encoder not implemented yet")
                dict_features_out = self.compute_feature_distance(y, p_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="AFxRep":
                dict_features_out = self.compute_feature_distance(y, p_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="AFxRep-mid":
                dict_features_out = self.compute_feature_distance(y, p_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="AFxRep-side":
                dict_features_out = self.compute_feature_distance(y, p_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            

        # Compute the mean of the features across all keys

        mean_features = sum(dict_features.values()) / len(dict_features)

        return  mean_features, {}

class PairwiseLDR(PairwiseMetric):
    """
    Class for computing the pairwise LDR metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise LDR metric.
    """
    def __init__(self, mode=None, *args, **kwargs):
        """
        Initialize the PairwiseLDR instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        assert mode is not None, "Mode must be specified for PairwiseLDR"

        if mode == "mldr-lr":
            from evaluation.ldr import MLDRLoss
            self.metric= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
            ).cuda()
        elif mode == "mldr-ms":
            from evaluation.ldr import MLDRLoss
            self.metric= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
                mid_side=True
            ).cuda()

    def compute(self, dict_y, dict_y_hat, dict_x,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """

        dict_metrics={}

        for key in dict_y.keys():
            y= dict_y[key]
            y_hat= dict_y_hat[key]

            assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"

            c, d=y.shape
            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            y=y.T
            y_hat=y_hat.T
            
            y=torch.from_numpy(y).cuda().unsqueeze(0)
            y_hat=torch.from_numpy(y_hat).cuda().unsqueeze(0)

            metric=self.metric(y_hat, y)

            dict_metrics[key] = metric.item()

        mean_features = sum(dict_metrics.values()) / len(dict_metrics)

        return  mean_features, {}


class PairwiseAuraloss(PairwiseMetric):
    """
    Class for computing the pairwise MSS metric.
    
    This class inherits from PairwiseMetric and implements the compute method
    to calculate the pairwise MSS metric.
    """
    def __init__(self, mode=None, *args, **kwargs):
        """
        Initialize the PairwiseMSS instance.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        assert mode is not None, "Mode must be specified for PairwiseMSS"

        if mode == "mss-lr":
            from auraloss.freq import MultiResolutionSTFTLoss
            self.metric=MultiResolutionSTFTLoss(
                [128, 512, 2048],
                [32, 128, 512],
                [128, 512, 2048],
                sample_rate=44100,
                perceptual_weighting=True,
            ).cuda()
        elif mode == "mss-ms":
            from auraloss.freq import  SumAndDifferenceSTFTLoss
            self.metric=SumAndDifferenceSTFTLoss(
            [128, 512, 2048],
            [32, 128, 512],
            [128, 512, 2048],
            sample_rate=44100,
            perceptual_weighting=True,
            ).cuda()

    def compute(self, dict_y, dict_y_hat, dict_x,   *args, **kwargs):
        """
        Compute the pairwise spectral metric.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The computed pairwise spectral metric.
        """

        dict_metrics={}

        for key in dict_y.keys():
            y= dict_y[key]
            y_hat= dict_y_hat[key]

            assert y.shape == y_hat.shape, f"Shape mismatch for key {key}: {y.shape} vs {y_hat.shape}"

            c, d=y.shape
            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            #y=y.T
            #y_hat=y_hat.T
            
            y=torch.from_numpy(y).cuda().unsqueeze(0)
            y_hat=torch.from_numpy(y_hat).cuda().unsqueeze(0)

            metric=self.metric(y_hat, y)

            dict_metrics[key] = metric.item()

        mean_features = sum(dict_metrics.values()) / len(dict_metrics)

        return  mean_features, {}

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
    if metric_name == "pairwise-spectral":
        return PairwiseFeatures(*args, **kwargs, type="spectral", sample_rate=sample_rate)
    elif metric_name == "pairwise-panning":
        return PairwiseFeatures(*args, **kwargs, type="panning", sample_rate=sample_rate)
    elif metric_name == "pairwise-loudness":
        return PairwiseFeatures(*args, **kwargs, type="loudness", sample_rate=sample_rate)
    elif metric_name == "pairwise-dynamic":
        return PairwiseFeatures(*args, **kwargs, type="dynamic", sample_rate=sample_rate)
    elif metric_name == "pairwise-mss-lr":
        return PairwiseAuraloss(mode="mss-lr",*args, **kwargs)
    elif metric_name == "pairwise-mss-ms":
        return PairwiseAuraloss(mode="mss-ms",*args, **kwargs)
    elif metric_name == "pairwise-mldr-lr":
        return PairwiseLDR(mode="mldr-lr",*args, **kwargs)
    elif metric_name == "pairwise-mldr-ms":
        return PairwiseLDR(mode="mldr-ms",*args, **kwargs)
    elif metric_name == "pairwise-fx_encoder":
        return PairwiseFeatures(*args, **kwargs, type="fx_encoder", sample_rate=sample_rate, model_args=kwargs.get('fx_encoder_args', None))
    elif metric_name == "pairwise-AFxRep":
        return PairwiseFeatures(*args, **kwargs, type="AFxRep", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-AFxRep-mid":
        return PairwiseFeatures(*args, **kwargs, type="AFxRep-mid", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-AFxRep-side":
        return PairwiseFeatures(*args, **kwargs, type="AFxRep-side", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-style-AFxRep":
        return PairwiseStyleFeatures(*args, **kwargs, type="AFxRep", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-style-AFxRep-mid":
        return PairwiseStyleFeatures(*args, **kwargs, type="AFxRep-mid", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-style-AFxRep-side":
        return PairwiseStyleFeatures(*args, **kwargs, type="AFxRep-side", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

# Example usage:
#metric_instance = metric_factory("pairwise-spectral")
#```
