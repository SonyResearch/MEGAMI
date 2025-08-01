
import math
import torchaudio
import os
from importlib import import_module
import yaml
import torch
from evaluation.feature_extractors import load_AFxRep, load_fx_encoder, load_fx_encoder_plusplus, load_fx_encoder_plusplus_2048

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
        elif self.type == "fxenc++":
            self.model_args= kwargs.get("fx_encoder_plusplus_args", None)

            assert self.model_args is not None, "model_args must be provided for fx_encoder type"

            self.distance_type=self.model_args.distance_type

            self.feat_extractor = load_fx_encoder_plusplus(self.model_args, self.device)


            #self.feat_extractor = load_effects_encoder(ckpt_path=ckpt_path).to(self.device)
        
        elif self.type == "logrms":
            self.model_args= kwargs.get("logrms_args", None)

            assert self.model_args is not None, "model_args must be provided for logrms type"
            self.distance_type=self.model_args.distance_type

            from utils.ITOMaster_loss import compute_log_rms

            self.feat_extractor = lambda x: compute_log_rms(x)

        elif self.type == "crestfactor":
            self.model_args= kwargs.get("crestfactor_args", None)
            assert self.model_args is not None, "model_args must be provided for crestfactor type"

            self.distance_type=self.model_args.distance_type

            from utils.ITOMaster_loss import compute_crest_factor

            self.feat_extractor = lambda x: compute_crest_factor(x)
        elif self.type == "logspread":
            self.model_args= kwargs.get("logspread_args", None)
            assert self.model_args is not None, "model_args must be provided for logspread type"

            self.distance_type=self.model_args.distance_type

            from utils.ITOMaster_loss import compute_log_spread

            self.feat_extractor = lambda x: compute_log_spread(x).view(-1, 1)
        elif self.type == "stereowidth":
            self.model_args= kwargs.get("stereowidth_args", None)
            assert self.model_args is not None, "model_args must be provided for stereowidth type"

            self.distance_type=self.model_args.distance_type

            from utils.ITOMaster_loss import compute_stereo_width
            self.feat_extractor = lambda x: compute_stereo_width(x).view(-1, 1)

        elif self.type == "stereoimbalance":
            self.model_args= kwargs.get("stereoimbalance_args", None)
            assert self.model_args is not None, "model_args must be provided for stereoimbalance type"  
            self.distance_type=self.model_args.distance_type
            from utils.ITOMaster_loss import compute_stereo_imbalance
            self.feat_extractor = lambda x: compute_stereo_imbalance(x).view(-1, 1)
        
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
        elif self.distance_type == "l1":
            l1_dist = torch.abs(feat_y_hat - feat_y).mean(dim=1)

            return {"distance": l1_dist.mean().item()}


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
            elif self.type=="fxenc++":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="logrms":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="crestfactor":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="logspread":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="stereowidth":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="stereoimbalance":
                dict_features_out = self.compute_feature_distance(y, y_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            

        # Compute the mean of the features across all keys

        mean_features = sum(dict_features.values()) / len(dict_features)

        return  mean_features, {}

class PairwiseStyleMultitrackFeatures(PairwiseMetric):
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

        elif self.type == "fxenc2048AFv3-fxenc++":

            self.model_args= kwargs.get("fx_encoder_plusplus_args", None)
            assert self.model_args is not None, "model_args must be provided for fxencAFv2-fxenc++ type"

            self.distance_type= self.model_args.distance_type

            fxencoder = load_fx_encoder_plusplus_2048(self.model_args, self.device)

            def feat_extractor_fn(x):
                z= fxencoder(x)
                z=torch.nn.functional.normalize(z, dim=-1, p=2)  # normalize to unit variance
                return z
            
            self.feat_extractor = feat_extractor_fn

        elif self.type == "fxenc2048AFv3-AF":

            from utils.AF_features_embedding_v2 import AF_fourier_embedding
            AFembedding= AF_fourier_embedding(device=self.device)

            self.distance_type="cosine"

            def feat_extracfor_fn(x):
                z, _ = AFembedding.encode(x)
                return z

            self.feat_extractor = feat_extracfor_fn
        

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
        
        index=0

        for key in dict_y.keys():
            y= dict_y[key]

            embed= dict_p_hat[key]
            embed=torch.tensor(embed).to(self.device).unsqueeze(0)

            print(f"Processing key: {key}, embed shape: {embed.shape}")
            if "AFxRep" in self.type:
                embed_mid, embed_side = torch.chunk(embed, 2, dim=-1)

                if self.type== "AFxRep-mid":
                        p_hat= embed_mid
                elif self.type== "AFxRep-side":
                        p_hat= embed_side
                elif self.type== "AFxRep":
                        p_hat= embed
            elif "fxenc2048AFv3" in self.type:

                embed=embed*math.sqrt(embed.shape[-1])  # Scale the embedding
                embed_fxenc=embed[...,:2048]/ math.sqrt(2048)  # Scale the first 128 dimensions
                embed_AF=embed[...,2048:]/ math.sqrt(64)  # Scale the last 128 dimensions

                if self.type == "fxenc2048AFv3-fxenc++":
                        p_hat= embed_fxenc
                elif self.type == "fxenc2048AFv3-AF":
                        p_hat= embed_AF
    

            n, c, d=y.shape

            #assert b==1, f"Expected batch size of 1, got {b} for key {key}"

            assert c==2, f"Expected 2 channels, got {c} for key {key}"

            for i in range(n):
                y_i=y[i].T

                if self.type=="fx_encoder":
                    raise NotImplementedError("Style features for fx_encoder not implemented yet")
                    dict_features_out = self.compute_feature_distance(y, p_hat, self.sample_rate, type=self.type)
                    dict_features[key] = dict_features_out["distance"]
                elif self.type=="AFxRep":
                    dict_features_out = self.compute_feature_distance(y_i, p_hat[i], self.sample_rate, type=self.type)
                    dict_features[index] = dict_features_out["distance"]
                elif self.type=="AFxRep-mid":
                    dict_features_out = self.compute_feature_distance(y_i, p_hat[i], self.sample_rate, type=self.type)
                    dict_features[index] = dict_features_out["distance"]
                elif self.type=="AFxRep-side":
                    dict_features_out = self.compute_feature_distance(y_i, p_hat[i], self.sample_rate, type=self.type)
                    dict_features[index] = dict_features_out["distance"]
                elif self.type=="fxenc2048AFv3-fxenc++":
                    dict_features_out = self.compute_feature_distance(y_i, p_hat[i], self.sample_rate, type=self.type)
                    dict_features[index] = dict_features_out["distance"]
                elif self.type=="fxenc2048AFv3-AF":
                    dict_features_out = self.compute_feature_distance(y_i, p_hat, self.sample_rate, type=self.type)
                    dict_features[index] = dict_features_out["distance"]
            

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

        elif self.type == "fxenc2048AFv3-fxenc++":

            self.model_args= kwargs.get("fx_encoder_plusplus_args", None)
            assert self.model_args is not None, "model_args must be provided for fxencAFv2-fxenc++ type"

            self.distance_type= self.model_args.distance_type

            fxencoder = load_fx_encoder_plusplus_2048(self.model_args, self.device)

            def feat_extractor_fn(x):
                z= fxencoder(x)
                z=torch.nn.functional.normalize(z, dim=-1, p=2)  # normalize to unit variance
                return z
            
            self.feat_extractor = feat_extractor_fn

        elif self.type == "fxenc2048AFv3-AF":

            from utils.AF_features_embedding_v2 import AF_fourier_embedding
            AFembedding= AF_fourier_embedding(device=self.device)

            self.distance_type="cosine"

            def feat_extracfor_fn(x):
                z, _ = AFembedding.encode(x)
                return z

            self.feat_extractor = feat_extracfor_fn
        

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
            if "AFxRep" in self.type:
                embed_mid, embed_side = torch.chunk(embed, 2, dim=-1)

                if self.type== "AFxRep-mid":
                        p_hat= embed_mid
                elif self.type== "AFxRep-side":
                        p_hat= embed_side
                elif self.type== "AFxRep":
                        p_hat= embed
            elif "fxenc2048AFv3" in self.type:

                embed=embed*math.sqrt(embed.shape[-1])  # Scale the embedding
                embed_fxenc=embed[...,:2048]/ math.sqrt(2048)  # Scale the first 128 dimensions
                embed_AF=embed[...,2048:]/ math.sqrt(64)  # Scale the last 128 dimensions

                if self.type == "fxenc2048AFv3-fxenc++":
                        p_hat= embed_fxenc
                elif self.type == "fxenc2048AFv3-AF":
                        p_hat= embed_AF
    

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
            elif self.type=="fxenc2048AFv3-fxenc++":
                dict_features_out = self.compute_feature_distance(y, p_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            elif self.type=="fxenc2048AFv3-AF":
                dict_features_out = self.compute_feature_distance(y, p_hat, self.sample_rate, type=self.type)
                dict_features[key] = dict_features_out["distance"]
            
            

        # Compute the mean of the features across all keys

        mean_features = sum(dict_features.values()) / len(dict_features)

        return  mean_features, {}

class PairwiseIMMSS(PairwiseMetric):
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

        from utils.ITOMaster_loss import MultiScale_Spectral_Loss_MidSide_DDSP
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        if mode == "mss-lr":
            multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=1e-6, device=device)
            self.metric= lambda y_hat, y: multi_scale_spectral_ori(y_hat, y)
        elif mode == "mss-ms":
            multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=1e-6, device=device)
            self.metric= lambda y_hat, y: multi_scale_spectral_midside(y_hat, y)

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

            y=torch.from_numpy(y).cuda().unsqueeze(0)
            y_hat=torch.from_numpy(y_hat).cuda().unsqueeze(0)

            metric=self.metric(y_hat, y)

            dict_metrics[key] = metric.item()

        mean_features = sum(dict_metrics.values()) / len(dict_metrics)

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
            raw_metric= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
            ).cuda()
            self.metric=lambda y_hat, y: 0.5*raw_metric(y_hat, y)
        elif mode == "mldr-ms":
            from evaluation.ldr import MLDRLoss
            raw_metric= MLDRLoss(
                sr=44100,
                s_taus=[50, 100],
                l_taus=[1000, 2000],
                mid_side=True
            ).cuda()
            self.metric=lambda y_hat, y: 0.25*raw_metric(y_hat, y)

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
            raw_metric=MultiResolutionSTFTLoss(
                [128, 512, 2048],
                [32, 128, 512],
                [128, 512, 2048],
                sample_rate=44100,
                perceptual_weighting=True,
            ).cuda()
            self.metric=lambda y_hat, y: raw_metric(y_hat, y)
        elif mode == "mss-ms":
            from auraloss.freq import  SumAndDifferenceSTFTLoss
            raw_metric=SumAndDifferenceSTFTLoss(
            [128, 512, 2048],
            [32, 128, 512],
            [128, 512, 2048],
            sample_rate=44100,
            perceptual_weighting=True,
            ).cuda()
            self.metric=lambda y_hat, y: 0.5*raw_metric(y_hat, y)

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
    elif metric_name == "pairwise-IM-mss-lr":
        return PairwiseIMMSS(mode="mss-lr",*args, **kwargs)
    elif metric_name == "pairwise-IM-mss-ms":
        return PairwiseIMMSS(mode="mss-ms",*args, **kwargs)
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
    elif metric_name == "pairwise-style-fxenc2048AFv3-fxenc++":
        return PairwiseStyleFeatures(*args, **kwargs, type="fxenc2048AFv3-fxenc++", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-style-fxenc2048AFv3-AF":
        return PairwiseStyleFeatures(*args, **kwargs, type="fxenc2048AFv3-AF", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-style-multitrack-fxenc2048AFv3-fxenc++":
        return PairwiseStyleMultitrackFeatures(*args, **kwargs, type="fxenc2048AFv3-fxenc++", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-style-multitrack-fxenc2048AFv3-AF":
        return PairwiseStyleMultitrackFeatures(*args, **kwargs, type="fxenc2048AFv3-AF", sample_rate=sample_rate, model_args=kwargs.get('AFxRep_args', None))
    elif metric_name == "pairwise-fxenc++":
        return PairwiseFeatures(*args, **kwargs, type="fxenc++", sample_rate=sample_rate, model_args=kwargs.get('fx_encoder_plusplus_args', None))
    elif metric_name == "pairwise-logrms":
        return PairwiseFeatures(*args, **kwargs, type="logrms", sample_rate=sample_rate, model_args=kwargs.get('logrms_args', None))
    elif metric_name == "pairwise-crestfactor":
        return PairwiseFeatures(*args, **kwargs, type="crestfactor", sample_rate=sample_rate, model_args=kwargs.get('crestfactor_args', None))
    elif metric_name == "pairwise-logspread":
        return PairwiseFeatures(*args, **kwargs, type="logspread", sample_rate=sample_rate, model_args=kwargs.get('logspread_args', None))
    elif metric_name == "pairwise-stereowidth":
        return PairwiseFeatures(*args, **kwargs, type="stereowidth", sample_rate=sample_rate, model_args=kwargs.get('stereowidth_args', None))
    elif metric_name == "pairwise-stereoimbalance":
        return PairwiseFeatures(*args, **kwargs, type="stereoimbalance", sample_rate=sample_rate, model_args=kwargs.get('stereoimbalance_args', None))
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

# Example usage:
#metric_instance = metric_factory("pairwise-spectral")
#```
