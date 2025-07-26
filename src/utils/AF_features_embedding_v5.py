import math
from utils.ITOMaster_loss import compute_log_rms_gated, compute_crest_factor, compute_stereo_width, compute_stereo_imbalance, compute_log_spread
import torch

class AF_fourier_embedding:
    def __init__(self,
                input_dim=8,
                output_dim=64,
                sigma=0.2,
                log_rms_shift=-26.5, #calculated as the mean from the dataset
                log_rms_scale=7.0, #calculated as the std from the dataset
                crest_shift=16.7, #calculated as the mean from the dataset
                crest_scale=6.3,
                log_spread_shift=-20.0, #calculated as the mean from the dataset
                log_spread_scale=20.0, #calculated as the std from the dataset
                stereo_width_shift=0.28,
                stereo_width_scale=0.39,
                stereo_imbalance_shift=0.0,
                stereo_imbalance_scale=0.35,
                device="cpu"
                ):
        """
        Deterministic Fourier feature transformer using fixed cosine-based projection
        """

        self.device = device
        # Ensure output_dim is even and >= 2 * input_dim
        self.output_dim = max(input_dim * 2, output_dim)
        if self.output_dim % 2 != 0:
            self.output_dim += 1
        
        self.input_dim = input_dim
        self.sigma = sigma
        
        # Create deterministic projection matrix
        self.projection = self._create_deterministic_projection(input_dim, self.output_dim // 2, sigma)
        self.projection = self.projection.to(self.device)
        
        # Normalization factor
        self.scale_factor = math.sqrt(2.0 / self.output_dim)

        self.log_rms_shift = log_rms_shift
        self.log_rms_scale = log_rms_scale
        self.crest_shift = crest_shift
        self.crest_scale = crest_scale
        self.log_spread_shift = log_spread_shift
        self.log_spread_scale = log_spread_scale
        self.stereo_width_shift = stereo_width_shift
        self.stereo_width_scale = stereo_width_scale
        self.stereo_imbalance_shift = stereo_imbalance_shift
        self.stereo_imbalance_scale = stereo_imbalance_scale
    
    def _create_deterministic_projection(self, input_dim, proj_dim, sigma):
        """
        Create a deterministic projection matrix using a cosine basis
        """
        # Cosine-based matrix (like DCT type-II)
        projection = torch.zeros(input_dim, proj_dim)
        for i in range(input_dim):
            for j in range(proj_dim):
                projection[i, j] = math.cos(math.pi * (i + 0.5) * (j + 1) / proj_dim)
        
        return projection * sigma
    
    def encode(self, x):

        log_rms=compute_log_rms_gated(x)
        crest_factor= compute_crest_factor(x)
        log_spread= compute_log_spread(x)
        stereo_width= compute_stereo_width(x)
        stereo_imbalance= compute_stereo_imbalance(x)


        log_rms_std, crest_factor_std, log_spread_std, stereo_width_std, stereo_imbalance_std = self.standardize_features(
            log_rms, crest_factor, log_spread, stereo_width, stereo_imbalance
        )

        embedding= self.transform(
            log_rms_std, crest_factor_std, log_spread_std, stereo_width_std, stereo_imbalance_std
        )


        return embedding, (log_rms, crest_factor, log_spread, stereo_width, stereo_imbalance)
    
    def decode(self, fourier_features):
        """
        Invert Fourier features back to original feature space
        (approximate due to phase-only reconstruction)
        """
        reconstructed = self.inverse_transform(fourier_features)
        
        # Reshape back to original feature dimensions
        log_rms= reconstructed[:,0:2]
        crest_factor = reconstructed[:,2:4]
        log_spread= reconstructed[:,4:6]
        stereo_width = reconstructed[:,6:7]
        stereo_imbalance = reconstructed[:,7:8]

        log_rms, crest_factor, log_spread,stereo_width, stereo_imbalance = self.destandardize_features(
            log_rms, crest_factor, log_spread, stereo_width, stereo_imbalance
        )
        
        return log_rms, crest_factor, log_spread, stereo_width, stereo_imbalance

    def standardize_features(self, log_rms, crest_factor, log_spread, stereo_width, stereo_imbalance):
        """
        Standardize features using pre-computed mean and std
        """
        log_rms = (log_rms - self.log_rms_shift) / self.log_rms_scale
        crest_factor = (crest_factor - self.crest_shift) / self.crest_scale
        log_spread = (log_spread - self.log_spread_shift) / self.log_spread_scale
        stereo_width = (stereo_width - self.stereo_width_shift) / self.stereo_width_scale
        stereo_imbalance = (stereo_imbalance - self.stereo_imbalance_shift) / self.stereo_imbalance_scale
        
        return log_rms, crest_factor, log_spread, stereo_width, stereo_imbalance
    
    def destandardize_features(self, log_rms, crest_factor, log_spread, stereo_width, stereo_imbalance):
        """
        Reverse standardization to get back to original feature space
        """
        log_rms = log_rms * self.log_rms_scale + self.log_rms_shift
        crest_factor = crest_factor * self.crest_scale + self.crest_shift
        log_spread = log_spread * self.log_spread_scale + self.log_spread_shift
        stereo_width = stereo_width * self.stereo_width_scale + self.stereo_width_shift
        stereo_imbalance = stereo_imbalance * self.stereo_imbalance_scale + self.stereo_imbalance_shift
        
        return log_rms, crest_factor, log_spread, stereo_width, stereo_imbalance
        
    def transform(self, log_rms, crest_factor,log_spread, stereo_width, stereo_imbalance):
        """
        Transform features using the stored projection matrix
        """

        flat_features=torch.cat([log_rms, crest_factor, log_spread, stereo_width.unsqueeze(-1), stereo_imbalance.unsqueeze(-1)], dim=-1)
        
        # Project and transform
        projected = flat_features @ self.projection
        cos_features = torch.cos(projected)
        sin_features = torch.sin(projected)
        
        # Concatenate and normalize
        return torch.cat([cos_features, sin_features], dim=-1) * self.scale_factor
    
    def inverse_transform(self, fourier_features):
        """
        Invert Fourier features back to original feature space
        (approximate due to phase-only reconstruction)
        """
        # Split into cosine and sine components
        feature_dim = fourier_features.shape[-1] // 2
        cos_features = fourier_features[:, :feature_dim]
        sin_features = fourier_features[:, feature_dim:]
        
        # Denormalize
        cos_features = cos_features / self.scale_factor
        sin_features = sin_features / self.scale_factor
        
        # Compute phase angles
        phases = torch.atan2(sin_features, cos_features)
        
        # Use pseudo-inverse for approximate inversion
        projection_pinv = torch.pinverse(self.projection)
        reconstructed = phases @ projection_pinv
        
        return reconstructed

