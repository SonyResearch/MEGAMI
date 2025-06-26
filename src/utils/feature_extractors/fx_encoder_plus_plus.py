import torch
from fxencoder_plusplus import FxEncoderPlusPlus
from fxencoder_plusplus.model import get_model_path


def load_model(model_name="default", model_path=None, device="cuda", auto_download=True, cache_dir=None):
    """
    Load FxEncoderPlusPlus model.
    
    Args:
        model_name: Name of pretrained model ('default', 'musdb', 'medleydb')
        model_path: Custom checkpoint path. If provided, ignores model_name
        device: Device to load model on ('cuda' or 'cpu')
        auto_download: Automatically download if model not found
        cache_dir: Custom cache directory for downloaded models
        
    Returns:
        Loaded FxEncoderPlusPlus model
        
    Examples:
        # Load default base model
        model = load_model()
        
        # Load musdb model
        model = load_model(model_name="musdb")
        
        # Load medleydb model
        model = load_model(model_name="medleydb")
        
        # Load custom checkpoint
        model = load_model(model_path="/path/to/custom.pt")
        
        # List available models
        list_available_models()
    """
    # Handle device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    
    # Determine model path
    if model_path is None:
        if auto_download:
            model_path = get_model_path(model_name, cache_dir=cache_dir)
        else:
            raise ValueError("model_path is None and auto_download is False")
    
    # Create model instance with specified device
    model = FxEncoderPlusPlus(
        embed_dim=2048, 
        audio_clap_module=False, 
        text_clap_module=False,
        extractor_module=False,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False )
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)
        print("Loaded model checkpoint")
    
    model.to(device)
    model.eval()
    
    # Freeze parameters for inference
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded successfully on {device}")
    return model