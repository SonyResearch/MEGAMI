import argparse
import os
import torch

OFFENDING_KEY = "text_clap_model.encoder.model.text_branch.embeddings.position_ids"

def main(args):
    # Load the checkpoint from the given path
    checkpoint = torch.load(
        args.input_checkpoint, map_location="cpu", weights_only=False
    )

    # Extract the state_dict from the checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(state_dict.keys())

    # Delete the specific key from the state_dict
    if OFFENDING_KEY in state_dict:
        print("key found")
        del state_dict[OFFENDING_KEY]

    # Save the modified state_dict back to the checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint["state_dict"] = state_dict

    # Create the output checkpoint filename by replacing the ".pt" suffix with ".patched.pt"
    output_checkpoint_path = args.input_checkpoint.replace('.pt', '.patched.pt')

    # Save the modified checkpoint
    torch.save(checkpoint, output_checkpoint_path)
    print(f"Saved patched checkpoint to {output_checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patch a PyTorch checkpoint by removing a specific key.')
    parser.add_argument('input_checkpoint', type=str, help='Path to the input PyTorch checkpoint (.pt) file.')

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()
    main(args)
