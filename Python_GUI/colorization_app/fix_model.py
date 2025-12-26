#!/usr/bin/env python3
"""
Model Compatibility Checker and Fixer
Fixes PyTorch model loading issues by cleaning up the checkpoint and removing unsafe operations
"""

import logging
import os
import sys
from pathlib import Path

import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_model_checkpoint(input_path, output_path=None):
    """
    Fix a PyTorch checkpoint to make it compatible with weights_only=True loading

    Args:
        input_path: Path to the problematic checkpoint
        output_path: Path for the fixed checkpoint (optional)

    Returns:
        str: Path to the fixed checkpoint
    """
    try:
        logger.info(f"Fixing model checkpoint: {input_path}")

        # Load the checkpoint with weights_only=False to read everything
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

        # Create a clean checkpoint with only the essential data
        fixed_checkpoint = {}

        # Extract generator state dict
        if "generator_state_dict" in checkpoint:
            state_dict = checkpoint["generator_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and any(
            "conv" in key or "fc" in key or "bn" in key for key in checkpoint.keys()
        ):
            # Checkpoint is already a state dict
            state_dict = checkpoint
        else:
            # Try to find the state dict in the checkpoint
            state_dict = None
            for key, value in checkpoint.items():
                if isinstance(value, dict) and any("weight" in k for k in value.keys()):
                    state_dict = value
                    break

            if state_dict is None:
                raise ValueError("Cannot find state_dict in checkpoint")

        # Clean the state dict by converting all tensors to CPU and ensuring they're contiguous
        clean_state_dict = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                # Move to CPU and make contiguous
                clean_value = value.cpu().contiguous()
                clean_state_dict[key] = clean_value
            else:
                logger.warning(f"Non-tensor value in state_dict: {key} = {type(value)}")
                # Skip non-tensor values to ensure compatibility

        # Add the clean state dict
        fixed_checkpoint["generator_state_dict"] = clean_state_dict

        # Add safe metadata
        if "config" in checkpoint:
            config = checkpoint["config"]
            # Ensure config contains only basic types
            safe_config = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool, list, tuple)):
                    safe_config[k] = v
                elif hasattr(v, "__dict__"):
                    # Convert objects to string representation
                    safe_config[k] = str(v)
                else:
                    safe_config[k] = str(v)
            fixed_checkpoint["config"] = safe_config
        else:
            # Add default config for Enhanced Pix2Pix
            fixed_checkpoint["config"] = {
                "IN_CHANNELS": 1,
                "OUT_CHANNELS": 3,
                "USE_ATTENTION": True,
                "USE_SE_BLOCKS": True,
            }

        # Add basic metadata
        if "epoch" in checkpoint:
            fixed_checkpoint["epoch"] = (
                int(checkpoint["epoch"]) if checkpoint["epoch"] is not None else 1
            )
        else:
            fixed_checkpoint["epoch"] = 1

        # Add model architecture info
        total_params = sum(
            p.numel() for p in clean_state_dict.values() if torch.is_tensor(p)
        )
        fixed_checkpoint["model_info"] = {
            "total_parameters": int(total_params),
            "architecture": "Enhanced Pix2Pix",
            "fixed_by": "fix_model.py",
        }

        # Determine output path
        if output_path is None:
            input_file = Path(input_path)
            output_path = (
                input_file.parent / f"{input_file.stem}_fixed{input_file.suffix}"
            )

        # Save the fixed checkpoint
        logger.info(f"Saving fixed checkpoint to: {output_path}")
        torch.save(fixed_checkpoint, output_path)

        # Verify the fixed checkpoint can be loaded with weights_only=True
        try:
            test_load = torch.load(output_path, map_location="cpu", weights_only=True)
            logger.info(
                "✅ Fixed checkpoint verified - can be loaded with weights_only=True"
            )
        except Exception as e:
            logger.warning(f"⚠️  Fixed checkpoint still has issues: {e}")
            # Try to save with even stricter cleaning
            torch.save(clean_state_dict, output_path)
            logger.info("Saved state_dict only as fallback")

        return str(output_path)

    except Exception as e:
        logger.error(f"❌ Error fixing checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return None


def validate_checkpoint(checkpoint_path):
    """
    Validate if a checkpoint can be loaded safely

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        tuple: (is_safe, can_load_weights_only, error_message)
    """
    try:
        # Test weights_only loading
        try:
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            return True, True, "Checkpoint is safe"
        except Exception as weights_error:
            # Test unsafe loading
            try:
                torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                return (
                    True,
                    False,
                    f"Checkpoint loads but not with weights_only: {weights_error}",
                )
            except Exception as unsafe_error:
                return False, False, f"Checkpoint cannot be loaded: {unsafe_error}"

    except Exception as e:
        return False, False, f"Error validating checkpoint: {e}"


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix PyTorch model checkpoints for safe loading"
    )
    parser.add_argument("input", help="Path to the input checkpoint file")
    parser.add_argument("-o", "--output", help="Path for the output fixed checkpoint")
    parser.add_argument(
        "-v", "--validate-only", action="store_true", help="Only validate, don't fix"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    print(f"🔍 Processing: {args.input}")

    # Validate first
    is_safe, weights_only_ok, message = validate_checkpoint(args.input)
    print(f"📋 Validation result: {message}")

    if args.validate_only:
        if is_safe and weights_only_ok:
            print("✅ Checkpoint is already safe")
            return 0
        else:
            print("❌ Checkpoint needs fixing")
            return 1

    if is_safe and weights_only_ok:
        print("✅ Checkpoint is already safe - no fixing needed")
        return 0

    if not is_safe:
        print("❌ Checkpoint is corrupted and cannot be fixed")
        return 1

    # Fix the checkpoint
    print("🔧 Fixing checkpoint...")
    fixed_path = fix_model_checkpoint(args.input, args.output)

    if fixed_path:
        print(f"✅ Fixed checkpoint saved to: {fixed_path}")
        print("\n💡 Usage:")
        print(f"   python run.py")
        print(f"   Then load the fixed model: {fixed_path}")
        return 0
    else:
        print("❌ Failed to fix checkpoint")
        return 1


if __name__ == "__main__":
    sys.exit(main())
