import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import json
from model import ModelArgs


def zero_weight(
    ckpt_path: str,
    config_path: str,
    output_path: str
):
    """Zero out potential super weight and save modified model."""

    # Set default tensor type to CPU
    torch.set_default_tensor_type(torch.FloatTensor)

    # Load config to get dimensions
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))

    # Find all model shards
    shards = [f for f in os.listdir(ckpt_path) if f.startswith("model") and f.endswith(".safetensors")]
    print(f"Found {len(shards)} model shards")

    for shard in shards:
        print(f"\nProcessing {shard}")

        # Parse shard info from filename (modelX-mpY.safetensors)
        rank = int(shard.split('-')[0].replace('model', ''))
        world_size = int(shard.split('-')[1].replace('mp', '').split('.')[0])

        # Calculate which rows this shard owns
        rows_per_shard = args.dim // world_size
        shard_start = rank * rows_per_shard
        shard_end = shard_start + rows_per_shard

        # Check if our target row (6660) falls in this shard
        target_row = 6660
        if shard_start <= target_row < shard_end:
            print(f"Found target row {target_row} in shard {shard}")
            print(f"Loading shard...")

            # Load the shard
            shard_path = os.path.join(ckpt_path, shard)
            state_dict = {}
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)

            # Get local row index
            local_row = target_row - shard_start

            # Modify the weight
            w2_weight = state_dict['layers.1.ffn.w2.weight']
            print(f"Original weight value at [{local_row}, 24]: {w2_weight[local_row, 24]}")

            # Convert to fp32, modify, then back to float8
            w2_fp32 = w2_weight.to(torch.float32)
            w2_fp32[local_row, 24] = 0.0
            w2_weight = w2_fp32.to(torch.float8_e4m3fn)
            state_dict['layers.1.ffn.w2.weight'] = w2_weight

            print(f"New weight value at [{local_row}, 24]: {w2_weight[local_row, 24]}")

            # Save modified shard
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, shard)
            save_file(state_dict, output_file)
            print(f"Saved modified shard to {output_file}")
        else:
            # Just copy the shard unchanged
            print(f"Target row not in this shard, copying unchanged")
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, shard)

            # Load and save without modification
            state_dict = {}
            with safe_open(os.path.join(ckpt_path, shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            save_file(state_dict, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    zero_weight(args.ckpt_path, args.config, args.output_path)