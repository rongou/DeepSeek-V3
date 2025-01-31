import os
import torch
import torch.distributed as dist
from safetensors.torch import load_model
import json
from model import Transformer, ModelArgs, MLP, MoE


class ActivationTracker:
    def __init__(self, rank=0):
        self.mlp_inputs = {}
        self.w1_outputs = {}
        self.w3_outputs = {}
        self.w2_outputs = {}
        self.hooks = []
        self.rank = rank

    def hook_fn(self, name, storage):
        def hook(module, input, output):
            # Only collect on rank 0 to avoid duplicates
            if self.rank == 0:
                storage[name] = {
                    'input': input[0].detach(),
                    'output': output.detach()
                }

        return hook

    def attach_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList):
                for i, layer in enumerate(module):
                    if hasattr(layer, 'ffn'):
                        if isinstance(layer.ffn, MLP):
                            # For regular MLP layers
                            self.hooks.append(
                                layer.ffn.w1.register_forward_hook(
                                    self.hook_fn(f"layer{i}.mlp.w1", self.w1_outputs)
                                )
                            )
                            self.hooks.append(
                                layer.ffn.w3.register_forward_hook(
                                    self.hook_fn(f"layer{i}.mlp.w3", self.w3_outputs)
                                )
                            )
                            self.hooks.append(
                                layer.ffn.w2.register_forward_hook(
                                    self.hook_fn(f"layer{i}.mlp.w2", self.w2_outputs)
                                )
                            )
                        elif isinstance(layer.ffn, MoE):
                            # For MoE layers, track the local experts and shared experts
                            for expert_idx, expert in enumerate(layer.ffn.experts):
                                if expert is not None:  # Only track experts assigned to this rank
                                    self.hooks.append(
                                        expert.w1.register_forward_hook(
                                            self.hook_fn(f"layer{i}.moe.expert{expert_idx}.w1", self.w1_outputs)
                                        )
                                    )
                                    self.hooks.append(
                                        expert.w3.register_forward_hook(
                                            self.hook_fn(f"layer{i}.moe.expert{expert_idx}.w3", self.w3_outputs)
                                        )
                                    )
                                    self.hooks.append(
                                        expert.w2.register_forward_hook(
                                            self.hook_fn(f"layer{i}.moe.expert{expert_idx}.w2", self.w2_outputs)
                                        )
                                    )
                            # Also track shared experts
                            self.hooks.append(
                                layer.ffn.shared_experts.w1.register_forward_hook(
                                    self.hook_fn(f"layer{i}.moe.shared.w1", self.w1_outputs)
                                )
                            )
                            self.hooks.append(
                                layer.ffn.shared_experts.w3.register_forward_hook(
                                    self.hook_fn(f"layer{i}.moe.shared.w3", self.w3_outputs)
                                )
                            )
                            self.hooks.append(
                                layer.ffn.shared_experts.w2.register_forward_hook(
                                    self.hook_fn(f"layer{i}.moe.shared.w2", self.w2_outputs)
                                )
                            )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def analyze_activations(self):
        """Analyze activation patterns to identify potential super weights"""
        results = []

        # Only analyze on rank 0 where we collected the activations
        if self.rank == 0:
            # Get all layer names and sort them
            layer_names = sorted(self.w2_outputs.keys())

            for layer_name in layer_names:
                # Parse layer name parts
                parts = layer_name.split('.')
                layer_num = int(parts[0].replace('layer', ''))  # Extract number from 'layerX'
                layer_type = parts[1]  # 'mlp' or 'moe'
                if layer_type == 'moe':
                    sublayer = '.'.join(parts[2:-1])  # 'expertX' or 'shared'
                else:
                    sublayer = None

                w2_data = self.w2_outputs[layer_name]
                w2_input = w2_data['input']
                w2_output = w2_data['output']

                max_input_val, max_input_idx = torch.max(w2_input.abs().view(-1), dim=0)
                max_output_val, max_output_idx = torch.max(w2_output.abs().view(-1), dim=0)

                in_features = w2_input.size(-1)
                out_features = w2_output.size(-1)

                input_row = max_input_idx.item() // in_features
                input_col = max_input_idx.item() % in_features
                output_row = max_output_idx.item() // out_features
                output_col = max_output_idx.item() % out_features

                result = {
                    'layer': layer_num,
                    'layer_type': layer_type,
                    'sublayer': sublayer,
                    'input_max': max_input_val.item(),
                    'input_pos': (input_row, input_col),
                    'output_max': max_output_val.item(),
                    'output_pos': (output_row, output_col),
                }

                # Statistical analysis
                input_mean = w2_input.abs().mean().item()
                input_std = w2_input.abs().std().item()
                output_mean = w2_output.abs().mean().item()
                output_std = w2_output.abs().std().item()

                is_input_outlier = (max_input_val.item() - input_mean) / input_std > 5
                is_output_outlier = (max_output_val.item() - output_mean) / output_std > 5

                result.update({
                    'input_stats': {
                        'mean': input_mean,
                        'std': input_std,
                        'zscore': (max_input_val.item() - input_mean) / input_std
                    },
                    'output_stats': {
                        'mean': output_mean,
                        'std': output_std,
                        'zscore': (max_output_val.item() - output_mean) / output_std
                    },
                    'is_potential_super_weight': is_input_outlier and is_output_outlier
                })

                results.append(result)

        return results


def find_super_weights(
    ckpt_path: str,
    config_path: str,
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0
):
    """
    Analyze DeepSeek model for potential super weights in distributed setting.
    """
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    # Setup model
    torch.set_default_dtype(torch.bfloat16)

    # Load config and create model
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))

    if rank == 0:
        print("Loading model...")

    model = Transformer(args).cuda()
    checkpoint_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    load_model(model, checkpoint_file)
    model.eval()

    # Setup activation tracking
    tracker = ActivationTracker(rank)
    tracker.attach_hooks(model)

    # Create input and broadcast from rank 0
    if rank == 0:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device="cuda")
        objects = [input_ids]
        dist.broadcast_object_list(objects, 0)
    else:
        objects = [None]
        dist.broadcast_object_list(objects, 0)
        input_ids = objects[0].cuda()

    # Run inference
    with torch.inference_mode():
        output = model.forward(input_ids, start_pos=0)

    # Analyze activations
    results = tracker.analyze_activations()

    # Clean up
    tracker.remove_hooks()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    # Print results only on rank 0
    if rank == 0:
        print("\nAnalyzing potential super weights:")

        # First check MLP layers
        mlp_results = [r for r in results if r.get('layer_type') == 'mlp']
        if mlp_results:
            print("\nMLP Layer Results:")
            for result in mlp_results:
                print(f"\nLayer {result['layer']} (MLP):")
                print(f"Input activation: {result['input_max']:.2f} at position {result['input_pos']}")
                print(f"Output activation: {result['output_max']:.2f} at position {result['output_pos']}")
                print(f"Input Z-score: {result['input_stats']['zscore']:.2f}")
                print(f"Output Z-score: {result['output_stats']['zscore']:.2f}")

                if result['is_potential_super_weight']:
                    print("⚠️ Potential super weight detected!")
                    print(f"Weight coordinates in w2: [{result['output_pos'][1]}, {result['input_pos'][1]}]")

        # Then check MoE layers
        moe_results = [r for r in results if r.get('layer_type') == 'moe']
        if moe_results:
            print("\nMoE Layer Results:")
            for result in moe_results:
                print(f"\nLayer {result['layer']} (MoE):")
                if 'shared' in result.get('sublayer', ''):
                    print("Shared Expert:")
                else:
                    expert_idx = result.get('expert_idx', 'unknown')
                    print(f"Expert {expert_idx}:")
                print(f"Input activation: {result['input_max']:.2f} at position {result['input_pos']}")
                print(f"Output activation: {result['output_max']:.2f} at position {result['output_pos']}")
                print(f"Input Z-score: {result['input_stats']['zscore']:.2f}")
                print(f"Output Z-score: {result['output_stats']['zscore']:.2f}")

                if result['is_potential_super_weight']:
                    print("⚠️ Potential super weight detected!")
                    print(f"Weight coordinates in w2: [{result['output_pos'][1]}, {result['input_pos'][1]}]")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Get distributed setup from environment
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Override print on non-zero ranks
    if rank != 0:
        print = lambda *args, **kwargs: None

    results = find_super_weights(
        args.ckpt_path,
        args.config,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )
