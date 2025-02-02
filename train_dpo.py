import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.commands.cli_utils import TrlParser
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'attention-gym')))
from attn_gym import visualize_attention_scores
from modeling.dpo_flex_attn_masks import construct_causal_mask, construct_dpo_mask, construct_dpo_mask_with_packing, \
    construct_causal_mask_with_packing
from trainer import DPOTrainer
from config import DPOConfig, DPOScriptArguments

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    model = model_config.model_name_or_path
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    if training_args.prefix_sharing:
        assert model_config.attn_implementation == "flex_attention", "Must set --attn_implementation=flex_attention for prefix sharing attention mask support"
    if model_config.attn_implementation == "flex_attention":
        # because of compilation, batch sizes need to match so we don't trigger a recompilation
        assert training_args.per_device_eval_batch_size == training_args.per_device_train_batch_size, "Must have equal train and eval batch sizes for FlexAttention support"
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = model_config.model_name_or_path
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]


    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name)

    # train_dataset = dataset[script_args.dataset_train_split]
    # if script_args.dataset_test_split in dataset:
    #     eval_dataset = dataset[script_args.dataset_test_split]
    # else:
    #     eval_dataset = None
    #
    # if script_args.keep_columns:
    #     train_dataset = train_dataset.select_columns(script_args.keep_columns)
    #     if eval_dataset is not None:
    #         eval_dataset = eval_dataset.select_columns(script_args.keep_columns)
    #
    # if training_args.max_train_samples and training_args.max_train_samples < len(train_dataset):
    #     train_dataset = train_dataset.select(range(training_args.max_train_samples))
    #
    # if eval_dataset and training_args.max_eval_samples and training_args.max_eval_samples < len(eval_dataset):
    #     eval_dataset = eval_dataset.select(range(training_args.max_eval_samples))

    ################
    # Training
    ################

    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Generate random text for prompt, chosen, and rejected
    prompt = "This is a random prompt text. Predict next numbers."
    chosen = "7, 15, 29, 50. This is a chosen random text."
    rejected = "22, 131, 90, 33, 209. This is a rejected random text."

    # Encode sequences
    prompt_tokens = tokenizer.encode(prompt)
    chosen_tokens = tokenizer.encode(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer.encode(rejected, add_special_tokens=False)

    # Combine sequences and pad to 128 tokens
    def pad_to_128(tokens):
        pad_length = (128 - len(tokens) % 128) % 128
        return tokens + [tokenizer.pad_token_id] * pad_length


    dpo_tokens = pad_to_128(prompt_tokens + chosen_tokens + rejected_tokens)
    causal_chosen_tokens = pad_to_128(prompt_tokens + chosen_tokens)
    causal_rejected_tokens = pad_to_128(prompt_tokens + rejected_tokens)

    # Convert to tensors
    dpo_input = torch.tensor(dpo_tokens).unsqueeze(0).to('cuda')
    causal_chosen_input = torch.tensor(causal_chosen_tokens).unsqueeze(0).to('cuda')
    causal_rejected_input = torch.tensor(causal_rejected_tokens).unsqueeze(0).to('cuda')

    # Calculate indices
    chosen_index = len(prompt_tokens)
    rejected_index = len(prompt_tokens) + len(chosen_tokens)
    seq_len = len(dpo_tokens)
    batch_size = 1

    # Create sequence_id for DPO mask
    sequence_id = torch.zeros(batch_size * seq_len, dtype=torch.long, device='cuda')
    sequence_id[len(prompt_tokens + chosen_tokens + rejected_tokens):] = -1  # Set padding tokens to -1

    # Create masks
    dpo_mask = construct_dpo_mask_with_packing(
        sequence_id,
        torch.full((batch_size * seq_len,), chosen_index, dtype=torch.long, device='cuda'),
        torch.full((batch_size * seq_len,), rejected_index, dtype=torch.long, device='cuda'),
        None,
        batch_size,
        seq_len,
        0
    )

    sequence_id_causal_chosen = torch.zeros(batch_size * seq_len, dtype=torch.long, device='cuda')
    sequence_id_causal_chosen[len(prompt_tokens + chosen_tokens):] = -1  # Set padding tokens to -1
    causal_chosen_mask = construct_causal_mask_with_packing(
        sequence_id_causal_chosen,
        batch_size,
        len(causal_chosen_tokens),
        len(causal_chosen_tokens)
    )

    sequence_id_causal_rejected = torch.zeros(batch_size * seq_len, dtype=torch.long, device='cuda')
    sequence_id_causal_rejected[len(prompt_tokens + rejected_tokens):] = -1  # Set padding tokens to -1
    causal_rejected_mask = construct_causal_mask_with_packing(
        sequence_id_causal_rejected,
        batch_size,
        len(causal_rejected_tokens),
        0
    )


    # Visualization function
    def visualize_mask(mask, name):
        B, H, HEAD_DIM = 1, 1, 64
        query = torch.randn(B, H, mask.shape[-1], HEAD_DIM).to('cuda')
        key = torch.randn(B, H, mask.shape[-1], HEAD_DIM).to('cuda')
        visualize_attention_scores(query, key, mask_mod=mask.mask_mod, device="cuda", name=name)


    visualize_mask(dpo_mask, "dpo_mask")
    visualize_mask(causal_chosen_mask, "causal_chosen_mask")
    visualize_mask(causal_rejected_mask, "causal_rejected_mask")


    # Perform forward passes
    def forward_pass(input_ids, attention_mask):
        with torch.no_grad():
            outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.logits


    dpo_logits = forward_pass(dpo_input, dpo_mask)
    causal_chosen_logits = forward_pass(causal_chosen_input, causal_chosen_mask)
    causal_rejected_logits = forward_pass(causal_rejected_input, causal_rejected_mask)

    # Extract relevant parts for comparison
    dpo_logits_prompt_chosen = dpo_logits[:, :rejected_index]
    dpo_logits_prompt_rejected = torch.cat(
        [dpo_logits[:, :chosen_index], dpo_logits[:, rejected_index:rejected_index + len(rejected_tokens)]], dim=1)


    # Compare logits
    def compare_logits_detailed(masked, unmasked, name, tokenizer, token_str=True):
        unmasked = unmasked.to(masked.device, masked.dtype)
        diff = torch.abs(masked - unmasked)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        per_position_max = diff.max(dim=-1).values
        per_position_mean = diff.mean(dim=-1)

        print(f"\n{name} Comparison:")
        print(f"Overall - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        print("Per-position statistics and next token predictions:")
        for pos in range(diff.shape[1]):  # Iterate over sequence length
            masked_next_token = masked[0, pos].argmax().item()
            unmasked_next_token = unmasked[0, pos].argmax().item()

            print(f"Position {pos:3d} - "
                        f"Max diff: {per_position_max[0, pos].item():.6f}, "
                        f"Mean diff: {per_position_mean[0, pos].item():.6f}, "
                        f"Masked next: (id={masked_next_token}), "
                        f"Unmasked next: (id={unmasked_next_token})")

            if masked_next_token != unmasked_next_token:
                print(f"WARNING! Token prediction mismatch at position {pos} in {name}!")


    compare_logits_detailed(dpo_logits_prompt_chosen, causal_chosen_logits[:, :rejected_index], "Prompt + Chosen",
                            tokenizer)
    compare_logits_detailed(dpo_logits_prompt_rejected, causal_rejected_logits[:, :len(prompt_tokens) + len(rejected_tokens)],
                            "Prompt + Rejected", tokenizer)

    print(f"\nPrompt length: {len(prompt_tokens)}")
    print(f"Chosen length: {len(chosen_tokens)}")
    print(f"Rejected length: {len(rejected_tokens)}")
    print(f"DPO sequence length: {len(dpo_tokens)}")
    print(f"Causal chosen sequence length: {len(causal_chosen_tokens)}")
    print(f"Causal rejected sequence length: {len(causal_rejected_tokens)}")
    print(f"Chosen index: {chosen_index}")
    print(f"Rejected index: {rejected_index}")

    # trainer.train()

    # if eval_dataset is not None:
    #     metrics = trainer.evaluate()
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)
