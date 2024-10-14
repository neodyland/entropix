from entropixing.model import forward
from entropixing.sampler import sample
from entropixing.kv_cache import KVCache
from entropixing.utils import build_attn_mask, precompute_freqs_cis
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PretrainedConfig,
)
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Default device: {device}")

torch.set_float32_matmul_precision("high")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, default="google/gemma-2-2b-jpn-it"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--prompt", type=str, default="大規模言語モデルとは、")
    parser.add_argument("--device", type=str, default=device.type)
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    arch: PretrainedConfig = AutoConfig.from_pretrained(args.model)
    if arch.architectures[0] not in [
        "Gemma2ForCausalLM",
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
    ]:
        raise ValueError(f"Unsupported model architecture: {arch.architectures[0]}")
    if arch.architectures[0] == "Qwen2ForCausalLM":
        raise NotImplementedError("Qwen2ForCausalLM is not supported yet.")
    with torch.inference_mode():
        dtype = getattr(torch, args.dtype)
        weights = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype=dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        inputs = tokenizer.encode(args.prompt, return_tensors="pt")

        def generate(weights: AutoModelForCausalLM, tokens: torch.tensor):
            config: AutoConfig = weights.config
            gen_tokens = None
            cur_pos = 0
            bsz, seqlen = tokens.shape
            attn_mask = build_attn_mask(
                seqlen,
                cur_pos,
                device=device,
            )
            if not hasattr(config, "head_dim"):
                setattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                )
                weights.config = config
            freqs_cis = precompute_freqs_cis(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                (
                    config.rope_scaling is not None
                    if hasattr(config, "rope_scaling")
                    else False
                ),
                device=device,
            )
            kvcache = KVCache(
                config.num_hidden_layers,
                bsz,
                config.max_position_embeddings,
                config.num_key_value_heads,
                config.head_dim,
                device,
                dtype,
            ).to(device)
            logits, kvcache, _, _ = forward(
                weights,
                config,
                tokens,
                cur_pos,
                freqs_cis[:seqlen],
                kvcache,
                attn_mask=attn_mask,
                device=device,
            )
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(
                torch.int32
            )
            gen_tokens = next_token
            print(tokenizer.decode([next_token.item()]), end="", flush=True)
            cur_pos = seqlen
            stop = torch.tensor(
                [tokenizer.eos_token_id], device=device, dtype=torch.int32
            )
            while cur_pos < args.max_length:
                cur_pos += 1
                logits, kvcache, scores, _stats = forward(
                    weights,
                    config,
                    next_token,
                    cur_pos,
                    freqs_cis[cur_pos : cur_pos + 1],
                    kvcache,
                    device=device,
                )
                next_token = sample(gen_tokens, logits, scores)
                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                print(tokenizer.decode(next_token.tolist()[0]), end="", flush=True)
                if torch.isin(next_token, stop).any():
                    break

        print(args.prompt)
        generate(weights, inputs)


if __name__ == "__main__":
    main()
