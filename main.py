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

print(f"Using device: {device}")

torch.set_float32_matmul_precision("high")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, default="google/gemma-2-2b-jpn-it"
    )
    args = parser.parse_args()
    arch: PretrainedConfig = AutoConfig.from_pretrained(args.model)
    if arch.architectures[0] not in ["Gemma2ForCausalLM", "LlamaForCausalLM"]:
        raise ValueError(f"Unsupported model architecture: {arch.architectures[0]}")
    with torch.inference_mode():
        dtype = torch.bfloat16
        weights = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompt = "ビションフリーゼとは、"
        inputs = tokenizer.encode(prompt, return_tensors="pt")

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
            while cur_pos < 8192:
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

        print(prompt)
        generate(weights, inputs)


if __name__ == "__main__":
    main()
