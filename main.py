from entropixing.model import forward
from entropixing.sampler import sample
from entropixing.kv_cache import KVCache
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.set_float32_matmul_precision("high")


def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (
            HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR
        )
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled,  # Apply smooth scaling
            ),
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)

    return scaled_freqs


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500000.0,
    use_scaled: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim)
    )
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = (
            torch.hstack([torch.zeros((seqlen, start_pos)), mask])
            .to(torch.float32)
            .to(device)
        )
    return mask


def main():
    model = "meta-llama/Llama-3.2-1B-Instruct"
    model = "google/gemma-2-2b-jpn-it"
    with torch.inference_mode():
        dtype = torch.bfloat16
        weights = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model)
        prompt = "ビションフリーゼとは、"
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        def generate(weights: AutoModelForCausalLM, tokens: torch.tensor):
            config: AutoConfig = weights.config
            gen_tokens = None
            cur_pos = 0
            bsz, seqlen = tokens.shape
            attn_mask = build_attn_mask(seqlen, cur_pos)
            freqs_cis = precompute_freqs_cis(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                (
                    config.rope_scaling is not None
                    if hasattr(config, "rope_scaling")
                    else False
                ),
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
