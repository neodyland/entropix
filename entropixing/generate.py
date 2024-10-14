from .model import forward
from .sampler import sample
from .kv_cache import KVCache
from .utils import build_attn_mask, precompute_freqs_cis
import torch
from transformers import AutoModelForCausalLM, AutoConfig


@torch.inference_mode()
def generate(
    weights: AutoModelForCausalLM,
    tokens: torch.tensor,
    device: torch.device,
    dtype: torch.dtype,
    stop_tokens: list[int],
    max_length: int,
):
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
        setattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        weights.config = config
    freqs_cis = precompute_freqs_cis(
        config.head_dim,
        config.max_position_embeddings,
        config.rope_theta,
        (config.rope_scaling is not None if hasattr(config, "rope_scaling") else False),
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
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
    gen_tokens = next_token
    yield next_token.item()
    cur_pos = seqlen
    stop = torch.tensor(stop_tokens, device=device, dtype=torch.int32)
    while cur_pos < max_length:
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
        yield next_token.tolist()[0]
        if torch.isin(next_token, stop).any():
            break


def is_valid_str(s: str):
    try:
        s.encode("utf-8").decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def stream(it, tokenizer):
    text_cache = ""
    for token in it:
        text_cache += tokenizer.decode(token, skip_special_tokens=True)
        if is_valid_str(text_cache):
            yield text_cache
            text_cache = ""
