from typing import Optional

from .model import forward
from .sampler import sample, calculate_metrics
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
    top_p: float = 0.95,
    top_k: int = 27,
    min_p: int = 0,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None,
):
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
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
    logits, kvcache, scores, _stats = forward(
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
    metrics = calculate_metrics(logits, scores)
    ent, vent = metrics.logits_entropy, metrics.logits_varentropy
    yield {
        "token": next_token.item(),
        "temperature": -1.0,
        "entropy": ent.item(),
        "varentropy": vent.item(),
    }
    cur_pos = seqlen
    stop = torch.tensor(stop_tokens, device=device, dtype=torch.int32)
    num_recent_deletes = 0
    should_noise = False
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
        if should_noise:
            logits = logits + torch.randn_like(logits) * 0.1
        metrics = calculate_metrics(logits, scores)
        ent, vent = metrics.logits_entropy, metrics.logits_varentropy
        del metrics

        # basic weighting to prevent backspacing too much
        threshold = 5.0 + 2 * num_recent_deletes
        if ent > threshold and vent > threshold and cur_pos > seqlen + 4:
            #    backspace and pop the last token
            num_recent_deletes += 1
            # reset to the position before the last token, regenerate the token
            cur_pos -= 2
            next_token = gen_tokens[:, -2].unsqueeze(0)
            gen_tokens = gen_tokens[:, :-1]
            yield {"back": True}
            should_noise = True
            continue
        else:
            num_recent_deletes = max(0, num_recent_deletes - 0.5)
            should_noise = False

        temperature = 0.7 + (0.5 * num_recent_deletes)
        next_token = sample(
            gen_tokens,
            logits,
            scores,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        yield {
            "token": next_token.tolist()[0],
            "temperature": temperature,
            "entropy": ent.item(),
            "varentropy": vent.item(),
        }
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
        if "back" in token:
            yield {"back": True}
        else:
            text_cache += tokenizer.decode(token["token"], skip_special_tokens=True)
            if is_valid_str(text_cache):
                yield {
                    "text": text_cache,
                    "tempeature": token["temperature"],
                    "entropy": token["entropy"],
                    "varentropy": token["varentropy"],
                }
                text_cache = ""
