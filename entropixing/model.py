import math
import torch
import torch.nn.functional as F
from .kv_cache import KVCache
from .attn_stats import AttnStats
from transformers import (
    PretrainedConfig,
    AutoModelForCausalLM,
    Gemma2ForCausalLM,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

from typing import Tuple, Optional


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(
        *xq_out.shape[:-1], -1
    )
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(
        *xk_out.shape[:-1], -1
    )
    return xq_out.to(dtype), xk_out.to(dtype)


def reverse_permute(
    tensor: torch.Tensor, n_heads: int = 32, dim1: int = 4096, dim2: int = 4096
) -> torch.Tensor:
    return (
        tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def attention(
    weights: AutoModelForCausalLM,
    x: torch.Tensor,
    layer_weights,
    model_params: PretrainedConfig,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    bsz, _, _ = x.shape
    n_rep = model_params.num_attention_heads // model_params.num_key_value_heads
    xq = F.linear(
        x,
        reverse_permute(
            layer_weights.q_proj.weight,
            model_params.num_attention_heads,
            layer_weights.q_proj.weight.size(-2),
            layer_weights.q_proj.weight.size(-1),
        ),
        bias=(
            reverse_permute(
                layer_weights.q_proj.bias.view(1, -1),
                model_params.num_attention_heads,
                layer_weights.q_proj.bias.size(-1),
                1,
            ).squeeze()
            if layer_weights.q_proj.bias is not None
            else None
        ),
    ).reshape(bsz, -1, model_params.num_attention_heads, model_params.head_dim)
    xk = F.linear(
        x,
        reverse_permute(
            layer_weights.k_proj.weight,
            model_params.num_key_value_heads,
            layer_weights.k_proj.weight.size(-2),
            layer_weights.k_proj.weight.size(-1),
        ),
        bias=(
            reverse_permute(
                layer_weights.k_proj.bias.view(1, -1),
                model_params.num_key_value_heads,
                layer_weights.k_proj.bias.size(-1),
                1,
            ).squeeze()
            if layer_weights.k_proj.bias is not None
            else None
        ),
    ).reshape(bsz, -1, model_params.num_key_value_heads, model_params.head_dim)
    xv = layer_weights.v_proj(x).reshape(
        bsz, -1, model_params.num_key_value_heads, model_params.head_dim
    )
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=xq.dtype)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = torch.permute(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
    keys = torch.permute(
        keys, (0, 2, 3, 1)
    )  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = torch.permute(
        values, (0, 2, 1, 3)
    )  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys)
    if isinstance(weights, Gemma2ForCausalLM):
        scores = scores * (model_params.query_pre_attn_scalar**-0.5)
        if model_params.attn_logit_softcapping is not None:
            scores = scores / model_params.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * model_params.attn_logit_softcapping
    elif isinstance(weights, LlamaForCausalLM) or isinstance(weights, Qwen2ForCausalLM):
        scores = scores / math.sqrt(model_params.head_dim)
    pre_scores = scores
    scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32
    if cur_pos == 0:
        scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    masked_logits = torch.where(
        (mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE
    )
    scores = F.softmax(masked_logits, dim=-1).to(values.dtype)
    if (
        hasattr(model_params, "attention_dropout")
        and model_params.attention_dropout is not None
    ):
        scores = F.dropout(scores, p=model_params.attention_dropout)
    output = torch.matmul(scores, values)
    output = output.transpose(1, 2)
    output = output.reshape(xq.shape[0], xq.shape[2], -1)
    output = layer_weights.o_proj(output)
    return output, kvcache, pre_scores


def forward(
    weights: AutoModelForCausalLM,
    model_params: PretrainedConfig,
    tokens: torch.Tensor,
    cur_pos: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    attn_mask: Optional[torch.Tensor] = None,
    device: torch.device = "cpu",
) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    h = weights.model.embed_tokens.weight[tokens]
    if isinstance(weights, Gemma2ForCausalLM):
        normalizer = torch.tensor(model_params.hidden_size**0.5, dtype=h.dtype)
        h = h * normalizer
    attn_stats = AttnStats.init(
        bsz=tokens.shape[0],
        n_layers=model_params.num_hidden_layers,
        n_heads=model_params.num_attention_heads,
        device=device,
    )
    for i in range(model_params.num_hidden_layers):
        layer = weights.model.layers[i]
        if isinstance(weights, Gemma2ForCausalLM):
            if not bool(i % 2) and attn_mask is not None:
                min_dtype = torch.finfo(h.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attn_mask, dtype=torch.bool),
                    diagonal=-model_params.sliding_window,
                )
                attn_mask = torch.where(sliding_window_mask, min_dtype, attn_mask)
                if attn_mask.shape[-1] <= 1:  # when decoding
                    attn_mask = attn_mask[:, :, :, -model_params.sliding_window :]
        norm_x = layer.input_layernorm(h)
        h_attn, kvcache, scores = attention(
            weights,
            norm_x,
            layer.self_attn,
            model_params,
            cur_pos,
            i,
            freqs_cis,
            kvcache,
            attn_mask=attn_mask,
        )
        if isinstance(weights, LlamaForCausalLM) or isinstance(
            weights, Qwen2ForCausalLM
        ):
            h = h + h_attn
            h_mlp = layer.post_attention_layernorm(h)
        elif isinstance(weights, Gemma2ForCausalLM):
            h = h + layer.post_attention_layernorm(h_attn)
            h_mlp = layer.pre_feedforward_layernorm(h)
        attn_stats = attn_stats.update(scores[:, :, -1, :], i)
        h_mlp = layer.mlp(h_mlp)
        if isinstance(weights, Gemma2ForCausalLM):
            h_mlp = layer.post_feedforward_layernorm(h_mlp)
        h = h + h_mlp
    logits = weights.lm_head(weights.model.norm(h))
    if isinstance(weights, Gemma2ForCausalLM):
        if model_params.final_logit_softcapping is not None:
            logits = logits / model_params.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * model_params.final_logit_softcapping
    return logits, kvcache, scores, attn_stats
