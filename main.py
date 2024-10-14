from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PretrainedConfig,
)
import torch
from entropixing.generate import generate, stream
from rich.console import Console

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

    global device
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
    parser.add_argument("--prompt", type=str, default="Hello, my name is ")
    parser.add_argument("--device", type=str, default=device.type)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--min_p", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    console = Console()
    print(f"Using device: {device}")
    arch: PretrainedConfig = AutoConfig.from_pretrained(args.model)
    if arch.architectures[0] not in [
        "Gemma2ForCausalLM",
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
    ]:
        raise ValueError(f"Unsupported model architecture: {arch.architectures[0]}")
    dtype = getattr(torch, args.dtype)
    weights = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=dtype,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer.encode(args.prompt, return_tensors="pt")

    console.print(args.prompt, style="green", end="", flush=True)
    it = generate(
        weights,
        inputs,
        device,
        dtype,
        [tokenizer.eos_token_id],
        args.max_length,
        args.top_p,
        args.top_k,
        args.min_p,
        args.repetition_penalty,
        args.seed,
    )
    for token in stream(it, tokenizer):
        if "text" in token:
            style = ""
            if token["entropy"] > 3:
                style = "bold"
            elif token["varentropy"] > 15:
                style += "blue"
            console.print(token["text"], style=style, end="")
        elif "back" in token:
            console.print("⌫", style="red", end="")


if __name__ == "__main__":
    main()
