from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from entropixing.generate import generate, stream
from rich.console import Console

from entropixing.utils import is_supported_model

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Default device: {device}")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def main():
    from argparse import ArgumentParser

    global device
    console = Console()
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
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--device", type=str, default=device.type)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--min_p", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--print_back", action="store_true")
    parser.add_argument("--go_back", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if not is_supported_model(args.model):
        raise ValueError("Unsupported model")
    dtype = getattr(torch, args.dtype)
    weights = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=dtype,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conv = []
    while True:
        console.print("User: ", end="", style="green")
        inp = input("").strip()
        if inp == "exit":
            break
        elif inp == "clear":
            conv.clear()
            continue
        conv.append({"role": "user", "content": inp})
        inputs = tokenizer.apply_chat_template(
            conv, return_tensors="pt", add_generation_prompt=True
        )
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
            args.go_back,
            args.context_length,
        )
        console.print("Assistant: ", end="", style="green")
        text = ""
        for token in stream(it, tokenizer):
            if "text" in token:
                style = ""
                if token["entropy"] > 3:
                    style = "bold"
                elif token["varentropy"] > 15:
                    style += "blue"
                console.print(token["text"], style=style, end="")
                text += token["text"]
            elif "back" in token:
                if args.print_back:
                    console.print("⌫", style="red", end="")
                else:
                    console.print("\b \b", end="")
        conv.append({"role": "assistant", "content": text.strip()})
        console.print()


if __name__ == "__main__":
    main()
