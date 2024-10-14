from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PretrainedConfig,
)
import torch
from entropixing.generate import generate, stream

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
    dtype = getattr(torch, args.dtype)
    weights = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=dtype,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conv = []
    while True:
        inp = input("User: ").strip()
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
            weights, inputs, device, dtype, [tokenizer.eos_token_id], args.max_length
        )
        print("Assistant: ", end="")
        text = ""
        for token in stream(it, tokenizer):
            print(token, end="", flush=True)
            text += token
        conv.append({"role": "assistant", "content": text.strip()})
        print()


if __name__ == "__main__":
    main()
