from entropixing.llama_cpp_impl import generate_response
from rich.console import Console
from transformers import AutoTokenizer
from llama_cpp import Llama, GGML_TYPE_Q4_0


def main():
    from argparse import ArgumentParser

    global device
    console = Console()
    parser = ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--model", type=str, required=True, default="./model.gguf")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    conv = []
    weights = Llama(
        args.model,
        n_gpu_layers=args.ngl,
        n_ctx=args.context_length,
        verbose=False,
        flash_attn=True,
        type_k=GGML_TYPE_Q4_0,
        type_v=GGML_TYPE_Q4_0,
    )
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
            conv, tokenize=False, add_generation_prompt=True
        )[len(tokenizer.bos_token) if tokenizer.bos_token else 0 :]
        it = generate_response(
            weights,
            inputs,
            args.max_length,
            stop=[tokenizer.eos_token],
        )
        console.print("Assistant: ", end="", style="green")
        text = ""
        for token in it:
            console.print(token, end="")
            text += token
        conv.append({"role": "assistant", "content": text.strip()})
        console.print()


if __name__ == "__main__":
    main()
