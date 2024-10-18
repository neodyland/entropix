from pydantic import TypeAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig
import torch
import json
from entropixing.generate import generate, stream
from asyncio import Lock
from uvicorn import run
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from openai.types.model import Model
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.chat.chat_completion import Choice as NostreamChoice
from openai.types.chat.completion_create_params import CompletionCreateParams
from uuid import uuid4
import time

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
adapter: TypeAdapter[CompletionCreateParams] = TypeAdapter(CompletionCreateParams)


def main():
    from argparse import ArgumentParser

    global dtype
    global device
    global weights
    global tokenizer
    global lock
    lock = Lock()
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
    parser.add_argument("--device", type=str, default=device.type)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--min_p", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quantize", action="store_true")
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
        quantization_config=(
            TorchAoConfig("int4_weight_only", ["self_attn"], group_size=64)
            if args.quantize
            else None
        ),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    app = FastAPI()

    @app.post("/chat/completions")
    async def chat_completion(body: Request) -> Response:
        j = adapter.validate_python(await body.json())
        max_length = j.get("max_completion_tokens") or args.max_length
        messages = list(j["messages"])
        top_p = j.get("top_p", args.top_p)
        top_k = j.get("top_logprobs", args.top_k)
        min_p = args.min_p
        repetition_penalty = j.get("frequency_penalty", args.repetition_penalty)
        seed = j.get("seed", args.seed)
        if j.get("stream"):

            async def stream_generator():
                async for chunk in gen(
                    messages,
                    max_length,
                    top_p,
                    top_k,
                    min_p,
                    repetition_penalty,
                    seed,
                    args.context_length,
                ):
                    if "text" in chunk:
                        yield f"data: {ChatCompletionChunk(
                            id=str(uuid4()),
                            choices=[Choice(delta=ChoiceDelta(content=chunk["text"]), index=0)],
                            created=time.time() // 1000,
                            model=j["model"],
                            object="chat.completion.chunk",
                        ).model_dump_json()}\n\n"
                    else:
                        yield f"data: {ChatCompletionChunk(
                            id=str(uuid4()),
                            choices=[Choice(delta=ChoiceDelta(content="⌫"), index=0)],
                            created=time.time() // 1000,
                            model=j["model"],
                            object="chat.completion.chunk",
                        ).model_dump_json()}\n\n"
                yield f"data: {ChatCompletionChunk(
                    id=str(uuid4()),
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                    ],
                    created=time.time() // 1000,
                    model=j["model"],
                    object="chat.completion.chunk",
                ).model_dump_json()}\n\n"

            return StreamingResponse(
                content=stream_generator(), media_type="text/event-stream"
            )
        else:
            text = await gen_no_stream(
                messages,
                max_length,
                top_p,
                top_k,
                min_p,
                repetition_penalty,
                seed,
                args.context_length,
            )
            return Response(
                content=ChatCompletion(
                    id=str(uuid4()),
                    choices=[
                        NostreamChoice(
                            finish_reason="stop",
                            message=ChatCompletionMessage(
                                content=text, role="assistant"
                            ),
                            index=0,
                        )
                    ],
                    created=time.time() // 1000,
                    model=j["model"],
                    object="chat.completion",
                ).model_dump_json(),
                media_type="application/json",
            )

    @app.get("/models")
    async def models():
        return Response(
            content=json.dumps(
                {
                    "data": [
                        json.loads(
                            Model(
                                id="entropix-any",
                                object="model",
                                created=1,
                                owned_by="someone",
                            ).model_dump_json()
                        )
                    ]
                }
            ),
            media_type="application/json",
        )

    run(app, host=args.host, port=args.port)


async def gen_no_stream(
    conv,
    max_length,
    top_p,
    top_k,
    min_p,
    repetition_penalty,
    seed,
    context_length,
):
    text = ""
    async for chunk in gen(
        conv, max_length, top_p, top_k, min_p, repetition_penalty, seed, context_length
    ):
        if "text" in chunk:
            text += chunk["text"]
    return text


async def gen(
    conv,
    max_length,
    top_p,
    top_k,
    min_p,
    repetition_penalty,
    seed,
    context_length,
):
    inputs = tokenizer.apply_chat_template(
        conv, return_tensors="pt", add_generation_prompt=True
    )
    async with lock:
        it = generate(
            weights,
            inputs,
            device,
            dtype,
            [tokenizer.eos_token_id],
            max_length,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            seed,
            False,
            context_length,
        )
        for token in stream(it, tokenizer):
            yield token


if __name__ == "__main__":
    main()
