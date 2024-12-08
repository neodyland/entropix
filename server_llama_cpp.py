from pydantic import TypeAdapter
from transformers import AutoTokenizer
from llama_cpp import Llama, GGML_TYPE_Q4_0
import json
from entropixing.llama_cpp_impl import generate_response
from asyncio import Lock
from uvicorn import run
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from openai.types.model import Model
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.chat.chat_completion import Choice as NostreamChoice
from openai.types.chat.completion_create_params import (
    CompletionCreateParams,
)
from uuid import uuid4
import time

adapter: TypeAdapter[CompletionCreateParams] = TypeAdapter(CompletionCreateParams)


def main():
    from argparse import ArgumentParser

    global weights
    global tokenizer
    global lock
    lock = Lock()
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--tokenizer", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--model", type=str, required=True, default="./model.gguf")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()
    weights = Llama(
        args.model,
        n_gpu_layers=args.ngl,
        n_ctx=args.context_length,
        verbose=False,
        flash_attn=True,
        type_k=GGML_TYPE_Q4_0,
        type_v=GGML_TYPE_Q4_0,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    app = FastAPI()

    @app.post("/chat/completions")
    async def chat_completion(body: Request) -> Response:
        j = adapter.validate_python(await body.json())
        max_length = j.get("max_completion_tokens") or args.max_length
        messages = list(j["messages"])
        if j.get("stream") == True:

            async def stream_generator():
                async for chunk in gen(
                    messages,
                    max_length,
                ):
                    yield f"data: {ChatCompletionChunk(
                            id=str(uuid4()),
                            choices=[Choice(delta=ChoiceDelta(content=chunk), index=0)],
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
    stop=None,
):
    text = ""
    async for chunk in gen(
        conv,
        max_length,
        stop,
    ):
        text += chunk
    return text


async def gen(conv, max_length, stop=None):
    inputs = (
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)[
            len(tokenizer.bos_token) if tokenizer.bos_token else 0 :
        ]
        if isinstance(conv, list)
        else conv
    )
    stops = [tokenizer.eos_token]
    if stop:
        stops.extend(stop)
    async with lock:
        it = generate_response(
            weights,
            inputs,
            max_new_tokens=max_length,
            stop=stops,
        )
        for token in it:
            print(token, end="", flush=True)
            yield token


if __name__ == "__main__":
    main()
