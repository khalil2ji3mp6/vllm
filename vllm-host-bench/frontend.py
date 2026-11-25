"""Local frontend with per-request image and text preprocessing."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns, process_time_ns

import msgspec
import zmq
import zmq.asyncio
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.hf import HfRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from engine import config
from measure import Clock
from tokenizer_rc5 import GlmRc5Tokenizer


def prompt_factory(tokenizer, request_indices, tokens):
    """Fit source text to an exact encoded length using one repeatable token."""
    messages_by_request = []
    for i in request_indices:
        prefix = f"Request {i}. Describe the measurement."
        repeat = max(0, tokens - 30)
        for _ in range(20):
            messages = [{"role": "user", "content": prefix + " test" * repeat}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            delta = tokens - len(ids)
            if not delta:
                break
            repeat = max(0, repeat + delta)
        else:
            raise ValueError(f"Cannot make exactly {tokens} input tokens")
        messages_by_request.append(messages)
    return messages_by_request


def workload_messages(args, tokenizer, request_indices):
    if args.request_file is None:
        return prompt_factory(tokenizer, request_indices, args.prompt_tokens)
    entries = [
        msgspec.json.decode(line)["messages"]
        for line in args.request_file.read_bytes().splitlines()
        if line.strip()
    ]
    if len(entries) != args.concurrency:
        raise ValueError(f"Expected {args.concurrency} request file entries, found {len(entries)}")
    for index in request_indices:
        messages = entries[index]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if count != args.prompt_tokens:
            raise ValueError(f"Request {index + 1} has {count} tokens, expected {args.prompt_tokens}")
    return [entries[index] for index in request_indices]


def render_prompt(tokenizer, messages, submitted):
    clock = Clock()
    start = perf_counter_ns()
    clock.rows["prefill", "renderer_queue_wait"].append((submitted, start, 0))
    with clock.span("prefill", "chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with clock.span("prefill", "tokenize"):
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    return rendered, token_ids, (start, perf_counter_ns()), clock.rows


def peak_parallel(intervals):
    events = sorted((t, change) for start, end in intervals for t, change in ((start, 1), (end, -1)))
    peak = active = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    return peak


async def preprocess_requests(args, front_index, entries, images, tokenizer, pool, name,
                              input_processor, processor, encoder, sock, batch_clock):
    loop = asyncio.get_running_loop()
    slots = asyncio.Semaphore(args.frontend_slots)

    async def prepare(index, messages):
        clock = Clock()
        start = perf_counter_ns()
        with clock.span("prefill", "request_validate"):
            payload = msgspec.json.encode(dict(
                model="glm", messages=messages, max_tokens=args.new_tokens, stream=True,
            ))
            request = ChatCompletionRequest.model_validate_json(payload)
            params = SamplingParams(
                max_tokens=request.max_tokens, ignore_eos=True, stop=["<never-stop>"],
                output_kind=RequestOutputKind.DELTA, temperature=0.0,
            )
        # A request holds one shared CPU slot from image decode through rendering.
        # Either executor can use the slots currently free of other work.
        async with slots:
            image_interval = await images.decode_request(clock) if images else None
            rendered, token_ids, render_interval, render_rows = await loop.run_in_executor(
                pool, render_prompt, tokenizer, messages, perf_counter_ns(),
            )
        for key, values in render_rows.items():
            clock.rows[key].extend(values)
        if len(token_ids) != args.prompt_tokens:
            raise RuntimeError(f"Unexpected prompt token count for request {index}")
        with clock.span("prefill", "input_validate_and_enqueue"):
            req = input_processor.process_inputs(
                f"{name}-{index}", {"type": "token", "prompt_token_ids": token_ids},
                params, ("generate",),
            )
            req.external_req_id = req.request_id
            req.client_index = front_index
            processor.add_request(req, rendered)
        preprocess_cpu = sum(cpu for samples in clock.rows.values() for _, _, cpu in samples)
        clock.rows["prefill", "front_prefill_preprocess"].append(
            (start, perf_counter_ns(), preprocess_cpu)
        )
        with clock.span("prefill", "front_engine_encode_send"):
            await sock.send_multipart(encoder.encode(req))
        for key, values in clock.rows.items():
            batch_clock.rows[key].extend(values)
        return image_interval, render_interval

    tasks = await asyncio.gather(*(prepare(index, messages) for index, messages in entries))
    image_intervals = [image for image, _ in tasks if image is not None]
    render_intervals = [render for _, render in tasks]
    return {
        "front_ready_ns": perf_counter_ns(),
        "image_end_ns": max((end for _, end in image_intervals), default=None),
        "renderer_peak_threads": peak_parallel(render_intervals),
        "media_peak_threads": peak_parallel(image_intervals),
        "media_renderer_overlap": any(
            max(image_start, render_start) < min(image_end, render_end)
            for image_start, image_end in image_intervals
            for render_start, render_end in render_intervals
        ),
    }


async def stream_outputs(sock, decoder, processor, clock, batch_start_ns, request_count):
    emitted = 0
    serialized_bytes = 0
    first_token_ns = {}
    finished_ns = {}
    while True:
        # Idle socket waits are not counted as frontend computation.
        phase, *frames = await sock.recv_multipart()
        if phase == b"done":
            break
        phase = phase.decode()
        with clock.span(phase, "front_output_postprocess"):
            with clock.span(phase, "front_output_ipc_decode"):
                update = decoder.decode(frames)
            with clock.span(phase, "detokenize_and_stop_strings"):
                output = processor.process_outputs(update.outputs, update.timestamp)
            if output.reqs_to_abort:
                raise RuntimeError("Unexpected stop-string hit in fixed workload")
            with clock.span(phase, "serialize_stream_chunks"):
                for item in output.request_outputs:
                    for choice in item.outputs:
                        emitted += len(choice.token_ids)
                        chunk = msgspec.json.encode({
                            "id": item.request_id, "object": "chat.completion.chunk",
                            "choices": [{"index": choice.index, "delta": {"content": choice.text},
                                         "finish_reason": choice.finish_reason}],
                        })
                        serialized_bytes += len(b"data: " + chunk + b"\n\n")
        now = perf_counter_ns() - batch_start_ns
        for item in output.request_outputs:
            for choice in item.outputs:
                if choice.token_ids:
                    first_token_ns.setdefault(item.request_id, now)
                if choice.finish_reason is not None:
                    finished_ns[item.request_id] = now
        await asyncio.sleep(0)
    if len(first_token_ns) != request_count or len(finished_ns) != request_count:
        raise RuntimeError("Missing first token or completion timestamp")
    return emitted, serialized_bytes, list(first_token_ns.values()), list(finished_ns.values())


async def run_front_round(args, front_index, entries, images, tokenizer, pool, name,
                          input_processor, processor, encoder, decoder, sock, clock, batch_start_ns):
    return await asyncio.gather(
        preprocess_requests(args, front_index, entries, images, tokenizer, pool, name,
                            input_processor, processor, encoder, sock, clock),
        stream_outputs(sock, decoder, processor, clock, batch_start_ns, len(entries)),
    )


def front_worker(args, front_index, ipc_path, control):
    tokenizer = GlmRc5Tokenizer(args.model_metadata)
    cfg = config(args, skip_tokenizer_init=False)
    input_processor = InputProcessor(cfg, renderer=HfRenderer(cfg, tokenizer))
    request_indices = range(front_index, args.concurrency, args.api_server_count)
    entries = list(zip(request_indices, workload_messages(args, tokenizer, request_indices), strict=True))
    dummy_token_id = tokenizer.encode(" hello", add_special_tokens=False)[0]
    images = None
    if args.image_file:
        from image_codec import ImageCodec

        images = ImageCodec(args.image_file)
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(ipc_path)
    encoder, decoder = MsgpackEncoder(), MsgpackDecoder(EngineCoreOutputs)
    control.send((dummy_token_id, images.info if images else None))
    try:
        render_workers = min(args.renderer_num_workers, args.frontend_slots)
        with ThreadPoolExecutor(max_workers=render_workers, thread_name_prefix="renderer") as pool:
            with asyncio.Runner() as runner:
                while (job := control.recv()) is not None:
                    if job == "encode-reference":
                        control.send(images.measure_encode(len(entries), pool))
                        continue
                    name, batch_start_ns = job
                    cpu_start = process_time_ns()
                    clock = Clock()
                    processor = OutputProcessor(tokenizer, log_stats=False)
                    parallelism, responses = runner.run(run_front_round(
                        args, front_index, entries, images, tokenizer, pool, name,
                        input_processor, processor, encoder, decoder, sock, clock, batch_start_ns,
                    ))
                    emitted, serialized_bytes, first_token_ns, finished_ns = responses
                    metrics = dict(frontend_cpu_ns=process_time_ns() - cpu_start,
                                   host_ttft_ns=first_token_ns,
                                   host_request_latency_ns=finished_ns)
                    control.send((dict(clock.rows), emitted, serialized_bytes, metrics | parallelism))
    finally:
        sock.close()
        ctx.term()
