"""Local frontend with per-request image and text preprocessing."""
import asyncio
from contextvars import ContextVar
from time import perf_counter_ns, process_time_ns

import msgspec
import uvloop
import zmq
import zmq.asyncio
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.renderers.hf import HfRenderer, safe_apply_chat_template
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.gc_utils import freeze_gc_heap
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


# The event loop carries this clock with the request. Capture it on submit:
# run_in_executor does not propagate context variables to executor threads.
render_clock = ContextVar("render_clock")


class MeasuredRenderer(HfRenderer):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg, tokenizer)
        self._apply_chat_template_async = self._measure("chat_template", safe_apply_chat_template)
        self._tokenize_prompt_async = self._measure("tokenize", self._tokenize_prompt)

    def _measure(self, stage, function):
        def submit(*args, **kwargs):
            clock = render_clock.get()
            submitted = perf_counter_ns()

            def run():
                clock.rows["prefill", "renderer_queue_wait"].append((submitted, perf_counter_ns(), 0))
                with clock.prefill(stage):
                    return function(*args, **kwargs)

            return asyncio.get_running_loop().run_in_executor(self._executor, run)

        return submit


def peak_parallel(intervals):
    events = sorted((t, change) for start, end in intervals for t, change in ((start, 1), (end, -1)))
    peak = active = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    return peak


async def preprocess_requests(args, front_index, entries, images, renderer, name,
                              input_processor, processor, encoder, sock, batch_clock):
    async def prepare(index, messages):
        clock = Clock()
        start = perf_counter_ns()
        with clock.prefill("request_validate"):
            payload = msgspec.json.encode(dict(
                model="glm", messages=messages, max_tokens=args.new_tokens, stream=True,
            ))
            request = ChatCompletionRequest.model_validate_json(payload)
            params = SamplingParams(
                max_tokens=request.max_tokens, ignore_eos=True, stop=["<never-stop>"],
                output_kind=RequestOutputKind.DELTA, temperature=0.0,
            )
        image_interval = await images.decode_request(clock) if images else None
        token = render_clock.set(clock)
        try:
            result = await renderer.render_chat(request)
        finally:
            render_clock.reset(token)
        if isinstance(result, ErrorResponse):
            raise ValueError(result.error.message)
        _, (engine_input,) = result
        if len(engine_input["prompt_token_ids"]) != args.prompt_tokens:
            raise RuntimeError(f"Unexpected prompt token count for request {index}")
        with clock.prefill("input_validate_and_enqueue"):
            req = input_processor.process_inputs(
                f"{name}-{index}", engine_input, params, ("generate",),
            )
            req.external_req_id = req.request_id
            req.client_index = front_index
            processor.add_request(req, engine_input["prompt"])
        preprocess_cpu = sum(cpu for samples in clock.rows.values() for _, _, cpu in samples)
        clock.rows["prefill", "front_prefill_preprocess"].append(
            (start, perf_counter_ns(), preprocess_cpu)
        )
        with clock.span("prefill", "front_engine_encode_send"):
            await sock.send_multipart(encoder.encode(req))
        for key, values in clock.rows.items():
            batch_clock.rows[key].extend(values)
        batch_clock.merge_resources(clock)
        return image_interval

    images_done = await asyncio.gather(*(prepare(index, messages) for index, messages in entries))
    image_intervals = [interval for interval in images_done if interval is not None]
    render_intervals = [(start, end) for stage in ("chat_template", "tokenize")
                        for start, end, _ in batch_clock.rows["prefill", stage]]
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


def serialize_chunks(items):
    emitted = serialized_bytes = 0
    for item in items:
        for choice in item.outputs:
            emitted += len(choice.token_ids)
            chunk = msgspec.json.encode({
                "id": item.request_id, "object": "chat.completion.chunk",
                "choices": [{"index": choice.index, "delta": {"content": choice.text},
                             "finish_reason": choice.finish_reason}],
            })
            serialized_bytes += len(b"data: " + chunk + b"\n\n")
    return emitted, serialized_bytes


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
                count, size = serialize_chunks(output.request_outputs)
                emitted += count
                serialized_bytes += size
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
    for rid, first in first_token_ns.items():
        clock.rows["prefill", "request_phase_lifetime"].append((batch_start_ns, batch_start_ns + first, 0))
        clock.rows["decode", "request_phase_lifetime"].append((batch_start_ns + first, batch_start_ns + finished_ns[rid], 0))
    return emitted, serialized_bytes, list(first_token_ns.values()), list(finished_ns.values())


async def run_front_round(args, front_index, entries, images, renderer, name,
                          input_processor, processor, encoder, decoder, sock, clock, batch_start_ns):
    return await asyncio.gather(
        preprocess_requests(args, front_index, entries, images, renderer, name,
                            input_processor, processor, encoder, sock, clock),
        stream_outputs(sock, decoder, processor, clock, batch_start_ns, len(entries)),
    )


def front_worker(args, front_index, ipc_path, control):
    tokenizer = GlmRc5Tokenizer(args.model_metadata)
    cfg = config(args, skip_tokenizer_init=False)
    renderer = MeasuredRenderer(cfg, tokenizer)
    online_renderer = OnlineRenderer(
        cfg.model_config, renderer, request_logger=None,
        chat_template=None, chat_template_content_format="auto",
    )
    input_processor = InputProcessor(cfg, renderer=renderer)
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
    # Match the API server lifespan before accepting timed requests.
    freeze_gc_heap()
    control.send((dummy_token_id, images.info if images else None))
    try:
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            while (job := control.recv()) is not None:
                if job == "encode-reference":
                    control.send(images.measure_encode(len(entries), renderer._executor))
                    continue
                name, batch_start_ns = job
                cpu_start = process_time_ns()
                clock = Clock()
                processor = OutputProcessor(tokenizer, log_stats=False)
                parallelism, responses = runner.run(run_front_round(
                    args, front_index, entries, images, online_renderer, name,
                    input_processor, processor, encoder, decoder, sock, clock, batch_start_ns,
                ))
                emitted, serialized_bytes, first_token_ns, finished_ns = responses
                metrics = dict(frontend_cpu_ns=process_time_ns() - cpu_start,
                               host_ttft_ns=first_token_ns,
                               host_request_latency_ns=finished_ns)
                if clock.resources:
                    metrics["prefill_resources"] = dict(clock.resources)
                control.send((dict(clock.rows), emitted, serialized_bytes, metrics | parallelism))
    finally:
        renderer.shutdown()
        sock.close()
        ctx.term()
