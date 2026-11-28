"""One CPU core, one request, and direct calls between frontend, scheduler and TP1 worker."""
import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter_ns

import bench  # Apply the same CPU-only environment before loading vLLM.
from cpu_topology import benchmark_topology
from flow_groups import ENGINE, IMAGE, IPC, OUTPUT, TEXT, WORKER, groups
from host_capacity import reserve_output_dir
from measure import Clock


def verify_affinity(cpu):
    tasks = list(Path("/proc/self/task").iterdir())
    for task in tasks:
        if os.sched_getaffinity(int(task.name)) != {cpu}:
            raise RuntimeError(f"Thread {task.name} is outside the selected physical core")
        if (task / "children").read_text().strip():
            raise RuntimeError("Unexpected child process in single-process mode")


async def run_request(args, cfg, renderer, inputs, tokenizer, entries, images, token):
    from engine import add, build_scheduler, classify, dummy_result
    from frontend import OutputProcessor, preprocess_requests, serialize_chunks
    from worker import CpuBatch

    scheduler, hasher = build_scheduler(cfg, args)
    processor = OutputProcessor(tokenizer, log_stats=False)
    batch = CpuBatch(args.block_size, token)
    clock = Clock()

    async def admit(req):
        # The frontend and the scheduler share the request object. No IPC codec.
        add(scheduler, req, hasher)

    started = perf_counter_ns()
    await preprocess_requests(args, 0, entries, images, renderer, "single", inputs,
                              processor, admit, clock)
    emitted = prompt = decode = 0
    steps = Counter()
    first = last = None
    finished = False
    while not finished:
        phase = "decode" if emitted else "prefill"
        with clock.span(phase, ENGINE):
            scheduled = scheduler.schedule()
            kind, p, d, ready = classify(scheduler, scheduled)
        if not 0 < scheduled.total_num_scheduled_tokens <= args.chunk_tokens or kind != phase:
            raise RuntimeError(f"Unexpected {phase} scheduling step: {scheduled.total_num_scheduled_tokens}")
        if scheduled.preempted_req_ids:
            raise RuntimeError("Unexpected preemption of the single request")
        with clock.span(phase, WORKER):
            batch.update(scheduled)
            batch.prepare(scheduled)
        result = dummy_result(ready, token)
        with clock.span(phase, ENGINE):
            updates = scheduler.update_from_output(scheduled, result)
        for update in updates.values():
            if not update.outputs:
                continue
            with clock.span(phase, OUTPUT):
                response = processor.process_outputs(update.outputs, update.timestamp)
                if response.reqs_to_abort:
                    raise RuntimeError("Unexpected stop-string hit")
                n, _ = serialize_chunks(response.request_outputs)
            timestamp = perf_counter_ns()
            if n and first is None:
                first = timestamp
            emitted += n
            if any(choice.finish_reason is not None for item in response.request_outputs
                   for choice in item.outputs):
                finished = True
                last = timestamp
        prompt += p
        decode += d
        steps[phase] += 1

    if (first is None or last is None or prompt != args.prompt_tokens or
            emitted != args.new_tokens or decode != emitted - 1 or scheduler.get_num_unfinished_requests()):
        raise RuntimeError(f"Incomplete request: prompt={prompt}, decode={decode}, outputs={emitted}")

    def duration(phase, key):
        return sum(b - a for a, b, _ in clock.rows[phase, key])

    phases = {}
    for phase, wall in (("prefill", first - started), ("decode", last - first)):
        times = {g: duration(phase, g) for g in groups(phase.capitalize())}
        if phase == "prefill":
            times[TEXT] = duration(phase, "request_text_elapsed")
            times[IMAGE] = duration(phase, "request_image_elapsed")
            times[IPC] = duration(phase, "front_engine_encode_send")
        if sum(times.values()) > wall:
            raise RuntimeError(f"Overlapping {phase} timers")
        phases[phase] = dict(stages_ns=times, wall_ns=wall, steps=steps[phase])
    return dict(**phases, e2e_ns=last - started, tokenizer_ns=duration("prefill", "tokenize"),
                prompt_tokens=prompt, generated=emitted, decode_tokens=decode)


def report(result):
    lines = [
        "# 单进程单核整网",
        "",
        f"DP1/TP1，Linux CPU {result['cpu']}，一个请求输入 {result['prompt_tokens']:,} token，"
        f"输出 {result['generated']:,} token。预热 {result['warmup']} 次后计时一次。",
        f"Prefill 共 {result['prefill']['steps']} 个调度步，Decode 共 {result['decode']['steps']} 个输出步。",
        "",
    ]
    for phase in ("prefill", "decode"):
        data = result[phase]
        header = "| 流程 | 分项耗时（ms） | 占比 |"
        if phase == "decode":
            header += " 每输出步（µs） |"
        lines += [f"## {phase.capitalize()}", "", header, "|---|---:|---:|" + ("---:|" if phase == "decode" else "")]
        measured = sum(data["stages_ns"].values())
        rows = {**data["stages_ns"], "阶段合计": measured}
        for group, ns in rows.items():
            share = f"{100 * ns / measured:.2f}%" if measured else "—"
            line = f"| {group} | {ns/1e6:.3f} | {share} |"
            if phase == "decode":
                avg = f"{ns/1000/result['decode_tokens']:.2f}" if result["decode_tokens"] else "—"
                line += f" {avg} |"
            lines.append(line)
        lines.append("")
    lines.append(f"Tokenizer {result['tokenizer_ns']/1e6:.3f} ms（含在文本前处理），"
                 f"端到端 {result['e2e_ns']/1e6:.3f} ms。")
    return "\n".join(lines) + "\n"


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-tokens", type=int, default=204800)
    parser.add_argument("--new-tokens", type=int, default=1024)
    parser.add_argument("--chunk-tokens", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=1)
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--text-only", action="store_true")
    media.add_argument("--image-file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("host-single-process"))
    args = parser.parse_args()
    limit = json.loads((root / "model" / "config.json").read_text())["max_position_embeddings"]
    length = args.prompt_tokens + args.new_tokens
    if args.prompt_tokens < 20 or args.new_tokens < 1 or length > limit or args.warmup < 0:
        parser.error("Use at least 20 prompt tokens, positive output, a valid context and nonnegative warmup")
    if args.chunk_tokens <= 0 or args.chunk_tokens % 128 or args.chunk_tokens > limit:
        parser.error("Chunk size must be positive, block-aligned and fit the model context")
    args.model_metadata = root / "model"
    args.tp = args.renderer_num_workers = args.max_num_seqs = 1
    args.block_size = 128
    args.image_barrier_parties = 0
    args.max_model_len = min(limit, max(135000, length))
    args.blocks = max(8192, 1 + (length + 127) // 128)
    cpu = benchmark_topology()["physical_core_cpus"][0]
    os.sched_setaffinity(0, {cpu})
    os.environ.update(RAYON_NUM_THREADS="1", VLLM_MEDIA_LOADING_THREAD_COUNT="1")
    import torch
    import uvloop
    from engine import config
    from frontend import GlmRc5Tokenizer, InputProcessor, MeasuredRenderer, OnlineRenderer, prompt_factory
    from image_codec import ImageCodec

    torch.set_num_threads(1)
    cfg = config(args, skip_tokenizer_init=False)
    scheduler_cfg = config(args, skip_tokenizer_init=True)
    tokenizer = GlmRc5Tokenizer(args.model_metadata)
    renderer = MeasuredRenderer(cfg, tokenizer)
    try:
        online = OnlineRenderer(cfg.model_config, renderer, request_logger=None,
                                chat_template=None, chat_template_content_format="auto")
        inputs = InputProcessor(cfg, renderer=renderer)
        messages = prompt_factory(tokenizer, [0], args.prompt_tokens)
        images = None if args.text_only else ImageCodec(args.image_file or root / "assets/input.jpg")
        token = tokenizer.encode(" hello", add_special_tokens=False)[0]
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as loop:
            for _ in range(args.warmup + 1):
                result = loop.run(run_request(args, scheduler_cfg, online, inputs, tokenizer,
                                              [(0, messages[0])], images, token))
                verify_affinity(cpu)
    finally:
        renderer.shutdown()
    result.update(cpu=cpu, pid=os.getpid(), dp=1, tp=1, concurrency=1, chunk_tokens=args.chunk_tokens,
                  warmup=args.warmup, image=images.info if images else None)
    out = reserve_output_dir(args.out)
    md = report(result)
    (out / "coverage.md").write_text(md)
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(md, end="")


if __name__ == "__main__":
    main()
