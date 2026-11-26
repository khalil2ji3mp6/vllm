#!/usr/bin/env python3
"""vLLM host benchmark with one scheduler and CPU-only rank workers."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter_ns, process_time_ns, sleep
from unittest.mock import patch

# These are inherited by spawned children before importing vLLM or PyTorch.
os.environ.update(
    TORCH_DEVICE_BACKEND_AUTOLOAD="0", VLLM_TARGET_DEVICE="cpu", VLLM_PLUGINS="",
    VLLM_USE_V2_MODEL_RUNNER="0", ASCEND_RT_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
)
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
for env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[env] = "1"
os.environ.setdefault("PYTHONHASHSEED", "0")

# Transformers probes accelerator packages even on CPU. Mark the optional
# NPU package unavailable before it can load any driver libraries.
sys.modules["torch_npu"] = None

from cpu_topology import cpu_topology
from measure import Clock, distribution
from request_times import request_times


def affinity(args):
    topology = cpu_topology(args.numa_node)
    nodes = topology["numa_cpus"]
    representatives = set(topology["physical_core_cpus"])
    if args.affinity == "inherit":
        return [], [], [[] for _ in range(args.tp)], topology
    if len(representatives) < args.tp + 2:
        raise ValueError(f"{args.affinity} binding needs {args.tp + 2} available physical cores; "
                         f"found {len(representatives)}. Use --affinity inherit to share cores.")
    nodes = [[cpu for cpu in group if cpu in representatives] for group in nodes]
    nodes = [group for group in nodes if group]
    if args.affinity == "compact":
        preferred = next((group for group in nodes if len(group) >= args.tp + 2), None)
        cpus = preferred if preferred is not None else [cpu for group in nodes for cpu in group]
        return [cpus[0]], [cpus[1]], [[cpu] for cpu in cpus[2:args.tp + 2]], topology
    engine, front = nodes[0][:2] if len(nodes[0]) >= 3 else sorted(representatives)[:2]
    reserved = {engine, front}
    pools = [[cpu for cpu in node if cpu not in reserved] for node in nodes]
    worker_cpus = []
    for rank in range(args.tp):
        for offset in range(len(pools)):
            pool = pools[(rank + offset) % len(pools)]
            if pool:
                worker_cpus.append([pool.pop(0)])
                break
    assert len(worker_cpus) == args.tp
    return [engine], [front], worker_cpus, topology


def frontend_plan(cores, tp, concurrency, with_image, dedicated=False,
                  api_servers=None, renderer_workers=None, media_threads=None):
    # Leave room for the scheduler, model workers and one event loop per frontend.
    # Budgets size pools; in inherit mode neither the roles nor pool threads own dedicated CPUs.
    budget = 1 if dedicated else max(1, cores - tp - 1)
    if api_servers is None:
        # Validation and request-state creation run on each frontend's event loop.
        # Split long request batches across processes, with CPU room for their pools.
        api_servers = max(1, min(budget // 4, (concurrency + 15) // 16))
    per_frontend = (concurrency + api_servers - 1) // api_servers
    pool_budget = min(per_frontend, max(1, (budget - api_servers) // api_servers))
    if not with_image:
        media_threads = 0
    elif media_threads is None:
        media_threads = pool_budget
    if renderer_workers is None:
        renderer_workers = pool_budget
    return dict(api_server_count=api_servers, renderer_num_workers=renderer_workers,
                media_loading_threads=media_threads)


def get(pipe, timeout):
    if not pipe.poll(timeout):
        raise TimeoutError("A frontend/worker failed or timed out; see worker traceback")
    return pipe.recv()


def launch(ctx, name, target, params):
    parent, child = ctx.Pipe()
    proc = ctx.Process(name=name, target=target, args=(*params, child))
    proc.start()
    child.close()
    return proc, parent


def worker_entry(args, rank, handle, token_id, child):
    if args.worker_cpus[rank]:
        os.sched_setaffinity(0, args.worker_cpus[rank])
    from worker import run_worker
    try:
        run_worker(args, rank, handle, child, token_id)
    finally:
        child.close()


def front_entry(args, front_index, endpoint, child):
    if args.front_cpus:
        os.sched_setaffinity(0, args.front_cpus)
    from frontend import front_worker
    try:
        front_worker(args, front_index, endpoint, child)
    finally:
        child.close()


def await_replica_round(args, name):
    round_dir = args.round_gate / name
    (round_dir / f"ready-{args.replica_id}").touch()
    deadline = monotonic() + args.round_timeout
    while not (round_dir / "start").is_file():
        if monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting to start {name}")
        sleep(0.002)


@contextmanager
def measure_reply_acquire(reply, clock, active):
    acquire = reply.acquire_read

    @contextmanager
    def timed_acquire(*args, **kwargs):
        with ExitStack() as stack:
            # Retain the SHM read lease through upstream unpacking and release.
            with clock.span(active["phase"], "engine_worker_reply_acquire"):
                buf = stack.enter_context(acquire(*args, **kwargs))
            with clock.span(active["phase"], "engine_worker_unpack"):
                yield buf

    with patch.object(reply, "acquire_read", timed_acquire):
        yield


def timed_scenario(args, name, cfg, case, fronts, token_id, sockets, tx, reply, workers):
    import zmq

    from engine import add, build_scheduler, classify, seed_prefixes, time_scheduler_children
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    sched, hasher = build_scheduler(cfg, args)
    clock = Clock()
    if args.round_gate:
        await_replica_round(args, name)
    engine_cpu_start = process_time_ns()
    start = perf_counter_ns()
    tx.enqueue(("begin",), timeout=args.timeout)
    for front in fronts:
        front.send((name, start))
    decoder, encoder = MsgpackDecoder(EngineCoreRequest), MsgpackEncoder()
    poller = zmq.Poller()
    for sock in sockets:
        poller.register(sock, zmq.POLLIN)

    def receive_requests(wait):
        with clock.span("prefill", "engine_request_receive_wait"):
            events = poller.poll(args.timeout * 1000 if wait else 0)
        if wait and not events:
            raise TimeoutError("Timed out waiting for frontend requests")
        requests = []
        for sock, _ in events:
            while True:
                try:
                    frames = sock.recv_multipart(zmq.DONTWAIT)
                except zmq.Again:
                    break
                with clock.span("prefill", "engine_request_deserialize"):
                    requests.append(decoder.decode(frames))
        return requests

    step_weights = defaultdict(list)
    send_weights = defaultdict(list)
    generated = set()
    received = 0
    seed_ns = 0
    if case == "l1":
        requests = []
        while len(requests) < args.concurrency:
            requests.extend(receive_requests(wait=True))
        requests.sort(key=lambda request: int(request.request_id.rsplit("-", 1)[1]))
        received = len(requests)
        seed_start = perf_counter_ns()
        seed_prefixes(sched, requests, hasher, args.hit_tokens, token_id)
        seed_ns = perf_counter_ns() - seed_start
        for req in requests:
            with clock.span("prefill", "engine_add_request_and_hash"):
                add(sched, req, hasher)

    counts = Counter()
    with time_scheduler_children(sched, clock) as active, measure_reply_acquire(reply, clock, active):
        while received < args.concurrency or sched.has_requests():
            if received < args.concurrency:
                requests = receive_requests(wait=not sched.has_requests())
                if requests:
                    received += len(requests)
                    for req in requests:
                        with clock.span("prefill", "engine_add_request_and_hash"):
                            add(sched, req, hasher)
            if not sched.has_requests():
                continue
            step_start = perf_counter_ns()
            with clock.span("pending", "schedule"):
                output = sched.schedule()
            phase, prefill, decode, sample = classify(sched, output)
            clock.move_last("schedule", phase)
            for child in ("prefix_lookup", "kv_allocate_or_extend"):
                clock.rows[phase, child].extend(clock.rows["pending", child])
                clock.rows["pending", child].clear()
            if not output.total_num_scheduled_tokens:
                if sched.get_num_unfinished_requests():
                    raise RuntimeError("Scheduler stalled on unfinished requests")
                continue
            if output.total_num_scheduled_tokens > args.chunk_tokens:
                raise RuntimeError("Scheduler exceeded configured token budget")
            if output.preempted_req_ids:
                raise RuntimeError("Logical KV cache capacity was insufficient")
            n_decode = len(generated.intersection(sample))
            step_weights[phase].append((len(sample) - n_decode, n_decode))
            active["phase"] = phase
            counts.setdefault("first_scheduled_step_ns", step_start)
            counts[phase + "_steps"] += 1
            counts["scheduled_prefill_tokens"] += prefill
            counts["scheduled_decode_tokens"] += decode
            # One outstanding scheduler step per replica. The next step needs this reply.
            with clock.span(phase, "engine_worker_enqueue"):
                tx.enqueue((phase, output, sample), timeout=args.timeout)
            with clock.span(phase, "engine_worker_reply_wait"):
                result = reply.dequeue(timeout=args.timeout)
            with clock.span(phase, "engine_output_postprocess"):
                with clock.span(phase, "update_request_and_output"):
                    outputs = sched.update_from_output(output, result)
                for client, group in outputs.items():
                    if not group.outputs:
                        continue
                    ids = [o.request_id for o in group.outputs if o.new_token_ids]
                    decoded = sum(rid in generated for rid in ids)
                    send_weights[phase].append((len(ids) - decoded, decoded))
                    generated.update(ids)
                    counts["output_tokens"] += sum(len(o.new_token_ids) for o in group.outputs)
                    with clock.span(phase, "engine_front_output_pack"):
                        sockets[client].send_multipart([phase.encode(), *encoder.encode(group)])
            counts[phase + "_step_wall_ns"] += perf_counter_ns() - step_start

    for sock in sockets:
        sock.send_multipart([b"done"])
    engine_cpu_ns = process_time_ns() - engine_cpu_start
    tx.enqueue(("end",), timeout=args.timeout)
    front_rows = defaultdict(list)
    front_tokens = serialized_bytes = 0
    front_metrics = dict(frontend_cpu_ns=0, host_ttft_ns=[], host_request_latency_ns=[],
                         renderer_peak_threads=0, media_peak_threads=0, image_text_overlap=False)
    ready = start
    image_end = None
    output_times = {"Prefill": Counter(), "Decode": Counter()}
    for front in fronts:
        rows, tokens, size, metrics = get(front, args.timeout)
        for section, totals in metrics["output_times"].items():
            output_times[section].update(totals)
        for key, values in rows.items():
            front_rows[key].extend(values)
        if "prefill_resources" in metrics:
            totals = front_metrics.setdefault("prefill_resources", {})
            for stage, counts_by_resource in metrics["prefill_resources"].items():
                totals.setdefault(stage, Counter()).update(counts_by_resource)
        front_tokens += tokens
        serialized_bytes += size
        front_metrics["frontend_cpu_ns"] += metrics["frontend_cpu_ns"]
        for field in ("host_ttft_ns", "host_request_latency_ns"):
            front_metrics[field].extend(metrics[field])
        for field in ("renderer_peak_threads", "media_peak_threads"):
            front_metrics[field] = max(front_metrics[field], metrics[field])
        front_metrics["image_text_overlap"] |= metrics["media_renderer_overlap"]
        ready = max(ready, metrics["front_ready_ns"])
        if metrics["image_end_ns"] is not None:
            image_end = max(image_end or start, metrics["image_end_ns"])
    front_rows["prefill", "front_prefill_batch"].append((start, ready, 0))
    if image_end is not None:
        front_rows["prefill", "image_ingress_batch"].append((start, image_end, 0))
    end = start + max(front_metrics["host_request_latency_ns"])
    worker_results = [get(pipe, args.timeout) for pipe in workers]
    expected_outputs = args.concurrency * args.new_tokens
    expected_prefill = args.concurrency * (args.prompt_tokens - (args.hit_tokens if case == "l1" else 0))
    if counts["output_tokens"] != expected_outputs or front_tokens != expected_outputs:
        raise RuntimeError(f"Wrong output count: engine={counts['output_tokens']}, frontend={front_tokens}")
    scheduled_prefill = counts["scheduled_prefill_tokens"]
    if args.request_file is not None:
        if not 0 < scheduled_prefill <= expected_prefill:
            raise RuntimeError(f"Unexpected scheduled prompt token count: {scheduled_prefill}")
    elif scheduled_prefill != expected_prefill:
        raise RuntimeError(f"Wrong prefix reuse: {counts['scheduled_prefill_tokens']} != {expected_prefill}")
    counts["frontend_serialized_bytes"] = serialized_bytes
    counts["input_tokens"] = args.concurrency * args.prompt_tokens
    counts["prefix_hit_tokens"] = counts["input_tokens"] - counts["scheduled_prefill_tokens"]
    counts["scenario_wall_ns_excluding_seed"] = end - start - seed_ns
    counts["front_prefill_wall_ns"] = ready - start
    counts["scheduling_frontend_overlap_ns"] = max(
        0, ready - counts.get("first_scheduled_step_ns", ready),
    )
    counts["start_ns"] = start
    counts["end_ns"] = end
    counts["diagnostic_tail_ns"] = perf_counter_ns() - end
    counts["engine_cpu_ns"] = engine_cpu_ns
    expected_decode = expected_outputs - args.concurrency
    scheduled_decode = sum(d for weights in step_weights.values() for _, d in weights)
    if len(generated) != args.concurrency or scheduled_decode != expected_decode:
        raise RuntimeError("Request-stage weights disagree with emitted token counts")
    metrics = dict(counts) | front_metrics
    metrics["request_stage_ns"] = request_times(
        clock.rows, front_rows, [rows for rows, _, _ in worker_results],
        step_weights, send_weights, output_times,
    )
    metrics["worker_cpu_ns"] = [cpu_ns for _, cpu_ns, _ in worker_results]
    return metrics, {"engine": clock.rows, "frontend": front_rows,
                     **{f"worker{rank}": rows for rank, (rows, _, _) in enumerate(worker_results)}}


def write_report(args, samples, timed_rounds, topology, masks, frontend_masks, image):
    import vllm
    from report import capture_stage_intervals, write_coverage

    intervals = {}
    raw = defaultdict(list)
    for case, rounds in timed_rounds.items():
        intervals[case] = [capture_stage_intervals(processes) for processes in rounds]
        for processes in rounds:
            for process, phases in processes.items():
                for (phase, stage), times in phases.items():
                    if times:
                        raw[case, process, phase, stage].extend(times)
    # Combine the original occurrences before computing medians. Prefill
    # includes every step that scheduled at least one prompt token.
    for (case, process, phase, stage), times in list(raw.items()):
        if phase in ("prefill", "mixed"):
            raw[case, process, "prefill_inclusive", stage].extend(times)
    rows = []
    for (case, process, phase, stage), times in sorted(raw.items()):
        rows.append(dict(case=case, process=process, phase=phase, stage=stage, **distribution(times)))
    summary = {
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "hardware": {
            "platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine(), "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
            "cpu_model": next((line.split(":", 1)[1].strip() for line in
                               Path("/proc/cpuinfo").read_text().splitlines()
                               if line.startswith(("model name", "CPU part"))), None),
            "affinity_before_binding": topology, "worker_affinity": masks,
            "engine_affinity": sorted(os.sched_getaffinity(0)),
            "frontend_affinity": frontend_masks[0],  # Preserve the original single-mask field.
            "frontend_affinities": frontend_masks,
        },
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "transformers", "tokenizers", "pyzmq", "msgspec",
                                  "pillow", "pybase64")}
                    | {"vllm": vllm.__version__,
                       "vllm_source_commit": os.environ.get("VLLM_HOST_SOURCE_COMMIT")},
        "metadata_sha256": {
            name: hashlib.sha256((args.model_metadata / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        },
        "request_file_sha256": (
            hashlib.sha256(args.request_file.read_bytes()).hexdigest() if args.request_file else None
        ),
        "image": image,
        "workload": samples,
        "timings": rows,
        "stage_intervals": intervals,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "stages.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.round_gate is None:
        write_coverage(summary, args.out)


def wait_for_shutdown(children, timeout):
    deadline = monotonic() + timeout
    for child in children:
        child.join(max(0, deadline - monotonic()))
        if child.exitcode is None:
            raise TimeoutError(f"{child.name} (PID {child.pid}) did not exit within "
                               f"the {timeout}s shutdown budget")
        if child.exitcode != 0:
            raise RuntimeError(f"{child.name} (PID {child.pid}) exited with code {child.exitcode}")


def run(args):
    args.engine_cpus, args.front_cpus, args.worker_cpus, topology = affinity(args)
    vars(args).update(frontend_plan(
        len(topology["physical_core_cpus"]), args.tp, args.concurrency,
        bool(args.image_file), bool(args.front_cpus),
        args.api_server_count, args.renderer_num_workers, args.media_loading_threads,
    ))
    os.environ["VLLM_MEDIA_LOADING_THREAD_COUNT"] = str(max(1, args.media_loading_threads))
    if args.engine_cpus:
        os.sched_setaffinity(0, args.engine_cpus)
    import torch
    import zmq
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.platforms import current_platform
    from engine import config

    if sys.modules.get("torch_npu") is not None or not current_platform.is_cpu():
        raise RuntimeError("Run the harness with CPU vLLM and hardware backend autoload disabled")
    torch.set_num_threads(1)
    ctx = multiprocessing.get_context("spawn")
    children = []
    sockets = []
    tx = reply = None
    with TemporaryDirectory(prefix="vllm-host-") as ipc_dir:
        zctx = zmq.Context()
        try:
            fronts = []
            for index in range(args.api_server_count):
                sock = zctx.socket(zmq.PAIR)
                sockets.append(sock)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, args.timeout * 1000)
                sock.setsockopt(zmq.SNDTIMEO, args.timeout * 1000)
                endpoint = f"ipc://{ipc_dir}/frontend-{index}.sock"
                sock.bind(endpoint)
                front_proc, front = launch(ctx, f"frontend-{index}", front_entry,
                                           (args, index, endpoint))
                children.append(front_proc)
                fronts.append(front)
            frontend_info = [get(front, args.timeout) for front in fronts]
            token_id, image, _ = frontend_info[0]
            frontend_masks = [mask for _, _, mask in frontend_info]
            for front_token, front_image, _ in frontend_info:
                if front_token != token_id or front_image != image:
                    raise RuntimeError("Frontends disagreed on the synthetic output token or fixture")
            if image is not None:
                image["media_loading_threads"] = args.media_loading_threads
            tx = MessageQueue(args.tp, args.tp, max_chunk_bytes=4 << 20, max_chunks=2)
            handle = tx.export_handle()
            workers = []
            handles = []
            for rank in range(args.tp):
                proc, control = launch(ctx, f"worker-{rank}", worker_entry, (args, rank, handle, token_id))
                children.append(proc)
                workers.append(control)
            for control in workers:
                handles.append(get(control, args.timeout))
            reply = MessageQueue.create_from_handle(handles[0]["reply"], 0)
            tx.wait_until_ready()
            reply.wait_until_ready()
            masks = {rank: info["affinity"] for rank, info in enumerate(handles)}
            cfg = config(args, skip_tokenizer_init=True)
            samples = {}
            timed_rounds = {}
            for case in args.cases:
                samples[case] = []
                timed_rounds[case] = []
                for iteration in range(-args.warmup, args.runs):
                    name = f"{case}-{iteration}"
                    counts, processes = timed_scenario(
                        args, name, cfg, case, fronts, token_id, sockets, tx, reply, workers,
                    )
                    if args.round_gate is not None:
                        (args.round_gate / name / f"done-{args.replica_id}").touch()
                    if iteration < 0:
                        continue
                    samples[case].append(dict(counts))
                    timed_rounds[case].append(processes)
                    name = f"{case} " if len(args.cases) > 1 else ""
                    print(f"{name}run {iteration + 1}/{args.runs} completed", flush=True)
            tx.enqueue(("stop",), timeout=args.timeout)
            # Synchronized reference rounds follow all timed Host rounds.
            if image is not None:
                for case, rounds in timed_rounds.items():
                    for index, processes in enumerate(rounds):
                        name = f"encode-{case}-{index}"
                        if args.round_gate is not None:
                            await_replica_round(args, name)
                        for front in fronts:
                            front.send("encode-reference")
                        for front in fronts:
                            for key, values in get(front, args.timeout).items():
                                processes["frontend"][key].extend(values)
                        if args.round_gate is not None:
                            (args.round_gate / name / f"done-{args.replica_id}").touch()
            for front in fronts:
                front.send(None)
            write_report(args, samples, timed_rounds, topology, masks, frontend_masks, image)
            wait_for_shutdown(children, args.timeout)
            if args.round_gate is None:
                print(f"Prefill and Decode tables saved to {args.out / 'coverage.md'}", flush=True)
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                child.join(3)
                if child.is_alive():
                    child.kill()
                    child.join()
            for queue in (tx, reply):
                if queue is not None:
                    queue.local_socket.context.destroy(linger=0)
            for sock in sockets:
                sock.close()
            zctx.term()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-metadata", type=Path, default=Path("model").resolve())
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--prompt-tokens", type=int, default=32768)
    p.add_argument("--request-file", type=Path, help="JSONL with one messages array per request")
    p.add_argument("--image-file", type=Path, help="Local JPEG/PNG used once per request")
    p.add_argument("--api-server-count", type=int, help="Local frontend processes for this replica")
    p.add_argument("--renderer-num-workers", type=int, help="Tokenizer threads per frontend")
    p.add_argument("--media-loading-threads", type=int, help="Media threads per frontend")
    p.add_argument("--new-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--cases", nargs="+", choices=("cold", "l1"), default=["cold"])
    p.add_argument("--hit-tokens", type=int, default=16384)
    p.add_argument("--chunk-tokens", type=int, default=16384)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--blocks", type=int, help="Override the automatic logical KV block budget")
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=135000)
    p.add_argument("--affinity", choices=("compact", "numa", "inherit"), default="compact")
    p.add_argument("--numa-node", type=int, help="Restrict compact binding to this Linux NUMA node ID")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--timeout", type=int, default=45)
    p.add_argument("--round-gate", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--replica-id", type=int, help=argparse.SUPPRESS)
    p.add_argument("--round-timeout", type=int, default=600, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.request_file is not None:
        args.request_file = args.request_file.resolve()
        if not args.request_file.is_file():
            p.error(f"Request file not found: {args.request_file}")
    if args.image_file is not None:
        args.image_file = args.image_file.resolve()
        if not args.image_file.is_file():
            p.error(f"Image file not found: {args.image_file}")
    if args.round_gate is not None and (args.replica_id is None or args.round_timeout <= 0):
        p.error("A round gate needs a replica ID and positive timeout")
    if args.numa_node is not None and (args.affinity != "compact" or args.numa_node < 0):
        p.error("--numa-node needs --affinity compact and a nonnegative node ID")
    if min(args.tp, args.max_num_seqs, args.prompt_tokens, args.new_tokens, args.concurrency,
           args.block_size, args.chunk_tokens, args.runs, args.timeout) < 1 or args.warmup < 0:
        p.error("Counts must be positive and warmup nonnegative")
    if args.blocks is None:
        # Each live sequence can grow to its output limit. vLLM reserves one null block.
        tokens = args.prompt_tokens + args.new_tokens
        blocks_per_seq = (tokens + args.block_size - 1) // args.block_size
        args.blocks = max(8192, 1 + min(args.concurrency, args.max_num_seqs) * blocks_per_seq)
    elif args.blocks < 1:
        p.error("--blocks must be positive")
    if any(n is not None and n < 1 for n in
           (args.api_server_count, args.renderer_num_workers, args.media_loading_threads)):
        p.error("Frontend process and thread counts must be positive")
    if args.api_server_count is not None and args.api_server_count > args.concurrency:
        p.error("--api-server-count cannot exceed replica concurrency")
    if args.affinity != "inherit" and args.api_server_count not in (None, 1):
        p.error("Multiple API frontends require --affinity inherit")
    if args.prompt_tokens + args.new_tokens > args.max_model_len:
        p.error("Workload exceeds max_model_len")
    if args.chunk_tokens % args.block_size:
        p.error("chunk-tokens must be block aligned")
    if "l1" in args.cases and (not 0 < args.hit_tokens < args.prompt_tokens or args.hit_tokens % args.block_size):
        p.error("l1 hit length must be block aligned and strictly below input length")
    if args.out.exists():
        p.error("use a new output directory")
    args.model_metadata = args.model_metadata.resolve()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (args.model_metadata / name).is_file():
            p.error(f"Missing model metadata {args.model_metadata / name}")
    return args


if __name__ == "__main__":
    run(parse_args())
