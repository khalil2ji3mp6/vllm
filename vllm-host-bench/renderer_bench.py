"""Isolated service-layout tokenizer: API processes, one vLLM executor per API."""
import asyncio
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace

from host_capacity import make_plan


def service_plan(topology, cores):
    # Match the whole-host --core-sweep planner for the supported DP8 core counts.
    dp = min(8, cores)
    width = cores // dp
    if cores % dp:
        # Nonuniform masks are allowed by the isolated CLI. Keep eight service
        # roles and distribute the requested CPUs with the host all-core planner.
        physical = set(topology["physical_core_cpus"])
        nodes = topology["numa_nodes"] or {0: topology["allowed_cpus"]}
        pools = [[c for c in node if c in physical] for node in nodes.values()]
        order = [c for stripe in zip_longest(*pools) for c in stripe if c is not None]
        selected = set(order[:cores])
        topology = dict(topology, allowed_cpus=sorted(selected), physical_core_cpus=sorted(selected),
                        numa_nodes={i: [c for c in node if c in selected] for i, node in nodes.items()
                                    if selected.intersection(node)})
        width = None
    args = SimpleNamespace(replicas=dp, cores_per_replica=width, tp=1,
                           concurrency_per_replica=48, text_only=False, full_frontend_pools=True,
                           renderer_num_workers=None, media_loading_threads=None)
    return make_plan(args, topology, None)


def set_gate(gate):
    global round_gate
    round_gate = gate


def measure_api(cpus, api_index, api_count, workers, rounds):
    # Pin before tokenizer/torch imports and before the renderer creates threads.
    try:
        os.sched_setaffinity(0, cpus)
        import uvloop
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            return runner.run(encode_rounds(api_index, api_count, workers, rounds))
    except BaseException:
        round_gate.abort()
        raise


async def encode_rounds(api_index, api_count, workers, rounds):
    from engine import config
    from frontend import MeasuredRenderer, peak_parallel, prompt_factory, render_clock
    from measure import Clock
    from tokenizer_rc5 import GlmRc5Tokenizer
    from vllm.renderers.params import TokenizeParams
    from vllm.utils.gc_utils import freeze_gc_heap

    model = Path(__file__).resolve().parent / "model"
    cfg = config(SimpleNamespace(model_metadata=model, renderer_num_workers=workers,
                                 max_model_len=135000, tp=1, block_size=128, max_num_seqs=48,
                                 chunk_tokens=16384, blocks=16384), skip_tokenizer_init=False)
    tokenizer = GlmRc5Tokenizer(model)
    renderer = MeasuredRenderer(cfg, tokenizer)
    try:
        messages = prompt_factory(tokenizer, range(api_index, 48, api_count), 32768)
        prompts = [dict(prompt=tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True)) for m in messages]
        params = TokenizeParams(max_total_tokens=135000, add_special_tokens=False)
        freeze_gc_heap()
        results = []
        for index in range(rounds + 1):
            clock = Clock()
            marker = render_clock.set(clock)
            try:
                round_gate.wait(timeout=300)
                start = perf_counter_ns()
                outputs = await renderer.tokenize_prompts_async(prompts, params)
                end = perf_counter_ns()
            finally:
                render_clock.reset(marker)
            spans = clock.rows["prefill", "tokenize"]
            if len(spans) != len(prompts) or any(len(o["prompt_token_ids"]) != 32768 for o in outputs):
                raise RuntimeError("Each submitted prompt must complete exactly one 32K encode")
            if index == 0:
                if any(o["prompt_token_ids"] != tokenizer.encode(p["prompt"], add_special_tokens=False)
                       for p, o in zip(prompts, outputs, strict=True)):
                    raise RuntimeError("Async renderer returned unexpected token IDs")
            else:
                results.append(dict(start=start, end=end, calls=len(spans),
                                    wall_sum_ns=sum(b - a for a, b, _ in spans),
                                    peak=peak_parallel([(a, b) for a, b, _ in spans])))
            del outputs
        return results
    finally:
        renderer.shutdown()


def measure_tokenizer(plan, rounds):
    roles = []
    for dp in plan["replicas"]:
        frontend = dp["frontend"]
        n = frontend["api_server_count"]
        roles.extend((dp["cpus"], i, n, frontend["renderer_num_workers"]) for i in range(n))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(roles), mp_context=ctx, initializer=set_gate,
                             initargs=(ctx.Barrier(len(roles)),)) as pool:
        jobs = [pool.submit(measure_api, *role, rounds) for role in roles]
        results = [job.result() for job in jobs]
    throughputs = []
    for batch in zip(*results, strict=True):
        calls = sum(r["calls"] for r in batch)
        if calls != 48 * len(plan["replicas"]):
            raise RuntimeError("Tokenizer workload differs from 48 requests per DP")
        throughputs.append(calls * 1e9 / (max(r["end"] for r in batch) - min(r["start"] for r in batch)))
    return dict(calls=sum(r["calls"] for rows in results for r in rows),
                wall_sum_ns=sum(r["wall_sum_ns"] for rows in results for r in rows),
                ops_s=median(throughputs), round_ops_s=throughputs,
                api_count=len(roles), peak_by_api=[max(r["peak"] for r in rows) for rows in results])
