"""Compare GLM-5.2 native tokenizer versions with equal text and CPU masks."""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import threading
from array import array
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter_ns


def freeze_environment():
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def digest(ids):
    return hashlib.sha256(array("I", ids).tobytes()).hexdigest()


def prepare(model):
    from frontend import prompt_factory
    from tokenizer_rc5 import GlmRc5Tokenizer

    tokenizer = GlmRc5Tokenizer(model)
    prompts = []
    for messages in prompt_factory(tokenizer, range(32), 32768):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompts.append({"text": text, "sha256": digest(ids), "tokens": len(ids)})
    return prompts


def measure_dp(model, prompts, cores, workers, rounds, gate):
    from tokenizers import Tokenizer

    os.sched_setaffinity(0, cores)
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    compiled = hasattr(tokenizer, "compile")
    if compiled:
        tokenizer = tokenizer.compile()
    local = threading.local()

    def initialize():
        local.tokenizer = tokenizer if compiled else copy.deepcopy(tokenizer)

    def encode(prompt):
        return local.tokenizer.encode(prompt["text"], add_special_tokens=False).ids

    def timed_encode(prompt):
        start = perf_counter_ns()
        ids = encode(prompt)
        end = perf_counter_ns()
        if len(ids) != prompt["tokens"]:
            raise ValueError(f"Expected {prompt['tokens']} tokens, got {len(ids)}")
        return start, end

    with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        # Start every worker and load its tokenizer before timed rounds.
        ready = threading.Barrier(workers)
        initializers = [pool.submit(ready.wait, 60) for _ in range(workers)]
        for worker in initializers:
            worker.result()
        for ids, prompt in zip(pool.map(encode, prompts), prompts, strict=True):
            if digest(ids) != prompt["sha256"]:
                raise ValueError(f"Token IDs differ for {prompt['text'][:72]!r}")
        spans = []
        for _ in range(rounds):
            gate.wait(timeout=60)
            spans.append(list(pool.map(timed_encode, prompts)))
    return spans


def elapsed(intervals):
    duration = end = 0
    for start, stop in sorted(intervals):
        duration += max(0, stop - max(start, end))
        end = max(end, stop)
    return duration


def run(args, prompts, plans, output):
    from importlib.metadata import version

    raw = json.dumps(prompts).encode()
    results = {"tokenizers": version("tokenizers"), "fixture_sha256": hashlib.sha256(raw).hexdigest(),
               "rounds": args.rounds, "requests_per_dp": len(prompts), "cases": {}}
    ctx = mp.get_context("spawn")
    for name, layout in plans.items():
        # The isolated tokenizer can use the entire DP mask.
        plan = [{"cpus": dp["cpus"], "workers": len(dp["cpus"])} for dp in layout]
        with ctx.Manager() as manager, ProcessPoolExecutor(max_workers=len(plan), mp_context=ctx) as pool:
            gate = manager.Barrier(len(plan))
            work = [pool.submit(measure_dp, args.model, prompts, dp["cpus"], dp["workers"], args.rounds, gate)
                    for dp in plan]
            replicas = [task.result() for task in work]
        requests = len(prompts) * len(plan)
        batches = [[span for dp in replicas for span in dp[i]] for i in range(args.rounds)]
        round_us = [elapsed(batch) / (1000 * requests) for batch in batches]
        result = {"plan": plan, "amortized_us_per_request": sum(round_us) / len(round_us),
                  "round_us_per_request": round_us,
                  "mean_encode_call_us": sum(end - start for batch in batches for start, end in batch)
                  / (1000 * requests * args.rounds)}
        results["cases"][name] = result
    (output / "tokenizer-results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["| 物理核数 | DP 数 | 每 DP 分词线程 | 1.0.0rc5（µs/请求） |",
             "|---:|---:|---:|---:|"]
    for name, data in results["cases"].items():
        plan = data["plan"]
        lines.append(f"| {name} | {len(plan)} | {plan[0]['workers']} | {data['amortized_us_per_request']:,.2f} |")
    table = "\n".join(lines) + "\n"
    (output / "tokenizer-results.md").write_text(table)
    print(table, end="")


def main():
    freeze_environment()
    from cpu_topology import cpu_topology
    from host_capacity import capacity_plan, reserve_output_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--core-sweep", nargs="+", type=int, default=(1, 48, 96))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    if args.rounds < 1 or any(cores != 1 and (cores < 8 or cores % 8) for cores in args.core_sweep):
        parser.error("Use positive rounds and core counts of 1 or a positive multiple of 8")
    topology = cpu_topology()
    try:
        plans = {}
        for cores in args.core_sweep:
            replicas = 1 if cores == 1 else 8
            plan = capacity_plan(topology, replicas, cores // replicas)
            plans[str(cores)] = plan["replicas"]
        output = reserve_output_dir(args.out)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    run(args, prepare(args.model), plans, output)


if __name__ == "__main__":
    main()
