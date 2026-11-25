"""Measure the two business IPC boundaries using up to eight real process pairs."""

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns, process_time_ns

BOUNDARIES = {
    "frontend_ipc": "前端→Core，32K 输入消息与 ACK",
    "worker_ipc": "Core↔worker，首个 Prefill 调度步与固定返回",
}


def queues_close(*queues):
    for queue in queues:
        if queue is not None:
            queue.local_socket.context.destroy(linger=0)


def peer(cpus, endpoint, token, handle, control):
    os.sched_setaffinity(0, cpus)
    import zmq
    from engine import dummy_result
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder

    decoder = MsgpackDecoder(EngineCoreRequest)
    context = zmq.Context()
    sock = context.socket(zmq.PAIR)
    sock.setsockopt(zmq.RCVTIMEO, 30000)
    sock.setsockopt(zmq.SNDTIMEO, 30000)
    sock.connect(endpoint)
    rx = MessageQueue.create_from_handle(handle, 0)
    reply = MessageQueue(1, 1, max_chunk_bytes=1 << 20, max_chunks=2)
    try:
        control.send(reply.export_handle())
        rx.wait_until_ready()
        reply.wait_until_ready()
        while (mode := control.recv()) is not None:
            n = 0
            cpu = process_time_ns()
            if mode == "frontend_ipc":
                while (frames := sock.recv_multipart()) != [b""]:
                    assert len(decoder.decode(frames).prompt_token_ids) == 32768
                    sock.send(b"ok")
                    n += 1
            else:
                while (item := rx.dequeue(timeout=30)) is not None:
                    _, _, sample = item
                    reply.enqueue(dummy_result(sample, token), timeout=30)
                    n += 1
            control.send((n, process_time_ns() - cpu))
    finally:
        sock.close(linger=0)
        context.term()
        queues_close(rx, reply)


def pair(cpus, args, gate, deadline, directory):
    os.setsid()
    os.sched_setaffinity(0, cpus)
    import zmq
    from stage_cases import Cases
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    cases = Cases(["worker_prefill"], 1)
    cases.initialize_lane(["worker_prefill"])
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    context = zmq.Context()
    sock = context.socket(zmq.PAIR)
    sock.setsockopt(zmq.RCVTIMEO, 30000)
    sock.setsockopt(zmq.SNDTIMEO, 30000)
    # Short path works even when the result directory is under a long mount path.
    with TemporaryDirectory(prefix="vh-ipc-") as tmp:
        endpoint = f"ipc://{tmp}/front"
        sock.bind(endpoint)
        tx = MessageQueue(1, 1, max_chunk_bytes=4 << 20, max_chunks=2)
        proc = ctx.Process(
            target=peer, args=(cpus, endpoint, cases.token, tx.export_handle(), child)
        )
        reply = None
        proc.start()
        child.close()
        try:
            if not parent.poll(120):
                raise RuntimeError("IPC peer did not start")
            reply = MessageQueue.create_from_handle(parent.recv(), 0)
            tx.wait_until_ready()
            reply.wait_until_ready()
            sample = {rid: False for rid in cases.first_chunk.num_scheduled_tokens}
            payload = ("prefill", cases.first_chunk, sample)
            result = {}
            for mode in args.stages:
                result[mode] = []
                for iteration in range(-1, args.rounds):
                    if gate.wait() == 0:
                        deadline.value = perf_counter_ns() + int(args.seconds * 1e9)
                    gate.wait()
                    parent.send(mode)
                    start = perf_counter_ns()
                    cpu = process_time_ns()
                    n = 0
                    while perf_counter_ns() < deadline.value:
                        if mode == "frontend_ipc":
                            sock.send_multipart(
                                cases.encoder.encode(cases.requests[n % 32])
                            )
                            assert sock.recv() == b"ok"
                        else:
                            tx.enqueue(payload, timeout=30)
                            assert reply.dequeue(timeout=30).req_ids == list(sample)
                        n += 1
                    end = perf_counter_ns()
                    cpu = process_time_ns() - cpu
                    if mode == "frontend_ipc":
                        sock.send(b"")
                    else:
                        tx.enqueue(None, timeout=30)
                    if not parent.poll(30):
                        raise RuntimeError("IPC peer did not finish")
                    received, peer_cpu = parent.recv()
                    assert received == n and n
                    if iteration >= 0:
                        result[mode].append((start, end, n, cpu + peer_cpu))
            parent.send(None)
            proc.join(30)
            if proc.exitcode != 0:
                raise RuntimeError(f"IPC peer exit code {proc.exitcode}")
            result["fixture"] = dict(
                token_ids_sha256=cases.ids_hash.hexdigest(),
                image_sha256=hashlib.sha256(cases.raw).hexdigest(),
            )
            (directory / f"{cpus[0]}.json").write_text(json.dumps(result))
        except BaseException:
            gate.abort()
            raise
        finally:
            if proc.is_alive():
                proc.terminate()
            proc.join()
            sock.close(linger=0)
            context.term()
            queues_close(tx, reply)
            parent.close()


def measure_pairs(layout, args, out):
    ctx = mp.get_context("spawn")
    gate = ctx.Barrier(len(layout), timeout=180)
    deadline = ctx.RawValue("q", 0)
    with TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        processes = []
        try:
            for group in layout:
                p = ctx.Process(
                    target=pair, args=(group, args, gate, deadline, directory)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                if p.exitcode:
                    raise RuntimeError(f"IPC pair exited {p.exitcode}")
            lanes = [
                json.loads((directory / f"{cpus[0]}.json").read_text())
                for cpus in layout
            ]
        finally:
            for p in processes:
                if p.is_alive():
                    try:
                        os.killpg(p.pid, 15)
                    except ProcessLookupError:
                        p.terminate()
            for p in processes:
                p.join()
    result = {}
    for mode in args.stages:
        rounds = []
        for samples in zip(*(s[mode] for s in lanes), strict=True):
            seconds = (max(s[1] for s in samples) - min(s[0] for s in samples)) / 1e9
            n = sum(s[2] for s in samples)
            rounds.append(
                dict(
                    ops_s=n / seconds,
                    effective_cores=sum(s[3] for s in samples) / 1e9 / seconds,
                    operations=n,
                    wall_s=seconds,
                )
            )
        result[mode] = dict(
            rounds=rounds,
            ops_s=median(r["ops_s"] for r in rounds),
            effective_cores=median(r["effective_cores"] for r in rounds),
        )
    assert all(s["fixture"] == lanes[0]["fixture"] for s in lanes)
    return dict(cpus_by_pair=layout, fixture=lanes[0]["fixture"], stages=result)
