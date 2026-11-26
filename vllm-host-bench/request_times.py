"""Request-weighted service spans. The first token completes Prefill."""

from collections import Counter

from flow_groups import ENGINE, IMAGE, IPC, OUTPUT, TEXT, WORKER
from report import merge_intervals


def charge(totals, group, spans, weights):
    """Charge each participating request the wall time of its shared operation."""
    for (start, end, *_), (prefills, decodes) in zip(spans, weights, strict=True):
        if prefills:
            totals["Prefill"][group] += (end - start) * prefills
        if decodes:
            totals["Decode"][group] += (end - start) * decodes


def request_times(engine, front, workers, steps, sends, frontend_times):
    totals = {phase: Counter(frontend_times[phase]) for phase in ("Prefill", "Decode")}
    for group, stages in (
        (TEXT, ("request_validate", "chat_template", "tokenize", "input_validate_and_enqueue")),
        (IMAGE, ("image_base64_decode", "image_decode_rgb")),
        (IPC, ("front_engine_encode_send",)),
    ):
        for stage in stages:
            spans = front.get(("prefill", stage), ())
            if spans:
                totals["Prefill"][group] += sum(end - start for start, end, _ in spans)
    # Ingress decoding and KV hashing are serial per-request calls.
    for stage in ("engine_request_deserialize", "engine_add_request_and_hash"):
        totals["Prefill"][IPC] += sum(end - start for start, end, _ in engine["prefill", stage])
    for phase, weights in steps.items():
        for group, stage in ((ENGINE, "schedule"), (ENGINE, "update_request_and_output"),
                             (IPC, "engine_worker_enqueue"), (IPC, "engine_worker_unpack")):
            charge(totals, group, engine[phase, stage], weights)
        charge(totals, IPC, workers[0][phase, "worker_result_enqueue"], weights)
        # TP ranks prepare the same scheduled batch concurrently on the host.
        batches = zip(*(w[phase, "worker_prepare_cpu"] for w in workers), strict=True)
        spans = []
        for ranks in batches:
            intervals = merge_intervals((start, end) for start, end, _ in ranks)
            spans.append((0, sum(end - start for start, end in intervals)))
        charge(totals, WORKER, spans, weights)
    for phase, weights in sends.items():
        charge(totals, IPC, engine[phase, "engine_front_output_pack"], weights)
    return totals


def frontend_output_times(clock, batches):
    totals = {"Prefill": Counter(), "Decode": Counter()}
    for phase, weights in batches.items():
        for group, stage in ((IPC, "front_output_ipc_decode"),
                             (OUTPUT, "detokenize_and_stop_strings"),
                             (OUTPUT, "serialize_stream_chunks")):
            charge(totals, group, clock.rows[phase, stage], weights)
    return totals
