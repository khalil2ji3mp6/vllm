"""Shared host-only business stages, in request order."""

TEXT = "请求与文本前处理"
IMAGE = "图片接入与处理"
IPC = "进程通信与入队"
ENGINE = "引擎调度与请求状态"
WORKER = "Worker 输入准备"
OUTPUT = "输出处理与响应"

# Display only. JSON counters and interval lookups use the stable group names.
SERVICE_LABELS = {
    TEXT: "跨请求多核，模板与分词在线程池，校验与建状态在 API 事件循环",
    IMAGE: "跨图片多核，媒体线程池",
    IPC: "跨 DP 并行，各连接独立收发",
    ENGINE: "跨 DP 并行，每个 Core 同步调度",
    WORKER: "跨 DP 并行，每个 worker 步内同步",
    OUTPUT: "跨 API 进程并行，每个事件循环同步处理",
}
MICRO_LAYOUTS = {
    "plans": "跨任务多核，每物理核一个独立进程",
    "thread_plans": "跨任务线程并发，共享进程内 GIL",
    "ipc_plans": "跨 DP 收发对并行",
}

MICRO = {
    "text_request": (TEXT, "32K 请求"),
    "image_request": (IMAGE, "1024×1024 图"),
    "frontend_ipc": (IPC + "（前端↔Core）", "32K 请求往返"),
    "worker_ipc": (IPC + "（Core↔worker）", "Prefill 步往返"),
    "drive_scheduler": (ENGINE, "32K 输入 / 1K 输出请求"),
    "worker_prefill": (WORKER + "（Prefill）", "16K-token 步"),
    "worker_decode": (WORKER + "（Decode）", "单序列步"),
    "stream": (OUTPUT, "1K-token 请求"),
}

TARGETS = {
    TEXT: [
        ("frontend", s)
        for s in (
            "request_validate",
            "chat_template",
            "tokenize",
            "input_validate_and_enqueue",
        )
    ],
    IMAGE: [("frontend", "image_base64_decode"), ("frontend", "image_decode_rgb")],
    IPC: [
        ("frontend", "front_engine_encode_send"),
        ("engine", "engine_request_deserialize"),
        ("engine", "engine_add_request_and_hash"),
        ("engine", "engine_worker_enqueue"),
        ("worker0", "worker_result_enqueue"),
        ("engine", "engine_worker_unpack"),
        ("engine", "engine_front_output_pack"),
        ("frontend", "front_output_ipc_decode"),
    ],
    ENGINE: [
        ("engine", s)
        for s in ("schedule", "update_request_and_output", "kv_free", "check_stop")
    ],
    WORKER: [("worker", "worker_prepare_cpu")],
    OUTPUT: [
        ("frontend", "detokenize_and_stop_strings"),
        ("frontend", "serialize_stream_chunks"),
    ],
}


def groups(phase):
    return [g for g in TARGETS if phase == "Prefill" or g not in (TEXT, IMAGE)]
