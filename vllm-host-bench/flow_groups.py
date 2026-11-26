"""Shared host-only business stages, in request order."""

TEXT = "请求与文本前处理"
IMAGE = "图片接入与处理"
IPC = "进程通信与入队"
ENGINE = "引擎调度与请求状态"
WORKER = "Worker 输入准备"
OUTPUT = "输出处理与响应"
TOTAL = "阶段整体"

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
    TOTAL: [("frontend", "request_phase_lifetime")],
}


def groups(phase):
    return [g for g in TARGETS if phase == "Prefill" or g not in (TEXT, IMAGE)]
