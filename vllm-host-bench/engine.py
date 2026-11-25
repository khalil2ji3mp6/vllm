"""vLLM scheduler, prefix cache/KV block management, and output-state update."""
from contextlib import contextmanager

import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


class HostGlmMetadataModel(torch.nn.Module):
    """Declare the text-only driver interface without importing model kernels."""

    def __init__(self, vllm_config, prefix=""):
        raise RuntimeError("Model must never be instantiated by the host benchmark")

    def embed_input_ids(self, input_ids):
        raise RuntimeError("Embedding is outside the host benchmark")

    def forward(self, input_ids, positions):
        raise RuntimeError("Forward is outside the host benchmark")

    def compute_logits(self, hidden_states):
        raise RuntimeError("Logits are outside the host benchmark")


def config(args, *, skip_tokenizer_init):
    ModelRegistry.register_model("GlmMoeDsaForCausalLM", HostGlmMetadataModel)
    model = ModelConfig(
        model=str(args.model_metadata), max_model_len=args.max_model_len,
        trust_remote_code=False, skip_tokenizer_init=skip_tokenizer_init,
        dtype="bfloat16", generation_config="vllm",
    )
    cache = CacheConfig(
        block_size=args.block_size, enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )
    scheduler = SchedulerConfig(
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.chunk_tokens, enable_chunked_prefill=True,
        async_scheduling=False, watermark=0.0, policy="fcfs",
        is_encoder_decoder=False,
    )
    cfg = VllmConfig(
        model_config=model, cache_config=cache, scheduler_config=scheduler,
        parallel_config=ParallelConfig(tensor_parallel_size=args.tp),
    )
    # CpuPlatform adjusts these for its MLA kernel. No kernel is used here,
    # so restore the benchmark's Ascend-recipe scheduling and logical geometry.
    cfg.cache_config.block_size = args.block_size
    cfg.cache_config.enable_prefix_caching = True
    cfg.cache_config.num_gpu_blocks = args.blocks
    cfg.scheduler_config.enable_chunked_prefill = True
    cfg.scheduler_config.max_num_batched_tokens = args.chunk_tokens
    cfg.scheduler_config.max_num_scheduled_tokens = args.chunk_tokens
    return cfg


def build_scheduler(cfg, args):
    spec = MLAAttentionSpec(
        block_size=args.block_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16,
    )
    logical_kv = KVCacheConfig(
        num_blocks=args.blocks, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(
            [f"model.layers.{i}.self_attn" for i in range(cfg.model_config.hf_config.num_hidden_layers)],
            spec,
        )],
    )
    register_all_kvcache_specs(cfg)
    sched = scheduler_module.Scheduler(
        vllm_config=cfg, kv_cache_config=logical_kv, block_size=args.block_size,
        log_stats=False, structured_output_manager=StructuredOutputManager(cfg),
    )
    init_none_hash(sha256)
    return sched, get_request_block_hasher(args.block_size, sha256)


def add(sched, req, hasher):
    req = Request.from_engine_core_request(req, hasher)
    sched.add_request(req)
    return req


def classify(sched, output):
    prefill_tokens = 0
    decode_tokens = 0
    ready_to_sample = {}
    for rid, n in output.num_scheduled_tokens.items():
        req = sched.requests[rid]
        end = req.num_computed_tokens
        prompt_remaining = max(0, req.num_prompt_tokens - (end - n))
        prefill = min(n, prompt_remaining)
        prefill_tokens += prefill
        decode_tokens += n - prefill
        ready_to_sample[rid] = end >= req.num_prompt_tokens
    if not prefill_tokens:
        phase = "decode" if decode_tokens else "idle"
    else:
        phase = "mixed" if decode_tokens else "prefill"
    return phase, prefill_tokens, decode_tokens, ready_to_sample


def dummy_result(ready_to_sample, token_id):
    ids = list(ready_to_sample)
    return ModelRunnerOutput(
        req_ids=ids, req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[[token_id] if ready_to_sample[rid] else [] for rid in ids],
    )


def seed_prefixes(sched, requests, hasher, hit, token_id):
    for req in requests:
        seed = type(req)(
            request_id="warm-" + req.request_id,
            prompt_token_ids=req.prompt_token_ids[:hit],
            mm_features=None, sampling_params=req.sampling_params.clone(),
            pooling_params=None, arrival_time=req.arrival_time, lora_request=None,
            cache_salt=None, data_parallel_rank=None,
        )
        seed.sampling_params.max_tokens = 1
        add(sched, seed, hasher)
    while sched.has_requests():
        out = sched.schedule()
        if out.total_num_scheduled_tokens == 0:
            break
        sched.update_from_output(out, dummy_result(classify(sched, out)[3], token_id))


@contextmanager
def time_scheduler_children(sched, clock):
    cache = sched.kv_cache_manager
    methods = (
        (cache, "get_computed_blocks", "prefix_lookup"),
        (cache, "allocate_slots", "kv_allocate_or_extend"),
        (cache, "free", "kv_free"),
        (scheduler_module, "check_stop", "check_stop"),
    )
    originals = []
    active = {"phase": "idle"}

    def timed(fn, label):
        def called(*args, **kwargs):
            phase = "pending" if label in ("prefix_lookup", "kv_allocate_or_extend") else active["phase"]
            with clock.span(phase, label):
                return fn(*args, **kwargs)
        return called

    try:
        for obj, key, label in methods:
            fn = getattr(obj, key)
            originals.append((obj, key, fn))
            setattr(obj, key, timed(fn, label))
        yield active
    finally:
        for obj, key, fn in originals:
            setattr(obj, key, fn)
