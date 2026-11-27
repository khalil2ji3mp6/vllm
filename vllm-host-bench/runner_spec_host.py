"""Isolated CPU slices of one synchronous verifier/drafter step (B=1, K=3)."""
from copy import copy
from dataclasses import replace
from functools import partial
from itertools import islice
from types import SimpleNamespace

import numpy as np
import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN, RejectionSampler
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class SpecHostCases:
    def __init__(self, cfg):
        self.vocab = cfg.model_config.get_vocab_size()
        self.ids = torch.tensor([7, 11, 12, 13], dtype=torch.int32)
        self.positions = torch.arange(32768, 32772)
        self.model = torch.nn.Module()  # callable for argument binding only
        hf = cfg.model_config.hf_config
        self.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(hf.num_hidden_layers))
        for layer in self.layers:
            layer.use_sequence_parallel_moe = False
        self.aux_hidden_state_layers = ()
        self.hidden_states = torch.empty((4, hf.hidden_size), dtype=torch.bfloat16)
        self.logits = torch.empty((4, self.vocab))
        self.verified = torch.tensor([[11, 17, -1, -1]], dtype=torch.int32)
        self.sampled, _ = self.parse_verified_ids()
        self.spec = SpecDecodeMetadata(
            draft_token_ids=self.ids[1:], num_draft_tokens=[3],
            cu_num_draft_tokens=torch.tensor([3]), cu_num_sampled_tokens=torch.tensor([4]),
            target_logits_indices=torch.arange(3), bonus_logits_indices=torch.tensor([3]),
            logits_indices=torch.arange(4),
        )
        empty = torch.empty(0)
        self.sampling = SamplingMetadata(
            temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
            generators={}, max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
            frequency_penalties=empty, presence_penalties=empty, repetition_penalties=empty,
            output_token_ids=[[]], allowed_token_ids_mask=None, bad_words_token_ids={},
            logitsprocs=LogitsProcessors(), spec_token_ids=[[11, 12, 13]],
        )
        self.query_start = torch.tensor([0, 4], dtype=torch.int32)
        self.seq_lens = torch.tensor([32772], dtype=torch.int32)
        self.arange = np.arange(4, dtype=np.int32)
        self.batch = SimpleNamespace(req_ids=["r0"])
        self.scheduled = {"r0": 4}
        self.proposer_state = SimpleNamespace(input_ids=self.ids)
        graph_cfg = copy(cfg)
        compilation = copy(cfg.compilation_config)
        graph_cfg.compilation_config = compilation
        compilation.mode = CompilationMode.VLLM_COMPILE
        compilation.cudagraph_mode = CUDAGraphMode.PIECEWISE
        compilation.splitting_ops = list(compilation._attention_ops)
        compilation.cudagraph_capture_sizes = [1, 4, 8]
        compilation.max_cudagraph_capture_size = 8
        compilation.compile_sizes = []
        self.dispatcher = CudagraphDispatcher(graph_cfg)
        self.dispatcher.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE, uniform_decode_query_len=4)
        # One accepted draft followed by the target replacement, computed before timing.
        self.draft_input_tokens = len(self.sampled[0])

    def model_forward(self):
        # NPUModelRunner._model_forward, plain text, no Engram, replay or ENPU.
        # Use the eager target's outer layer loop. Frozen tensors retain their
        # shapes across layers. Do not count calls to dummy layer operators.
        model_inputs = dict(input_ids=self.ids, positions=self.positions,
                            intermediate_tensors=None, inputs_embeds=None)
        if getattr(self.model, "prepare_engram_inputs", None) is not None:
            raise ValueError("Expected the plain model entry")
        bound = partial(self.model, **model_inputs)
        hidden = self.hidden_states
        for index, layer in enumerate(islice(self.layers, 0, len(self.layers))):
            if hidden.shape[0] != self.positions.shape[0] and not layer.use_sequence_parallel_moe:
                raise ValueError("Fixture requires TP1 with matching token dimensions")
            if index in self.aux_hidden_state_layers:
                raise ValueError("Fixture has no auxiliary hidden-state outputs")
        return bound

    def compute_logits(self):
        # _get_logits_normal, replicated TP1 head: host view of the LM-head
        # output. No advanced indexing, projection, collective or data copy.
        return self.logits[..., :self.vocab]

    def rej_sample(self):
        # Host arguments for the bonus sampler and host result container from
        # AscendRejectionSampler.forward. Device indices/outputs are fixtures.
        if self.spec.max_spec_len > MAX_SPEC_LEN:
            raise ValueError("Draft length exceeds rejection sampler limit")
        bonus_sampling = replace(self.sampling, max_num_logprobs=-1)
        return bonus_sampling, SamplerOutput(sampled_token_ids=self.verified, logprobs_tensors=None)

    def draft_padding(self):
        # Ascend _propose dispatches locally, then again after DP agreement.
        # An independent DP replica keeps the local count; no collective.
        _, local = self.dispatcher.dispatch(self.draft_input_tokens, uniform_decode=True, has_lora=False)
        return self.dispatcher.dispatch(local.num_tokens, uniform_decode=True, has_lora=False)

    def propose(self):
        # Ascend disable_padded_drafter_batch path. The original next-token
        # helper materializes on self.input_ids.device; here it is CPU.
        next_ids = SpecDecodeBaseProposer.prepare_next_token_ids_cpu(
            self.proposer_state, self.sampled, {}, self.batch, self.scheduled)
        rejected = torch.tensor([
            n + 1 - len(tokens) if n else 0
            for n, tokens in zip(self.spec.num_draft_tokens, self.sampled, strict=True)
        ], dtype=torch.int32)
        old = self.query_start
        lengths = self.seq_lens - rejected
        counts = (old[1:] - old[:-1] - rejected).numpy()
        starts = torch.zeros(old.shape, dtype=torch.int32)
        starts_np = starts.numpy()
        np.cumsum(counts, out=starts_np[1:])
        offsets = self.arange[:starts_np[-1]] - np.repeat(starts_np[:-1], counts)
        indices = torch.from_numpy(offsets + np.repeat(old[:-1].numpy(), counts))
        return next_ids, lengths, starts, indices

    def parse_verified_ids(self):
        # Synchronous sampler bookkeeping, after rejection_sampler.forward.
        return RejectionSampler.parse_output(self.verified, self.vocab, [])

    def operations(self):
        return dict(model_forward=self.model_forward, compute_logits=self.compute_logits,
                    rej_sample=self.rej_sample, draft_padding=self.draft_padding, propose=self.propose)
