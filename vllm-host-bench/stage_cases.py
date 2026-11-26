"""Inputs and isolated CPU operations for the GLM host benchmark."""

import hashlib
from array import array
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import pybase64
from engine import add, build_scheduler, classify, config, dummy_result
from frontend import prompt_factory, serialize_chunks
from image_codec import ImageCodec
from PIL import Image
from tokenizer_rc5 import GlmRc5Tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.inputs import tokens_input
from vllm.multimodal.media.image import ImageMediaIO
from vllm.renderers.hf import HfRenderer, safe_apply_chat_template
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from worker import CpuBatch


def sampling():
    return SamplingParams(
        max_tokens=1024,
        ignore_eos=True,
        stop=["<never-stop>"],
        output_kind=RequestOutputKind.DELTA,
        temperature=0.0,
    )


class Cases:
    """Prepared once per NUMA domain. Each lane owns its mutable state after fork."""

    def __init__(self, stages, render_threads):
        root = Path(__file__).resolve().parent
        self.args = SimpleNamespace(
            model_metadata=root / "model",
            max_model_len=135000,
            renderer_num_workers=render_threads,
            block_size=128,
            max_num_seqs=32,
            chunk_tokens=16384,
            tp=1,
            blocks=8192,
        )
        self.cfg = config(self.args, skip_tokenizer_init=False)
        self.tokenizer = GlmRc5Tokenizer(self.args.model_metadata)
        self.renderer = HfRenderer(self.cfg, self.tokenizer)
        self.input = InputProcessor(self.cfg, renderer=self.renderer)
        self.messages = prompt_factory(self.tokenizer, range(32), 32768)
        self.texts = [self.template(i) for i in range(32)]
        self.prompts = [
            tokens_input(self.tokenize(i), prompt=text)
            for i, text in enumerate(self.texts)
        ]
        self.backend = self.tokenizer._tokenizer
        if "ids_export" in stages:
            self.encodings = [
                self.backend.encode(t, add_special_tokens=False) for t in self.texts
            ]
            assert all(
                e.ids == p["prompt_token_ids"]
                for e, p in zip(self.encodings, self.prompts, strict=True)
            )
        self.payloads = [
            msgspec.json.encode(
                dict(model="glm", messages=m, max_tokens=1024, stream=True)
            )
            for m in self.messages
        ]
        self.token = self.tokenizer.encode(" hello", add_special_tokens=False)[0]
        if {"worker_prefill", "worker_decode", "drive_scheduler"}.intersection(stages):
            self.engine_cfg = config(self.args, skip_tokenizer_init=True)
        self.image_path = root / "assets/input.jpg"
        self.raw = self.image_path.read_bytes()
        self.b64 = pybase64.b64encode(self.raw).decode("ascii")
        self.media = ImageMediaIO()
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreRequest)
        self.requests = [self.request(i) for i in range(32)]
        self.ids_hash = hashlib.sha256()
        for prompt in self.prompts:
            ids = prompt["prompt_token_ids"]
            assert len(ids) == 32768
            self.ids_hash.update(array("I", ids).tobytes())

    def text_request(self, i):
        messages = self.messages[i % 32]
        payload = msgspec.json.encode(
            dict(model="glm", messages=messages, max_tokens=1024, stream=True)
        )
        request = ChatCompletionRequest.model_validate_json(payload)
        assert request.max_tokens == 1024
        params = sampling()
        text = safe_apply_chat_template(
            self.cfg.model_config,
            self.tokenizer,
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) == 32768
        req = self.input.process_inputs(
            str(i), tokens_input(ids, prompt=text), params, ("generate",)
        )
        req.external_req_id = req.request_id
        OutputProcessor(self.tokenizer, log_stats=False).add_request(req, text)

    def image_request(self, i):
        with self.media.load_bytes(self.base64(i)).media as image:
            assert image.size == (1024, 1024) and image.mode == "RGB"

    def schema(self, i):
        request = ChatCompletionRequest.model_validate_json(self.payloads[i % 32])
        assert request.max_tokens == 1024
        return sampling()

    def template(self, i):
        return safe_apply_chat_template(
            self.cfg.model_config,
            self.tokenizer,
            self.messages[i % 32],
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize(self, i):
        return self.tokenizer.encode(self.texts[i % 32], add_special_tokens=False)

    def tokenize_native(self, i):
        return self.backend.encode(self.texts[i % 32], add_special_tokens=False)

    def ids_export(self, i):
        return self.encodings[i % 32].ids

    def base64(self, _):
        return pybase64.b64decode(self.b64, validate=True)

    def jpeg(self, _):
        with self.media.load_bytes(self.raw).media as image:
            assert image.size == (1024, 1024) and image.mode == "RGB"

    def jpeg_pixels(self, _):
        with Image.open(BytesIO(self.raw)) as image:
            image.load()

    def rgb_copy(self, _):
        with self.rgb.copy():
            pass

    def encode_image(self, _):
        return self.codec._encode_one(0)

    def request(self, i):
        req = self.input.process_inputs(
            str(i), self.prompts[i % 32], sampling(), ("generate",)
        )
        req.external_req_id = req.request_id
        return req

    def admit(self, i):
        req = self.request(i)
        # No retained states across independent admissions.
        OutputProcessor(self.tokenizer, log_stats=False).add_request(
            req, self.texts[i % 32]
        )

    def msgpack(self, i):
        req = self.decoder.decode(self.encoder.encode(self.requests[i % 32]))
        assert len(req.prompt_token_ids) == 32768

    def drive_scheduler(self, i, capture=False):
        req = self.requests[i % 32]
        req.cache_salt = f"stage-{self.serial}"
        self.serial += 1
        add(self.sched, req, self.hasher)
        emitted = prompt_tokens = 0
        while self.sched.has_requests():
            output = self.sched.schedule()
            phase, prompt, _, sample = classify(self.sched, output)
            if not output.total_num_scheduled_tokens:
                assert not self.sched.get_num_unfinished_requests()
                continue
            if capture:
                self.batch.update(output)
                if self.first_chunk is None:
                    self.first_chunk = output
                if phase == "decode" and self.decode_step is None:
                    # Keep both the step and the exact preceding worker state.
                    import copy

                    self.decode_step = output
                    self.decode_batch = copy.deepcopy(self.batch)
            groups = self.sched.update_from_output(
                output, dummy_result(sample, self.token)
            )
            prompt_tokens += prompt
            emitted += sum(
                len(o.new_token_ids) for g in groups.values() for o in g.outputs
            )
        assert prompt_tokens == 32768 and emitted == 1024

    def worker_prefill(self, _):
        self.batch.update(self.first_chunk)
        self.batch.prepare(self.first_chunk)

    def worker_decode(self, _):
        self.decode_batch.prepare(self.decode_step)

    def stream(self, i):
        processor = OutputProcessor(self.tokenizer, log_stats=False)
        req = self.requests[i % 32]
        processor.add_request(req, self.texts[i % 32])
        tokens = 0
        for n in range(1024):
            event = EngineCoreOutput(
                request_id=req.request_id,
                new_token_ids=[self.token],
                finish_reason=FinishReason.LENGTH if n == 1023 else None,
            )
            results = processor.process_outputs([event], 0.0)
            emitted, _ = serialize_chunks(results.request_outputs)
            tokens += emitted
        assert tokens == 1024 and not processor.request_states

    def copy64(self, _):
        np.copyto(self.dst, self.src)

    def initialize_lane(self, stages):
        if "copy64" in stages:
            # Private, first-touched on the pinned CPU. 32 MiB read + 32 MiB written.
            self.src = np.full(32 << 20, 113, np.uint8)
            self.dst = np.zeros_like(self.src)
            self.dst.fill(1)
        if "rgb_copy" in stages:
            self.rgb = self.media.load_bytes(self.raw).media
        if "encode_image" in stages:
            self.codec = ImageCodec(self.image_path)
        if {"worker_prefill", "worker_decode", "drive_scheduler"}.intersection(stages):
            self.sched, self.hasher = build_scheduler(self.engine_cfg, self.args)
            self.serial = 0
            self.batch = CpuBatch(self.args.block_size, self.token)
            self.first_chunk = self.decode_step = None
            self.drive_scheduler(0, capture=True)
