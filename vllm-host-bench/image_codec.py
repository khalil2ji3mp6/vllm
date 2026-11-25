"""Measure vLLM image ingress and a separate local encoding reference."""

import hashlib
from collections import defaultdict
from io import BytesIO
from time import perf_counter_ns

import pybase64
from PIL import Image
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.media.image import ImageMediaIO

from measure import Clock


class TimedImageMediaIO(ImageMediaIO):
    def __init__(self):
        super().__init__()
        self.clock = Clock()

    def load_base64(self, media_type, data):
        start = perf_counter_ns()
        with self.clock.span("prefill", "image_base64_decode"):
            image_bytes = pybase64.b64decode(data, validate=True)
        with self.clock.span("prefill", "image_decode_rgb"):
            result = self.load_bytes(image_bytes)
        self.decode_interval = (start, perf_counter_ns())
        return result


class ImageCodec:
    def __init__(self, path):
        data = path.read_bytes()
        with Image.open(BytesIO(data)) as image:
            self.image_format = image.format
            mime = Image.MIME.get(self.image_format)
        if self.image_format not in ("JPEG", "PNG"):
            raise ValueError(f"Expected a JPEG or PNG benchmark image, found {self.image_format}")
        self.image = ImageMediaIO().load_bytes(data).media
        self.data_url = f"data:{mime};base64," + pybase64.b64encode(data).decode("ascii")
        self.connector = MediaConnector()
        self.info = dict(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
            format=self.image_format, width=self.image.width, height=self.image.height,
            images_per_request=1,
        )

    def _encode_one(self, _):
        # Reference path for client/edge encoding, outside the timed Host rounds.
        options = {"quality": 85} if self.image_format == "JPEG" else {}
        clock = Clock()
        with self.image.copy() as image, BytesIO() as buffer:
            with clock.span("prefill", "image_encode"):
                image.save(buffer, self.image_format, **options)
                data = buffer.getvalue()
            with clock.span("prefill", "image_base64_encode"):
                pybase64.b64encode(data).decode("ascii")
        return clock.rows

    def measure_encode(self, count, pool):
        rows = defaultdict(list)
        for sample in pool.map(self._encode_one, range(count)):
            for key, values in sample.items():
                rows[key].extend(values)
        return dict(rows)

    async def decode_request(self, clock):
        loader = TimedImageMediaIO()
        result = await self.connector.load_from_url_async(self.data_url, loader)
        with result.media as image:
            if image.size != self.image.size or image.mode != "RGB":
                raise RuntimeError("vLLM image decode returned unexpected dimensions or mode")
        for key, values in loader.clock.rows.items():
            clock.rows[key].extend(values)
        return loader.decode_interval
