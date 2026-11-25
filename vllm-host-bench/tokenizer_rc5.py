"""GLM-5.2 adapter for the immutable Tokenizers 1.0.0rc5 backend."""
import json
from importlib.metadata import version

from tokenizers import Tokenizer
from transformers import TokenizersBackend
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class GlmRc5Tokenizer(TokenizersBackend):
    def __init__(self, model_dir):
        installed = version("tokenizers")
        if installed not in ("1.0.0rc5", "1.0.0rc5+host.serial1"):
            raise RuntimeError(f"Expected tokenizers 1.0.0rc5, found {installed}")
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json")).compile()
        settings = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
        # The upstream fast constructor configures a mutable v0 Rust tokenizer.
        # Reuse the Transformers special-token metadata and chat template API.
        PreTrainedTokenizerBase.__init__(self, chat_template=template, **settings)
        vocab = self.get_vocab()
        self.max_token_id = max(vocab.values())
        self.max_chars_per_token = max(map(len, vocab))

    def encode(self, text, add_special_tokens=True):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text, *, add_special_tokens, truncation=False, max_length=None):
        # The pinned renderer requests a single string without padding or offsets.
        # RC5 exposes immutable encodings; Transformers v5 expects mutable v0 settings.
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None and len(ids) > max_length:
            if self.truncation_side == "left" and max_length > 0:
                ids = ids[-max_length:]
            else:
                ids = ids[:max_length]
        return {"input_ids": ids}
