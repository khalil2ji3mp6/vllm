"""Set source version metadata and keep unused grammar libraries lazy."""
from pathlib import Path
import sys


root = Path(sys.argv[1]) / "vllm"
parsers = root / "parser" / "__init__.py"
text = parsers.read_text(encoding="utf-8")
eager_import = "from vllm.parser.harmony import HarmonyParser\n"
if eager_import in text and "def __getattr__(" not in text:
    text = text.replace(eager_import, "", 1)
    text += '''
def __getattr__(name):
    if name == "HarmonyParser":
        from vllm.parser.harmony import HarmonyParser

        return HarmonyParser
    raise AttributeError(name)
'''
    parsers.write_text(text, encoding="utf-8")

for backend in ("backend_xgrammar.py", "backend_guidance.py"):
    path = root / "v1" / "structured_output" / backend
    text = path.read_text(encoding="utf-8")
    deferred_annotations = "from __future__ import annotations\n"
    if deferred_annotations not in text:
        license_header, body = text.split("\n\n", 1)
        path.write_text(
            license_header + "\n\n" + deferred_annotations + "\n" + body,
            encoding="utf-8",
        )

(root / "_version.py").write_text(
    '__version__ = "0.30.0+host"\n__version_tuple__ = (0, 30, 0, "host")\n',
    encoding="utf-8",
)
