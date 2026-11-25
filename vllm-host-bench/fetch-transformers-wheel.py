"""Fetch and verify the exact upstream wheel used for the local RC5 build."""
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen


output = Path(sys.argv[1])
with urlopen("https://pypi.org/pypi/transformers/5.14.1/json", timeout=30) as response:
    release = json.load(response)
artifact = next(item for item in release["urls"] if item["filename"] == output.name)
with urlopen(artifact["url"], timeout=60) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != artifact["digests"]["sha256"]:
    raise RuntimeError("Transformers wheel checksum mismatch")
output.write_bytes(data)
