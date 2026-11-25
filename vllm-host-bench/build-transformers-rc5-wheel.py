"""Repack the pinned Transformers wheel for this GLM-only host benchmark."""
import base64
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def replace_once(data, before, after):
    if data.count(before) != 1:
        raise ValueError(f"Unexpected upstream Transformers content: {before!r}")
    return data.replace(before, after, 1)


def repack(source, output_dir, serial=False):
    upstream = "transformers-5.14.1"
    suffix = "+host.rc5.serial1" if serial else "+host.rc5"
    local = upstream + suffix
    dependency = b"tokenizers==1.0.0rc5" + (b"+host.serial1" if serial else b"")
    if source.name != f"{upstream}-py3-none-any.whl":
        raise ValueError("Expected the pinned upstream Transformers 5.14.1 wheel")
    output = output_dir / f"{local}-py3-none-any.whl"
    record = local + ".dist-info/RECORD"
    records = []

    with ZipFile(source) as original, ZipFile(output, "w", ZIP_DEFLATED) as wheel:
        for path in original.namelist():
            if path.endswith("/RECORD") or path.endswith("/"):
                continue
            data = original.read(path)
            path = path.replace(upstream + ".dist-info/", local + ".dist-info/", 1)
            if path.endswith(".dist-info/METADATA"):
                data = replace_once(data, b"Version: 5.14.1\n", (f"Version: 5.14.1{suffix}\n").encode())
                data = replace_once(data, b"Requires-Dist: tokenizers<=0.23.0,>=0.22.0\n",
                                    b"Requires-Dist: " + dependency + b"\n")
            elif path == "transformers/__init__.py":
                data = replace_once(data, b'__version__ = "5.14.1"', f'__version__ = "5.14.1{suffix}"'.encode())
            elif path == "transformers/dependency_versions_table.py":
                data = replace_once(data, b"tokenizers>=0.22.0,<=0.23.0", dependency)
            elif path == "transformers/tokenization_utils_tokenizers.py":
                trainers_import = b"from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer\n"
                trainer_map = (b'MODEL_TO_TRAINER_MAPPING = {\n    "BPE": BpeTrainer,\n'
                               b'    "Unigram": UnigramTrainer,\n    "WordLevel": WordLevelTrainer,\n'
                               b'    "WordPiece": WordPieceTrainer,\n}\n')
                data = replace_once(data, trainers_import, b"")
                data = replace_once(data, trainer_map, b"")
                lookup = b'        trainer_class = MODEL_TO_TRAINER_MAPPING[tokenizer_json["model"]["type"]]\n'
                data = replace_once(
                    data, lookup, b"        " + trainers_import
                    + b"\n".join(b"        " + line for line in trainer_map.splitlines())
                    + b"\n" + lookup,
                )
            checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            records.append((path, f"sha256={checksum}", len(data)))
            wheel.writestr(path, data)
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows([*records, (record, "", "")])
        wheel.writestr(record, stream.getvalue())
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()
    print(repack(args.wheel, args.out, args.serial))
