# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from pathlib import Path
from io import TextIOWrapper


def write_copyright(f: TextIOWrapper):
    f.write("# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)\n")
    f.write("#\n")
    f.write("# MIT License\n\n")

def get_codecs() -> list[str]:
    filepath = str(Path(args.file).resolve())
    lines = []
    with open(filepath, "r") as file:
        lines = file.readlines()
    codecs: list[str] = []
    for line in lines:
        if 'class="odd"' in line or 'class="even"' in line:
            # get data between <strong> and </strong>
            codec = line.split("<strong>")[1].split("</strong>")[0]
            if len(codec) == 4:
                # cannot handle codec start with number
                if not codec[0].isdigit():
                    codecs.append(codec)
            elif "/" in codec:
                sub_codecs = codec.split("/")
                for sub_codec in sub_codecs:
                    codecs.append(sub_codec.replace(" ", ""))
            elif "THROUGH" in codec:
                sub_codecs = [s.replace(" ", "") for s in codec.split("THROUGH")]
                start = sub_codecs[0][0:3]
                idx = int(sub_codecs[0][3])
                end = int(sub_codecs[1][3])
                for i in range(idx, end + 1):
                    codecs.append(f"{start}{i}")
            elif len(codec) == 9 and " " in codec:
                sub_codecs = codec.split(" ")
                for sub_codec in sub_codecs:
                    codecs.append(sub_codec)
            elif "- " in codec:
                sub_codecs = [s.replace(" ", "") for s in codec.split("- ")]
                start = sub_codecs[0][0:3]
                idx = int(sub_codecs[0][3])
                end = int(sub_codecs[1][3])
                for i in range(idx, end + 1):
                    codecs.append(f"{start}{i}")
            elif len(codec) == 6 and "-" in codec:
                start = codec[0:3]
                idx = int(codec[3])
                end = int(codec[5])
                for i in range(idx, end + 1):
                    new_codec = f"{start}{i}"
                    codecs.append(new_codec)
            elif len(codec) == 9 and "-" in codec:
                start = codec[0:3]
                idx = int(codec[3])
                end = int(codec[8])
                for i in range(idx, end + 1):
                    new_codec = f"{start}{i}"
                    codecs.append(new_codec)
            elif len(codec) < 4:
                pass
            else:
                raise ValueError(f"Unexpected codec: {codec}")

    # add lowercase versions
    codecs.extend([codec.lower() for codec in codecs])

    return sorted(list(set(codecs)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse codecs from a file.")
    parser.add_argument("--file", type=str, required=True, default="data/codecs.html", help="Path to the file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file.")
    args = parser.parse_args()

    codecs = get_codecs()

    with open(args.output, "w") as f:
        write_copyright(f)
        # add big comment saying this is an auto-generated file
        f.write(
            "# =============================================================================\n"
        )
        f.write("# This file is auto-generated by scripts/parse_codec.py\n")
        f.write(
            "# =============================================================================\n\n"
        )
        f.write("from __future__ import annotations\n\n")
        f.write("from enum import Enum\n\n")
        f.write("import cv2\n\n\n")
        f.write("class Fourcc(Enum):\n")
        for codec in codecs:
            f.write(f'    {codec} = cv2.VideoWriter_fourcc(*"{codec}")  # type: ignore[attr-defined]\n')
