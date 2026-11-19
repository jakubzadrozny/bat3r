# Author: Jakub Zadrozny (2026). Licensed under BSD-3-Clause.
# Downloads CC0 HDRI environment maps from Poly Haven (https://polyhaven.com).

import argparse
import hashlib
import json
from pathlib import Path

import requests
from tqdm import tqdm


def get_md5(path: Path) -> str:
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "md5")
    return digest.hexdigest()


def get_hdr(s: requests.Session, hdr_filename: str, target_dir: Path) -> None:
    fname, ext = hdr_filename.split(".")
    res = fname[fname.rfind("_") + 1:]
    asset = fname[:fname.rfind("_")]

    files_url = "https://api.polyhaven.com/files/" + asset

    file_js = json.loads(s.get(files_url).content)
    url = file_js["hdri"][res][ext]["url"]
    md5 = file_js["hdri"][res][ext]["md5"]

    target_path = target_dir / hdr_filename
    if target_path.exists() and get_md5(target_path) == md5:
        print(f"Skipping {hdr_filename}")
    else:
        print(f"Downloading {hdr_filename}")
        try:
            r = s.get(url, timeout=15)
            with target_path.open("wb") as f:
                f.write(r.content)
        except requests.exceptions.Timeout:
            print(f"error downloading {hdr_filename}")
            return

        if not (get_md5(target_path) == md5):
            print(f"error downloading {hdr_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CC0 HDRI environment maps from Poly Haven (https://polyhaven.com)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("hdrs"),
        help="Output directory for downloaded HDRI files (default: hdrs).",
    )
    parser.add_argument(
        "--hdr-list",
        type=Path,
        default=Path(__file__).resolve().parent / "all_hdrs.txt",
        help="Path to text file listing HDRI filenames to download.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers.update({"User-Agent": "BAT3R/1.0 (https://github.com/jakubzadrozny/bat3r)"})

    with args.hdr_list.open() as f:
        hdr_filenames = [line.strip() for line in f if line.strip()]

    for hdr_filename in tqdm(hdr_filenames):
        get_hdr(s, hdr_filename, args.out_dir)


if __name__ == "__main__":
    main()
