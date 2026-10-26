#!/usr/bin/env python3
"""Fetch pinned, free SHFE annual futures archives into the ignored local cache.

Standard library only: Excel/parser dependencies are not needed to acquire data.
No account, agreement click or payment is involved. The raw exchange files are
for local research and must not be included in public evidence or Git commits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

CATALOGUE_URL = "https://www.shfe.com.cn/reports/tradedata/datadownload/download.json"
DISCOVERY_CATALOGUE_SHA256 = "42bded46596f52331ce2d40a51c64bb36e615c11af4d8f89ac90112c60efbd55"
SOURCE_PAGE = "https://www.shfe.com.cn/reports/tradedata/datadownload/"
PINNED_ARCHIVES = {
    2022: {
        "url": (
            "https://www.shfe.com.cn/reports/tradedata/datadownload/"
            "historicaldata/futures/202401/P020240119587093379978.zip"
        ),
        "bytes": 18_790_402,
        "sha256": "38be8b5ff728a6aa34aa1e25a961857cac319a194db8f9c161d3e1af10b5399c",
    },
    2023: {
        "url": (
            "https://www.shfe.com.cn/reports/tradedata/datadownload/"
            "historicaldata/futures/202401/P020240119587353273027.zip"
        ),
        "bytes": 24_187_534,
        "sha256": "605e119b8f6e7cd73b02fa18aa75ce1b388d5628aa024d7f5b7961d028a9ec2c",
    },
    2024: {
        "url": (
            "https://www.shfe.com.cn/reports/tradedata/datadownload/"
            "historicaldata/futures/202501/P020250103380580546440.zip"
        ),
        "bytes": 35_087_891,
        "sha256": "68505aefe460ffa3167bd200477a5ebb05a8aaac5e477bb013e97ffbcf7bc99e",
    },
    2025: {
        "url": (
            "https://www.shfe.com.cn/reports/tradedata/datadownload/"
            "historicaldata/futures/202601/P020260317382428557825.zip"
        ),
        "bytes": 76_397_695,
        "sha256": "fe0122e1ff0e072fc200ce924c488407a0305d3e5790692a19efc999709d4902",
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _request(url: str, *, start: int = 0):
    headers = {"User-Agent": "Mozilla/5.0", "Referer": SOURCE_PAGE}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urlopen(Request(url, headers=headers), timeout=45)


def _check_cache(cache: Path, repository: Path) -> Path:
    cache = cache.resolve()
    allowed = (repository / "data/local").resolve()
    if not cache.is_relative_to(allowed) or cache == allowed:
        raise ValueError("raw cache must be a subdirectory of this checkout's data/local/")
    relative = (cache / "ignore-probe").relative_to(repository)
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(relative)], cwd=repository, check=False
    )
    if result.returncode != 0:
        raise ValueError("raw cache is not ignored by Git; refuse to download")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _safe_target(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"refuse symlink in raw cache: {path.name}")


def _download_archive(year: int, cache: Path) -> dict:
    pin = PINNED_ARCHIVES[year]
    path = cache / f"shfe-futures-{year}.zip"
    part = path.with_suffix(".zip.part")
    part_meta = path.with_suffix(".zip.part.json")
    for target in (path, part, part_meta):
        _safe_target(target)
    downloaded = False
    response_headers = {}
    if not path.exists():
        expected_metadata = {"year": year, **pin}
        if part.exists():
            if not part_meta.exists() or json.loads(part_meta.read_text()) != expected_metadata:
                raise ValueError(
                    f"unverified partial archive for {year}; inspect it before retrying"
                )
        else:
            part_meta.write_text(json.dumps(expected_metadata, indent=2) + "\n")
        start = part.stat().st_size if part.exists() else 0
        if start > pin["bytes"]:
            raise ValueError(f"partial archive for {year} is larger than the pinned file")
        if start < pin["bytes"]:
            with _request(pin["url"], start=start) as response:
                status = response.status
                if start:
                    expected_range = rf"bytes {start}-{pin['bytes'] - 1}/{pin['bytes']}"
                    if status != 206 or not re.fullmatch(
                        expected_range, response.headers.get("Content-Range", "")
                    ):
                        raise ValueError("server did not honor the exact pinned resume range")
                elif status != 200:
                    raise ValueError(f"unexpected archive response status {status}")
                response_headers = {
                    name: response.headers[name]
                    for name in ("Content-Type", "Content-Length", "Last-Modified", "ETag")
                    if response.headers.get(name)
                }
                count = start
                with part.open("ab" if start else "wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > pin["bytes"]:
                            raise ValueError("archive response exceeded its pinned size")
                        stream.write(chunk)
        if not part.exists() or part.stat().st_size != pin["bytes"] or _hash(part) != pin["sha256"]:
            raise ValueError(f"archive {year} failed pinned size/SHA-256 verification")
        part.replace(path)
        part_meta.unlink(missing_ok=True)
        downloaded = True
    if path.stat().st_size != pin["bytes"] or _hash(path) != pin["sha256"]:
        raise ValueError(f"cached archive {year} failed pinned size/SHA-256 verification")
    return {
        "year": year,
        **pin,
        "path": str(path),
        "verified_utc": _utc(),
        "downloaded_this_run": downloaded,
        "http_headers": response_headers,
    }


def acquire_archives(years: list[int], cache: Path, *, repository: Path) -> dict:
    """Acquire or verify pinned archives; repeated calls do not redownload them."""
    if not years or len(set(years)) != len(years) or any(y not in PINNED_ARCHIVES for y in years):
        raise ValueError("years must be unique values among 2022, 2023, 2024 and 2025")
    cache = _check_cache(cache, repository)
    catalogue = cache / "download_catalogue.json"
    manifest_path = cache / "acquisition_manifest.json"
    _safe_target(catalogue)
    _safe_target(manifest_path)
    if not catalogue.exists():
        with _request(CATALOGUE_URL) as response:
            raw = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(raw) > 1024 * 1024:
                raise ValueError("invalid or oversized SHFE download catalogue")
        json.loads(raw)
        catalogue.write_bytes(raw)
    catalogue_hash = _hash(catalogue)
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous = {entry["year"]: entry for entry in prior.get("archives", [])}
    manifest = {
        "catalogue_url": CATALOGUE_URL,
        "catalogue_sha256": catalogue_hash,
        "discovery_catalogue_sha256": DISCOVERY_CATALOGUE_SHA256,
        "catalogue_matches_discovery": catalogue_hash == DISCOVERY_CATALOGUE_SHA256,
        "source_page": SOURCE_PAGE,
        "raw_data_policy": "private noncommercial research cache; do not commit raw data",
        "retrieved_utc": prior.get("retrieved_utc", _utc()),
        "archives": list(previous.values()),
    }
    for index, year in enumerate(years):
        record = _download_archive(year, cache)
        old = previous.get(year, {})
        if old.get("sha256") == record["sha256"]:
            if "retrieved_utc" in old:
                record["retrieved_utc"] = old["retrieved_utc"]
            if not record["http_headers"]:
                record["http_headers"] = old.get("http_headers", {})
        elif record["downloaded_this_run"]:
            record["retrieved_utc"] = _utc()
        previous[year] = record
        manifest["archives"] = [previous[key] for key in sorted(previous)]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Verified SHFE {year}: {record['bytes']} bytes, SHA-256 {record['sha256']}", flush=True
        )
        if index + 1 < len(years) and record["downloaded_this_run"]:
            time.sleep(1)
    return manifest


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=sorted(PINNED_ARCHIVES))
    parser.add_argument("--cache-dir", type=Path, default=repository / "data/local/shfe")
    args = parser.parse_args()
    cache = args.cache_dir if args.cache_dir.is_absolute() else repository / args.cache_dir
    try:
        acquire_archives(args.years, cache, repository=repository)
    except (ValueError, OSError) as error:
        parser.exit(1, f"SHFE acquisition failed: {error}\n")
    print(cache.resolve() / "acquisition_manifest.json")


if __name__ == "__main__":
    main()
