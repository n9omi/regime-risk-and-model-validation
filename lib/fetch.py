"""
fetch.py | Download-with-cache used by every data script.

- Files are cached under data/raw/ and re-downloaded at most once per `max_age_hours`.
- If a download fails, the cached copy is used and the console says so.
- Several URLs can be given: they are tried in order (primary source first, mirrors after).
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0 (research pipeline; +https://github.com/n9omi)"}


def fetch(urls, dest, max_age_hours: float = 24, timeout: int = 60, min_bytes: int = 200) -> tuple[Path, str]:
    """Return (path, source_used). `urls` is a URL or a list of URLs tried in order."""
    urls = [urls] if isinstance(urls, str) else list(urls)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_file = dest.with_name(dest.name + ".source")
    cached_from = src_file.read_text().strip() if src_file.exists() else "unknown source"
    if dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_hours * 3600:
        return dest, cached_from
    errors = []
    for url in urls:
        for attempt in range(2):
            try:
                r = requests.get(url, headers=UA, timeout=timeout)
                r.raise_for_status()
                if len(r.content) < min_bytes:
                    raise ValueError(f"only {len(r.content)} bytes")
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                src_file.write_text(url)
                return dest, url
            except Exception as e:  # network, HTTP, SSL
                errors.append(f"{url.split('?')[0][:90]}: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(1.5 * (attempt + 1))
    if dest.exists():
        print(f"  ! download failed, using cached {dest.name}")
        return dest, cached_from
    raise RuntimeError("All sources failed for " + dest.name + ":\n  " + "\n  ".join(errors) +
                       f"\nTip: download the file by hand and save it as {dest}")
