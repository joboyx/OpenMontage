"""Shared Replicate helpers for the audio tools: file upload, prediction polling, download."""

from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

API = "https://api.replicate.com/v1"
# Replicate recommends data URIs only for small files; larger ones go through /v1/files.
_DATA_URI_LIMIT = 256 * 1024


def to_input_uri(token: str, ref: str) -> str:
    """Return a URL Replicate can read for a local path, URL or data URI."""
    import base64

    import requests

    if ref.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(ref)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    if len(data) <= _DATA_URI_LIMIT:
        return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
    resp = requests.post(
        f"{API}/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"content": (path.name, data, mime_type)},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["urls"]["get"]


def run_prediction(token: str, body: dict[str, Any], url: str = f"{API}/predictions") -> dict[str, Any]:
    """Create a prediction and poll until it settles; raise on failure."""
    import requests

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait=60",
    }
    submit = requests.post(url, headers=headers, json=body, timeout=90)
    submit.raise_for_status()
    pred = submit.json()
    while pred.get("status") in ("starting", "processing"):
        time.sleep(3)
        get_url = pred.get("urls", {}).get("get")
        if not get_url:
            raise RuntimeError("Replicate response missing poll URL")
        poll = requests.get(get_url, headers=headers, timeout=30)
        poll.raise_for_status()
        pred = poll.json()
    if pred.get("status") != "succeeded":
        raise RuntimeError(f"prediction {pred.get('status')}: {pred.get('error')}")
    return pred


def download(pred: dict[str, Any], output_path: str | None, default_stem: str) -> Path:
    """Save the prediction output file; keep the provider's extension when no path is given."""
    import requests

    output = pred.get("output")
    file_url = output[0] if isinstance(output, list) else output
    if not isinstance(file_url, str):
        raise RuntimeError(f"Unexpected output shape from Replicate: {output!r}")
    suffix = Path(urlparse(file_url).path).suffix or ".bin"
    path = Path(output_path) if output_path else Path(default_stem + suffix)
    resp = requests.get(file_url, timeout=180)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return path
