"""
Event-attachment sink for Jimi JT/T 808 dashcams.

Traccar answers an ADAS/DMS alarm (0x64/0x65) with
`VIDEOUPLOAD,<host>,<port>,<alarmLabel>,<channel>,<type>#`; the camera then
does `POST /upload` here with the event clip (.mp4) and snapshots (.jpg),
named `<imei>_<alarmLabel>_<xy>.<ext>` (xy = channel + serial, per Jimi docs).

Files are stored under STORE_DIR/<imei>/<alarmLabel>/ and served back over
HTTP so the frontend can list and play an event's media by its label.
"""
import json
import logging
import os
import re
import time
from pathlib import Path

from aiohttp import web

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upload")

PORT = int(os.getenv("UPLOAD_PORT", "10005"))
STORE_DIR = Path(os.getenv("STORE_DIR", "/data/attachments"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
ALLOWED_EXT = {".mp4", ".h264", ".jpg", ".jpeg", ".png"}

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
NAME_PARTS = re.compile(r"^(\d{10,17})_([0-9A-Fa-f]{8,64})_(.+)$")


def sanitize(name: str) -> str:
    return SAFE_NAME.sub("_", os.path.basename(name or "")).lstrip(".")


async def _write(path: Path, chunks) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as handle:
        async for chunk in chunks:
            written += len(chunk)
            if written > MAX_BYTES:
                handle.close()
                tmp.unlink(missing_ok=True)
                raise web.HTTPRequestEntityTooLarge(max_size=MAX_BYTES, actual_size=written)
            handle.write(chunk)
    tmp.replace(path)
    return written


async def _iter_part(part, size=64 * 1024):
    while True:
        chunk = await part.read_chunk(size)
        if not chunk:
            return
        yield chunk


async def _iter_body(request, size=64 * 1024):
    while True:
        chunk = await request.content.read(size)
        if not chunk:
            return
        yield chunk


IMEI_KEYS = ("imei", "deviceImei", "deviceimei", "sn", "terminal")
LABEL_KEYS = ("alarmLabel", "alarmlabel", "label", "alarm", "alarmId", "alarmid", "warnId")


def _pick(mapping, keys):
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def dest_dir(request, filename):
    """Route by explicit imei/label (query or header), else the filename pattern,
    else _unsorted/. Also accept them as extra path segments after /upload."""
    extra = [s for s in request.match_info.get("tail", "").split("/") if s]
    headers = {k.lower().replace("x-", "").replace("-", ""): v for k, v in request.headers.items()}
    imei = (extra[0] if len(extra) > 0 else None) or _pick(request.query, IMEI_KEYS) or _pick(headers, ("imei", "deviceimei", "sn"))
    label = (extra[1] if len(extra) > 1 else None) or _pick(request.query, LABEL_KEYS) or _pick(headers, ("alarmid", "alarmlabel", "label"))
    if imei and label:
        return STORE_DIR / sanitize(imei) / sanitize(label).lower()
    match = NAME_PARTS.match(filename)
    if match:
        return STORE_DIR / match.group(1) / match.group(2).lower()
    return STORE_DIR / "_unsorted"


async def handle_upload(request: web.Request) -> web.Response:
    log.info(
        "POST %s query=%s ct=%r len=%s ua=%r xhdr=%s",
        request.path, dict(request.query), request.content_type,
        request.headers.get("Content-Length"), request.headers.get("User-Agent"),
        {k: v for k, v in request.headers.items() if k.lower().startswith("x-")})

    saved = []
    if request.content_type and request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        async for part in reader:
            log.info("  part name=%r filename=%r headers=%s", part.name, part.filename, dict(part.headers))
            if not part.filename:
                text = (await part.text())[:200]
                log.info("  field %r = %r", part.name, text)
                continue
            name = sanitize(part.filename)
            if Path(name).suffix.lower() not in ALLOWED_EXT:
                log.warning("rejected part %r (extension)", part.filename)
                continue
            path = dest_dir(request, name) / name
            size = await _write(path, _iter_part(part))
            saved.append({"name": name, "bytes": size})
            log.info("stored %s (%d bytes)", path.relative_to(STORE_DIR), size)
    else:
        disp = request.headers.get("Content-Disposition", "")
        match = re.search(r'filename="?([^"]+)"?', disp)
        raw = match.group(1) if match else request.query.get("filename") or request.query.get("name")
        name = sanitize(raw or f"{int(time.time() * 1000)}.bin")
        if Path(name).suffix.lower() not in ALLOWED_EXT:
            raise web.HTTPUnsupportedMediaType(text="extension not allowed")
        path = dest_dir(request, name) / name
        size = await _write(path, _iter_body(request))
        saved.append({"name": name, "bytes": size})
        log.info("stored %s (%d bytes)", path.relative_to(STORE_DIR), size)

    if not saved:
        raise web.HTTPBadRequest(text="no file in request")
    return web.json_response({"stored": saved})


async def handle_list(request: web.Request) -> web.Response:
    imei = sanitize(request.match_info["imei"])
    label = sanitize(request.match_info["label"]).lower()
    directory = STORE_DIR / imei / label
    if not directory.is_dir():
        return web.json_response({"files": []})
    files = [
        {
            "name": entry.name,
            "bytes": entry.stat().st_size,
            "url": f"/attachments/{imei}/{label}/{entry.name}",
        }
        for entry in sorted(directory.iterdir())
        if entry.is_file() and not entry.name.endswith(".part")
    ]
    return web.json_response({"files": files})


async def handle_file(request: web.Request) -> web.StreamResponse:
    imei = sanitize(request.match_info["imei"])
    label = sanitize(request.match_info["label"]).lower()
    name = sanitize(request.match_info["name"])
    path = STORE_DIR / imei / label / name
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Access-Control-Allow-Origin": "*"})


async def handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def build_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BYTES + 1024 * 1024)
    app.router.add_post("/upload", handle_upload)
    app.router.add_post(r"/upload/{tail:.*}", handle_upload)  # tolerate /upload/<imei>/<label>
    for pattern in ("/attachments/{imei}/{label}", "/attachments/{imei}/{label}/"):
        app.router.add_get(pattern, handle_list)
    app.router.add_get("/attachments/{imei}/{label}/{name}", handle_file)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("attachment sink on :%d  store=%s", PORT, STORE_DIR)
    web.run_app(build_app(), host="0.0.0.0", port=PORT, print=None)
