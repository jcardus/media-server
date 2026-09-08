"""
Event-attachment sink for Jimi JT/T 808 dashcams.

Traccar answers an ADAS/DMS alarm (0x64/0x65) with
`VIDEOUPLOAD,<host>,<port>,<alarmLabel>,<channel>,<type>#`; the camera then
`POST /upload`s the event clip (.mp4) and snapshots (.jpg) as multipart:
  file       the binary (its own filename is just <channel>_<epoch>)
  filename   <imei>_<alarmLabel(32 hex)>_<channel>_<seq>.<ext>   <- the real key
  timestamp  epoch ms
  sign       base64 HMAC (not verified here)
Reply must be {"code": 0, ...} or the camera re-POSTs every few minutes.

Files are stored under STORE_DIR/<imei>/<alarmLabel>/<filename> and served
back over HTTP so the frontend can list and play an event's media.
"""
import logging
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from aiohttp import web

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upload")

PORT = int(os.getenv("UPLOAD_PORT", "10005"))
STORE_DIR = Path(os.getenv("STORE_DIR", "/data/attachments"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
ALLOWED_EXT = {".mp4", ".h264", ".jpg", ".jpeg", ".png"}

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
# The camera's `filename` form field: <imei>_<alarmLabel(32 hex)>_<channel>_<seq>.<ext>
FILENAME_FIELD = re.compile(r"^(\d{10,17})_([0-9A-Fa-f]{8,64})_")


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


def _dest(stored_name):
    """<imei>/<alarmLabel>/<name> from the camera's `filename` field, else _unsorted/."""
    match = FILENAME_FIELD.match(stored_name)
    if match:
        return STORE_DIR / match.group(1) / match.group(2).lower() / stored_name
    return STORE_DIR / "_unsorted" / stored_name


def _ok():
    # The camera treats any body without `code == 0` as failure and re-POSTs
    # the file every few minutes forever.
    return web.json_response({"code": 0, "msg": "success"})


def _fail(msg, status=200):
    return web.json_response({"code": 1, "msg": msg}, status=status)


async def handle_upload(request: web.Request) -> web.Response:
    log.info(
        "POST %s query=%s ct=%r len=%s ua=%r",
        request.path, dict(request.query), request.content_type,
        request.headers.get("Content-Length"), request.headers.get("User-Agent"))

    if not (request.content_type or "").startswith("multipart/"):
        return await _handle_raw(request)

    reader = await request.multipart()
    pending = None
    suffix = ".bin"
    fields = {}
    async for part in reader:
        if part.name == "file" and part.filename:
            suffix = Path(sanitize(part.filename)).suffix.lower() or ".bin"
            pending = STORE_DIR / "_pending" / (uuid4().hex + suffix)
            await _write(pending, _iter_part(part))
        else:
            fields[part.name] = (await part.text())[:256]
    log.info("  fields=%s pending=%s", fields, bool(pending))

    if pending is None or not pending.exists():
        return _fail("no file", 400)

    # The `filename` field is <imei>_<alarmLabel>_<channel>_<seq>.<ext>.
    stored = sanitize(fields.get("filename") or (uuid4().hex + suffix))
    if Path(stored).suffix.lower() not in ALLOWED_EXT:
        pending.unlink(missing_ok=True)
        return _fail("extension not allowed", 415)
    dest = _dest(stored)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pending.replace(dest)
    log.info("stored %s (%d bytes)", dest.relative_to(STORE_DIR), dest.stat().st_size)
    return _ok()


async def _handle_raw(request: web.Request) -> web.Response:
    disp = request.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^"]+)"?', disp)
    name = sanitize((match.group(1) if match else request.query.get("filename"))
                    or f"{int(time.time() * 1000)}.bin")
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        return _fail("extension not allowed", 415)
    dest = _dest(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = await _write(dest, _iter_body(request))
    log.info("stored %s (%d bytes)", dest.relative_to(STORE_DIR), size)
    return _ok()


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
    return web.FileResponse(path)


async def handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


@web.middleware
async def cors(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            exc.headers["Access-Control-Allow-Origin"] = "*"
            raise
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.setdefault("Access-Control-Allow-Headers", "*")
    return response


def build_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BYTES + 1024 * 1024, middlewares=[cors])
    app.router.add_post("/upload", handle_upload)
    for pattern in ("/attachments/{imei}/{label}", "/attachments/{imei}/{label}/"):
        app.router.add_get(pattern, handle_list)
    app.router.add_get("/attachments/{imei}/{label}/{name}", handle_file)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    (STORE_DIR / "_pending").mkdir(parents=True, exist_ok=True)
    for leftover in (STORE_DIR / "_pending").glob("*"):
        leftover.unlink(missing_ok=True)
    log.info("attachment sink on :%d  store=%s", PORT, STORE_DIR)
    web.run_app(build_app(), host="0.0.0.0", port=PORT, print=None)
