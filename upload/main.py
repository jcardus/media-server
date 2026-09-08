"""
Event-attachment sink for Jimi JT/T 808 dashcams.

Traccar answers an ADAS/DMS alarm (0x64/0x65) with
`VIDEOUPLOAD,<host>,<port>,<alarmLabel>,<channel>,<type>#`; the camera then
`POST /upload`s the event clip (.mp4) and snapshots (.jpg) as multipart:
  file       the binary (its own filename is just <channel>_<epoch>)
  filename   <imei>_<alarmLabel(32 hex)>_<channel>_<seq>.<ext>   <- the real key
  timestamp  epoch ms
  sign       md5(filename + timestamp + "jimidvr@123!443") hex, base64-wrapped
Reply must be Jimi dvr-upload's exact body
  {"code": 200, "message": "File upload success", "data": <filename>}
or the camera reports UPLOADFILEFAIL and re-POSTs every ~5 min.

Files are stored under STORE_DIR/<imei>/<alarmLabel>/<filename> and served
back over HTTP so the frontend can list and play an event's media.
"""
import hashlib
import json
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
# Jimi dvr-upload contract: sign = md5(filename + timestamp + SECRET), lowercase
# hex. The camera then base64-wraps that hex string. Reject bad signs only when
# STRICT_SIGN is set - otherwise just log.
SIGN_SECRET = os.getenv("SIGN_SECRET", "jimidvr@123!443")
STRICT_SIGN = os.getenv("STRICT_SIGN", "") not in ("", "0", "false")

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


def _ok(filename):
    # dvr-upload's exact success body - anything else and the camera reports
    # UPLOADFILEFAIL and re-POSTs every ~5 min.
    return web.json_response({"code": 200, "message": "File upload success", "data": filename})


def _fail(message, status=400):
    return web.json_response({"code": status, "message": message}, status=status)


def _sign_ok(filename, timestamp, sign):
    if not sign:
        return False
    want = hashlib.md5((filename + timestamp + SIGN_SECRET).encode("utf-8")).hexdigest()
    got = sign.strip().lower()
    if len(got) != 32:  # camera also sends base64(hex)
        try:
            import base64
            got = base64.b64decode(sign).decode().strip().lower()
        except Exception:
            pass
    return got == want


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
        return _fail("The file content is empty", 400)

    name_field = fields.get("filename", "")
    if not name_field:
        pending.unlink(missing_ok=True)
        return _fail("The filename cannot be empty", 400)
    if not _sign_ok(name_field, fields.get("timestamp", ""), fields.get("sign", "")):
        log.warning("sign mismatch for %s (ts=%s)", name_field, fields.get("timestamp"))
        if STRICT_SIGN:
            pending.unlink(missing_ok=True)
            return _fail("Signature error", 400)

    # The `filename` field is <imei>_<alarmLabel>_<channel>_<seq>.<ext>.
    stored = sanitize(name_field)
    if Path(stored).suffix.lower() not in ALLOWED_EXT:
        pending.unlink(missing_ok=True)
        return _fail("extension not allowed", 400)
    dest = _dest(stored)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pending.replace(dest)
    log.info("stored %s (%d bytes)", dest.relative_to(STORE_DIR).as_posix(), dest.stat().st_size)
    return _ok(name_field)


async def _handle_raw(request: web.Request) -> web.Response:
    disp = request.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^"]+)"?', disp)
    name = sanitize((match.group(1) if match else request.query.get("filename"))
                    or f"{int(time.time() * 1000)}.bin")
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        return _fail("extension not allowed", 400)
    dest = _dest(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = await _write(dest, _iter_body(request))
    log.info("stored %s (%d bytes)", dest.relative_to(STORE_DIR), size)
    return _ok(name)


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
        if entry.is_file() and not entry.name.startswith(".") and not entry.name.endswith(".part")
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


def _label_time_ms(label):
    """Alarm identification number: termId[7] time-BCD[6] seq[1] count[1] rsv[1].
    Bytes 7..12 (hex offsets 14..26) are YYMMDDHHMMSS in BCD."""
    try:
        bcd = label[14:26]
        y, mo, d = 2000 + int(bcd[0:2]), int(bcd[2:4]), int(bcd[4:6])
        h, mi, s = int(bcd[6:8]), int(bcd[8:10]), int(bcd[10:12])
        import datetime
        # camera stamps local time (America/Sao_Paulo, UTC-3)
        return int((datetime.datetime(y, mo, d, h, mi, s) + datetime.timedelta(hours=3)).timestamp() * 1000)
    except (ValueError, IndexError):
        return None


async def handle_event_meta(request: web.Request) -> web.Response:
    """Traccar POSTs {imei, identifier, type, alarm, level, time} when it decodes
    an ADAS/DMS alarm - the label alone can't tell fatigue from lane departure."""
    try:
        body = await request.json()
    except Exception:
        return _fail("bad json", 400)
    imei = sanitize(str(body.get("imei", "")))
    identifier = sanitize(str(body.get("identifier", ""))).lower()
    if not imei or not identifier:
        return _fail("imei and identifier required", 400)
    folder = STORE_DIR / imei / identifier
    folder.mkdir(parents=True, exist_ok=True)
    meta = {k: body[k] for k in ("type", "alarm", "level", "time", "kind") if body.get(k) is not None}
    (folder / ".meta.json").write_text(json.dumps(meta))
    log.info("event meta %s/%s %s", imei, identifier, meta)
    return _ok(identifier)


async def handle_events(request: web.Request) -> web.Response:
    """Every event folder we hold media for (or have metadata for), newest first
    - the frontend uses this because this Traccar instance can't return
    historical alarm positions."""
    imei = sanitize(request.match_info["imei"])
    root = STORE_DIR / imei
    events = []
    if root.is_dir():
        for entry in root.iterdir():
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            files = [f for f in entry.iterdir()
                     if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".part")]
            meta_path = entry / ".meta.json"
            if not files and not meta_path.is_file():
                continue
            event = {
                "identifier": entry.name,
                "time": _label_time_ms(entry.name),
                "fileCount": len(files),
                "hasVideo": any(f.suffix.lower() in (".mp4", ".h264") for f in files),
            }
            if meta_path.is_file():
                try:
                    event.update({k: v for k, v in json.loads(meta_path.read_text()).items() if v is not None})
                except Exception:
                    pass
            events.append(event)
    events.sort(key=lambda e: e["time"] or 0, reverse=True)
    return web.json_response({"events": events})


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
    app.router.add_post("/event", handle_event_meta)
    for pattern in ("/attachments/{imei}", "/attachments/{imei}/"):
        app.router.add_get(pattern, handle_events)
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
