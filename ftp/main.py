"""
FTP receiver for JT/T 1078 0x9206 file uploads.

Traccar's `videoUpload` command sends a camera 0x9206 with this server's
host/port, the shared FTP account and a path; the camera then uploads its
SD-card recordings for the requested time range here over plain FTP, and
reports 0x1206 to Traccar when it's done. The 0x9206 path is what tells
uploads apart (the camera chooses the file names), so the trigger should put
the IMEI in it, e.g. /<imei>/.

Every completed file is pushed to S3_BUCKET as S3_PREFIX + <path as
uploaded> and then removed from the local spool, so the disk only holds
uploads in flight. If the S3 push fails the file is moved to RETRY_DIR and
retried every RETRY_INTERVAL seconds. A transfer that breaks off half-way is
left where it is (the camera may resume it with REST) and never pushed - unless
INCOMPLETE_PUSH_AFTER is set: some cameras (Hikvision G40) drop the control
connection as soon as the data is sent, which the FTP server counts as incomplete
although the whole file arrived. Then a file untouched for that many seconds is
pushed anyway, with the S3 object metadata partial=true.

Hikvision G40 cameras write their clips with an "IMKH" header and an .mp4 name, but the
content is an MPEG program stream that QuickTime and browsers cannot open. Such files are
rewritten into a standard MP4 (video copied, not re-encoded; HEVC tagged hvc1 for QuickTime,
audio to AAC) before they are pushed, under the same S3 key. If ffmpeg fails the original is
pushed instead. CONVERT_IMKH=0 turns this off.

While a file is uploading, every PROGRESS_INTERVAL seconds a small marker
{"bytes": <size so far>} is written to S3_BUCKET as PROGRESS_PREFIX + <path as
uploaded> + ".json", but only for files that grew since the last sweep. The
marker's S3 LastModified is therefore "bytes last arrived", which the camera
recording sync (pinme-backend-v2 cameraSync.js) reads so it doesn't time out
and re-request a slow upload that is still going (ticket #296).

One shared account (FTP_USER / FTP_PASSWORD) for every camera. Cameras reach
this port directly (Cloudflare doesn't proxy FTP), and passive mode needs
FTP_MASQUERADE_ADDRESS set to the host's public IP because of docker's NAT.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ftp")

PORT = int(os.getenv("FTP_PORT", "21"))
SPOOL_DIR = Path(os.getenv("SPOOL_DIR", "/data/ftp"))
RETRY_DIR = Path(os.getenv("RETRY_DIR", "/data/ftp-retry"))
RETRY_INTERVAL = int(os.getenv("RETRY_INTERVAL", "300"))
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "recordings/")
# Outside S3_PREFIX on purpose: anything under recordings/<imei>/<request>/ counts as
# an uploaded file for the sync.
PROGRESS_PREFIX = os.getenv("PROGRESS_PREFIX", "recordings-progress/")
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", "30"))
# Seconds an incomplete upload may sit untouched before it is pushed anyway; 0 = never.
INCOMPLETE_PUSH_AFTER = int(os.getenv("INCOMPLETE_PUSH_AFTER", "0"))
# Rewrite Hikvision "IMKH" files as standard MP4 before pushing them.
CONVERT_IMKH = os.getenv("CONVERT_IMKH", "1") not in ("0", "false", "")
CONVERT_TIMEOUT = int(os.getenv("CONVERT_TIMEOUT", "300"))
# Read when the server starts: tests import this module without them.
FTP_USER = os.getenv("FTP_USER", "")
FTP_PASSWORD = os.getenv("FTP_PASSWORD", "")
PASSIVE_PORTS = os.getenv("FTP_PASSIVE_PORTS", "30000-30009")
MASQUERADE_ADDRESS = os.getenv("FTP_MASQUERADE_ADDRESS", "")
# e: cd, l: list, m: mkdir, w: store, a: append (resume). No delete or rename:
# a camera only ever needs to write.
PERMISSIONS = "elmwa"

SAFE_PART = re.compile(r"[^A-Za-z0-9._-]")

_s3 = None


def s3():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client("s3")
    return _s3


def s3_key(relative: str, prefix: str = None) -> str:
    """S3 key for a spool-relative path: each segment sanitized, no '..'."""
    parts = [SAFE_PART.sub("_", part) for part in PurePosixPath(relative).parts
             if part not in ("", "/", ".", "..")]
    return (S3_PREFIX if prefix is None else prefix) + "/".join(part for part in parts if part)


def is_imkh(path: Path) -> bool:
    """True for a Hikvision "IMKH" file (a program stream that has an .mp4 name)."""
    try:
        with path.open("rb") as file:
            return file.read(4) == b"IMKH"
    except OSError:
        return False


def convert_imkh(path: Path):
    """Standard MP4 copy of a Hikvision IMKH file in a temporary file, or None.

    The video is copied, not re-encoded (HEVC gets the hvc1 tag QuickTime needs) and any audio
    becomes AAC. None when the file is not IMKH, conversion is off, or ffmpeg fails: the caller
    then pushes the original.
    """
    if not CONVERT_IMKH or not is_imkh(path):
        return None
    codec = ""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
             "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
        codec = probe.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("ffprobe failed file=%s: %s", path, exc)
    handle, name = tempfile.mkstemp(suffix=".mp4")
    os.close(handle)
    output = Path(name)
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-map", "0:a?", "-c:v", "copy"]
    if codec == "hevc":
        command += ["-tag:v", "hvc1"]
    command += ["-c:a", "aac", "-movflags", "+faststart", str(output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=CONVERT_TIMEOUT)
        if result.returncode == 0 and output.stat().st_size > 0:
            return output
        log.warning("ffmpeg could not convert file=%s (exit %s): %s", path, result.returncode,
                    (result.stderr or "").strip()[-300:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("ffmpeg failed file=%s: %s", path, exc)
    output.unlink(missing_ok=True)
    return None


def push(path: Path, root: Path, extra_args: dict = None) -> bool:
    """Upload `path` (under `root`) to S3 and delete it locally; True on success."""
    relative = path.relative_to(root).as_posix()
    key = s3_key(relative)
    converted = convert_imkh(path)
    source = converted or path
    size = source.stat().st_size
    try:
        if extra_args:
            s3().upload_file(str(source), S3_BUCKET, key, ExtraArgs=extra_args)
        else:
            s3().upload_file(str(source), S3_BUCKET, key)
    except Exception as exc:
        log.error("s3 upload failed file=%s bucket=%s key=%s: %s", relative, S3_BUCKET, key, exc)
        return False
    finally:
        if converted:
            converted.unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    log.info("stored file=%s bytes=%d s3://%s/%s%s", relative, size, S3_BUCKET, key,
             " (converted from IMKH)" if converted else "")
    return True


def received(path: Path):
    """A camera finished uploading `path`: push it, or park it for a retry."""
    if push(path, SPOOL_DIR):
        return
    target = RETRY_DIR / path.relative_to(SPOOL_DIR)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    log.warning("parked for retry: %s", target)


def report_progress(now: float = None):
    """Write a progress marker for every spool file written to in the last sweep.

    A file is "growing" if its mtime is within PROGRESS_INTERVAL; incomplete
    uploads left behind for a REST resume stop changing, so they stop being
    reported. Failures are only logged: the marker is advisory.
    """
    now = time.time() if now is None else now
    for path in SPOOL_DIR.rglob("*"):
        try:
            stat = path.stat()
        except FileNotFoundError:  # pushed and removed meanwhile
            continue
        if not path.is_file() or now - stat.st_mtime > PROGRESS_INTERVAL:
            continue
        relative = path.relative_to(SPOOL_DIR).as_posix()
        key = s3_key(relative, PROGRESS_PREFIX) + ".json"
        try:
            s3().put_object(Bucket=S3_BUCKET, Key=key, ContentType="application/json",
                            Body=json.dumps({"bytes": stat.st_size}).encode())
        except Exception as exc:
            log.warning("progress marker failed file=%s key=%s: %s", relative, key, exc)


def _progress_loop(stop: threading.Event):
    while not stop.wait(PROGRESS_INTERVAL):
        try:
            report_progress()
        except Exception:
            log.exception("progress sweep failed")


def push_stale_incomplete(now: float = None) -> int:
    """Push spool files untouched for INCOMPLETE_PUSH_AFTER seconds, marked partial.

    Anything still in the spool is an upload in flight or one that broke off; a file
    that has not changed for that long is not being written any more. Returns how many
    were pushed. A failed push leaves the file for the next sweep.
    """
    if INCOMPLETE_PUSH_AFTER <= 0:
        return 0
    now = time.time() if now is None else now
    pushed = 0
    for path in sorted(SPOOL_DIR.rglob("*")):
        try:
            stat = path.stat()
        except FileNotFoundError:  # pushed and removed meanwhile
            continue
        if not path.is_file() or stat.st_size == 0 or now - stat.st_mtime < INCOMPLETE_PUSH_AFTER:
            continue
        log.warning("pushing incomplete upload untouched for %ds: %s", now - stat.st_mtime, path)
        if push(path, SPOOL_DIR, {"Metadata": {"partial": "true"}}):
            pushed += 1
    return pushed


def _incomplete_loop(stop: threading.Event):
    while not stop.wait(min(INCOMPLETE_PUSH_AFTER, 30)):
        try:
            push_stale_incomplete()
        except Exception:
            log.exception("incomplete sweep failed")


def retry_once():
    for path in sorted(RETRY_DIR.rglob("*")):
        if path.is_file():
            push(path, RETRY_DIR)


def _retry_loop(stop: threading.Event):
    while not stop.wait(RETRY_INTERVAL):
        try:
            retry_once()
        except Exception:
            log.exception("retry sweep failed")


def build_server():
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.servers import ThreadedFTPServer

    class Handler(FTPHandler):
        def on_login(self, username):
            log.info("login user=%s from %s", username, self.remote_ip)

        def on_file_received(self, file):
            log.info("received %s from %s", file, self.remote_ip)
            received(Path(file))

        def on_incomplete_file_received(self, file):
            # Left in place for a REST resume; never pushed to S3 as-is.
            log.warning("incomplete upload %s from %s", file, self.remote_ip)

    authorizer = DummyAuthorizer()
    authorizer.add_user(FTP_USER, FTP_PASSWORD, str(SPOOL_DIR), perm=PERMISSIONS)
    Handler.authorizer = authorizer
    Handler.banner = "media-server ftp ready"
    low, high = (int(port) for port in PASSIVE_PORTS.split("-"))
    Handler.passive_ports = range(low, high + 1)
    if MASQUERADE_ADDRESS:
        Handler.masquerade_address = MASQUERADE_ADDRESS
    return ThreadedFTPServer(("0.0.0.0", PORT), Handler)


if __name__ == "__main__":
    missing = [name for name, value in (("FTP_USER", FTP_USER), ("FTP_PASSWORD", FTP_PASSWORD),
                                        ("S3_BUCKET", S3_BUCKET)) if not value]
    if missing:
        raise SystemExit("missing required setting(s): " + ", ".join(missing))
    if not MASQUERADE_ADDRESS:
        log.warning("FTP_MASQUERADE_ADDRESS unset: passive mode will advertise the container IP")
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    RETRY_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_retry_loop, args=(threading.Event(),), daemon=True).start()
    threading.Thread(target=_progress_loop, args=(threading.Event(),), daemon=True).start()
    if INCOMPLETE_PUSH_AFTER > 0:
        threading.Thread(target=_incomplete_loop, args=(threading.Event(),), daemon=True).start()
    log.info("ftp receiver on :%d passive=%s spool=%s -> s3://%s/%s",
             PORT, PASSIVE_PORTS, SPOOL_DIR, S3_BUCKET, S3_PREFIX)
    build_server().serve_forever()
