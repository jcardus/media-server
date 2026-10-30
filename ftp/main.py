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
left where it is (the camera may resume it with REST) and never pushed.

One shared account (FTP_USER / FTP_PASSWORD) for every camera. Cameras reach
this port directly (Cloudflare doesn't proxy FTP), and passive mode needs
FTP_MASQUERADE_ADDRESS set to the host's public IP because of docker's NAT.
"""
import logging
import os
import re
import shutil
import threading
from pathlib import Path, PurePosixPath

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ftp")

PORT = int(os.getenv("FTP_PORT", "21"))
SPOOL_DIR = Path(os.getenv("SPOOL_DIR", "/data/ftp"))
RETRY_DIR = Path(os.getenv("RETRY_DIR", "/data/ftp-retry"))
RETRY_INTERVAL = int(os.getenv("RETRY_INTERVAL", "300"))
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "recordings/")
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


def s3_key(relative: str) -> str:
    """S3 key for a spool-relative path: each segment sanitized, no '..'."""
    parts = [SAFE_PART.sub("_", part) for part in PurePosixPath(relative).parts
             if part not in ("", "/", ".", "..")]
    return S3_PREFIX + "/".join(part for part in parts if part)


def push(path: Path, root: Path) -> bool:
    """Upload `path` (under `root`) to S3 and delete it locally; True on success."""
    relative = path.relative_to(root).as_posix()
    key = s3_key(relative)
    size = path.stat().st_size
    try:
        s3().upload_file(str(path), S3_BUCKET, key)
    except Exception as exc:
        log.error("s3 upload failed file=%s bucket=%s key=%s: %s", relative, S3_BUCKET, key, exc)
        return False
    path.unlink(missing_ok=True)
    log.info("stored file=%s bytes=%d s3://%s/%s", relative, size, S3_BUCKET, key)
    return True


def received(path: Path):
    """A camera finished uploading `path`: push it, or park it for a retry."""
    if push(path, SPOOL_DIR):
        return
    target = RETRY_DIR / path.relative_to(SPOOL_DIR)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    log.warning("parked for retry: %s", target)


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
    log.info("ftp receiver on :%d passive=%s spool=%s -> s3://%s/%s",
             PORT, PASSIVE_PORTS, SPOOL_DIR, S3_BUCKET, S3_PREFIX)
    build_server().serve_forever()
