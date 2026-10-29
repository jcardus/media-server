import asyncio
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("jt1078")

MAGIC = b"\x30\x31\x63\x64"
MAX_PAYLOAD_SIZE = 65535
AAC_SAMPLE_RATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000)
# Bump on every change to handle_status()/publisher_status() so a deploy can
# be confirmed with `curl -i` (X-JT1078-Version header) instead of guessing
# from behavior - this is what caught the /media-status prefix-stripping bug.
STATUS_ENDPOINT_VERSION = "3"


def audio_input_options(codec: str, sample_rate: str) -> list[str]:
    options = ["-f", codec]
    if codec in {"alaw", "mulaw", "s16be", "s16le"}:
        options.extend(("-ar", sample_rate, "-ac", "1"))
    return options


def audio_codec_for_payload_type(payload_type: int) -> tuple[str, str]:
    """JT/T 1078's payload-type field is vendor-extended: 19 is Jimi's own
    extension for AAC (see git history - JT1078_AUDIO_CODEC used to default
    to "alaw", the standard's own G.711A baseline, until the Jimi cameras
    needed AAC). A single env var for the whole service can't represent
    that once a second vendor's cameras share it: an iStartek DC600 reports
    payload type 6, which got wrapped in a synthetic AAC ADTS header meant
    for type 19 and fed ffmpeg's AAC decoder nothing but garbage ("Invalid
    data found", "SBR was found before the first channel element").
    Fall back to alaw, not aac, for anything that isn't the known Jimi
    AAC type."""
    if payload_type == 19:
        return (
            os.getenv("JT1078_AUDIO_CODEC", "aac"),
            os.getenv("JT1078_AUDIO_SAMPLE_RATE", "8000"))
    return (
        os.getenv("JT1078_AUDIO_CODEC_FALLBACK", "alaw"),
        os.getenv("JT1078_AUDIO_SAMPLE_RATE_FALLBACK", "8000"))


def prepare_audio_frame(frame: bytes, codec: str, sample_rate: str, channels: int = 1) -> bytes:
    if codec != "aac" or frame.startswith((b"\xff\xf0", b"\xff\xf1", b"\xff\xf8", b"\xff\xf9")):
        return frame
    frequency_index = AAC_SAMPLE_RATES.index(int(sample_rate))
    frame_length = len(frame) + 7
    profile = 1  # AAC Low Complexity (Audio Object Type 2)
    header = bytes((
        0xFF,
        0xF1,
        (profile << 6) | (frequency_index << 2) | (channels >> 2),
        ((channels & 3) << 6) | (frame_length >> 11),
        (frame_length >> 3) & 0xFF,
        ((frame_length & 7) << 5) | 0x1F,
        0xFC,
    ))
    return header + frame


def luhn(number: int) -> int:
    total = 0
    for index, value in enumerate(reversed(str(number))):
        digit = int(value)
        if index % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (10 - total % 10) % 10


def decode_terminal_id(value: bytes) -> str:
    serial = value.hex()
    if serial.isdigit():
        return serial
    number = int.from_bytes(value, "big")
    return f"{number}{luhn(number)}"


@dataclass
class Packet:
    sequence: int
    imei: str
    channel: int
    payload_type: int
    data_type: int
    fragment_type: int
    timestamp: int
    payload: bytes


class PacketParser:
    def __init__(self, peer=None):
        self.buffer = bytearray()
        self.peer = peer
        self.packets_parsed = 0
        self.unrecognized_logged = False

    def _log_unrecognized(self):
        # Bytes that never contain the JT1078 frame magic used to be
        # discarded in total silence: the receiver logged "camera
        # connected" and then nothing at all, forever. That is exactly
        # what a device speaking some *other* framing looks like from the
        # outside -- a JT/T 808 0x7e message, a vendor handshake, TLS, or
        # a JT1078 variant with a header we do not parse -- and it is
        # indistinguishable from a camera that simply sent nothing. Dump
        # the head of the stream once so a new device model can be
        # identified from the log instead of guessed at.
        if self.unrecognized_logged or self.packets_parsed or len(self.buffer) < 16:
            return
        self.unrecognized_logged = True
        LOGGER.warning(
            "no JT1078 frame magic in first %d bytes peer=%s head=%s",
            len(self.buffer), self.peer, bytes(self.buffer[:64]).hex())

    def feed(self, data: bytes) -> list[Packet]:
        self.buffer.extend(data)
        packets = []
        while True:
            start = self.buffer.find(MAGIC)
            if start < 0:
                self._log_unrecognized()
                if len(self.buffer) > len(MAGIC) - 1:
                    del self.buffer[: -(len(MAGIC) - 1)]
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < 16:
                break

            data_type = self.buffer[15] >> 4
            fragment_type = self.buffer[15] & 0x0F
            if data_type <= 2:
                header_size = 30
                length_offset = 28
                timestamp = int.from_bytes(self.buffer[16:24], "big")
            elif data_type == 3:
                header_size = 26
                length_offset = 24
                timestamp = int.from_bytes(self.buffer[16:24], "big")
            else:
                LOGGER.warning("unsupported JT1078 data type %d", data_type)
                del self.buffer[:4]
                continue

            if len(self.buffer) < header_size:
                break
            payload_size = int.from_bytes(self.buffer[length_offset:length_offset + 2], "big")
            if payload_size > MAX_PAYLOAD_SIZE:
                LOGGER.warning("invalid JT1078 payload size %d", payload_size)
                del self.buffer[:4]
                continue
            packet_size = header_size + payload_size
            if len(self.buffer) < packet_size:
                break

            packets.append(Packet(
                sequence=int.from_bytes(self.buffer[6:8], "big"),
                imei=decode_terminal_id(bytes(self.buffer[8:14])),
                channel=self.buffer[14],
                payload_type=self.buffer[5] & 0x7F,
                data_type=data_type,
                fragment_type=fragment_type,
                timestamp=timestamp,
                payload=bytes(self.buffer[header_size:packet_size]),
            ))
            del self.buffer[:packet_size]
            self.packets_parsed += 1
        return packets


class Publisher:
    def __init__(self, imei: str, channel: int, namespace: str):
        channel_offset = int(os.getenv("JT1078_CHANNEL_OFFSET", "1"))
        output_channel = max(0, channel - channel_offset)
        self.path = f"{namespace}/{output_channel}/{imei}"
        self.process = None
        self.audio_input = None
        self.audio_payload_type = None
        self.video_payload_type = None
        # Decided from the first audio packet's payload type (see write()),
        # not fixed at construction - audio_codec_for_payload_type() is what
        # tells a Jimi camera's AAC (payload type 19) apart from everything
        # else, including the DC600's G.711A (payload type 6).
        self.audio_codec = None
        self.audio_sample_rate = None
        # Whether to include an audio input in the ffmpeg command. Channels
        # that never send audio must not block video: FFmpeg blocks opening
        # a second -i until it can probe it. A new publisher waits a short
        # grace period for a first audio packet before committing to a
        # pipeline shape and spawning ffmpeg, buffering frames meanwhile —
        # restarting ffmpeg after the fact throws away the camera's only
        # SPS/PPS, which breaks the decoder for the rest of the connection.
        self.has_audio = False
        self.decided = False
        self.pending = []
        self.grace_handle = None
        self.grace_task = None
        # Liveness telemetry for the /status endpoint: monotonic timestamps
        # of the last frame received from the camera, when ffmpeg was last
        # (re)started, and how many times it has been restarted.
        self.created_at = time.monotonic()
        self.started_at = None
        self.last_video_at = None
        self.last_audio_at = None
        self.restart_count = 0
        # Set once this publisher is evicted or its connection ends. A
        # coroutine still stuck in a write when that happens wakes up with
        # a broken pipe, and its retry must not spawn a fresh ffmpeg on
        # the path the replacing connection is about to publish to.
        self.closed = False

    async def start(self):
        if self.started_at is not None:
            self.restart_count += 1
        self.started_at = time.monotonic()
        if self.audio_input:
            self.audio_input.close()
            self.audio_input = None
        target = f"rtsp://mediamtx:8554/rtc/{self.path}"
        codec = os.getenv("JT1078_VIDEO_CODEC", "h264")
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", os.getenv("FFMPEG_LOG_LEVEL", "warning"),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            # Raw H.264 over the pipe carries no timestamps of its own.
            # A synthetic frame-counter clock (the old setts filter) can't
            # track these cameras' variable, sub-nominal frame rate, so
            # video time drifts against audio's sample-count clock until
            # the RTSP muxer starves waiting for the "behind" stream.
            # Stamp each frame with its arrival wall-clock time instead, so
            # video is paced by real elapsed time; the audio input keeps
            # its own ADTS-derived PTS (forcing wall-clock onto the aac
            # demuxer produced non-monotonic DTS -- see git history) and
            # aresample=async below pulls audio onto this same timeline.
            "-use_wallclock_as_timestamps", "1",
            "-f", codec,
            "-i", "pipe:0",
        ]
        pass_fds = ()
        audio_read = None
        audio_write = None
        if self.has_audio:
            # Normally already set by write()'s first audio packet, which is
            # the only caller that knows the payload type; fall back to the
            # same default audio_codec_for_payload_type() uses for an
            # unrecognized type, for anything that flips has_audio on without
            # going through write() (tests do this directly).
            if self.audio_codec is None:
                self.audio_codec, self.audio_sample_rate = audio_codec_for_payload_type(-1)
            audio_options = audio_input_options(self.audio_codec, self.audio_sample_rate)
            audio_read, audio_write = os.pipe()
            args += [
                "-thread_queue_size", "512",
                # Sparse/gappy voice audio (silence gaps, VAD-gated
                # transmission) can take minutes of wall-clock time to
                # reach even a modest probesize, stalling this input open
                # indefinitely (and with it the whole muxer, video
                # included) -- confirmed in production: one connection
                # took 8 minutes to satisfy a 32768-byte probesize. The
                # format is already known (-f aac) and one ADTS frame is
                # enough to learn the stream parameters, so keep both
                # bounds as small as FFmpeg allows.
                "-analyzeduration", "100000",
                "-probesize", "4096",
                *audio_options,
                "-i", f"pipe:{audio_read}",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "libopus",
                "-b:a", "32k",
                # Resample audio onto video's wall-clock timeline,
                # inserting/dropping samples across the gaps that VAD-gated
                # voice audio leaves in the sample-count clock instead of
                # letting them accumulate into a muxer stall. Matches
                # mediamtx.yml's transcode steps.
                "-af", "aresample=async=1000",
            ]
            pass_fds = (audio_read,)
        else:
            args += ["-an", "-c:v", "copy"]
        if self.has_audio:
            # Bound how long the RTSP muxer will hold the "ahead" stream
            # waiting for the other to catch up in DTS order. FFmpeg's
            # default is 10s and "0" means *unlimited* (a previous version
            # of this line set 0 and made the wedge permanent instead of
            # self-clearing). With both streams now on a wall-clock pace,
            # imbalance stays small; cap it low so any transient (a network
            # hiccup, a rate change mid-stream) is flushed rather than
            # allowed to back up until our audio-write timeout kills the
            # process.
            args += ["-max_interleave_delta", "500000"]
        args += [
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            target,
        ]
        LOGGER.info(
            "starting publisher path=rtc/%s videoCodec=%s audio=%s",
            self.path, codec, self.has_audio)
        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            pass_fds=pass_fds,
        )
        asyncio.ensure_future(self._log_stderr(self.process))
        if audio_read is not None:
            os.close(audio_read)
            # A plain blocking write() to this pipe (as used to be done
            # here) can freeze the entire event loop -- all cameras, not
            # just this one -- if ffmpeg's audio reader ever stalls (e.g.
            # backpressure from the RTSP connection to MediaMTX) long
            # enough to fill the kernel pipe buffer. Route writes through
            # asyncio instead so a stalled reader just backs up this
            # publisher's writes rather than blocking the whole process.
            loop = asyncio.get_event_loop()
            transport, protocol = await loop.connect_write_pipe(
                asyncio.streams.FlowControlMixin, os.fdopen(audio_write, "wb", buffering=0))
            self.audio_input = asyncio.StreamWriter(transport, protocol, None, loop)

    async def _log_stderr(self, process):
        # FFmpeg block-buffers stderr when it isn't a TTY, so the useful
        # startup diagnostics (input/output stream info, mapping, and any
        # "Non-monotonic DTS" / "Queue input backward" warnings) are lost
        # when we kill a wedged process. Drain it line by line instead.
        stream = process.stderr
        if not isinstance(stream, asyncio.StreamReader):
            return
        buffer = ""
        try:
            while chunk := await stream.read(4096):
                buffer += chunk.decode("utf-8", "replace")
                buffer = buffer.replace("\r", "\n")
                *lines, buffer = buffer.split("\n")
                for line in lines:
                    if line.strip():
                        LOGGER.info("ffmpeg[%s] %s", self.path, line)
        except Exception:
            pass

    async def write(self, frame: bytes, data_type: int, payload_type: int):
        is_audio = data_type == 3
        if is_audio:
            self.last_audio_at = time.monotonic()
            if self.audio_codec is None:
                self.audio_codec, self.audio_sample_rate = audio_codec_for_payload_type(payload_type)
            frame = prepare_audio_frame(frame, self.audio_codec, self.audio_sample_rate)
        else:
            self.last_video_at = time.monotonic()
            if not frame.startswith((b"\x00\x00\x01", b"\x00\x00\x00\x01")):
                frame = b"\x00\x00\x00\x01" + frame

        if not self.decided:
            self.pending.append((frame, data_type, payload_type))
            if is_audio:
                self.has_audio = True
                if self.grace_handle is not None:
                    self.grace_handle.cancel()
                    self.grace_handle = None
                await self._commit()
            elif self.grace_handle is None:
                grace = float(os.getenv("JT1078_AUDIO_GRACE_SECONDS", "0.5"))
                self.grace_handle = asyncio.get_event_loop().call_later(
                    grace, self._on_grace_expired)
            return

        if is_audio and not self.has_audio:
            self.has_audio = True
            if self.process is not None and self.process.returncode is None:
                LOGGER.info("audio detected, restarting publisher path=%s", self.path)
                await self.close(final=False)

        await self._deliver(frame, data_type, payload_type)

    def _on_grace_expired(self):
        self.grace_handle = None
        self.grace_task = asyncio.ensure_future(self._commit())

    async def _commit(self):
        if self.decided or self.closed:
            return
        self.decided = True
        pending, self.pending = self.pending, []
        await self.start()
        for frame, data_type, payload_type in pending:
            await self._deliver(frame, data_type, payload_type)

    async def _deliver(self, frame: bytes, data_type: int, payload_type: int):
        is_audio = data_type == 3
        for attempt in range(2):
            if self.closed:
                return
            if self.process is None or self.process.returncode is not None:
                if self.process is not None:
                    LOGGER.warning(
                        "restarting publisher path=%s status=%s", self.path, self.process.returncode)
                await self.start()
            try:
                if is_audio:
                    if self.audio_payload_type != payload_type:
                        self.audio_payload_type = payload_type
                        LOGGER.info("JT1078 audio path=%s payloadType=%d", self.path, payload_type)
                    # Already ADTS-framed in write().
                    self.audio_input.write(frame)
                    # If FFmpeg's audio pipeline ever stalls mid-stream
                    # (the same class of internal wedge seen during
                    # startup probing, just happening later), drain()
                    # would otherwise wait forever for backpressure that
                    # will never clear -- silently hanging this
                    # connection's entire packet loop with no error and
                    # no way to recover. Bound it so a stall becomes a
                    # detected, recoverable failure instead.
                    await asyncio.wait_for(
                        self.audio_input.drain(),
                        timeout=float(os.getenv("JT1078_AUDIO_WRITE_TIMEOUT", "2")))
                else:
                    if self.video_payload_type != payload_type:
                        self.video_payload_type = payload_type
                        # Vendors disagree wildly on this field (the Jimi
                        # cameras send 96, the standard says 98 for H.264),
                        # so it is logged next to the codec we actually
                        # configured rather than used to pick one: a new
                        # model streaming H.265 into "-f h264" fails in
                        # ffmpeg with nothing pointing at the cause.
                        LOGGER.info(
                            "JT1078 video path=%s payloadType=%d configuredCodec=%s",
                            self.path, payload_type,
                            os.getenv("JT1078_VIDEO_CODEC", "h264"))
                    self.process.stdin.write(frame)
                    # Same stall as audio above, on the video pipe. Left
                    # unbounded, it hangs the packet loop outside the read
                    # timeout, so the camera's socket is never released and
                    # /status keeps reporting the session as live. Larger
                    # default than audio: one I-frame can fill the pipe on
                    # its own, and ffmpeg's startup probe holds back reads.
                    await asyncio.wait_for(
                        self.process.stdin.drain(),
                        timeout=float(os.getenv("JT1078_VIDEO_WRITE_TIMEOUT", "5")))
                return
            except asyncio.TimeoutError:
                LOGGER.warning(
                    "publisher %s write stalled, restarting path=%s",
                    "audio" if is_audio else "video", self.path)
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                self.process = None
                if attempt:
                    raise
            except (BrokenPipeError, ConnectionResetError, ValueError):
                await self.process.wait()
                LOGGER.warning(
                    "publisher connection closed path=%s status=%s", self.path, self.process.returncode)
                self.process = None
                if attempt:
                    raise

    async def close(self, final=True):
        if final:
            self.closed = True
        if self.grace_handle is not None:
            self.grace_handle.cancel()
            self.grace_handle = None
        self.pending = []
        if self.process is None:
            return
        if self.audio_input:
            self.audio_input.close()
            self.audio_input = None
        if self.process.stdin:
            self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.process.terminate()
            await self.process.wait()


# Tracks the currently-active Publisher per (namespace, imei, channel),
# across all connections. A camera that reconnects without cleanly
# closing its old TCP connection (dropped by a NAT/firewall without a
# FIN/RST) leaves the old handle_connection() coroutine stuck forever in
# reader.read(), so its Connection.close() cleanup never runs and its
# publisher -- and the ffmpeg process it owns -- leaks, silently
# competing with the new connection for the same MediaMTX path. This
# registry lets a new connection evict that stale publisher immediately
# instead of waiting on a TCP-level cleanup that may never come.
ACTIVE_PUBLISHERS = {}


def publisher_status(namespace: str, imei: str, channel: str) -> dict:
    # channel is the *output* channel (post JT1078_CHANNEL_OFFSET), i.e. the
    # one that appears in the MediaMTX/WHEP path -- match Publisher.path
    # directly rather than trying to reverse the offset.
    target = f"{namespace}/{channel}/{imei}"
    publisher = next(
        (p for p in ACTIVE_PUBLISHERS.values() if p.path == target), None)
    now = time.monotonic()

    def age_ms(timestamp):
        return None if timestamp is None else round((now - timestamp) * 1000)

    if publisher is None:
        return {
            "connected": False, "publishing": False, "decided": False,
            "hasVideo": False, "hasAudio": False,
            "videoAgeMs": None, "audioAgeMs": None, "uptimeMs": None,
            "restarts": 0,
        }
    process = publisher.process
    return {
        "connected": True,
        "publishing": process is not None and process.returncode is None,
        "decided": publisher.decided,
        "hasVideo": publisher.last_video_at is not None,
        "hasAudio": publisher.has_audio,
        "videoAgeMs": age_ms(publisher.last_video_at),
        "audioAgeMs": age_ms(publisher.last_audio_at),
        "uptimeMs": age_ms(publisher.started_at),
        "restarts": publisher.restart_count,
    }


async def handle_status(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    # Minimal HTTP/1.1: GET /status/{imei}/{channel}
    #                or GET /status/{namespace}/{imei}/{channel}  (namespace: live|playback)
    body = b"{}"
    status = "404 Not Found"
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=5)
        while True:  # discard headers up to the blank line
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            if line in (b"\r\n", b"\n", b""):
                break
        parts = request_line.split()
        if len(parts) >= 2 and parts[0] == b"GET":
            segments = [s for s in parts[1].split(b"?")[0].decode(
                "ascii", "replace").strip("/").split("/") if s]
            if len(segments) in (3, 4) and segments[0] == "status":
                rest = segments[1:]
                namespace, imei, channel = (
                    ("live", *rest) if len(rest) == 2 else rest)
                if namespace in ("live", "playback"):
                    body = json.dumps(publisher_status(
                        namespace, imei, channel)).encode()
                    status = "200 OK"
        writer.write((
            f"HTTP/1.1 {status}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Cache-Control: no-store\r\n"
            f"X-JT1078-Version: {STATUS_ENDPOINT_VERSION}\r\n"
            "Connection: close\r\n\r\n"
        ).encode() + body)
        await writer.drain()
    except (asyncio.TimeoutError, ConnectionError):
        pass
    finally:
        writer.close()


class Connection:
    def __init__(self, peer, namespace: str):
        self.peer = peer
        self.namespace = namespace
        self.parser = PacketParser(peer)
        self.publishers = {}
        self.fragments = {}
        self.last_sequences = {}

    async def process(self, packet: Packet):
        key = (packet.imei, packet.channel)
        sequence_key = (packet.imei, packet.channel, packet.data_type)
        previous = self.last_sequences.get(sequence_key)
        if previous is not None and packet.sequence != (previous + 1) & 0xFFFF:
            LOGGER.debug(
                "sequence gap peer=%s imei=%s channel=%d dataType=%d expected=%d received=%d",
                self.peer, packet.imei, packet.channel, packet.data_type,
                (previous + 1) & 0xFFFF, packet.sequence)
        self.last_sequences[sequence_key] = packet.sequence

        fragment_key = (packet.imei, packet.channel, packet.timestamp, packet.data_type)
        if packet.fragment_type == 0:
            frame = packet.payload
        elif packet.fragment_type == 1:
            self.fragments[fragment_key] = bytearray(packet.payload)
            return
        elif packet.fragment_type == 3:
            self.fragments.setdefault(fragment_key, bytearray()).extend(packet.payload)
            return
        elif packet.fragment_type == 2:
            fragments = self.fragments.pop(fragment_key, bytearray())
            fragments.extend(packet.payload)
            frame = bytes(fragments)
        else:
            LOGGER.warning("unsupported fragment type %d", packet.fragment_type)
            return

        publisher = self.publishers.get(key)
        if publisher is None:
            registry_key = (self.namespace, *key)
            stale = ACTIVE_PUBLISHERS.get(registry_key)
            if stale is not None:
                LOGGER.warning("replacing stale publisher path=%s", stale.path)
                # Must finish before the new publisher connects: MediaMTX
                # only allows one publisher per path, so starting the new
                # ffmpeg process while the stale one is still tearing down
                # races and can get the new process rejected/killed too.
                await stale.close()
            publisher = Publisher(*key, self.namespace)
            self.publishers[key] = publisher
            ACTIVE_PUBLISHERS[registry_key] = publisher
        await publisher.write(frame, packet.data_type, packet.payload_type)

    async def close(self):
        for key, publisher in self.publishers.items():
            registry_key = (self.namespace, *key)
            if ACTIVE_PUBLISHERS.get(registry_key) is publisher:
                del ACTIVE_PUBLISHERS[registry_key]
        await asyncio.gather(*(publisher.close() for publisher in self.publishers.values()), return_exceptions=True)


async def handle_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter, namespace: str):
    peer = writer.get_extra_info("peername")
    connection = Connection(peer, namespace)
    LOGGER.info("camera connected peer=%s namespace=%s", peer, namespace)
    # Cameras on cellular/NAT networks can drop a connection without ever
    # sending a FIN/RST. Without a read timeout, reader.read() would then
    # block forever, leaking this connection's publisher (and the ffmpeg
    # process it owns) until the process is restarted -- observed in
    # production as a stream going dead mid-session with jt1078 never
    # logging anything about it again.
    read_timeout = float(os.getenv("JT1078_READ_TIMEOUT", "30"))
    try:
        while data := await asyncio.wait_for(reader.read(65536), timeout=read_timeout):
            for packet in connection.parser.feed(data):
                await connection.process(packet)
    except asyncio.TimeoutError:
        LOGGER.warning("camera idle timeout peer=%s namespace=%s", peer, namespace)
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    except Exception:
        LOGGER.exception("camera stream failed peer=%s", peer)
    finally:
        await connection.close()
        writer.close()
        await writer.wait_closed()
        LOGGER.info("camera disconnected peer=%s namespace=%s", peer, namespace)


# Active-safety (T/JSATL12) alarm attachment upload. After a 0x9208 the
# terminal (e.g. iStartek DC600) connects here and sends 0x7e-framed JT/T
# 808 messages - 0x1210 attachment list, 0x1211 file info, 0x1212 upload
# finished - interleaved with file-data packets that reuse the JT1078
# 0x30316364 magic but carry a different header (50-byte file name, offset,
# length). That is why this is not served on the playback port: PacketParser
# would misread the data packets as video. 0x1210 and 0x1211 get a 0x8001;
# 0x1212 gets a 0x9212 (complete, or the missing ranges to resend) - the
# terminal waits for it before sending the next file, and without it the
# DC600 only ever sent file 1 and then resent the whole alarm. Each complete
# file is handed to the upload sink, which files it under
# <terminal>/<alarm identifier>/ - the same folder Traccar's POST /event
# writes the alarm type to.
ATTACHMENT_FILE_NAME_SIZE = 50
ATTACHMENT_DATA_HEADER_SIZE = len(MAGIC) + ATTACHMENT_FILE_NAME_SIZE + 8
ATTACHMENT_ACKED_MESSAGES = (0x1210, 0x1211)
ATTACHMENT_MAX_BYTES = int(os.getenv("JT1078_ATTACHMENT_MAX_BYTES", str(64 * 1024 * 1024)))
ATTACHMENT_SINK_URL = os.getenv("JT1078_ATTACHMENT_SINK_URL", "http://upload:10005/upload")


def jt808_unescape(data: bytes) -> bytes:
    return data.replace(b"\x7d\x02", b"\x7e").replace(b"\x7d\x01", b"\x7d")


def jt808_escape(data: bytes) -> bytes:
    return data.replace(b"\x7d", b"\x7d\x01").replace(b"\x7e", b"\x7d\x02")


def jt808_checksum(data: bytes) -> int:
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


@dataclass
class Jt808Message:
    message_id: int
    serial: int
    terminal: bytes
    version: int | None
    body: bytes


def parse_jt808(frame: bytes) -> Jt808Message | None:
    """Parse one unescaped JT/T 808 frame (without the 0x7e delimiters)."""
    if len(frame) < 13 or jt808_checksum(frame[:-1]) != frame[-1]:
        return None
    message_id = int.from_bytes(frame[0:2], "big")
    attributes = int.from_bytes(frame[2:4], "big")
    offset = 4
    version = None
    if attributes & 0x4000:
        version = frame[offset]
        terminal = frame[offset + 1:offset + 11]
        offset += 11
    else:
        terminal = frame[offset:offset + 6]
        offset += 6
    serial = int.from_bytes(frame[offset:offset + 2], "big")
    offset += 2
    if attributes & 0x2000:
        offset += 4
    body = frame[offset:offset + (attributes & 0x03FF)]
    return Jt808Message(message_id, serial, terminal, version, body)


def build_jt808(message_id: int, serial: int, terminal: bytes, version: int | None, body: bytes) -> bytes:
    attributes = len(body)
    header = bytearray(message_id.to_bytes(2, "big"))
    if version is not None:
        header.extend((attributes | 0x4000).to_bytes(2, "big"))
        header.append(version)
    else:
        header.extend(attributes.to_bytes(2, "big"))
    header.extend(terminal)
    header.extend(serial.to_bytes(2, "big"))
    frame = bytes(header) + body
    return b"\x7e" + jt808_escape(frame + bytes((jt808_checksum(frame),))) + b"\x7e"


def general_response(message: Jt808Message, serial: int) -> bytes:
    body = message.serial.to_bytes(2, "big") + message.message_id.to_bytes(2, "big") + b"\x00"
    return build_jt808(0x8001, serial, message.terminal, message.version, body)


class AttachmentParser:
    """Split an attachment-upload stream into JT/T 808 frames and file-data packets."""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list:
        self.buffer.extend(data)
        items = []
        while self.buffer:
            if self.buffer[0] == 0x7e:
                end = self.buffer.find(b"\x7e", 1)
                if end < 0:
                    break
                if end == 1:
                    # Back-to-back delimiters: the second one opens the next frame.
                    del self.buffer[:1]
                    continue
                items.append(("message", jt808_unescape(bytes(self.buffer[1:end]))))
                del self.buffer[:end + 1]
            elif self.buffer[:len(MAGIC)] == MAGIC[:len(self.buffer)]:
                if len(self.buffer) < ATTACHMENT_DATA_HEADER_SIZE:
                    break
                name_end = len(MAGIC) + ATTACHMENT_FILE_NAME_SIZE
                name = bytes(self.buffer[len(MAGIC):name_end]).rstrip(b"\x00").decode("ascii", "replace")
                data_offset = int.from_bytes(self.buffer[name_end:name_end + 4], "big")
                length = int.from_bytes(self.buffer[name_end + 4:name_end + 8], "big")
                if len(self.buffer) < ATTACHMENT_DATA_HEADER_SIZE + length:
                    break
                payload = bytes(self.buffer[ATTACHMENT_DATA_HEADER_SIZE:ATTACHMENT_DATA_HEADER_SIZE + length])
                items.append(("data", name, data_offset, payload))
                del self.buffer[:ATTACHMENT_DATA_HEADER_SIZE + length]
            else:
                start = min((i for i in (self.buffer.find(b"\x7e"), self.buffer.find(MAGIC)) if i >= 0), default=-1)
                LOGGER.warning("attachment stream: skipping %d unrecognized bytes head=%s",
                               len(self.buffer) if start < 0 else start, bytes(self.buffer[:32]).hex())
                if start < 0:
                    del self.buffer[:-(len(MAGIC) - 1)]
                    break
                del self.buffer[:start]
        return items


def parse_attachment_file(body: bytes) -> tuple[str, int, int] | None:
    """0x1211 / 0x1212 body: name length, name, file type, file size."""
    if not body:
        return None
    end = 1 + body[0]
    if len(body) < end + 5:
        return None
    name = body[1:end].decode("ascii", "replace")
    return name, body[end], int.from_bytes(body[end + 1:end + 5], "big")


def missing_ranges(ranges: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """(offset, length) gaps in [0, size) not covered by the received ranges."""
    missing = []
    position = 0
    for offset, length in sorted(ranges):
        if offset > position:
            missing.append((position, offset - position))
        position = max(position, offset + length)
    if position < size:
        missing.append((position, size - position))
    return missing


def upload_complete_response(message: Jt808Message, serial: int, name: str, file_type: int,
                             missing: list[tuple[int, int]]) -> bytes:
    """0x9212: result 0 = complete, 1 = resend the listed (offset, length) ranges."""
    encoded = name.encode("ascii", "replace")
    missing = missing[:255]
    body = bytearray((len(encoded),))
    body.extend(encoded)
    body.extend((file_type, 1 if missing else 0, len(missing)))
    for offset, length in missing:
        body.extend(offset.to_bytes(4, "big") + length.to_bytes(4, "big"))
    return build_jt808(0x9212, serial, message.terminal, message.version, bytes(body))


def terminal_number(message: Jt808Message) -> str:
    """The BCD phone number the device is registered under (its Traccar uniqueId)."""
    number = message.terminal.hex()
    if message.version is not None:
        # 2019 headers zero-pad the number to 20 digits.
        number = number.lstrip("0").rjust(12, "0")
    return number


def post_attachment(stored_name: str, data: bytes) -> str:
    request = urllib.request.Request(
        ATTACHMENT_SINK_URL, data=data, method="POST",
        headers={"Content-Type": "application/octet-stream",
                 "Content-Disposition": f'attachment; filename="{stored_name}"'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(512).decode("utf-8", "replace")


async def store_attachment(terminal: str, identifier: str | None, name: str, data: bytes):
    # The sink files <terminal>_<identifier>_* under <terminal>/<identifier>/.
    stored_name = f"{terminal}_{identifier or 'unknown'}_{name}"
    try:
        reply = await asyncio.to_thread(post_attachment, stored_name, data)
        LOGGER.info("attachment stored file=%s bytes=%d sink=%s", stored_name, len(data), reply)
    except Exception:
        LOGGER.exception("attachment store failed file=%s bytes=%d", stored_name, len(data))


async def handle_attachment_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    parser = AttachmentParser()
    files = {}
    identifier = None
    serial = 0
    LOGGER.info("attachment terminal connected peer=%s", peer)
    read_timeout = float(os.getenv("JT1078_READ_TIMEOUT", "30"))

    def received():
        return {name: len(entry["data"]) for name, entry in files.items()}

    try:
        while data := await asyncio.wait_for(reader.read(65536), timeout=read_timeout):
            for item in parser.feed(data):
                if item[0] == "data":
                    _, name, data_offset, payload = item
                    entry = files.setdefault(name, {"data": bytearray(), "ranges": []})
                    if data_offset + len(payload) > ATTACHMENT_MAX_BYTES:
                        LOGGER.warning("attachment data beyond %d bytes dropped peer=%s file=%s offset=%d",
                                       ATTACHMENT_MAX_BYTES, peer, name, data_offset)
                        continue
                    if len(entry["data"]) < data_offset + len(payload):
                        entry["data"].extend(bytes(data_offset + len(payload) - len(entry["data"])))
                    entry["data"][data_offset:data_offset + len(payload)] = payload
                    entry["ranges"].append((data_offset, len(payload)))
                    LOGGER.debug("attachment data peer=%s file=%s offset=%d length=%d",
                                 peer, name, data_offset, len(payload))
                    continue
                message = parse_jt808(item[1])
                if message is None:
                    LOGGER.warning("attachment stream: bad JT808 frame peer=%s frame=%s", peer, item[1].hex())
                    continue
                LOGGER.info("attachment message 0x%04x peer=%s terminal=%s serial=%d body=%s",
                            message.message_id, peer, message.terminal.hex(), message.serial, message.body.hex())
                if message.message_id == 0x1210 and len(message.body) >= 23:
                    # terminal id[7], then the 16-byte alarm identifier from the 0x64/0x65 alarm.
                    identifier = message.body[7:23].hex()
                if message.message_id in ATTACHMENT_ACKED_MESSAGES:
                    writer.write(general_response(message, serial))
                    serial = (serial + 1) & 0xFFFF
                    await writer.drain()
                if message.message_id != 0x1212:
                    continue
                info = parse_attachment_file(message.body)
                if info is None:
                    LOGGER.warning("attachment stream: unparseable 0x1212 peer=%s body=%s", peer, message.body.hex())
                    continue
                name, file_type, size = info
                entry = files.get(name, {"data": bytearray(), "ranges": []})
                missing = missing_ranges(entry["ranges"], min(size, ATTACHMENT_MAX_BYTES))
                LOGGER.info("attachment upload finished peer=%s file=%s size=%d missing=%s bytes received per file=%s",
                            peer, name, size, missing, received())
                writer.write(upload_complete_response(message, serial, name, file_type, missing))
                serial = (serial + 1) & 0xFFFF
                await writer.drain()
                if not missing:
                    await store_attachment(terminal_number(message), identifier, name, bytes(entry["data"][:size]))
    except asyncio.TimeoutError:
        LOGGER.warning("attachment terminal idle timeout peer=%s", peer)
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    except Exception:
        LOGGER.exception("attachment stream failed peer=%s", peer)
    finally:
        writer.close()
        await writer.wait_closed()
        LOGGER.info("attachment terminal disconnected peer=%s bytes received per file=%s", peer, received())


async def main():
    live_port = int(os.getenv("JT1078_LIVE_PORT", "10002"))
    playback_port = int(os.getenv("JT1078_PLAYBACK_PORT", "10003"))
    status_port = int(os.getenv("JT1078_STATUS_PORT", "10004"))
    attachment_port = int(os.getenv("JT1078_ATTACHMENT_PORT", "10006"))
    live_server = await asyncio.start_server(
        lambda reader, writer: handle_connection(reader, writer, "live"),
        "0.0.0.0", live_port)
    playback_server = await asyncio.start_server(
        lambda reader, writer: handle_connection(reader, writer, "playback"),
        "0.0.0.0", playback_port)
    status_server = await asyncio.start_server(handle_status, "0.0.0.0", status_port)
    attachment_server = await asyncio.start_server(handle_attachment_connection, "0.0.0.0", attachment_port)
    LOGGER.info("JT1078 live receiver listening on port %d", live_port)
    LOGGER.info("JT1078 playback receiver listening on port %d", playback_port)
    LOGGER.info("JT1078 status endpoint listening on port %d", status_port)
    LOGGER.info("active-safety attachment receiver listening on port %d", attachment_port)
    async with live_server, playback_server, status_server, attachment_server:
        await asyncio.gather(
            live_server.serve_forever(),
            playback_server.serve_forever(),
            status_server.serve_forever(),
            attachment_server.serve_forever())


if __name__ == "__main__":
    asyncio.run(main())
