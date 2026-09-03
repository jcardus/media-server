import asyncio
import logging
import os
from dataclasses import dataclass


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("jt1078")

MAGIC = b"\x30\x31\x63\x64"
MAX_PAYLOAD_SIZE = 65535
AAC_SAMPLE_RATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000)


def audio_input_options(codec: str, sample_rate: str) -> list[str]:
    options = ["-f", codec]
    if codec in {"alaw", "mulaw", "s16be", "s16le"}:
        options.extend(("-ar", sample_rate, "-ac", "1"))
    return options


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
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[Packet]:
        self.buffer.extend(data)
        packets = []
        while True:
            start = self.buffer.find(MAGIC)
            if start < 0:
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
        return packets


class Publisher:
    def __init__(self, imei: str, channel: int, namespace: str):
        channel_offset = int(os.getenv("JT1078_CHANNEL_OFFSET", "1"))
        output_channel = max(0, channel - channel_offset)
        self.path = f"{namespace}/{output_channel}/{imei}"
        self.process = None
        self.audio_input = None
        self.audio_payload_type = None
        self.audio_codec = os.getenv("JT1078_AUDIO_CODEC", "aac")
        self.audio_sample_rate = os.getenv("JT1078_AUDIO_SAMPLE_RATE", "8000")
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

    async def start(self):
        if self.audio_input:
            self.audio_input.close()
            self.audio_input = None
        target = f"rtsp://mediamtx:8554/rtc/{self.path}"
        codec = os.getenv("JT1078_VIDEO_CODEC", "h264")
        frame_rate = os.getenv("JT1078_VIDEO_FRAME_RATE", "25")
        timestamp_step = round(90000 / float(frame_rate))
        timestamp_filter = (
            f"setts=pts=N*{timestamp_step}:dts=N*{timestamp_step}:"
            f"duration={timestamp_step}:time_base=1/90000"
        )
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", os.getenv("FFMPEG_LOG_LEVEL", "warning"),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            "-f", codec,
            "-i", "pipe:0",
        ]
        pass_fds = ()
        audio_read = None
        audio_write = None
        if self.has_audio:
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
                # Video runs on a synthetic, frame-count-based clock (see
                # the setts bitstream filter below) with no relation to
                # real elapsed time; left alone, audio's own AAC-sample-
                # count-based clock has no shared reference with it. Any
                # mismatch between each clock's assumed rate and the
                # camera's actual delivery rate accumulates until the
                # two streams' timelines have drifted apart enough that
                # the RTSP muxer -- which interleaves packets from both
                # streams in timestamp order -- stalls waiting for one
                # side to "catch up" (observed in production: a rock-
                # solid ~20s into every audio-enabled stream). Anchor
                # audio to real wall-clock time so it can't drift
                # indefinitely away from video's roughly-real-time pace.
                "-use_wallclock_as_timestamps", "1",
                *audio_options,
                "-i", f"pipe:{audio_read}",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "libopus",
                "-b:a", "32k",
                "-af", "aresample=async=1000",
            ]
            pass_fds = (audio_read,)
        else:
            args += ["-an", "-c:v", "copy"]
        args += [
            "-bsf:v", timestamp_filter,
        ]
        if self.has_audio:
            # The RTSP muxer interleaves packets from both streams in
            # timestamp order by default, buffering whichever stream is
            # "ahead" until the other catches up. With H.264 video at
            # ~1.2Mbps against ~32kbps Opus audio -- and neither stream
            # on a clock that exactly matches the camera's real delivery
            # rate -- that wait can grow unbounded and never resolve.
            # Disable strict interleaving so the muxer writes packets as
            # they arrive instead of holding one stream hostage to the
            # other.
            args += ["-max_interleave_delta", "0"]
        args += [
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            target,
        ]
        LOGGER.info(
            "starting publisher path=rtc/%s videoCodec=%s frameRate=%s audio=%s",
            self.path, codec, frame_rate, self.has_audio)
        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            pass_fds=pass_fds,
        )
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

    async def write(self, frame: bytes, data_type: int, payload_type: int):
        is_audio = data_type == 3
        if not is_audio and not frame.startswith((b"\x00\x00\x01", b"\x00\x00\x00\x01")):
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
                await self.close()

        await self._deliver(frame, data_type, payload_type)

    def _on_grace_expired(self):
        self.grace_handle = None
        self.grace_task = asyncio.ensure_future(self._commit())

    async def _commit(self):
        if self.decided:
            return
        self.decided = True
        pending, self.pending = self.pending, []
        await self.start()
        for frame, data_type, payload_type in pending:
            await self._deliver(frame, data_type, payload_type)

    async def _deliver(self, frame: bytes, data_type: int, payload_type: int):
        is_audio = data_type == 3
        for attempt in range(2):
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
                    self.audio_input.write(prepare_audio_frame(
                        frame, self.audio_codec, self.audio_sample_rate))
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
                    self.process.stdin.write(frame)
                    await self.process.stdin.drain()
                return
            except asyncio.TimeoutError:
                LOGGER.warning("publisher audio write stalled, restarting path=%s", self.path)
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

    async def close(self):
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


class Connection:
    def __init__(self, peer, namespace: str):
        self.peer = peer
        self.namespace = namespace
        self.parser = PacketParser()
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


async def main():
    live_port = int(os.getenv("JT1078_LIVE_PORT", "10002"))
    playback_port = int(os.getenv("JT1078_PLAYBACK_PORT", "10003"))
    live_server = await asyncio.start_server(
        lambda reader, writer: handle_connection(reader, writer, "live"),
        "0.0.0.0", live_port)
    playback_server = await asyncio.start_server(
        lambda reader, writer: handle_connection(reader, writer, "playback"),
        "0.0.0.0", playback_port)
    LOGGER.info("JT1078 live receiver listening on port %d", live_port)
    LOGGER.info("JT1078 playback receiver listening on port %d", playback_port)
    async with live_server, playback_server:
        await asyncio.gather(
            live_server.serve_forever(),
            playback_server.serve_forever())


if __name__ == "__main__":
    asyncio.run(main())
