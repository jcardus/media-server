import asyncio
import logging
import os
from dataclasses import dataclass


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("jt1078")

MAGIC = b"\x30\x31\x63\x64"
MAX_PAYLOAD_SIZE = 65535


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
                data_type=data_type,
                fragment_type=fragment_type,
                timestamp=timestamp,
                payload=bytes(self.buffer[header_size:packet_size]),
            ))
            del self.buffer[:packet_size]
        return packets


class Publisher:
    def __init__(self, imei: str, channel: int):
        channel_offset = int(os.getenv("JT1078_CHANNEL_OFFSET", "1"))
        output_channel = max(0, channel - channel_offset)
        self.path = f"live/{output_channel}/{imei}"
        self.process = None

    async def start(self):
        target = f"rtsp://mediamtx:8554/{self.path}"
        codec = os.getenv("JT1078_VIDEO_CODEC", "h264")
        LOGGER.info("starting publisher path=%s codec=%s", self.path, codec)
        self.process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel", os.getenv("FFMPEG_LOG_LEVEL", "warning"),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            "-f", codec,
            "-i", "pipe:0",
            "-an",
            "-c:v", "copy",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            target,
            stdin=asyncio.subprocess.PIPE,
        )

    async def write(self, frame: bytes):
        if self.process is None:
            await self.start()
        if self.process.returncode is not None:
            raise RuntimeError(f"FFmpeg exited with status {self.process.returncode}")
        if not frame.startswith((b"\x00\x00\x01", b"\x00\x00\x00\x01")):
            frame = b"\x00\x00\x00\x01" + frame
        self.process.stdin.write(frame)
        await self.process.stdin.drain()

    async def close(self):
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.process.terminate()
            await self.process.wait()


class Connection:
    def __init__(self, peer):
        self.peer = peer
        self.parser = PacketParser()
        self.publishers = {}
        self.fragments = {}
        self.last_sequences = {}

    async def process(self, packet: Packet):
        key = (packet.imei, packet.channel)
        previous = self.last_sequences.get(key)
        if previous is not None and packet.sequence != (previous + 1) & 0xFFFF:
            LOGGER.warning(
                "sequence gap peer=%s imei=%s channel=%d expected=%d received=%d",
                self.peer, packet.imei, packet.channel, (previous + 1) & 0xFFFF, packet.sequence)
        self.last_sequences[key] = packet.sequence

        if packet.data_type == 3:
            return  # Audio is intentionally omitted; browsers receive video-only WebRTC.

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
            publisher = Publisher(*key)
            self.publishers[key] = publisher
        await publisher.write(frame)

    async def close(self):
        await asyncio.gather(*(publisher.close() for publisher in self.publishers.values()), return_exceptions=True)


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    connection = Connection(peer)
    LOGGER.info("camera connected peer=%s", peer)
    try:
        while data := await reader.read(65536):
            for packet in connection.parser.feed(data):
                await connection.process(packet)
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    except Exception:
        LOGGER.exception("camera stream failed peer=%s", peer)
    finally:
        await connection.close()
        writer.close()
        await writer.wait_closed()
        LOGGER.info("camera disconnected peer=%s", peer)


async def main():
    port = int(os.getenv("JT1078_PORT", "10002"))
    server = await asyncio.start_server(handle_connection, "0.0.0.0", port)
    LOGGER.info("JT1078 TCP receiver listening on %s", ", ".join(str(sock.getsockname()) for sock in server.sockets))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
