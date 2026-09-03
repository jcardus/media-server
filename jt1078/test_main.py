import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main
from main import MAGIC, Connection, Packet, PacketParser, Publisher, audio_input_options, decode_terminal_id, \
    prepare_audio_frame


def packet(payload, data_type=0, fragment_type=0, sequence=1, channel=1, timestamp=123, payload_type=96):
    header = bytearray(MAGIC)
    header.extend((0x80, 0x80 | payload_type))
    header.extend(sequence.to_bytes(2, "big"))
    header.extend(bytes.fromhex("4e3a0b712725"))
    header.append(channel)
    header.append((data_type << 4) | fragment_type)
    header.extend(timestamp.to_bytes(8, "big"))
    if data_type <= 2:
        header.extend((0, 0, 0, 0))
    header.extend(len(payload).to_bytes(2, "big"))
    return bytes(header) + payload


class PacketParserTest(unittest.TestCase):
    def test_audio_input_options(self):
        self.assertEqual(["-f", "aac"], audio_input_options("aac", "8000"))
        self.assertEqual(
            ["-f", "alaw", "-ar", "8000", "-ac", "1"],
            audio_input_options("alaw", "8000"))

    def test_prepare_audio_frame(self):
        raw = b"\x21\x10\x04\x60"
        framed = prepare_audio_frame(raw, "aac", "8000")
        self.assertEqual(bytes.fromhex("fff16c40017ffc") + raw, framed)
        self.assertIs(framed, prepare_audio_frame(framed, "aac", "8000"))
        self.assertIs(raw, prepare_audio_frame(raw, "alaw", "8000"))

    def test_decode_terminal_id(self):
        self.assertEqual("860112070346616", decode_terminal_id(bytes.fromhex("4e3a0b712725")))
        self.assertEqual("013345678906", decode_terminal_id(bytes.fromhex("013345678906")))

    def test_fragmented_network_input(self):
        data = packet(b"\x00\x00\x00\x01\x67\x64")
        parser = PacketParser()
        self.assertEqual([], parser.feed(data[:17]))
        packets = parser.feed(data[17:])
        self.assertEqual(1, len(packets))
        self.assertEqual("860112070346616", packets[0].imei)
        self.assertEqual(1, packets[0].channel)
        self.assertEqual(96, packets[0].payload_type)
        self.assertEqual(b"\x00\x00\x00\x01\x67\x64", packets[0].payload)

    def test_audio_packet(self):
        packets = PacketParser().feed(packet(b"audio", data_type=3, payload_type=6))
        self.assertEqual(1, len(packets))
        self.assertEqual(3, packets[0].data_type)
        self.assertEqual(6, packets[0].payload_type)
        self.assertEqual(b"audio", packets[0].payload)

    def test_multiple_packets(self):
        parser = PacketParser()
        packets = parser.feed(packet(b"a", sequence=1) + packet(b"b", sequence=2))
        self.assertEqual([b"a", b"b"], [value.payload for value in packets])


class PublisherTest(unittest.IsolatedAsyncioTestCase):
    async def test_starts_video_only_without_blocking_on_audio(self):
        publisher = Publisher("860112070346616", 1, "live")
        with patch("main.asyncio.create_subprocess_exec", new=AsyncMock()) as create:
            await publisher.start()
        args = create.call_args.args
        self.assertIn("-an", args)
        self.assertEqual((), create.call_args.kwargs["pass_fds"])
        self.assertIsNone(publisher.audio_input)

    async def test_adds_audio_input_once_audio_is_detected(self):
        publisher = Publisher("860112070346616", 1, "live")
        with patch("main.asyncio.create_subprocess_exec", new=AsyncMock()) as create:
            publisher.has_audio = True
            await publisher.start()
        args = create.call_args.args
        self.assertNotIn("-an", args)
        self.assertIn("-c:a", args)
        self.assertNotEqual((), create.call_args.kwargs["pass_fds"])
        self.assertIsNotNone(publisher.audio_input)
        # The audio input must not rely on FFmpeg's default probesize
        # (5MB), which low-bitrate voice audio could take minutes to
        # reach, stalling the whole publisher.
        video_input_index = args.index("pipe:0")
        audio_input_index = args.index("-i", video_input_index + 1)
        probesize_index = args.index("-probesize", video_input_index, audio_input_index)
        self.assertLess(int(args[probesize_index + 1]), 1_000_000)
        publisher.audio_input.close()

    async def test_buffers_video_until_audio_arrives_then_commits_once(self):
        publisher = Publisher("860112070346616", 1, "live")
        process = MagicMock()
        process.returncode = None
        process.stdin = MagicMock()
        process.stdin.drain = AsyncMock()

        async def fake_start():
            publisher.process = process
            publisher.audio_input = MagicMock() if publisher.has_audio else None

        with patch.object(Publisher, "start", new=AsyncMock(side_effect=fake_start)) as start:
            await publisher.write(b"\x00\x00\x00\x01video", data_type=0, payload_type=96)
            start.assert_not_called()
            self.assertFalse(publisher.decided)

            await publisher.write(b"audio", data_type=3, payload_type=19)

        start.assert_called_once()
        self.assertTrue(publisher.decided)
        self.assertTrue(publisher.has_audio)
        process.stdin.write.assert_called_once_with(b"\x00\x00\x00\x01video")
        publisher.audio_input.write.assert_called_once()
        self.assertEqual([], publisher.pending)

    async def test_commits_video_only_after_grace_period(self):
        publisher = Publisher("860112070346616", 1, "live")
        process = MagicMock()
        process.returncode = None
        process.stdin = MagicMock()
        process.stdin.drain = AsyncMock()

        async def fake_start():
            publisher.process = process
            publisher.audio_input = MagicMock() if publisher.has_audio else None

        with patch.dict("os.environ", {"JT1078_AUDIO_GRACE_SECONDS": "0.01"}):
            with patch.object(Publisher, "start", new=AsyncMock(side_effect=fake_start)) as start:
                await publisher.write(b"\x00\x00\x00\x01video", data_type=0, payload_type=96)
                start.assert_not_called()
                await asyncio.sleep(0.05)

        start.assert_called_once()
        self.assertTrue(publisher.decided)
        self.assertFalse(publisher.has_audio)
        process.stdin.write.assert_called_once_with(b"\x00\x00\x00\x01video")


def make_packet(imei="860112070346616", channel=1, payload=b"\x00\x00\x00\x01video") -> Packet:
    return Packet(
        sequence=1, imei=imei, channel=channel, payload_type=96,
        data_type=0, fragment_type=0, timestamp=123, payload=payload)


class ConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_replaces_stale_publisher_from_a_dropped_connection(self):
        # A camera that reconnects without the OS ever closing its old
        # TCP socket (dropped silently by a NAT/firewall) leaves the old
        # Connection's publisher running forever unless the new
        # connection evicts it.
        main.ACTIVE_PUBLISHERS.clear()
        with patch.object(Publisher, "close", new=AsyncMock()) as close:
            first = Connection(("1.2.3.4", 1), "live")
            await first.process(make_packet())
            first_publisher = first.publishers[("860112070346616", 1)]

            second = Connection(("1.2.3.4", 2), "live")
            await second.process(make_packet())
            second_publisher = second.publishers[("860112070346616", 1)]
            await asyncio.sleep(0)

        close.assert_called_once()
        self.assertIsNot(first_publisher, second_publisher)
        self.assertIs(main.ACTIVE_PUBLISHERS[("live", "860112070346616", 1)], second_publisher)


if __name__ == "__main__":
    unittest.main()
