import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main
from main import MAGIC, Connection, Packet, PacketParser, Publisher, audio_codec_for_payload_type, \
    audio_input_options, decode_terminal_id, prepare_audio_frame


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

    def test_audio_codec_for_payload_type(self):
        # Payload type 19 is Jimi's own vendor extension for AAC; it is the
        # only type this codebase has ever confirmed as AAC.
        self.assertEqual(("aac", "8000"), audio_codec_for_payload_type(19))
        # Everything else - including an iStartek DC600's payload type 6 -
        # falls back to alaw, the standard's own G.711A baseline codec and
        # this project's original default before the Jimi cameras arrived
        # (see git history), not the Jimi-specific AAC default.
        self.assertEqual(("alaw", "8000"), audio_codec_for_payload_type(6))
        self.assertEqual(("alaw", "8000"), audio_codec_for_payload_type(0))

    def test_prepare_audio_frame(self):
        raw = b"\x21\x10\x04\x60"
        framed = prepare_audio_frame(raw, "aac", "8000")
        self.assertEqual(bytes.fromhex("fff16c40017ffc") + raw, framed)
        self.assertIs(framed, prepare_audio_frame(framed, "aac", "8000"))
        self.assertIs(raw, prepare_audio_frame(raw, "alaw", "8000"))

    def test_decode_terminal_id(self):
        self.assertEqual("860112070346616", decode_terminal_id(bytes.fromhex("4e3a0b712725")))
        self.assertEqual("013345678906", decode_terminal_id(bytes.fromhex("013345678906")))
        # An iStartek DC600 reports a 12-digit JT/T 808 terminal id rather
        # than a 15-digit IMEI: it must pass straight through, without the
        # luhn expansion the Jimi units' packed serial needs. The id this
        # yields is what appears in the MediaMTX path the frontend opens.
        self.assertEqual("806076813171", decode_terminal_id(bytes.fromhex("806076813171")))

    def test_logs_a_stream_that_never_shows_the_frame_magic(self):
        # A device speaking some other framing on the JT1078 port (a JT/T
        # 808 0x7e message, a vendor handshake) used to be discarded in
        # silence, indistinguishable from a camera that connected and sent
        # nothing. The head of the stream has to reach the log.
        parser = PacketParser(("1.2.3.4", 5))
        with self.assertLogs(main.LOGGER, level="WARNING") as logs:
            self.assertEqual([], parser.feed(b"\x7e\x02\x00" + bytes(30)))
        self.assertIn("no JT1078 frame magic", logs.output[0])
        self.assertIn("7e0200", logs.output[0])
        self.assertIn("1.2.3.4", logs.output[0])
        # Once per connection, however long the device keeps talking.
        self.assertEqual([], parser.feed(bytes(64)))
        self.assertEqual(1, len(logs.output))

    def test_stays_quiet_on_a_short_read_and_on_a_valid_stream(self):
        # A camera's first read can be a couple of bytes, and a valid
        # stream's tail is a partial packet on nearly every read; neither
        # is an unrecognized protocol.
        parser = PacketParser()
        parser.feed(MAGIC[:2])
        self.assertFalse(parser.unrecognized_logged)

        parser = PacketParser()
        self.assertEqual(1, len(parser.feed(packet(b"\x00\x00\x00\x01\x67\x64"))))
        parser.feed(b"\x11" * 32)
        self.assertFalse(parser.unrecognized_logged)

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
            publisher.audio_codec = "aac"
            publisher.audio_sample_rate = "8000"
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

    async def test_video_input_is_wallclock_timestamped_not_synthetic(self):
        # Video carries no timestamps of its own; a synthetic frame-counter
        # clock drifts against audio until the RTSP muxer starves. Stamp it
        # from the wall clock so both streams share a real-time pace.
        publisher = Publisher("860112070346616", 1, "live")
        publisher.has_audio = True
        publisher.audio_codec = "aac"
        publisher.audio_sample_rate = "8000"
        with patch("main.asyncio.create_subprocess_exec", new=AsyncMock()) as create:
            await publisher.start()
        args = create.call_args.args
        wallclock_index = args.index("-use_wallclock_as_timestamps")
        self.assertLess(wallclock_index, args.index("pipe:0"))
        self.assertNotIn("-bsf:v", args)
        self.assertFalse(any("setts" in str(a) for a in args))
        publisher.audio_input.close()

    async def test_picks_audio_codec_from_the_first_audio_packets_payload_type(self):
        # A non-Jimi camera reporting a payload type other than 19 must not
        # be fed through the AAC ADTS framer meant for Jimi's audio -- that
        # produces bytes an AAC decoder reads as valid-looking garbage
        # (confirmed against production ffmpeg logs for payload type 6).
        publisher = Publisher("806076813171", 1, "live")
        process = MagicMock()
        process.returncode = None
        process.stdin = MagicMock()
        process.stdin.drain = AsyncMock()

        async def fake_start():
            publisher.process = process
            publisher.audio_input = MagicMock()
            publisher.audio_input.drain = AsyncMock()

        with patch.object(Publisher, "start", new=AsyncMock(side_effect=fake_start)):
            await publisher.write(b"\x00\x00\x00\x01video", data_type=0, payload_type=96)
            await publisher.write(b"g711a-frame", data_type=3, payload_type=6)

        self.assertEqual("alaw", publisher.audio_codec)
        # Unlike AAC, alaw is never ADTS-framed -- the raw frame must reach
        # ffmpeg unchanged.
        publisher.audio_input.write.assert_called_once_with(b"g711a-frame")

    async def test_buffers_video_until_audio_arrives_then_commits_once(self):
        publisher = Publisher("860112070346616", 1, "live")
        process = MagicMock()
        process.returncode = None
        process.stdin = MagicMock()
        process.stdin.drain = AsyncMock()

        def make_audio_input():
            audio_input = MagicMock()
            audio_input.drain = AsyncMock()
            return audio_input

        async def fake_start():
            publisher.process = process
            publisher.audio_input = make_audio_input() if publisher.has_audio else None

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

    async def test_restarts_publisher_when_audio_write_stalls(self):
        # If FFmpeg's audio pipeline wedges mid-stream, drain() must not
        # be allowed to hang forever -- that would silently freeze this
        # connection's entire packet loop with no error and no recovery.
        publisher = Publisher("860112070346616", 1, "live")
        publisher.decided = True
        publisher.has_audio = True
        publisher.audio_codec = "aac"
        publisher.audio_sample_rate = "8000"

        stalled_process = MagicMock()
        stalled_process.returncode = None
        stalled_process.terminate = MagicMock()
        stalled_process.wait = AsyncMock(return_value=None)

        async def hang():
            await asyncio.sleep(3600)

        stalled_audio_input = MagicMock()
        stalled_audio_input.drain = AsyncMock(side_effect=hang)

        healthy_process = MagicMock()
        healthy_process.returncode = None
        healthy_audio_input = MagicMock()
        healthy_audio_input.drain = AsyncMock()

        starts = []

        async def fake_start():
            if not starts:
                starts.append(stalled_process)
                publisher.process = stalled_process
                publisher.audio_input = stalled_audio_input
            else:
                publisher.process = healthy_process
                publisher.audio_input = healthy_audio_input

        with patch.dict("os.environ", {"JT1078_AUDIO_WRITE_TIMEOUT": "0.01"}):
            with patch.object(Publisher, "start", new=AsyncMock(side_effect=fake_start)):
                await publisher.write(b"audio", data_type=3, payload_type=19)

        stalled_process.terminate.assert_called_once()
        self.assertIs(publisher.process, healthy_process)
        healthy_audio_input.write.assert_called_once()

    async def test_restarts_publisher_when_video_write_stalls(self):
        # An unbounded video drain hangs the packet loop outside the read
        # timeout, so the camera's socket is never released.
        publisher = Publisher("806076813171", 2, "playback")
        publisher.decided = True

        async def hang():
            await asyncio.sleep(3600)

        stalled_process = MagicMock()
        stalled_process.returncode = None
        stalled_process.terminate = MagicMock()
        stalled_process.wait = AsyncMock(return_value=None)
        stalled_process.stdin = MagicMock()
        stalled_process.stdin.drain = AsyncMock(side_effect=hang)

        healthy_process = MagicMock()
        healthy_process.returncode = None
        healthy_process.stdin = MagicMock()
        healthy_process.stdin.drain = AsyncMock()

        async def fake_start():
            publisher.process = healthy_process

        publisher.process = stalled_process
        with patch.dict("os.environ", {"JT1078_VIDEO_WRITE_TIMEOUT": "0.01"}):
            with patch.object(Publisher, "start", new=AsyncMock(side_effect=fake_start)):
                with self.assertLogs(main.LOGGER, level="WARNING") as logs:
                    await publisher.write(b"\x00\x00\x00\x01video", data_type=0, payload_type=98)

        stalled_process.terminate.assert_called_once()
        self.assertIs(publisher.process, healthy_process)
        healthy_process.stdin.write.assert_called_once_with(b"\x00\x00\x00\x01video")
        self.assertTrue(any("video write stalled" in line for line in logs.output))

    async def test_closed_publisher_does_not_respawn_ffmpeg(self):
        # An evicted publisher whose write then fails with a broken pipe
        # must not start a new ffmpeg on the path the replacing
        # connection is taking over.
        publisher = Publisher("806076813171", 2, "playback")
        publisher.decided = True
        process = MagicMock()
        process.returncode = None
        process.stdin = MagicMock()
        process.wait = AsyncMock(return_value=None)
        publisher.process = process

        async def evicted_mid_write():
            await publisher.close()
            raise BrokenPipeError

        process.stdin.drain = AsyncMock(side_effect=evicted_mid_write)
        with patch.object(Publisher, "start", new=AsyncMock()) as start:
            await publisher.write(b"\x00\x00\x00\x01video", data_type=0, payload_type=98)

        self.assertTrue(publisher.closed)
        start.assert_not_called()

    async def test_audio_restart_in_place_does_not_mark_publisher_closed(self):
        publisher = Publisher("806076813171", 1, "live")
        publisher.decided = True
        old_process = MagicMock(returncode=None)
        old_process.stdin = MagicMock()

        async def exits():
            old_process.returncode = 0

        old_process.wait = AsyncMock(side_effect=exits)
        publisher.process = old_process

        async def fake_start():
            publisher.process = MagicMock(returncode=None)
            publisher.process.stdin = MagicMock()
            publisher.audio_input = MagicMock()
            publisher.audio_input.drain = AsyncMock()

        with patch.object(Publisher, "start", new=AsyncMock(side_effect=fake_start)) as start:
            await publisher.write(b"audio", data_type=3, payload_type=6)

        self.assertFalse(publisher.closed)
        start.assert_called_once()
        publisher.audio_input.write.assert_called_once()

    async def test_logs_the_reported_video_payload_type_once(self):
        # A camera streaming H.265 into "-f h264" fails deep inside ffmpeg
        # with nothing pointing at the codec as the cause, so the type the
        # camera reports has to appear next to the one we configured.
        publisher = Publisher("806076813171", 1, "live")
        publisher.decided = True
        process = MagicMock()
        process.returncode = None
        process.stdin = MagicMock()
        process.stdin.drain = AsyncMock()
        publisher.process = process

        with self.assertLogs(main.LOGGER, level="INFO") as logs:
            await publisher.write(b"\x00\x00\x00\x01video", data_type=0, payload_type=99)
            await publisher.write(b"\x00\x00\x00\x01video", data_type=0, payload_type=99)

        lines = [line for line in logs.output if "JT1078 video" in line]
        self.assertEqual(1, len(lines))
        self.assertIn("payloadType=99", lines[0])
        self.assertIn("configuredCodec=h264", lines[0])

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

        close.assert_called_once()
        self.assertIsNot(first_publisher, second_publisher)
        self.assertIs(main.ACTIVE_PUBLISHERS[("live", "860112070346616", 1)], second_publisher)


class PublisherStatusTest(unittest.TestCase):
    def setUp(self):
        main.ACTIVE_PUBLISHERS.clear()
        self.addCleanup(main.ACTIVE_PUBLISHERS.clear)

    def test_reports_not_connected_when_no_publisher(self):
        status = main.publisher_status("live", "860112070346616", "0")
        self.assertEqual(status["connected"], False)
        self.assertEqual(status["publishing"], False)
        self.assertIsNone(status["videoAgeMs"])

    def test_reports_liveness_for_an_active_publisher(self):
        publisher = Publisher("860112070346616", 1, "live")  # channel 1 -> output 0
        process = MagicMock()
        process.returncode = None
        publisher.process = process
        publisher.decided = True
        publisher.has_audio = True
        publisher.started_at = main.time.monotonic() - 4.0
        publisher.last_video_at = main.time.monotonic()
        publisher.last_audio_at = main.time.monotonic()
        main.ACTIVE_PUBLISHERS[("live", "860112070346616", 1)] = publisher

        status = main.publisher_status("live", "860112070346616", "0")
        self.assertEqual(status["connected"], True)
        self.assertEqual(status["publishing"], True)
        self.assertEqual(status["hasVideo"], True)
        self.assertEqual(status["hasAudio"], True)
        self.assertLess(status["videoAgeMs"], 1000)
        self.assertGreaterEqual(status["uptimeMs"], 4000)

        # A different channel on the same camera has no publisher.
        self.assertEqual(
            main.publisher_status("live", "860112070346616", "1")["connected"], False)

    def test_publishing_false_once_ffmpeg_exits(self):
        publisher = Publisher("860112070346616", 1, "live")
        process = MagicMock()
        process.returncode = 0
        publisher.process = process
        main.ACTIVE_PUBLISHERS[("live", "860112070346616", 1)] = publisher
        status = main.publisher_status("live", "860112070346616", "0")
        self.assertEqual(status["connected"], True)
        self.assertEqual(status["publishing"], False)


class HandleConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_closes_connection_after_idle_timeout(self):
        # A camera on a cellular/NAT network can drop a connection
        # without ever sending a FIN/RST, leaving reader.read() blocked
        # forever unless it is bounded by a timeout.
        async def hang(size):
            await asyncio.sleep(3600)

        reader = AsyncMock()
        reader.read = AsyncMock(side_effect=hang)
        writer = MagicMock()
        writer.get_extra_info.return_value = ("1.2.3.4", 1)
        writer.wait_closed = AsyncMock()

        with patch.dict("os.environ", {"JT1078_READ_TIMEOUT": "0.01"}):
            await asyncio.wait_for(main.handle_connection(reader, writer, "live"), timeout=2)

        writer.close.assert_called_once()


class AttachmentReceiverTest(unittest.IsolatedAsyncioTestCase):
    # Header of a real DC600 0x1210 seen on the playback port on 2026-10-26
    # (2013-style header, phone 806076813171, serial 0x0180).
    TERMINAL = bytes.fromhex("806076813171")

    def message(self, message_id, body, serial=0x0180):
        return main.build_jt808(message_id, serial, self.TERMINAL, None, body)

    def data_packet(self, name, offset, payload):
        return (MAGIC + name.encode().ljust(main.ATTACHMENT_FILE_NAME_SIZE, b"\x00")
                + offset.to_bytes(4, "big") + len(payload).to_bytes(4, "big") + payload)

    def test_round_trips_and_escapes_a_jt808_frame(self):
        body = bytes.fromhex("7e7d01")
        frame = self.message(0x1210, body)
        self.assertEqual(frame.count(b"\x7e"), 2)
        parsed = main.parse_jt808(main.jt808_unescape(frame[1:-1]))
        self.assertEqual((0x1210, 0x0180, self.TERMINAL, None, body),
                         (parsed.message_id, parsed.serial, parsed.terminal, parsed.version, parsed.body))

    def test_rejects_a_bad_checksum(self):
        frame = bytearray(main.jt808_unescape(self.message(0x1210, b"\x01\x02")[1:-1]))
        frame[-1] ^= 0xFF
        self.assertIsNone(main.parse_jt808(bytes(frame)))

    def test_parses_a_2019_header(self):
        frame = main.build_jt808(0x1211, 7, bytes(10), 1, b"\xaa")
        parsed = main.parse_jt808(main.jt808_unescape(frame[1:-1]))
        self.assertEqual((0x1211, 7, 1, b"\xaa"), (parsed.message_id, parsed.serial, parsed.version, parsed.body))

    def test_general_response_acks_the_message(self):
        request = main.parse_jt808(main.jt808_unescape(self.message(0x1210, b"\x00")[1:-1]))
        response = main.parse_jt808(main.jt808_unescape(main.general_response(request, 5)[1:-1]))
        self.assertEqual(0x8001, response.message_id)
        self.assertEqual(self.TERMINAL, response.terminal)
        self.assertEqual(bytes.fromhex("0180121000"), response.body)

    def test_splits_messages_and_data_packets_across_reads(self):
        stream = (self.message(0x1211, b"\x01") + self.data_packet("a.jpg", 0, b"x" * 10)
                  + self.data_packet("a.jpg", 10, b"y" * 5) + self.message(0x1212, b"\x02"))
        parser = main.AttachmentParser()
        items = []
        for i in range(0, len(stream), 7):
            items.extend(parser.feed(stream[i:i + 7]))
        self.assertEqual(["message", "data", "data", "message"], [item[0] for item in items])
        self.assertEqual(("data", "a.jpg", 10, b"y" * 5), items[2])

    # A real DC600 0x1210 from 2026-10-28 (alarm 0x6402, 3 JPG + 1 MP4).
    ALARM_LIST = bytes.fromhex(
        "3000000000000030000000000000261028084945000400333030303030303030303030303032363130323830383439343530"
        "30303430300004"
        "3130305f36345f363430325f305f33303030303030303030303030303236313032383038343934353030303430302e6a706700007fcd"
        "3130305f36345f363430325f315f33303030303030303030303030303236313032383038343934353030303430302e6a706700007fc8"
        "3130305f36345f363430325f325f33303030303030303030303030303236313032383038343934353030303430302e6a706700007fdb"
        "3130325f36345f363430325f335f33303030303030303030303030303236313032383038343934353030303430302e6d703400200000")
    FILE_NAME = "00_64_6402_0_30000000000000261028084945000400.jpg"

    def file_body(self, size, name=FILE_NAME, file_type=0):
        return bytes((len(name),)) + name.encode() + bytes((file_type,)) + size.to_bytes(4, "big")

    async def run_connection(self, stream):
        reader = AsyncMock()
        reader.read = AsyncMock(side_effect=stream + [b""])
        writer = MagicMock()
        writer.get_extra_info.return_value = ("1.2.3.4", 1)
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()
        with patch.object(main, "post_attachment", return_value="ok") as post:
            await main.handle_attachment_connection(reader, writer)
        writer.close.assert_called_once()
        replies = [main.parse_jt808(main.jt808_unescape(call.args[0][1:-1])) for call in writer.write.call_args_list]
        return replies, post

    def test_parses_a_real_0x1211_body(self):
        body = bytes.fromhex(
            "3130305f36345f363430325f305f33303030303030303030303030303236313032383038343934353030303430302e6a70670000007fcd")
        self.assertEqual((self.FILE_NAME, 0, 0x7fcd), main.parse_attachment_file(body))
        self.assertIsNone(main.parse_attachment_file(b"\x05ab"))

    def test_missing_ranges(self):
        self.assertEqual([], main.missing_ranges([(0, 5), (5, 5)], 10))
        self.assertEqual([(3, 2), (8, 2)], main.missing_ranges([(5, 3), (0, 3)], 10))
        self.assertEqual([(0, 4)], main.missing_ranges([], 4))

    def test_terminal_number_strips_2019_padding(self):
        message = main.Jt808Message(0x1212, 0, bytes.fromhex("00000000806076813171"), 1, b"")
        self.assertEqual("806076813171", main.terminal_number(message))
        message = main.Jt808Message(0x1212, 0, self.TERMINAL, None, b"")
        self.assertEqual("806076813171", main.terminal_number(message))

    async def test_acks_list_and_file_info_then_completes_and_stores_the_file(self):
        stream = [self.message(0x1210, self.ALARM_LIST, 1), self.message(0x1211, self.file_body(8), 2),
                  self.data_packet(self.FILE_NAME, 0, b"abcd") + self.data_packet(self.FILE_NAME, 4, b"efgh"),
                  self.message(0x1212, self.file_body(8), 3)]
        replies, post = await self.run_connection(stream)

        self.assertEqual([0x8001, 0x8001, 0x9212], [reply.message_id for reply in replies])
        self.assertEqual([bytes.fromhex("0001121000"), bytes.fromhex("0002121100")], [r.body for r in replies[:2]])
        self.assertEqual(bytes((len(self.FILE_NAME),)) + self.FILE_NAME.encode() + b"\x00\x00\x00", replies[2].body)
        post.assert_called_once_with(
            "806076813171_30000000000000261028084945000400_" + self.FILE_NAME, b"abcdefgh")

    async def test_asks_for_missing_ranges_and_does_not_store(self):
        stream = [self.message(0x1210, self.ALARM_LIST, 1), self.message(0x1211, self.file_body(10), 2),
                  self.data_packet(self.FILE_NAME, 0, b"abcd"), self.message(0x1212, self.file_body(10), 3)]
        replies, post = await self.run_connection(stream)

        body = replies[-1].body
        self.assertEqual(0x9212, replies[-1].message_id)
        self.assertEqual(bytes.fromhex("0001010000000400000006"), body[1 + len(self.FILE_NAME):])
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
