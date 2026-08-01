import unittest

from main import MAGIC, PacketParser, audio_input_options, decode_terminal_id


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


if __name__ == "__main__":
    unittest.main()
