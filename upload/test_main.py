import json
import tempfile
import unittest
from pathlib import Path

import main
from main import _callback_event_meta, _merge_meta

# A real body as the 8654078… cameras send it, from the production logs.
CALLBACK_BODY = json.dumps({
    "businessType": "eventAttachment",
    "imei": "865478070008026",
    "camera": 2,
    "alarmTime": 1792490258,
    "lat": "-23.453370",
    "lng": "-51.801757",
    "localFileName": "865478070008026_30303038303236261020065728190700_2_00.jpg",
    "mimeType": "image/jpeg",
    "eventType": None,
    "videoBegin": 1792490248,
    "videoEnd": 1792490258,
})


class CallbackEventMetaTest(unittest.TestCase):
    def test_keeps_the_event_level_fields_and_floats_the_position(self):
        meta = _callback_event_meta(CALLBACK_BODY)
        self.assertEqual(-23.453370, meta["lat"])
        self.assertEqual(-51.801757, meta["lng"])
        self.assertEqual(2, meta["camera"])
        self.assertEqual(1792490258, meta["alarmTime"])
        self.assertEqual(1792490248, meta["videoBegin"])
        # Per-file noise and null fields are not event metadata.
        self.assertNotIn("localFileName", meta)
        self.assertNotIn("mimeType", meta)
        self.assertNotIn("eventType", meta)

    def test_a_truncated_body_is_reported_not_silently_dropped(self):
        # This is why the field has its own, larger length limit: clipped at
        # the old 256 chars the body is no longer valid JSON.
        with self.assertLogs(main.log, level="WARNING") as logs:
            self.assertEqual({}, _callback_event_meta(CALLBACK_BODY[:256]))
        self.assertIn("unparseable callbackBody", logs.output[0])

    def test_tolerates_a_non_object_body(self):
        self.assertEqual({}, _callback_event_meta("[]"))


class MergeMetaTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name) / "event"

    def read(self):
        return json.loads((self.folder / ".meta.json").read_text())

    def test_both_sources_survive_in_either_order(self):
        traccar = {"type": "fatigue", "alarm": "fatigueDriving", "level": 2,
                   "time": 1792490248000, "kind": "dms"}
        camera = _callback_event_meta(CALLBACK_BODY)

        _merge_meta(self.folder, camera, overwrite=False)
        _merge_meta(self.folder, traccar, overwrite=True)
        self.assertEqual("fatigue", self.read()["type"])
        self.assertEqual(-23.453370, self.read()["lat"])

        self.folder = self.folder.with_name("event2")
        _merge_meta(self.folder, traccar, overwrite=True)
        _merge_meta(self.folder, camera, overwrite=False)
        self.assertEqual("fatigue", self.read()["type"])
        self.assertEqual(-23.453370, self.read()["lat"])

    def test_traccar_may_correct_itself_but_the_camera_may_not(self):
        _merge_meta(self.folder, {"type": "general", "lat": 1.0}, overwrite=True)
        _merge_meta(self.folder, {"type": "fatigue"}, overwrite=True)
        self.assertEqual("fatigue", self.read()["type"])
        # The first file of an event sets the position; later files in the
        # same event report a slightly moved vehicle and must not shift it.
        _merge_meta(self.folder, {"lat": 2.0}, overwrite=False)
        self.assertEqual(1.0, self.read()["lat"])

    def test_nulls_are_never_written(self):
        _merge_meta(self.folder, {"type": "fatigue", "level": None}, overwrite=True)
        self.assertNotIn("level", self.read())

    def test_a_corrupt_meta_file_is_rewritten_rather_than_fatal(self):
        self.folder.mkdir(parents=True)
        (self.folder / ".meta.json").write_text("{truncated")
        with self.assertLogs(main.log, level="WARNING"):
            _merge_meta(self.folder, {"type": "fatigue"}, overwrite=True)
        self.assertEqual({"type": "fatigue"}, self.read())


if __name__ == "__main__":
    unittest.main()
