import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import main


class FakeS3:
    def __init__(self, fail=False):
        self.fail = fail
        self.uploads = []

    def upload_file(self, filename, bucket, key):
        if self.fail:
            raise OSError("s3 down")
        self.uploads.append((Path(filename).read_bytes(), bucket, key))

    def put_object(self, Bucket, Key, Body, ContentType):
        if self.fail:
            raise OSError("s3 down")
        self.uploads.append((Body, Bucket, Key))


class S3KeyTest(unittest.TestCase):
    def test_keeps_the_uploaded_path_under_the_prefix(self):
        self.assertEqual("recordings/869247060606701/ch1_20261030.mp4",
                         main.s3_key("869247060606701/ch1_20261030.mp4"))

    def test_cannot_climb_out_of_the_prefix(self):
        self.assertEqual("recordings/etc/passwd", main.s3_key("../../etc/passwd"))
        self.assertEqual("recordings/a_b/c_d.mp4", main.s3_key("/a b/./c?d.mp4"))


class ReceivedTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.spool, self.retry = root / "ftp", root / "ftp-retry"
        self.patch("SPOOL_DIR", self.spool)
        self.patch("RETRY_DIR", self.retry)
        self.patch("S3_BUCKET", "bucket")
        self.file = self.spool / "869247060606701" / "1.mp4"
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b"video")

    def patch(self, name, value):
        old = getattr(main, name)
        setattr(main, name, value)
        self.addCleanup(setattr, main, name, old)

    def test_a_finished_upload_goes_to_s3_and_leaves_the_spool(self):
        fake = FakeS3()
        self.patch("_s3", fake)
        main.received(self.file)
        self.assertEqual([(b"video", "bucket", "recordings/869247060606701/1.mp4")], fake.uploads)
        self.assertFalse(self.file.exists())

    def test_a_failed_push_is_parked_and_retried_later(self):
        self.patch("_s3", FakeS3(fail=True))
        with self.assertLogs(main.log, level="ERROR"):
            main.received(self.file)
        parked = self.retry / "869247060606701" / "1.mp4"
        self.assertFalse(self.file.exists())
        self.assertEqual(b"video", parked.read_bytes())

        fake = FakeS3()
        self.patch("_s3", fake)
        main.retry_once()
        self.assertEqual([(b"video", "bucket", "recordings/869247060606701/1.mp4")], fake.uploads)
        self.assertFalse(parked.exists())


class ProgressTest(ReceivedTest):
    def test_a_growing_upload_gets_a_marker_outside_the_recordings_prefix(self):
        fake = FakeS3()
        self.patch("_s3", fake)
        main.report_progress()
        self.assertEqual([(json.dumps({"bytes": 5}).encode(), "bucket",
                           "recordings-progress/869247060606701/1.mp4.json")], fake.uploads)

    def test_an_upload_that_stopped_growing_is_not_reported(self):
        fake = FakeS3()
        self.patch("_s3", fake)
        old = time.time() - main.PROGRESS_INTERVAL - 5
        os.utime(self.file, (old, old))
        main.report_progress()
        self.assertEqual([], fake.uploads)

    def test_a_failed_marker_is_only_logged(self):
        self.patch("_s3", FakeS3(fail=True))
        with self.assertLogs(main.log, level="WARNING"):
            main.report_progress()


if __name__ == "__main__":
    unittest.main()
