"""Python exception compatibility, using only small locally generated media."""

import os
import pickle
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import avtensor


class MediaErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.video = cls.root / "video.avi"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=32x32:r=4:d=1",
                "-an",
                "-c:v",
                "mpeg4",
                "-tag:v",
                "mp4v",
                str(cls.video),
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
        cls.audio = cls.root / "audio.wav"
        with wave.open(str(cls.audio), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\0\0" * 8000)
        cls.unsupported = cls.root / "unsupported.avi"
        encoded = cls.video.read_bytes()
        assert b"mp4v" in encoded
        cls.unsupported.write_bytes(encoded.replace(b"mp4v", b"zzzz"))

    def request(self, path, kind, index=None, max_output_bytes=None):
        request = avtensor.MediaDecodeRequest(
            str(path), max_output_bytes=max_output_bytes
        )
        if kind == "video":
            stream = avtensor.VideoStreamRequest()
            stream.index = index
            request.video_stream = stream
        else:
            stream = avtensor.AudioStreamRequest()
            stream.index = index
            request.audio_streams = [stream]
        return request

    def test_missing_streams_preserve_value_error_and_message(self):
        for function, operation in (
            (avtensor.decode_asset, "decode"),
            (avtensor.plan_asset, "plan"),
        ):
            for path, kind in ((self.video, "audio"), (self.audio, "video")):
                with self.subTest(operation=operation, kind=kind):
                    with self.assertRaises(ValueError) as raised:
                        function(self.request(path, kind))
                    error = raised.exception
                    self.assertIs(type(error), avtensor.MissingStreamError)
                    self.assertEqual(
                        str(error),
                        f"Failed to {operation} media: No {kind} stream found",
                    )
                    restored = pickle.loads(pickle.dumps(error))
                    self.assertIs(type(restored), type(error))
                    self.assertEqual(restored.args, error.args)

    def test_unavailable_decoder_preserves_value_error_and_message(self):
        for function, operation in (
            (avtensor.decode_asset, "decode"),
            (avtensor.plan_asset, "plan"),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError) as raised:
                    function(self.request(self.unsupported, "video"))
                self.assertIs(type(raised.exception), avtensor.DecoderUnavailableError)
                self.assertEqual(
                    str(raised.exception),
                    f"Failed to {operation} media: Failed to find best video stream!\n\n"
                    "Caused by:\n    AVERROR(-1128613112): `Decoder not found`",
                )
        with self.assertRaises(avtensor.DecoderUnavailableError) as raised:
            avtensor.decode_asset(self.request(self.unsupported, "video", index=0))
        self.assertEqual(
            str(raised.exception),
            "Failed to decode media: Failed to find decoder for video stream #0",
        )

    def test_missing_file_and_corruption_remain_plain_value_errors(self):
        corrupt = self.root / "corrupt.avi"
        corrupt.write_bytes(b"not a media container")
        for function in (avtensor.decode_asset, avtensor.plan_asset):
            for path in (self.root / "missing.avi", corrupt):
                with self.subTest(function=function.__name__, path=path.name):
                    with self.assertRaises(ValueError) as raised:
                        function(self.request(path, "video"))
                    self.assertIs(type(raised.exception), ValueError)

    def test_valid_decode_after_rejection(self):
        with self.assertRaises(avtensor.MissingStreamError):
            avtensor.decode_asset(self.request(self.video, "audio"))
        decoded = avtensor.decode_asset(self.request(self.video, "video"))
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["stream_type"], avtensor.StreamType.Video)
        self.assertEqual(tuple(decoded[0]["data"].shape), (4, 3, 32, 32))

    def test_invalid_explicit_index_remains_plain_value_error(self):
        for function in (avtensor.decode_asset, avtensor.plan_asset):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError) as raised:
                    function(self.request(self.video, "video", index=999))
                self.assertIs(type(raised.exception), ValueError)

    def test_output_limit_preserves_value_error_message_and_recovery(self):
        for path, kind, shape, size in (
            (self.video, "video", [4, 32, 32, 3], 12288),
            (self.audio, "audio", [1, 8000], 32000),
        ):
            for function, operation in (
                (avtensor.decode_asset, "decode"),
                (avtensor.plan_asset, "plan"),
            ):
                with self.subTest(operation=operation, kind=kind):
                    with self.assertRaises(ValueError) as raised:
                        function(self.request(path, kind, max_output_bytes=size - 1))
                    error = raised.exception
                    self.assertIs(type(error), avtensor.OutputLimitExceededError)
                    self.assertEqual(
                        str(error),
                        f"Failed to {operation} media: output tensor for shape {shape} "
                        f"would require {size} bytes, exceeding the request's "
                        f"max_output_bytes of {size - 1} bytes",
                    )
                    restored = pickle.loads(pickle.dumps(error))
                    self.assertIs(type(restored), type(error))
                    self.assertEqual(restored.args, error.args)
                    # The boundary is inclusive, and rejecting a request must not
                    # prevent the next valid plan or decode in the same process.
                    self.assertEqual(
                        len(function(self.request(path, kind, max_output_bytes=size))),
                        1,
                    )

    def test_environment_output_limit_and_request_override(self):
        with patch.dict(os.environ, {"AVTENSOR_MAX_OUTPUT_TENSOR_BYTES": "1"}):
            for function in (avtensor.decode_asset, avtensor.plan_asset):
                with self.subTest(function=function.__name__):
                    with self.assertRaises(avtensor.OutputLimitExceededError):
                        function(self.request(self.video, "video"))
                    self.assertEqual(
                        len(
                            function(
                                self.request(
                                    self.video, "video", max_output_bytes=12288
                                )
                            )
                        ),
                        1,
                    )


if __name__ == "__main__":
    unittest.main()
