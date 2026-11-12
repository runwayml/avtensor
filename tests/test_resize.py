"""Python surface of resize_filter/resize_params, on a small generated video."""

import subprocess
import tempfile
import unittest
from pathlib import Path

import avtensor
from avtensor import ResizeParams, VideoStreamRequest


class ResizeParamsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.video = Path(cls.directory.name) / "video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=128x72:r=4:d=1,format=yuv420p",
                "-c:v",
                "mpeg4",
                str(cls.video),
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )

    def decode(self, video_stream, path=None):
        request = avtensor.MediaDecodeRequest(
            str(path or self.video), video_stream=video_stream
        )
        return avtensor.decode_asset(request)[0]["data"]

    def test_params_fields_and_repr(self):
        params = ResizeParams(0.0, 0.5)
        self.assertEqual((params.a, params.b), (0.0, 0.5))
        self.assertIsNone(ResizeParams(4).b)
        self.assertEqual(ResizeParams(a=0, b=0.5), params)
        self.assertEqual(repr(params), "ResizeParams(a=0.0, b=0.5)")
        self.assertEqual(repr(ResizeParams(4)), "ResizeParams(a=4.0)")

    def test_params_are_immutable(self):
        with self.assertRaises(AttributeError):
            ResizeParams(4).a = 5

    def test_request_kwargs_and_attributes(self):
        request = VideoStreamRequest(
            resize_filter="lanczos", resize_params=ResizeParams(4)
        )
        self.assertEqual(request.resize_filter, "lanczos")
        self.assertEqual(request.resize_params, ResizeParams(4))
        request.resize_params = ResizeParams(5)
        self.assertEqual(request.resize_params.a, 5)
        request.resize_params = None
        self.assertIsNone(request.resize_params)

    def test_params_reach_the_kernel(self):
        def decode(params):
            return self.decode(
                VideoStreamRequest(
                    width=32, height=18, resize_filter="lanczos", resize_params=params
                )
            )

        default = decode(None)
        self.assertEqual(list(default.shape), [4, 3, 18, 32])
        self.assertTrue(decode(ResizeParams(3)).equal(default))
        self.assertFalse(decode(ResizeParams(5)).equal(default))

    def test_unknown_filter(self):
        with self.assertRaisesRegex(ValueError, "resize_filter must be one of"):
            self.decode(VideoStreamRequest(width=32, height=18, resize_filter="box"))

    def test_invalid_params_rejected_before_opening_input(self):
        missing = Path(self.directory.name) / "missing.mp4"
        cases = [
            ("bicubic", ResizeParams(0, 20), r"resize_params\.b in \[-2, 2\]"),
            ("lanczos", ResizeParams(0.5), r"\(taps\) in \[1, 32\]"),
            ("gauss", ResizeParams(1000), r"in \(0, 100\]"),
            ("sinc", ResizeParams(1), r"does not take resize_params\.a"),
            (None, ResizeParams(1), "need resize_filter"),
        ]
        for resize_filter, params, message in cases:
            with self.subTest(resize_filter=resize_filter, params=params):
                request = VideoStreamRequest(
                    width=32,
                    height=18,
                    resize_filter=resize_filter,
                    resize_params=params,
                )
                with self.assertRaisesRegex(ValueError, message):
                    self.decode(request, path=missing)


if __name__ == "__main__":
    unittest.main()
