"""Opt-in timeline alignment and untagged-HDR handling from Python: the defaults
keep the behaviour from before 0.6.0/0.7.0, the flags turn the newer one on."""

import subprocess
import tempfile
import unittest
from pathlib import Path

import avtensor
from avtensor import AudioStreamRequest, StreamType, VideoStreamRequest

RATE = 48000
OFFSET_S = 0.5


def ffmpeg(*args):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
        check=True,
        capture_output=True,
        timeout=30,
    )


class DefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        # Audio that starts 0.5 s after the video (a later first timestamp).
        cls.late_audio = root / "late_audio.mkv"
        ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=s=128x72:r=24:d=3,format=yuv420p",
            "-itsoffset", str(OFFSET_S),
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={RATE}:duration=2",
            "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "pcm_s16le",
            str(cls.late_audio),
        )  # fmt: skip
        # 10-bit PQ with BT.2020 primaries and no matrix tag.
        cls.untagged_pq = root / "untagged_pq.mkv"
        ffmpeg(
            "-f", "lavfi", "-i",
            "testsrc2=s=64x48:r=4:d=1,format=yuv420p10le,"
            "setparams=color_trc=smpte2084:color_primaries=bt2020",
            "-c:v", "ffv1", str(cls.untagged_pq),
        )  # fmt: skip

    def audio(self, **request):
        result = avtensor.decode_asset(
            avtensor.MediaDecodeRequest(
                str(self.late_audio),
                video_stream=VideoStreamRequest(),
                audio_streams=[AudioStreamRequest(**request)],
            )
        )
        return next(s for s in result if s["stream_type"] == StreamType.Audio)

    def test_request_fields_default_to_none(self):
        self.assertIsNone(AudioStreamRequest().align_to_video_timeline)
        self.assertIsNone(VideoStreamRequest().untagged_hdr_as_bt2020)
        self.assertNotEqual(
            AudioStreamRequest(), AudioStreamRequest(align_to_video_timeline=True)
        )

    def test_late_audio_is_not_padded_by_default(self):
        audio = self.audio()["data"]
        self.assertEqual(audio.shape[-1], 2 * RATE)
        self.assertNotEqual(float(audio[:, :10].abs().max()), 0.0)

    def test_late_audio_is_padded_when_aligned(self):
        stream = self.audio(align_to_video_timeline=True)
        audio = stream["data"]
        lead = int(OFFSET_S * RATE)
        self.assertEqual(audio.shape[-1], lead + 2 * RATE)
        self.assertEqual(float(audio[:, :lead].abs().max()), 0.0)
        self.assertEqual(stream["start_time"], 0.0)

    def decode_pq(self, **request):
        return avtensor.decode_asset(
            avtensor.MediaDecodeRequest(
                str(self.untagged_pq), video_stream=VideoStreamRequest(**request)
            )
        )[0]["data"]

    def test_untagged_hdr_fails_by_default(self):
        with self.assertRaises(ValueError):
            self.decode_pq()

    def test_untagged_hdr_decodes_as_bt2020_when_asked(self):
        frames = self.decode_pq(untagged_hdr_as_bt2020=True)
        self.assertEqual(tuple(frames.shape), (4, 3, 48, 64))


if __name__ == "__main__":
    unittest.main()
