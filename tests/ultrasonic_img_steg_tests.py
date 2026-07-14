import unittest
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from ultrasonic_img_steg import encode_spectrogram, encode_data, decode_data


class Ultrasonic_Img_Steg_Tests(unittest.TestCase):

    def setUp(self):
        """Set up temporary directories and dummy files for testing."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

        # Paths for test assets
        self.input_audio_path = self.dir_path / "input_audio.wav"
        self.input_image_path = self.dir_path / "input_image.png"
        self.output_audio_path = self.dir_path / "output_audio.wav"
        self.recovered_image_path = self.dir_path / "recovered_image.png"

        # 1. Create a dummy audio file (44.1 kHz, mono, 2 seconds of noise)
        # We need a high sample rate (>= 44100) to support ultrasonic bins up to 22kHz
        self.sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        # Pure sine wave at 1000Hz mixed with slight noise
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.01 * np.random.randn(len(t))
        sf.write(self.input_audio_path, audio_data, self.sample_rate)

        # 2. Create a dummy image file (10x10 pixels, grayscale)
        # Small dimensions ensure it fits into the data capacity effortlessly
        self.img_data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        img = Image.fromarray(self.img_data, mode="L")
        img.save(self.input_image_path)

    def tearDown(self):
        """Clean up temporary files after test completion."""
        self.test_dir.cleanup()

    def test_encode_spectrogram_execution(self):
        """Verify encode_spectrogram runs smoothly and outputs a valid audio file."""
        encode_spectrogram(
            audio_path=self.input_audio_path,
            image_path=self.input_image_path,
            output_path=self.output_audio_path,
            start_freq=19000,
            end_freq=22000,
            fft_size=1024,
            hop=256
        )

        # Assert output file was created
        self.assertTrue(self.output_audio_path.exists())
        
        # Verify the file is readable and matches original audio dimensions
        out_audio, out_sr = sf.read(self.output_audio_path)
        self.assertEqual(out_sr, self.sample_rate)
        self.assertGreater(len(out_audio), 0)

    def test_encode_and_decode_data_roundtrip(self):
        """Verify that generic data can be encoded into audio and identically recovered."""
        # Run encode
        encode_data(
            audio_path=self.input_audio_path,
            image_path=self.input_image_path,
            output_path=self.output_audio_path,
            start_freq=19000,
            end_freq=22000
        )
        
        self.assertTrue(self.output_audio_path.exists())

        # Run decode
        decode_data(
            encoded_audio_path=self.output_audio_path,
            output_image_path=self.recovered_image_path,
            start_freq=19000,
            end_freq=22000
        )

        self.assertTrue(self.recovered_image_path.exists())

        # Validate that recovered image data is structurally and byte-wise identical
        with Image.open(self.recovered_image_path) as recovered_img:
            recovered_data = np.asarray(recovered_img)
            
        np.testing.assert_array_equal(recovered_data, self.img_data)

    def test_encode_data_capacity_error(self):
        """Verify that encode_data raises a ValueError if the audio file is too short."""
        # Create a tiny 0.001-second audio file (not enough bins to hold a 10x10 image + header)
        short_audio_path = self.dir_path / "short_audio.wav"
        short_audio = np.zeros(int(self.sample_rate * 0.001))
        sf.write(short_audio_path, short_audio, self.sample_rate)

        # Expect ValueError because data size exceeds maximum frequency capacity
        with self.assertRaises(ValueError):
            encode_data(
                audio_path=short_audio_path,
                image_path=self.input_image_path,
                output_path=self.output_audio_path,
                start_freq=19000,
                end_freq=22000
            )


if __name__ == "__main__":
    unittest.main()