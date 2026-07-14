# ultrasonic_img_steg

A Python package designed to hide images in the ultrasonic and sub-ultrasonic frequency spectrum of lossless audio files.

This package supports two main functions:

1. Visual Spectrogram Embedding: Images are converted to grayscale, then embedded directly into the ultrasonic range of the audio file, making them visible to anyone analyzing the audio with a spectrogram viewer (like Audacity).

2. Raw Data Encoding: An entire image file (PNG, JPG, etc.) is encoded by converting it into Fourier coefficients and storing those in the ultrasonic range of the audio file, with a matching decoding function to extract the original file exactly as it was encoded.


## Important Notes

1. Use Lossless Audio Formats Only: Lossy compression formats (such as MP3, AAC, and the audio compression frequently used by messaging apps) heavily modify the data to reduce file size. Any compression will almost definitely break raw data encoding, and is likely to make any visual spectrogram embeddings unreadable as well. Always use lossless audio formats like `.wav` and `.flac` for both input and output files to prevent data corruption.

2. Sample Rate Requirements: To successfully hide data in the ultrasonic range (19 kHz to 22 kHz) without altering audible sound, the host audio file must have a sample rate of at least 44.1 kHz. If the sample rate is too low, the audio embeddings cannot be placed in ultrasonic frequencies, making them audible. If you wish to embed data or images into a specific range, ensure that the upper bound of that range is at most half of the sample rate of your input audio.

3. Audio Clipping Warning: If the `strength` parameter of the visual spectrogram function is set too high, the injected amplitudes can easily exceed the maximum headroom, causing harsh digital clipping and audio distortion. As a result, the signal meant for the ultrasonic range can bleed into the lower audible spectrum, turning the silent hidden image into a loud, piercing noise.

4. Data Encoding Click Issue: The data mode currently leaves a faint click sound at the beginning of the output audio. This is an artifact exclusive to the data encoding mode, and does not occur in the visual spectrogram mode.


## Motivation

Most existing image-to-audio embedding tools currently create an audio file with the image taking up the entire audio range, making a very loud noise when listened to. Thus, most past applications of this image spectrogram technique placed it in an audio track already composed of seemingly random noise. This package aims to allow embedding of these spectrogram images into audio without changing how the audio sounds.

Most available audio steganography tools use LSB to hide files in audio. This makes it harder to detect that anything is being hidden; however, the amount of data that can be stored before the audio becomes noticeably different is quite small. This package allows storing more data in audio files, especially ones with higher sample rates, while also keeping the sound of the audio exactly the same by placing all of the data in inaudible frequencies.


## Installation

```bash
pip install ultrasonic_img_steg
```

## Quickstart

### 1. Spectrogram Art

Drawing a visible image directly into the high-frequency spectrum of an audio track

```python
from ultrasonic_img_steg import encode_spectrogram

# Embeds image.png into audio.wav in the frequency range 19,000 - 22,000 Hz with strength 0.0003, and saves it to output.wav
encode_spectrogram(
    audio_path="audio.wav",
    image_path="image.png",
    output_path="output.wav",
    start_freq=19000,
    end_freq=22000,
    strength=0.0003
)
```

### 2. Data Encoding and Decoding

```python
from ultrasonic_img_steg import encode_data, decode_data

# Encodes image.jpg into audio.wav at 19,000 - 22,000 Hz, placing the output in audio_with_data.wav
encode_data(
    audio_path="audio.wav",
    image_path="image.jpg",
    output_path="audio_with_data.wav",
    start_freq=19000,
    end_freq=22000
)

# Extracts recovered_image.jpg from audio_with_data.wav at 19,000 - 22,000 Hz.
decode_data(
    encoded_audio_path="audio_with_data.wav",
    output_image_path="recovered_image.jpg",
    start_freq=19000,
    end_freq=22000
)
```


## User Guide

### 1. Adjusting Visual Spectrogram Strength
The `strength` parameter adjusts how bright the pixels in the spectrogram-embedded image will be. If it is set too high, it will cause digital clipping and audio distortion.

A strength of `1.0` represents the loudest possible sound that can be encoded.
Each time the strength is multiplied by 1/10, the sound decreases in volume by 20 decibels (dB).

For additional reference, these are some strength values and their approximate brightness:
`0.1` - Extremely Bright (only use with very quiet audio, or clipping will likely occur)
`0.01` - Very Bright (only use with quiet audio)
`0.001` - Normal brightness
`0.0003` - Default brightness (somewhat dim, but unlikely to cause clipping)
`0.0001` - Dim
`0.00001` - Generally invisible (too dark)


### 2. Spectrogram Image Duration and Cutoffs
**Sizing**: The image does not stretch to fit the entire audio file; instead, it is scaled so its height takes up the entire selected frequency band. For reference, a square image with a range of 3 kHz (19,000 - 22,000 Hz) will generally take up a little under 3.5 seconds.

**Padding**: If the audio is longer than the image, the image will finish drawing, and the remaining audio will be silent in that frequency band.

**Truncation**: If the audio file is too short to fit the entire image width, the image will be cut off on the right.

### 3. Data Mode Capacity Limits
Data mode has a capacity limit determined by the length of the host audio and the width of the frequency band.

For reference, using default settings, 1 second of audio with a range of 3 kHz (19,000 - 22,000 Hz) will store a little over 5.85 KB of data. This is equivalent to requiring around 3 minutes (180 seconds) of audio to store a 1 MB file.

If the file is too large to fit in the given range of the host audio, a ValueError will be raised:
```python
ValueError: Audio is too short! Need at least X bytes, but only have Y.
```

## Developer Guide

This guide outlines the internal mechanics of `ultrasonic_img_steg` and provides instructions for testing and modifying the codebase.

### 1. Mathematical Mapping & Architecture
The library uses two distinct Fourier domain techniques depending on the active mode:

#### Visual Spectrogram Mode (STFT)
This mode relies on the **Short-Time Fourier Transform (STFT)** from the `scipy` library to map a 2D grayscale image (Time vs. Frequency) into the audio spectrum.

* **Slicing and STFT**: The host audio is processed using a sliding window with a set `hop_length` (usually equal to the window size to prevent ghosting artifacts).
* **Magnitude Replacement**: The grayscale value of each pixel in the target column scales the magnitude of the FFT coefficients inside the $[k_{\text{start}}, k_{\text{end}}]$ frequency range.
* **Phase Preservation**: The original phase of the audio in those bins is preserved to keep transitions smooth, preventing harsh phase-cancellation clicks.
* **iSTFT Rebuild**: An Inverse Short-Time Fourier Transform (iSTFT) reconstructs the modified frequency bins back into raw time-domain audio samples.

#### Raw Data Mode (Global 1D FFT)
This mode bypasses time-slicing entirely. It treats the entire audio file as one single continuous block of data using a global **Real Fast Fourier Transform (RFFT)** from the `numpy` library.

* **Bin Mapping Formula**: A target physical frequency $f$ (in Hz) is mapped to its exact global bin index $k$ using the total sample count of the audio $N_{\text{samples}}$ and the sample rate $f_s$:
  $$k = \lfloor \frac{f \cdot N_{\text{samples}}}{f_s} \rfloor$$
* **Byte Pairing**: Every pair of bytes (values 0–255) is scaled to a float range $[-1.0, 1.0]$ and combined into a single complex number:
  $$\text{complexCoefficient} = \text{float}_1 + j \cdot \text{float}_2$$
* **Direct Injection**: These complex values are written directly into the flat frequency array between `start_bin` and `end_bin`.
* **Inverse FFT**: `np.fft.irfft` reconstructs the entire frequency array back into time-domain audio samples in a single mathematical operation, preserving exact precision.


### 2. Local Environment Setup
To modify `ultrasonic_img_steg`, clone the repository and install it locally in editable mode:

```bash
git clone https://github.com/nicholas-lua/ultrasonic_img_steg.git
cd ultrasonic_img_steg

# Install the package in editable mode
pip install -e .
```

### 3. Running the Test Suite

We use `pytest` to maintain codebase reliability. Before finalizing changes, ensure all tests pass cleanly by running the test suite.

```bash
# Execute tests
pytest
```