from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageOps
from scipy.signal import stft, istft


def encode_spectrogram(
    audio_path,
    image_path,
    output_path,
    start_freq=19000,
    end_freq=22000,
    fft_size=4096,
    hop=512,
    strength=0.0003,
    image_width=480,
):

    # Load audio
    audio, sample_rate = sf.read(audio_path)

    original_peak = np.max(np.abs(audio))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    mono_peak = np.max(np.abs(audio))
    if mono_peak > 0:
        audio = audio * (original_peak / mono_peak)

    # STFT
    freqs, times, Z = stft(
        audio,
        fs=sample_rate,
        nperseg=fft_size,
        noverlap=fft_size - hop,
        boundary=None,
    )

    magnitude = np.abs(Z)
    phase = np.angle(Z)

    # Frequency band
    start_bin = np.searchsorted(freqs, start_freq)
    end_bin = np.searchsorted(freqs, end_freq)

    usable_bins = end_bin - start_bin
    usable_frames = Z.shape[1]

    # Load image
    img = Image.open(image_path).convert("L")

    image_width = min(image_width, usable_frames)
    img = ImageOps.autocontrast(img)

    orig_width, orig_height = img.size
    scale = usable_bins / orig_height
    image_width = int(orig_width * scale)

    # Don't let it exceed the available time frames
    image_width = min(image_width, usable_frames)

    img = img.resize(
        (image_width, usable_bins),
        Image.Resampling.LANCZOS,
    )

    pixels = np.asarray(img, dtype=np.float32) / 255.0

    # Flip vertically so the top of the image is the highest frequency
    pixels = np.flipud(pixels)
    start_frame = 0

    magnitude[start_bin : start_bin + usable_bins, start_frame : start_frame + image_width] = 0.0

    # Draw into spectrogram
    for y in range(usable_bins):
        for x in range(image_width):
            brightness = pixels[y, x]
            magnitude[start_bin + y, start_frame + x] = brightness * strength

    # Recombine
    encoded = magnitude * np.exp(1j * phase)

    _, output = istft(
        encoded,
        fs=sample_rate,
        nperseg=fft_size,
        noverlap=fft_size - hop,
        input_onesided=True,
    )

    sf.write(output_path, output, sample_rate)


def encode_data(
    audio_path,
    image_path,
    output_path,
    start_freq=19000,
    end_freq=22000
):
    # Load Audio
    audio, sample_rate = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # Convert to mono
    
    total_samples = len(audio)
    
    # 2. Read Image into raw bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Prepend the length of the data (4 bytes)
    data_len = len(image_bytes)
    header = data_len.to_bytes(4, byteorder='big')
    full_data = header + image_bytes
    
    # Convert bytes to a numpy array of uint8
    data_array = np.frombuffer(full_data, dtype=np.uint8)
    if len(data_array) % 2 != 0:
        data_array = np.append(data_array, 0)
        
    # Capacity Check
    start_bin = int(start_freq * total_samples / sample_rate)
    end_bin = int(end_freq * total_samples / sample_rate)
    usable_bins = end_bin - start_bin
    max_bytes = usable_bins * 2
    
    if len(data_array) > max_bytes:
        raise ValueError(f"Audio is too short! Need at least {len(data_array)} bytes, but only have {max_bytes}.")

    # Compute FFT
    fft_data = np.fft.rfft(audio)
    
    # Use a fixed scaling factor
    scale_factor = 1000.0 
    
    # Map bytes directly to complex frequency coefficients
    floats = (data_array.astype(np.float32) / 255.0) * scale_factor
    complex_symbols = floats[0::2] + 1j * floats[1::2]
    
    # Inject Data into audio
    fft_data[start_bin:end_bin] = 0 
    fft_data[start_bin : start_bin + len(complex_symbols)] = complex_symbols
    
    # Inverse FFT back to time domain
    output_audio = np.fft.irfft(fft_data, n=total_samples)
    
    # Save as a 32-bit Float WAV file to maintain exact precision
    sf.write(output_path, output_audio, sample_rate, subtype='FLOAT')


def decode_data(
    encoded_audio_path,
    output_image_path,
    start_freq=19000,
    end_freq=22000
):
    # Load encoded audio
    audio, sample_rate = sf.read(encoded_audio_path)
    total_samples = len(audio)
    
    # Run FFT and extract target frequency bins
    fft_data = np.fft.rfft(audio)

    start_bin = int(start_freq * total_samples / sample_rate)
    end_bin = int(end_freq * total_samples / sample_rate)
    complex_symbols = fft_data[start_bin:end_bin]
    
    # Reconstruct the flattened float array
    floats = np.empty(len(complex_symbols) * 2, dtype=np.float32)
    floats[0::2] = np.real(complex_symbols)
    floats[1::2] = np.imag(complex_symbols)
    
    # Reverse scaling using the same fixed factor
    scale_factor = 1000.0
    data_array = np.round((floats / scale_factor) * 255.0)
    data_array = np.clip(data_array, 0, 255).astype(np.uint8)
    
    # Parse Header
    data_bytes = data_array.tobytes()
    data_len = int.from_bytes(data_bytes[0:4], byteorder='big')
    
    original_image_bytes = data_bytes[4:4 + data_len]
    
    # Save the recovered image
    with open(output_image_path, "wb") as f:
        f.write(original_image_bytes)