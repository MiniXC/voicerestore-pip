import sys
import time
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import torchaudio
import argparse
from tqdm import tqdm
import librosa

# Append BigVGAN to the system path
sys.path.append('./BigVGAN')

from BigVGAN.meldataset import get_mel_spectrogram
from model import OptimizedAudioRestorationModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def measure_gpu_memory(device):
    if device == 'cuda':
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    return 0


def apply_overlap_windowing_waveform(waveform, window_size_samples, overlap):
    step_size = int(window_size_samples * (1 - overlap))
    total_samples = waveform.shape[-1]
    # Use ceil to ensure the final segment is included even if it’s shorter.
    num_windows = math.ceil((total_samples - window_size_samples) / step_size) + 1
    windows = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size_samples
        if end_idx > total_samples:
            # Pad the last window to maintain consistent window size.
            pad_amount = end_idx - total_samples
            chunk = F.pad(waveform[..., start_idx:], (0, pad_amount))
        else:
            chunk = waveform[..., start_idx:end_idx]
        windows.append(chunk)
    
    return torch.stack(windows)


def reconstruct_waveform_from_windows(windows, window_size_samples, overlap, original_length=None):
    step_size = int(window_size_samples * (1 - overlap))
    shape = windows.shape
    if len(shape) == 2:
        num_windows, window_len = shape
        channels = 1
        windows = windows.unsqueeze(1)
    elif len(shape) == 3:
        num_windows, channels, window_len = shape
    else:
        raise ValueError(f"Unexpected windows.shape: {windows.shape}")

    output_length = (num_windows - 1) * step_size + window_size_samples
    reconstructed = torch.zeros((channels, output_length))
    window_sums = torch.zeros((channels, output_length))

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_len
        reconstructed[:, start_idx:end_idx] += windows[i]
        window_sums[:, start_idx:end_idx] += 1

    reconstructed = reconstructed / window_sums.clamp(min=1e-6)
    if original_length is not None:
        reconstructed = reconstructed[:, :original_length]
    if channels == 1:
        reconstructed = reconstructed.squeeze(0)
    return reconstructed



def load_bigvgan_model(device):
    from BigVGAN import bigvgan
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        'nvidia/bigvgan_v2_24khz_100band_256x',
        use_cuda_kernel=False,
        force_download=False
    )
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)
    return bigvgan_model


def load_model(save_path, device, decoder):
    """
    Load the optimized audio restoration model.
    
    Parameters:
    - save_path: Path to the checkpoint file.
    - device: Computation device.
    - decoder: 'bigvgan'
    """
    optimized_model = OptimizedAudioRestorationModel(device=device)
    
    if decoder == 'bigvgan':
        bigvgan_model = load_bigvgan_model(device)
        optimized_model.bigvgan_model = bigvgan_model
    else:
        raise ValueError(f"Unsupported decoder: {decoder}")
    
    state_dict = torch.load(save_path, map_location=device)

    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    optimized_model.voice_restore.load_state_dict(state_dict, strict=True)

    return optimized_model


def restore_audio(model, input_path, output_path, steps=16, cfg_strength=0.1, window_size_sec=5.0, overlap=0.1, batch_size=32, decoder='bigvgan'):  
    # Load the audio file
    start_time = time.time()

    initial_gpu_memory = measure_gpu_memory(device)
    wav, sr = librosa.load(input_path, mono=True, sr=model.bigvgan_model.h.sampling_rate)
    wav = torch.FloatTensor(wav).unsqueeze(0)  # Shape: [1, num_samples]

    window_size_samples = int(window_size_sec * sr)
    step_size = int(window_size_samples * (1 - overlap))

    # Apply overlapping windowing to the waveform
    wav_windows = apply_overlap_windowing_waveform(wav, window_size_samples, overlap)

    num_windows = wav_windows.size(0)
    restored_wav_windows = []

    for i in tqdm(range(0, num_windows, batch_size)):
        batch_wav_windows = wav_windows[i:i+batch_size]  # Shape: [batch_size, 1, window_size_samples]
        batch_wav_windows = batch_wav_windows.to(device)

        # Convert to Mel-spectrogram using BigVGAN's spectrogram configurations
        batch_processed_mel = get_mel_spectrogram(batch_wav_windows.squeeze(1), model.bigvgan_model.h).to(device)

        # Restore mel-spectrogram using voice_restore model
        with torch.no_grad():
            with torch.autocast(device):
                restored_mel = model.voice_restore.sample(
                    batch_processed_mel.transpose(1, 2),
                    steps=steps,
                    cfg_strength=cfg_strength
                )
                restored_mel = restored_mel.transpose(1, 2)  # Shape: [batch_size, mel_bins, time_steps]
    
        with torch.no_grad():
            if decoder == 'bigvgan':
                with torch.autocast(device):
                    restored_wav = model.bigvgan_model(restored_mel).float().cpu()  # Shape: [batch_size, num_samples]
            else:
                raise ValueError(f"Unsupported decoder: {decoder}")

        restored_wav_windows.append(restored_wav)
        del batch_wav_windows, batch_processed_mel, restored_wav
        torch.cuda.empty_cache()

    restored_wav_windows = torch.cat(restored_wav_windows, dim=0)  # Shape: [num_windows, num_samples]

    # Reconstruct the full waveform from the processed windows
    restored_wav = reconstruct_waveform_from_windows(restored_wav_windows, window_size_samples, overlap, original_length=wav.shape[-1])

    # Ensure the restored_wav has correct dimensions for saving
    if restored_wav.dim() == 1:
        restored_wav = restored_wav.unsqueeze(0)  # Shape: [1, num_samples]

    # Save the restored audio
    torchaudio.save(output_path, restored_wav, model.bigvgan_model.h.sampling_rate)

    end_time = time.time()
    total_time = end_time - start_time
    peak_gpu_memory = measure_gpu_memory(device)
    gpu_memory_used = peak_gpu_memory - initial_gpu_memory

    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Peak GPU memory usage: {peak_gpu_memory:.2f} MB")
    print(f"GPU memory used: {gpu_memory_used:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Audio restoration using OptimizedAudioRestorationModel for long-form audio.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--input', type=str, required=True, help="Path to the input audio file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the restored audio file")
    parser.add_argument('--steps', type=int, default=16, help="Number of sampling steps")
    parser.add_argument('--cfg_strength', type=float, default=0.5, help="CFG strength value")
    parser.add_argument('--window_size_sec', type=float, default=5.0, help="Window size in seconds for overlapping")
    parser.add_argument('--overlap', type=float, default=0.5, help="Overlap ratio for windowing")
    parser.add_argument('--decoder', type=str, choices=['bigvgan'], default='bigvgan', help="Decoder to use for waveform reconstruction")

    args = parser.parse_args()

    # Set device, handle MacBooks with M1 chip
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the optimized model with the selected decoder
    optimized_model = load_model(args.checkpoint, device, args.decoder)

    # Set model precision and move to device
    if args.decoder == 'bigvgan':
        if device == 'cuda':
            optimized_model.bigvgan_model.bfloat16()


    optimized_model.voice_restore = optimized_model.voice_restore.eval().to(device)
    optimized_model = optimized_model.eval().to(device)


    # Restore the audio
    restore_audio(
        optimized_model, 
        args.input, 
        args.output, 
        steps=args.steps, 
        cfg_strength=args.cfg_strength, 
        window_size_sec=args.window_size_sec, 
        overlap=args.overlap,
        batch_size=16,
        decoder=args.decoder
    )


if __name__ == "__main__":
    main()
