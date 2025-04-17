import torch
import torchaudio
from pathlib import Path
from typing import Optional, Union
from .model import VoiceRestoreModel
from .utils import download_checkpoint
from .bigvgan import bigvgan


def load_bigvgan_model(device):
    """Load and optimize BigVGAN model"""
    # Use cache directory for model loading
    cache_dir = Path("./model_cache")
    cache_dir.mkdir(exist_ok=True)

    # print(f"Loading BigVGAN model...")
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "nvidia/bigvgan_v2_24khz_100band_256x",
        use_cuda_kernel=False,
        force_download=False,
        cache_dir=str(cache_dir),
    )

    # Important: move model to device BEFORE removing weight norm
    bigvgan_model = bigvgan_model.to(device)
    bigvgan_model.remove_weight_norm()

    return bigvgan_model.eval()


class AudioRestorer:
    """Base class for audio restoration models."""

    def __init__(
        self,
        checkpoint_id: str = "1QJocOjqj2EUU3PP90eCGP3qAU_R75aYl",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the audio restorer.

        Args:
            checkpoint_id: Google Drive ID of the checkpoint
            device: Device to run the model on
        """
        self.device = device
        self.checkpoint_path = download_checkpoint(checkpoint_id)
        self.model = self._load_model()

    def _load_model(self) -> VoiceRestoreModel:
        """Load the model from checkpoint."""
        model = VoiceRestoreModel(device=self.device)
        model.bigvgan_model = load_bigvgan_model(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.voice_restore.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        self.mel_config = model.bigvgan_model.h
        return model

    def restore_audio(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """
        Restore audio from a file.

        Args:
            input_path: Path to input audio file
            output_path: Optional path to save restored audio

        Returns:
            Restored audio tensor
        """
        raise NotImplementedError("Subclasses must implement restore_audio")

    def _load_audio(self, path: Union[str, Path]) -> tuple[torch.Tensor, int]:
        """Load audio file and convert to tensor."""
        waveform, sample_rate = torchaudio.load(path)
        if waveform.shape[0] > 1:  # Convert to mono if stereo
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

    def _save_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        path: Union[str, Path],
    ) -> None:
        """Save audio tensor to file."""
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if len(waveform.shape) > 2:
            waveform = waveform.squeeze(0)
        torchaudio.save(path, waveform.cpu(), sample_rate)


class ShortAudioRestorer(AudioRestorer):
    """Restorer for short audio clips (up to a few seconds)."""

    def __init__(
        self,
        checkpoint_id: str = "1QJocOjqj2EUU3PP90eCGP3qAU_R75aYl",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(checkpoint_id, device)

    def restore_audio(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        steps: int = 32,
        cfg_strength: float = 0.5,
    ) -> torch.Tensor:
        """
        Restore a short audio clip.

        Args:
            input_path: Path to input audio file
            output_path: Optional path to save restored audio
            steps: Number of steps for the ODE solver
            cfg_strength: Strength of the classifier-free guidance

        Returns:
            Restored audio tensor
        """
        waveform, sample_rate = self._load_audio(input_path)
        waveform = waveform.to(self.device)

        # Get mel spectrogram
        # mel = get_mel_spectrogram(waveform, self.mel_config)
        # mel = mel.to(self.device)

        # Restore audio
        with torch.no_grad():
            restored = self.model(waveform, steps=steps, cfg_strength=cfg_strength)

        if output_path is not None:
            self._save_audio(restored, sample_rate, output_path)

        return restored


class LongAudioRestorer(AudioRestorer):
    """Restorer for long audio clips (several minutes or more)."""

    def __init__(
        self,
        checkpoint_id: str = "1QJocOjqj2EUU3PP90eCGP3qAU_R75aYl",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the long audio restorer.

        Args:
            checkpoint_id: Google Drive ID of the checkpoint
            device: Device to run the model on
        """
        super().__init__(checkpoint_id, device)

    def restore_audio(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        steps: int = 32,
        cfg_strength: float = 0.5,
        chunk_size: int = 16000 * 5,
        overlap: int = 16000,
    ) -> torch.Tensor:
        """
        Restore a long audio clip by processing it in chunks.

        Args:
            input_path: Path to input audio file
            output_path: Optional path to save restored audio
            steps: Number of steps for the ODE solver
            cfg_strength: Strength of the classifier-free guidance
            chunk_size: Size of audio chunks in samples
            overlap: Overlap between chunks in samples

        Returns:
            Restored audio tensor
        """
        waveform, sample_rate = self._load_audio(input_path)
        waveform = waveform.to(self.device)

        # Process in chunks
        restored_chunks = []
        for i in range(0, waveform.shape[1], chunk_size - overlap):
            chunk = waveform[:, i : i + chunk_size]
            if chunk.shape[1] < chunk_size:
                # Pad last chunk if needed
                pad_size = chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))

            # Restore chunk
            with torch.no_grad():
                restored_chunk = self.model(chunk)

            # Remove padding from last chunk
            if i + chunk_size > waveform.shape[1]:
                restored_chunk = restored_chunk[:, : waveform.shape[1] - i]

            restored_chunks.append(restored_chunk)

        # Combine chunks with overlap
        restored = self._combine_chunks(restored_chunks)

        if output_path is not None:
            self._save_audio(restored, sample_rate, output_path)

        return restored

    def _combine_chunks(
        self, chunks: list[torch.Tensor], overlap: int = 16000
    ) -> torch.Tensor:
        """Combine restored chunks with overlap."""
        if len(chunks) == 1:
            return chunks[0]

        chunks = [chunk.unsqueeze(0) for chunk in chunks]

        # Create window for crossfade
        window = torch.hann_window(overlap * 2, device=self.device)
        window1 = window[:overlap]
        window2 = window[overlap:]

        # Initialize output
        total_length = sum(chunk.shape[1] for chunk in chunks) - overlap * (
            len(chunks) - 1
        )
        output = torch.zeros((1, total_length), device=self.device)

        # Combine chunks with crossfade
        current_pos = 0
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk
                output[:, : chunk.shape[1]] = chunk
            else:
                # Crossfade with previous chunk
                prev_chunk = chunks[i - 1]
                output[:, current_pos : current_pos + overlap] = (
                    prev_chunk[:, -overlap:] * window2 + chunk[:, :overlap] * window1
                )
                output[:, current_pos + overlap : current_pos + chunk.shape[1]] = chunk[
                    :, overlap:
                ]
            current_pos += chunk.shape[1] - overlap

        return output
