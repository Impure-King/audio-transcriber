import torch
import torchaudio
from torch import nn
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")

# Loading the model:
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)


# Creating an inference pipeline:
class GreedyCTCDecoder(nn.Module):
    def __init__(self, labels, blank: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.blank = blank
    
    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())

def preprocess_audio(audio: torch.Tensor, sample_rate: int):
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(audio, sample_rate, bundle.sample_rate)
    return waveform

def forward_computation(audio:torch.Tensor):
    with torch.inference_mode():
        features, _ = model.extract_features(audio)
        emission, _ = model(audio)
    transcript = decoder(emission[0])
    return transcript

def postprocess_transcription(transcript: str):
    transcript = transcript.lower()
    new_str = ""
    for letter in transcript:
        if letter == "|":
            new_str += " "
        else:
            new_str += letter
    new_str += '.'
    transcript = new_str[0].upper() + new_str[1:-2] + '.'
    return transcript