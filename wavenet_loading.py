import torch
import torchaudio
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")

processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

def preprocess_audio(audio: torch.Tensor, sample_rate: int, audio_sample_rate: int = 16000):
    if sample_rate != audio_sample_rate:
        waveform = torchaudio.functional.resample(audio, sample_rate, audio_sample_rate)
    return waveform

def forward_computation(audio:torch.Tensor):
    audio = audio.reshape(-1)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription

# def postprocess_transcription(transcript: str):
#     transcript = transcript.lower()
#     new_str = ""
#     for letter in transcript:
#         if letter == "|":
#             new_str += " "
#         else:
#             new_str += letter
#     new_str += '.'
#     transcript = new_str[0].upper() + new_str[1:-2] + '.'
#     return transcript