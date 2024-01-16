import torch
import torchaudio

import torch

from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")

processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

raw_audio, raw_sr = torchaudio.load("./test_audio.wav")
raw_audio = raw_audio.reshape(-1)

def preprocess_audio(audio: torch.Tensor, sample_rate: int, audio_sample_rate: int = 16000):
    if sample_rate != audio_sample_rate:
        waveform = torchaudio.functional.resample(audio, sample_rate, audio_sample_rate)
    return waveform

raw_audio = preprocess_audio(raw_audio, raw_sr)

inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt")

generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(transcription)


# def preprocess_audio(audio: torch.Tensor, sample_rate: int, audio_sample_rate: int = 16000):
#     if sample_rate != audio_sample_rate:
#         waveform = torchaudio.functional.resample(audio, sample_rate, audio_sample_rate)
#     return waveform

# raw_audio, sr = torchaudio.load("test_audio.wav")
# audio_file = preprocess_audio(raw_audio, sr)

# from transformers import pipeline

# transcriber = pipeline("automatic-speech-recognition")
# transcriber(audio_file)





# model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")

# processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


# raw_audio, raw_sr = torchaudio.load("./test_audio.wav")
# print(raw_sr)
# inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt")

# generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

# transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
# print(transcription)