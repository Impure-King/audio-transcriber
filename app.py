from wavenet_loading import forward_computation, preprocess_audio
import torchaudio
from flask import Flask, render_template, request
import random
import os
import subprocess

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
@app.route('/home', methods=["GET", "POST"])
def home():
    return render_template("audio.html")

@app.route('/upload', methods=["POST"])
def uploaded():
    if "audioFile" not in request.files:
        return "No file found."
    
    audio_file = request.files['audioFile']

    if audio_file.filename == '':
        return "No selected file"
    
    if audio_file.mimetype not in ["audio/wav", "audio/x-wav", "audio/mpeg"]:
        return f"Invalid File Format. Please upload a WAV or a MP3 file."
    
    file_path = "uploads/" + str(random.randint(1, 1000)) + audio_file.filename # Adding some seeding


    audio_file.save(file_path)

    if audio_file.filename[-3:] == "mp3":
        subprocess.call(['ffmpeg', '-i', file_path, file_path[:-3] + 'wav'])
        os.remove(file_path)
        file_path = file_path[:-3] + 'wav'
    audio, sr = torchaudio.load(file_path)
    torchaudio.save("static/test_audio.wav", audio, sr)
    audio = preprocess_audio(audio, sr)
    transcript = forward_computation(audio)[0]
    # transcript = postprocess_transcription(transcript)
    os.remove(file_path)
    return render_template("analysed.html", transcription=transcript[0].upper() + transcript[1:] + '.')

if __name__ == "__main__":
    app.run(debug=True)

# Neural Network Things:

# # Creating a basic input system:
# file_name = "test.wav"

# # Getting wave from file:
# audio, sr = torchaudio.load(file_name)
# audio = preprocess_audio(audio, sr)
# audio = audio.to(device)
# transcript = forward_computation(audio)
# print(transcript)