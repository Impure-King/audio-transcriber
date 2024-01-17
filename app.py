## Main File for Route Management

# Important libraries:
import os # Gets file paths and manages directories
import subprocess # MP3 to Wav converter
import torchaudio # Audio Loader and Saver
from flask import Flask, render_template, request # Route managers and Input receivers:
from random import randint # Random seed generator

# Custom module file
from constants import AUDIO_PLAYBACK_PATH
from wavenet_loading import forward_computation, preprocess_audio

# Initiating app instance
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
@app.route('/home', methods=["GET", "POST"])
def home():
    """A function that manages home page of the website.
    At the home page, an upload button will be available for users to upload on.

    Returns:
        Appropriate HTML after Jinja handles inheritance.
    """
    return render_template("audio.html")


@app.route('/upload', methods=["POST"])
def uploaded():
    """A function that manages audio retrieval and transcription.
    It is a POST function and only activates when the user hits submit.
    
    Returns:
        Appropriate HTML page that has audio transcriptions and has audio playback."""
    
    # Input Handling:
    if "audioFile" not in request.files: # Checking if the user uploaded an audio file.
        return "No file found."
    
    # Retrieving the audio file:
    audio_file = request.files['audioFile']

    # Second Validation:
    if audio_file.filename == '':
        return "No selected file"
    
    # Checking if audio file type is MP3 or WAV:
    if audio_file.mimetype not in ["audio/wav", "audio/x-wav", "audio/mpeg"]:
        return f"Invalid File Format. Please upload a WAV or a MP3 file."
    
    
    # Saving file for appropriate processing:
    file_path = "uploads/" + str(randint(1, 1000)) + audio_file.filename # Adding some seeding
    audio_file.save(file_path)

    # Converting MP3 to WAV if needed:
    if audio_file.filename[-3:] == "mp3":
        subprocess.call(['ffmpeg', '-i', file_path, file_path[:-3] + 'wav']) # Conversion line
        os.remove(file_path) # Deleting MP3 version
        file_path = file_path[:-3] + 'wav' # Updating file_path
    
    # Loading data for preprocessing:    
    audio, sr = torchaudio.load(file_path)
    # Creating an audio playback file:
    torchaudio.save(AUDIO_PLAYBACK_PATH, audio, sr)
    
    # Preprocessing and transcribing with imported functions:
    audio = preprocess_audio(audio, sr)
    transcript = forward_computation(audio)[0]
    transcript = transcript[0].upper() + transcript[1:] + '.'
    
    # Removing the excess path:
    os.remove(file_path)
    return render_template("analysed.html", transcription=transcript, audio_path=f".{AUDIO_PLAYBACK_PATH}")

if __name__ == "__main__":
    app.run(debug=True)

