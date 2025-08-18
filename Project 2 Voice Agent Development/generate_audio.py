from pathlib import Path
from openai import OpenAI
import os
from datetime import datetime
from generate_HTTPs import upload_file_path, generate_audio_counter
from generate_SpeechtoText import get_transcription
from generate_HTTPs import generate_GPT_response
def get_filename_with_datetime(base_dir, prefix="audiofile", extension=".mp3"):
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")  # YYYY-MM-DD format
    time_str = now.strftime("%I-%M-%S-%p")  # HH-MM-SS format (%I=12hour, %H=24hour, %M=minutes, %S=seconds, %p=AM/PM)
    
    # Format: audiofile_20241201_143022.mp3
    filename = f"{prefix}_{date_str}_{time_str}{extension}"
    return base_dir / filename

#API Key
client = OpenAI()
# Base directory for audio files --> where the file goes
audio_dir = Path("D:/Inference Ai Stuff/Project 2 Voice Agent Development/Audio Generations")

# Get the current transcription for this audio file
'''current_transcription = get_transcription(upload_file_path)
print("hello \n\n")'''
#print(f"Current transcription: {current_transcription}")

# Get filename with datetime
speech_file_path = get_filename_with_datetime(audio_dir, "audiofile", ".mp3")

# Generate audio using the current transcription
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input= f"{generate_GPT_response}",
    instructions="Speak in a tone and manner that befits the input.",
) as response:
    response.stream_to_file(speech_file_path)

print(f"Audio generated and saved to: {speech_file_path}")

# Reset counter to 0 after generating audio
generate_audio_counter = 0
print(f"generate_audio_counter in generate_audio.py is now reset to: {generate_audio_counter}")
