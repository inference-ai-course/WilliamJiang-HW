import whisper

# Load model
model = whisper.load_model("base")  # or "small", "medium", "large"

# Transcribe audio
result = model.transcribe("D:/Inference Ai Stuff/Lecture 3/Lecture 3 Practice/test_data/audio/sample-0.mp3")
print("📄 Whisper Transcription:")
print(result["text"])