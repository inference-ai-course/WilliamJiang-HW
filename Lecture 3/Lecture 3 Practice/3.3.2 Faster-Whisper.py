from faster_whisper import WhisperModel

# Load model with float16 for speed
model = WhisperModel("base", device="cpu", compute_type="int8")  # For CPUs

# Transcribe
segments, _ = model.transcribe("D:/Inference Ai Stuff/Lecture 3/Lecture 3 Practice/test_data/audio/sample-1.mp3")

print("📄 Faster-Whisper Transcription:")
for segment in segments:
    print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
