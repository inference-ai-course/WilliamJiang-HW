from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
import os
from gtts import gTTS
import tempfile
import ollama

# Load models once when the app starts
asr_model = whisper.load_model("small")
response = ollama.chat(model='llama3.1:8b')
# Initialize conversation history
conversation_history = []

app = FastAPI()

def transcribe_audio(audio_bytes):
    #Transcribe audio bytes using Whisper ASR
    # Save audio bytes to temporary file
    temp_file = "temp.wav"
    with open(temp_file, "wb") as f:
        f.write(audio_bytes)
    
    # Transcribe the audio
    result = asr_model.transcribe(temp_file)
    
    # Clean up temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return result["text"]

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    
    # Construct prompt from conversation history
    prompt = ""
    for turn in conversation_history[-5:]:  # Keep last 5 turns
        prompt += f"{turn['role']}: {turn['text']}\n"
    
    # Use Ollama to generate response
    try:
        response = ollama.chat(model='llama3.1:8b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        bot_response = response['message']['content']
    except Exception as e:
        bot_response = f"I'm sorry, I encountered an error: {str(e)}"
    
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response

def synthesize_speech(text, filename="response.mp3"):
    # Use gTTS for text-to-speech
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)
    return filename

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    #Endpoint to handle audio file upload and return transcribed text
    # Read the uploaded audio file
    audio_bytes = await file.read()
    
    # Transcribe the audio
    user_text = transcribe_audio(audio_bytes)
    
    # Generate bot response using LLM
    bot_text = generate_response(user_text)
    
    # Synthesize speech from bot response
    audio_path = synthesize_speech(bot_text)
    
    # Return the audio file
    return FileResponse(audio_path, media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#http://localhost:8000/docs#/default/chat_endpoint_chat__post