from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from faster_whisper import WhisperModel

app = FastAPI()
BASE_DIR = Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global variable for file path
upload_file_path = None
transcribe_file_path = None
generate_audio_counter = 0
# Initialize Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

def get_unique_filename(upload_dir: Path, original_filename: str) -> Path:
    """Generate a unique filename by adding counter if file exists"""
    upload_filename = Path(original_filename).stem  # Get filename without extension
    upload_extension = Path(original_filename).suffix  # Get file extension
    
    upload_counter = 1
    new_filename = f"uploaded_{original_filename}"
    upload_file_path = upload_dir / new_filename
    
    # Keep incrementing counter until we find a unique filename
    while upload_file_path.exists():
        new_filename = f"uploaded_{upload_filename}_{upload_counter}{upload_extension}"
        upload_file_path = upload_dir / new_filename
        upload_counter += 1
    
    return upload_file_path

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper and save to file"""
    global generate_audio_counter
    try:
        segments, _ = model.transcribe(str(audio_path))
        transcription = ""
        #print("📄 Faster-Whisper Transcription:")
        for segment in segments:
            transcription += segment.text + " "
            print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
        
        # Save transcription to file
        transcriptions_dir = BASE_DIR / "transcriptions"
        transcriptions_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the audio filename without extension for the transcription filename
        audio_filename = Path(audio_path).stem
        transcription_filename = f"{audio_filename}_transcription.txt"
        transcribe_file_path = transcriptions_dir / transcription_filename
        
        with open(transcribe_file_path, "w", encoding="utf-8") as f:
            f.write(transcription.strip())
        
        print(f"Transcription saved to: {transcribe_file_path}")
        generate_audio_counter = 1
        print(f"generate_audio_counter set to: {generate_audio_counter}")
        return transcription.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return None
    
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-audio/")
async def upload_audio(audio_file: UploadFile = File(...)):
    # Handle audio file upload and automatically transcribe
    global upload_file_path, generate_audio_counter
    
    # Validate file type
    if not audio_file.content_type.startswith("audio/"):
        return {"error": "File must be an audio file"}
    
    # Create uploads directory if it doesn't exist
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique filename with counter
    upload_file_path = get_unique_filename(upload_dir, audio_file.filename)
    print(f"the file has been stored into {upload_file_path}")
    
    with open(upload_file_path, "wb") as buffer:
        content = await audio_file.read()
        buffer.write(content)
    
    # Automatically transcribe the uploaded audio
    print("Starting transcription...")
    transcription = transcribe_audio(upload_file_path)
    
    # Generate audio response after transcription
    if transcription and generate_audio_counter == 1:
        print("Transcription complete, generating audio response...")
        try:
            # This part is the area where it inputs the transcription into openAI gpt models and generate a response
            import importlib.util
            import openai
            from generate_SpeechtoText import get_transcription
            

            client = openai.OpenAI()

            latest_transcription = get_transcription(upload_file_path)
            #print(f"The audio message says: {latest_transcription}")
            # Use the transcription text directly instead of trying to read the audio file
            if latest_transcription:
                input_gpt_text = latest_transcription
            else:
                input_gpt_text = "No transcription available" 
            def get_completion_with_system_prompt(system_prompt, user_prompt, model="gpt-4o-mini"):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
                return response.choices[0].message.content

            # Define the system and user prompts
            system_prompt = "You are a helpful assistant that responds in a humorous tone."
            user_prompt = f"Can you answer this message? {input_gpt_text}"
            global generate_GPT_response

            generate_GPT_response = get_completion_with_system_prompt(system_prompt, user_prompt)
            print(generate_GPT_response)

            # Import and run generate_audio module
            # Note: change path when saved to a different location
            audio_response = importlib.util.spec_from_file_location("generate_audio", "D:/Inference Ai Stuff/Project 2 Voice Agent Development/generate_audio.py")
            generate_audio = importlib.util.module_from_spec(audio_response)
            audio_response.loader.exec_module(generate_audio)
            
            print("Response Audio generated successfully!")
            generate_audio_counter = 0
        except Exception as e:
            print(f"Error generating audio: {e}")
    
    return {
        "message": "Audio file uploaded and transcribed successfully",
        "filename": audio_file.filename,
        "file_path": str(upload_file_path),
        "final_filename": upload_file_path.name,
        "transcription": transcription
    }

@app.get("/transcribe/{filename}")
async def transcribe_file(filename: str):
    """Transcribe a specific uploaded file"""
    upload_dir = BASE_DIR / "uploads"
    upload_file_path = upload_dir / filename
    
    if not upload_file_path.exists():
        return {"error": "File not found"}
    
    transcription = transcribe_audio(upload_file_path)
    return {
        "filename": filename,
        "transcription": transcription
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "generate_HTTPs:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )