
from fileinput import filename
import generate_HTTPs
from openai.types import upload
from generate_HTTPs import upload_file_path, generate_audio_counter
import generate_SpeechtoText


#1. Get an audio file from the user from an HTTP frontend
#I want it to be able to also record audio and save it, live from a button TODO
#I might need to make a while loop or a loop so that it can exit and continue onto the next part of the code
    #make a global variable?
#inputaudio = file_path
#print(f"The Saved Audio Path is {inputaudio}")
# Start the HTTP server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(generate_HTTPs.app, host="127.0.0.1", port=8000)
#2. transcribe the audio file into text
#It automatically runs generate_SpeechtoText.py

#3. use the text to generate a response from a LLM, maybe openAI or Ollama

#4. Generate a response // TODO still need to be able to input the text into the "generate_audio.py" 
#bonus: change tone of the voice to fit the type of response

#5. TODO make sure that it can support 5-turn response memory.