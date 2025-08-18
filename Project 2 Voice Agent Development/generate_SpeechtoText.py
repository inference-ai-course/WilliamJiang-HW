from generate_HTTPs import upload_file_path, transcribe_audio

# This file now just provides access to the transcription functionality
# The actual transcription happens automatically when files are uploaded

def get_transcription(audio_path=None):
    """Get transcription for an audio file"""
    if audio_path is None:
        audio_path = upload_file_path
    
    if audio_path and audio_path != "None":
        return transcribe_audio(audio_path)
        
    else:
        print("No audio file available for transcription")
        return None

# Example usage
if __name__ == "__main__":
    transcription = get_transcription()
    if transcription:
        print(f"Transcription: {transcription}")


