import os
import json
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

def get_client():
    """Dynamically loads environmental variables and initializes the HF client."""
    # Build absolute path to .env file relative to this script's directory
    core_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(core_dir)
    env_path = os.path.join(project_root, ".env")
    
    load_dotenv(dotenv_path=env_path, override=True)
    api_key = os.getenv("HF_TOKEN")
    
    if not api_key or api_key == "your_hf_token_here" or api_key.strip() == "":
        return None
    
    return InferenceClient(api_key=api_key.strip())

def fetch_llm_response(prompt: str, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> dict:
    """
    Sends a prompt to the Hugging Face Inference API and expects a structured JSON response.
    """
    client = get_client()
    
    if not client:
        return {"error": "HF_TOKEN is missing or invalid. Please check your .env file."}
    
    try:
        messages = [{"role": "user", "content": prompt}]
        
        # Reverting to chat_completion as requested
        response = client.chat_completion(
            model=model_name,
            messages=messages,
            max_tokens=250,
            temperature=0.1
        )
        
        output_text = response.choices[0].message.content.strip()
        
        # Open source models sometimes wrap JSON in markdown (```json ... ```)
        start = output_text.find('{')
        end = output_text.rfind('}')
        if start != -1 and end != -1:
            output_text = output_text[start:end+1]
            
        return json.loads(output_text)
        
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg:
            return {"error": "Hugging Face Permission Error: Please ensure your token has 'Inference' permissions enabled on the HF website."}
        print(f"Error during LLM inference: {e}")
        return {"error": error_msg}

def transcribe_audio(audio_bytes: bytes, model_name: str = "openai/whisper-large-v3-turbo") -> str:
    """
    Sends raw audio bytes to Hugging Face's Whisper model to return transcribed text.
    """
    client = get_client()
    if not client:
        return "Error: HF_TOKEN is missing or invalid."
        
    import tempfile
    # Save bytes to a temporary wav file so the HF API can infer the Content-Type
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        # The InferenceClient's automatic_speech_recognition method handles audio to text
        text = client.automatic_speech_recognition(
            audio=tmp_path,
            model=model_name
        )
        # It typically returns a dict with a 'text' key or just a string
        if isinstance(text, dict):
            return text.get("text", "").strip()
        return str(text).strip()
    except Exception as e:
        print(f"Audio transcription error: {e}")
        return f"Error transcribing audio: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
