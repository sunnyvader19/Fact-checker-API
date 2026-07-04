import os
import yt_dlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
from google import genai
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load the secret variables from the .env file
load_dotenv()

# --- 1. Data Models ---
class VideoRequest(BaseModel):
    url: str

class FactCheckResponse(BaseModel):
    url: str
    transcript: str
    caption: str  
    verdict: str

# --- 2. Pipeline Classes ---
class VideoDownloader:
    def __init__(self, download_dir="./audio"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def extract_media_and_metadata(self, url: str, max_duration: int = 180):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
            'outtmpl': f'{self.download_dir}/%(id)s.%(ext)s',
            'quiet': True, 
            'no_warnings': True
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 1. Fetch metadata first to check length and platform locks
                try:
                    meta = ydl.extract_info(url, download=False)
                except Exception as meta_e:
                    error_str = str(meta_e).lower()
                    if "empty media response" in error_str or "logged-in" in error_str:
                        raise ValueError("The platform blocked the request. The content might be private, or the platform requires login credentials.")
                    raise ValueError(f"Could not fetch post information: {meta_e}")

                if not meta:
                    raise ValueError("Could not extract any metadata from this link.")

                duration = meta.get('duration', 0)
                if duration and duration > max_duration:
                    raise ValueError(f"Video is too long ({duration} seconds). Maximum allowed is {max_duration} seconds.")
                
                # 2. Extract caption or fallback to title
                caption = meta.get('description', '') 
                if not caption:
                    caption = meta.get('title', 'No caption found.')

                # 3. Try to download audio (fails safely if it's a text-only post like Twitter/Reddit)
                try:
                    info = ydl.extract_info(url, download=True)
                    file_path = f"{self.download_dir}/{info['id']}.wav"
                    return file_path, caption
                except Exception as dl_error:
                    print(f"[*] Note: Could not download audio (likely a text-only post). Error: {dl_error}")
                    return None, caption
                
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Failed to process media: {str(e)}")

class Transcriber:
    def __init__(self):
        # Initialized once on startup
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str) -> str:
        if not audio_path or not os.path.exists(audio_path):
            return ""
            
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        return " ".join([segment.text for segment in segments]).strip()

class FactChecker:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def verify_claims(self, transcript: str, caption: str) -> str:
        prompt = f"""
        You are an expert fact-checker and media analyzer. Analyze the following media transcript and/or post caption.

        CRITICAL RULES:
        1. If the content is clearly just a joke, a meme, a skit, or purely for entertainment with no serious factual claims, you MUST reply ONLY with: "That's just an entertaining media."
        2. If the content contains song lyrics, identify the song name and artist, and state that it is a song.
        3. If there are actual factual claims made, fact-check them using your internal knowledge. Provide a short verdict stating if the post is True, False, or Misleading, and why.
        
        Spoken Transcript: "{transcript}"
        Written Caption/Description: "{caption}"
        """
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

# --- 3. FastAPI Application Setup ---
app = FastAPI(title="Video Fact-Checking API", version="2.0")

# Enable CORS for the frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Securely load the API key from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Please check your .env file.")

# Initialize global engine components
downloader = VideoDownloader()
transcriber = Transcriber()
checker = FactChecker(api_key=GEMINI_API_KEY)

# --- 4. The API Endpoint ---
@app.post("/analyze", response_model=FactCheckResponse)
def analyze_video(request: VideoRequest):
    try:
        # Step 1: Ingestion
        audio_file, caption = downloader.extract_media_and_metadata(request.url)
        
        # Step 2: Transcription
        transcript = transcriber.transcribe(audio_file)
        
        # Step 3: Cleanup disk space
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
            
        # Step 4: AI Analysis
        verdict = checker.verify_claims(transcript, caption)
        
        return FactCheckResponse(
            url=request.url,
            transcript=transcript,
            caption=caption,
            verdict=verdict
        )
        
    except ValueError as ve:
        # Specific user errors (too long, blocked by platform)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Unexpected system failures
        raise HTTPException(status_code=500, detail=str(e))
