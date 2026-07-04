import os
import yt_dlp
from faster_whisper import WhisperModel
from google import genai # UPDATED IMPORT

class VideoDownloader:
    def __init__(self, download_dir="./audio"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def extract_audio(self, url: str) -> str:
        print(f"\n[*] Downloading audio from: {url}")
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
            'outtmpl': f'{self.download_dir}/%(id)s.%(ext)s',
            'quiet': True, 
            'no_warnings': True
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = f"{self.download_dir}/{info['id']}.wav"
                print(f"[+] Audio saved successfully to {file_path}")
                return file_path
        except Exception as e:
            print(f"[!] Error downloading video: {e}")
            return None

class Transcriber:
    def __init__(self):
        print("[*] Initializing Whisper Engine...")
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str) -> str:
        print(f"[*] Transcribing {audio_path}...")
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
        print("[+] Transcription complete!")
        return transcript.strip()

# --- UPDATED FACT CHECKER CLASS ---
class FactChecker:
    def __init__(self, api_key: str):
        print("[*] Initializing Fact-Checking AI...")
        # Initialize the new genai client
        self.client = genai.Client(api_key=api_key)

    def verify_claims(self, transcript: str) -> str:
        print("[*] Analyzing transcript for factual claims...")
        
        prompt = f"""
        You are an expert fact-checker. Analyze the following video transcript.
        1. Identify the core factual claims being made.
        2. Fact-check those claims using your internal knowledge.
        3. Provide a short, 3-sentence verdict stating if the video is True, False, or Misleading, and why.
        
        Transcript to analyze:
        "{transcript}"
        """
        
        try:
            # Generate content using the current model name
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            print("[+] Fact-check complete!")
            return response.text
        except Exception as e:
            print(f"[!] AI API Error: {e}")
            return "Could not verify claims due to an error."

# --- Execution Test ---
if __name__ == "__main__":
    # REPLACE THIS with your actual API key!
    GEMINI_API_KEY = "" 
    
    test_url = "https://www.youtube.com/watch?v=sAg07qAM7Xc" 
    
    downloader = VideoDownloader()
    transcriber = Transcriber()
    checker = FactChecker(api_key=GEMINI_API_KEY)

    audio_file = downloader.extract_audio(test_url)
    
    if audio_file:
        text = transcriber.transcribe(audio_file)
        print(f"\n[Transcript]: {text}\n")
        
        verdict = checker.verify_claims(text)
        
        print("=============================")
        print("       FINAL VERDICT         ")
        print("=============================")
        print(verdict)
        print("=============================\n")
