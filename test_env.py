from faster_whisper import WhisperModel

print("Loading the Whisper model into memory...")
# We use the 'tiny' model for maximum speed during development
model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("Model loaded successfully! Environment is ready.")