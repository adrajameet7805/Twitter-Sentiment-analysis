import os
import subprocess
from backend.audio_processor import extract_audio_ffmpeg

def test_ffmpeg():
    print("Checking FFmpeg version...")
    try:
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print("FFmpeg found:", res.stdout.split('\n')[0])
    except Exception as e:
        print("FFmpeg NOT found in PATH. Error:", e)

if __name__ == "__main__":
    test_ffmpeg()
