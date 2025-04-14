"""
Setup
    uv pip install piper-onnx soundfile

Prepare models
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json

Run
    uv run python simple.py
"""

from piper_onnx import Piper
import soundfile as sf

piper = Piper('en_US-ryan-medium.onnx', 'en_US-ryan-medium.onnx.json')
samples, sample_rate = piper.create('Hello world from Piper!')
sf.write('audio.wav', samples, sample_rate)
print("Created audio.wav")