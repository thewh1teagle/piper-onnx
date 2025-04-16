"""
Setup
    uv pip install piper-onnx soundfile

Prepare models
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json

Run
    uv run python simple.py
"""

from piper_onnx import Piper
import soundfile as sf

piper = Piper('en_US-arctic-medium.onnx', 'en_US-arctic-medium.onnx.json')

voices = piper.get_voices()
print(f'Found {len(voices)} voices.')
samples, sample_rate = piper.create('Hello world from Piper!', speaker_id=voices['awb'])
sf.write('audio.wav', samples, sample_rate)
print("Created audio.wav")