"""
Note: on Linux you need to run this as well: apt-get install portaudio19-dev

Setup
    uv pip install piper-onnx sounddevice

Prepare models
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json

Run
    uv run python simple.py
"""

from piper_onnx import Piper
import sounddevice as sd

piper = Piper('en_US-ryan-medium.onnx', 'en_US-ryan-medium.onnx.json')
samples, sample_rate = piper.create('Hello world from Piper!')
sd.play(samples, sample_rate)
print('Playing...')
sd.wait()