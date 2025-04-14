import soundfile as sf
import numpy as np
from numpy.typing import NDArray
import json
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader
import onnxruntime as ort


_BOS = "^"
_EOS = "$"
_PAD = "_"


class Piper:
    def __init__(
            self, 
            model_path: str, 
            config_path: str,
            providers = ["CPUExecutionProvider"]
        ):
        self.model_path = model_path
        with open(config_path) as fp:
            self.config: dict = json.load(fp)
        self.sample_rate: int = self.config['audio']['sample_rate']
        self.phoneme_id_map: dict = self.config['phoneme_id_map']
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        self.sess = ort.InferenceSession(
            model_path, 
            sess_options=ort.SessionOptions(),
            providers=providers
        )

    def create(
            self, 
            text: str, 
            speaker_id = 0, 
            is_phonemes = False
        ) -> tuple[NDArray[np.float32], int]:
        inference_cfg = self.config['inference']
        length_scale = inference_cfg['length_scale']
        noise_scale = inference_cfg['noise_scale']
        noise_w = inference_cfg['noise_w']
        
        phonemes = text if is_phonemes else phonemize(text)
        phonemes = list(phonemes)
        phonemes.insert(0, _BOS)

        ids = self._phoneme_to_ids(phonemes)
        inputs = self._create_input(ids, length_scale, noise_w, noise_scale, speaker_id)
        samples = self.sess.run(None, inputs)[0].squeeze((0,1)).squeeze()
        return samples, self.sample_rate
    
    def _phoneme_to_ids(self, phonemes: str) -> list[int]:
        ids = []
        for p in phonemes:
            if p in self.phoneme_id_map:
                ids.extend(self.phoneme_id_map[p])
                ids.extend(self.phoneme_id_map[_PAD])
        ids.extend(self.phoneme_id_map[_EOS])
        return ids
    
    def _create_input(self, ids, length_scale, noise_w, noise_scale, speaker_id) -> dict:
        ids = np.expand_dims(np.array(ids, dtype=np.int64), 0)
        length = np.array([ids.shape[1]], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w],dtype=np.float32)
        # speaker = np.array([speaker_id], dtype=np.int64) if speaker_id is not None else None
        return {
            'input': ids,
            'input_lengths': length,
            'scales': scales,
            # 'sid': speaker
        }