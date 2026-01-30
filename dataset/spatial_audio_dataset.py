import os
import random
import json
import copy

import numpy as np
import soundfile as sf
from scipy import signal

import torch

from slam_llm.datasets.base_dataset import BaseDataset

def format_prompt(instruction, input=None):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Based on the audio you've heard, refer to the instruction and provide a response.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


class SpatialAudioDatasetJsonl(BaseDataset):
    def __init__(
            self,
            dataset_config,
            tokenizer,
            split,
        ):
        super().__init__()
        dataset_path = os.path.join(dataset_config['qa_data_root'], dataset_config['stage'], split + '.json')
        self.data = json.load(open(dataset_path))["data"]

        self.anechoic_data_root = dataset_config['anechoic_data_root'] # which is AudioSet in this case
        self.reverb_data_root = dataset_config['reverb_data_root']
        self.channel_type = dataset_config['channel_type']
        
        self.ext_audio = dataset_config['ext_audio']
        self.max_words = dataset_config['max_words']
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)

        self.tokenizer = tokenizer

        self.normalize = dataset_config['normalize']
        self.inference_mode = dataset_config['inference_mode']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        audio_path = os.path.join(self.anechoic_data_root, sample['audio_id'] + self.ext_audio)
        reverb_path = os.path.join(self.reverb_data_root, self.channel_type, sample['reverb_id'])

        if sample['audio_id2'] is not None and sample['reverb_id2'] is not None:
            audio_path2 = os.path.join(self.anechoic_data_root, sample['audio_id2'] + self.ext_audio)
            reverb_path2 = os.path.join(self.reverb_data_root, self.channel_type, sample['reverb_id2'])
        else:
            audio_path2 = None
            reverb_path2 = None
            
        waveforms = self.load_waveform(audio_path, reverb_path, audio_path2, reverb_path2)
        
        prompt = sample['question']
        prompt = format_prompt(prompt, None)
        answer = sample['answer']        
        
        if not self.inference_mode:
            return super().__getitem__((waveforms, None, prompt, answer))
        else:
            base_sample = super().__getitem__((waveforms, None, prompt, answer))
            base_sample.update({
                "key": f"{sample['question_type']}-{sample['question_id']}",
                "target": sample['answer']
            })
            return base_sample
        
    @classmethod
    def normalize_audio(cls, audio_data, target_dBFS=-14.0):
        rms = np.sqrt(np.mean(audio_data**2)) # Calculate the RMS of the audio
    
        if rms == 0:  # Avoid division by zero in case of a completely silent audio
            return audio_data
        
        current_dBFS = 20 * np.log10(rms) # Convert RMS to dBFS
        gain_dB = target_dBFS - current_dBFS # Calculate the required gain in dB
        gain_linear = 10 ** (gain_dB / 20) # Convert gain from dB to linear scale
        normalized_audio = audio_data * gain_linear # Apply the gain to the audio data
        return normalized_audio

    @classmethod
    def load_waveform(cls, audio_path, reverb_path=None, audio_path2=None, reverb_path2=None, normalize=True):
        waveform, sr = sf.read(audio_path)
        
        if len(waveform.shape) > 1:
            waveform = waveform[:, 0] 
        if sr != 32000:
            waveform = signal.resample_poly(waveform, 32000, sr)
            sr = 32000
        if normalize:
            waveform = cls.normalize_audio(waveform, -14.0)

        waveform = waveform.reshape(1, -1)
        if reverb_path is not None:
            reverb = np.load(reverb_path)
            waveform = signal.fftconvolve(waveform, reverb, mode='full')

        waveform = torch.from_numpy(waveform).float()
        waveform = cls.padding(waveform, padding_length=10*sr-waveform.shape[1])

        if audio_path2 is not None and reverb_path2 is not None:
            waveform2, sr2 = sf.read(audio_path2)
            
            if len(waveform2.shape) > 1:
                waveform2 = waveform2[:, 0]
            if sr2 != 32000:
                waveform2 = signal.resample_poly(waveform2, 32000, sr2)
                sr2 = 32000
            if normalize:
                waveform2 = cls.normalize_audio(waveform2, -14.0)
            
            waveform2 = waveform2.reshape(1, -1)
            reverb2 = np.load(reverb_path2)
            waveform2 = signal.fftconvolve(waveform2, reverb2, mode='full')
            waveform2 = torch.from_numpy(waveform2).float()
            waveform2 = cls.padding(waveform2, padding_length=10*sr-waveform2.shape[1])

            waveform = (waveform + waveform2) / 2
        return waveform

    def collator(self, samples):
        audio = torch.stack([s['audio'] for s in samples])
        
        collated = super().collator(samples)
        collated['audio'] = audio
        
        return collated

def get_spatial_audio_dataset(dataset_config, tokenizer, split):
    dataset = SpatialAudioDatasetJsonl(dataset_config, tokenizer, split)
    return dataset


class SpatialAudioDatasetBinauralJsonl(BaseDataset):
    def __init__(
            self,
            dataset_config,
            tokenizer,
            split,
        ):
        super().__init__()
        dataset_path = os.path.join(dataset_config['qa_data_root'], dataset_config['stage'], split + '.json')
        raw = json.load(open(dataset_path))
        if isinstance(raw, dict) and "data" in raw:
            self.data = raw["data"]
        elif isinstance(raw, list):
            self.data = raw
        else:
            raise ValueError(f"Unsupported json format in {dataset_path}")

        self.ext_audio = dataset_config['ext_audio']
        self.max_words = dataset_config['max_words']
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)

        self.tokenizer = tokenizer
        self.normalize = dataset_config['normalize']
        self.inference_mode = dataset_config['inference_mode']

        self.target_sr = 32000
        self.target_len = 10 * self.target_sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        audio_path = sample.get("binaural_path") or sample.get("audio_path") or sample.get("audio")
        if audio_path is None:
            raise KeyError("Missing 'binaural_path' in sample")

        waveforms = self.load_waveform(audio_path)

        prompt = sample['question']
        prompt = format_prompt(prompt, None)
        answer = sample['answer']

        if not self.inference_mode:
            return super().__getitem__((waveforms, None, prompt, answer))
        else:
            base_sample = super().__getitem__((waveforms, None, prompt, answer))
            key = sample.get("binaural_path") or sample.get(
                "id",
                f"{sample.get('question_type', 'doa')}-{sample.get('question_id', index)}",
            )
            base_sample.update({
                "key": key,
                "target": sample['answer']
            })
            return base_sample

    @classmethod
    def normalize_audio(cls, audio_data, target_dBFS=-14.0):
        rms = np.sqrt(np.mean(audio_data**2))
        if rms == 0:
            return audio_data
        current_dBFS = 20 * np.log10(rms)
        gain_dB = target_dBFS - current_dBFS
        gain_linear = 10 ** (gain_dB / 20)
        return audio_data * gain_linear

    def load_waveform(self, audio_path):
        waveform, sr = sf.read(audio_path)

        if waveform.ndim == 1:
            raise ValueError(f"Expected binaural audio, got mono: {audio_path}")
        if waveform.ndim == 2:
            # soundfile returns (T, C); convert to (C, T)
            if waveform.shape[1] == 2:
                waveform = waveform.T
            elif waveform.shape[0] == 2:
                pass
            else:
                raise ValueError(f"Expected 2 channels, got {waveform.shape}: {audio_path}")

        if sr != self.target_sr:
            waveform = signal.resample_poly(waveform, self.target_sr, sr, axis=1)
            sr = self.target_sr

        if self.normalize:
            waveform = np.stack([self.normalize_audio(waveform[ch], -14.0) for ch in range(waveform.shape[0])], axis=0)

        waveform = torch.from_numpy(waveform).float()
        waveform = self.padding(waveform, padding_length=self.target_len - waveform.shape[1])
        return waveform

    def collator(self, samples):
        audio = torch.stack([s['audio'] for s in samples])
        collated = super().collator(samples)
        collated['audio'] = audio
        return collated


def get_spatial_audio_dataset_binaural(dataset_config, tokenizer, split):
    dataset = SpatialAudioDatasetBinauralJsonl(dataset_config, tokenizer, split)
    return dataset
