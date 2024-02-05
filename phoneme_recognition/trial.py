import torch
import torchaudio
import numpy as np
from scipy.io import wavfile

print(torch.__version__)
print(torchaudio.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

from dataclasses import dataclass

import IPython
import matplotlib.pyplot as plt

torch.random.manual_seed(0)

SPEECH_FILE = "/speech/dbwork/mul/spielwiese4/students/desengus/phoneme_recognition/Adele-Hello-vocals-F minor-59bpm-441hz.wav"

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
with torch.inference_mode():
    waveform, sample_rate = torchaudio.load(SPEECH_FILE)
    # waveform = mono_downmix(waveform.numpy())
    emissions, _ = model(torch.tensor(waveform).to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

print(labels)

waveform = wavfile.read(SPEECH_FILE)[1]


import whisper
model = whisper.load_model("tiny.en", device=device)
result = whisper.transcribe(model = model , audio = waveform)


print(result)