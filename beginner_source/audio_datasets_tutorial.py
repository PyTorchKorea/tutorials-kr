"""
오디오 데이터셋
==============

**저자**: `Moto Hira <moto@meta.com>`__
**번역**: `백승엽 <https://github.com/aromadsh>`__

``torchaudio``는 공용으로 접근할 수 있는 일반적인 데이터셋에 쉽게 접근할 수 있는 기능을 제공합니다. 
사용할 수 있는 데이터셋 목록은 공식 문서를 참고하세요. 
"""

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

######################################################################

import os

import IPython

import matplotlib.pyplot as plt


_SAMPLE_DIR = "_assets"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()


######################################################################
# 
# 여기서 :py:class:`torchaudio.datasets.YESNO` 데이터셋의 사용 방법을 볼 수 있습니다. 
#

dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)

######################################################################
#
i = 1
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#
i = 3
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#
i = 5
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)

i = 5
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)
