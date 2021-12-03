# -*- coding: utf-8 -*-
"""
오디오 데이터셋
========

``torchaudio`` 는 공개적으로 접근할 수 있는
공통 데이터 셋에 대한 쉬운 액세스를 제공합니다.
사용 가능한 데이터 셋 목록은 공식 설명서를 참조하십시오.
"""

# Google Colab에서 이 튜토리얼을 실행할 때 다음과 함께 필요한 패키지를 설치하세요.
# !pip install torchaudio

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# 데이터 및 utility 함수 준비 (skip this section)
# --------------------------------------------------------
#

#@title 데이터 및 utility 함수 준비 {display-mode: "form"}
#@markdown
#@markdown 이 부분을 자세히 살펴볼 필요는 없습니다.
#@markdown 한번만 실행해보면 쉽게 할 수 있습니다.

#-------------------------------------------------------------------------------
# 데이터 및 helper 함수 준비.
#-------------------------------------------------------------------------------
import multiprocessing
import os

import matplotlib.pyplot as plt
from IPython.display import Audio, display


_SAMPLE_DIR = "_sample_data"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)

def _download_yesno():
  if os.path.exists(os.path.join(YESNO_DATASET_PATH, "waves_yesno.tar.gz")):
    return
  torchaudio.datasets.YESNO(root=YESNO_DATASET_PATH, download=True)

YESNO_DOWNLOAD_PROCESS = multiprocessing.Process(target=_download_yesno)
YESNO_DOWNLOAD_PROCESS.start()

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

######################################################################
# 여기서는 ''YESNO'' 데이터 의 사용 방법을 보여 줍니다.
#

YESNO_DOWNLOAD_PROCESS.join()

dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)

for i in [1, 3, 5]:
  waveform, sample_rate, label = dataset[i]
  plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
  play_audio(waveform, sample_rate)
