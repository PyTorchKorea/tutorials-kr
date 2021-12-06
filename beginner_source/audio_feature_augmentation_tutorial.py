# -*- coding: utf-8 -*-
"""
오디오 기능 
====================
"""

# Google Colab에서 이 튜토리얼을 실행할 때 다음과 함께 필요한 패키지를 설치하세요.
# !pip install torchaudio librosa

import torch
import torchaudio
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# 데이터 및 유틸리티 함수 준비 (skip this section)
# --------------------------------------------------------
#

#@title 데이터 및 유틸리티 함수 준비 {display-mode: "form"}
#@markdown
#@markdown 이 부분을 자세히 살펴볼 필요는 없습니다.
#@markdown 한번만 실행해보면 쉽게 할 수 있습니다.
#@markdown
#@markdown 이 튜토리얼에서는 Creative Commos BY 4.0에 따라 라이선스가 부여된 [VOiCES dataset](https://iqtlabs.github.io/voices/)의 음성 데이터를 사용할 것입니다.

#-------------------------------------------------------------------------------
# 데이터 및 보조(helper) 함수 준비
#-------------------------------------------------------------------------------

import os
import requests

import librosa
import matplotlib.pyplot as plt


_SAMPLE_DIR = "_sample_data"

SAMPLE_WAV_SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SAMPLE_WAV_SPEECH_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")

os.makedirs(_SAMPLE_DIR, exist_ok=True)

def _fetch_data():
  uri = [
    (SAMPLE_WAV_SPEECH_URL, SAMPLE_WAV_SPEECH_PATH),
  ]
  for url, path in uri:
    with open(path, 'wb') as file_:
      file_.write(requests.get(url).content)

_fetch_data()

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_speech_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)

def get_spectrogram(
    n_fft = 400,
    win_len = None,
    hop_len = None,
    power = 2.0,
):
  waveform, _ = get_speech_sample()
  spectrogram = T.Spectrogram(
      n_fft=n_fft,
      win_length=win_len,
      hop_length=hop_len,
      center=True,
      pad_mode="reflect",
      power=power,
  )
  return spectrogram(waveform)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

######################################################################
# 사양증강
# -----------
#
# `SpecAugment <https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html>`__
# 는 널리 사용되는 스펙트로그램 확대 기법이다.
#
# ``torchaudio`` implements ``TimeStretch``, ``TimeMasking`` 및
# ``FrequencyMasking``.
#
# TimeStretch
# ~~~~~~~~~~
#

spec = get_spectrogram(power=None)
stretch = T.TimeStretch()

rate = 1.2
spec_ = stretch(spec, rate)
plot_spectrogram(torch.abs(spec_[0]), title=f"Stretched x{rate}", aspect='equal', xmax=304)

plot_spectrogram(torch.abs(spec[0]), title="Original", aspect='equal', xmax=304)

rate = 0.9
spec_ = stretch(spec, rate)
plot_spectrogram(torch.abs(spec_[0]), title=f"Stretched x{rate}", aspect='equal', xmax=304)

######################################################################
# TimeMasking
# ~~~~~~~~~~~
#

torch.random.manual_seed(4)

spec = get_spectrogram()
plot_spectrogram(spec[0], title="Original")

masking = T.TimeMasking(time_mask_param=80)
spec = masking(spec)

plot_spectrogram(spec[0], title="Masked along time axis")

######################################################################
# FrequencyMasking
# ~~~~~~~~~~~~~~~~
#


torch.random.manual_seed(4)

spec = get_spectrogram()
plot_spectrogram(spec[0], title="Original")

masking = T.FrequencyMasking(freq_mask_param=80)
spec = masking(spec)

plot_spectrogram(spec[0], title="Masked along frequency axis")
