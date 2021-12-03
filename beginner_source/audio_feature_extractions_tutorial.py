# -*- coding: utf-8 -*-
"""
오디오 기능 추출
===================

``torchaudio`` 는 오디오 도메인에서 일반적으로 사용되는 기능 추출을 구현합니다.
이들은 ``torchaudio.functional`` 와 ``torchaudio.transforms``에서 이용할 수 있습니다.

``functional`` 은 독립형 기능으로 기능을 구현합니다.
이들은 국적이 없습니다.

``transforms`` 은 ``functional`` 과 ``torch.nn.Module`` 의 구현을 사용하여
기능을 객체로 구현합니다. 모든 변환은 ``torch.nn.Module``의 하위 분류이기 때문에
TorchScript를 사용하여 직렬화할 수 있습니다.

사용 가능한 기능의 전체 목록은 설명서를 참조하십시오.
이 튜토리얼에서는 시간 영역과 주파수 영역 간의
변환을 살펴보겠습니다 (``Spectrogram``, ``GriffinLim``,
``MelSpectrogram``).
"""

# Google Colab에서 이 튜토리얼을 실행할 때 필요한 패키지를 다음과 같이 설치합니다.
# !pip install torchaudio librosa

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# 데이터 및 utility 함수 준비 (skip this section)
# --------------------------------------------------------
#

#@title 데이터 및 utility 함수 준비. {display-mode: "form"}
#@markdown
#@markdown 이 부분을 자세히 살펴볼 필요는 없습니다.
#@markdown 한번만 실행해보면 쉽게 할 수 있습니다.
#@markdown
#@markdown 이 튜토리얼에서는 Creative Commos BY 4.0에 따라 라이선스가 부여된 [VOiCES dataset](https://iqtlabs.github.io/voices/)의 음성데이터를 사용할 것입니다.

#-------------------------------------------------------------------------------
# 데이터 및 helper 함수 준비.
#-------------------------------------------------------------------------------

import os
import requests

import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display


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

def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

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

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
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

def plot_mel_fbank(fbank, title=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Filter bank')
  axs.imshow(fbank, aspect='auto')
  axs.set_ylabel('frequency bin')
  axs.set_xlabel('mel bin')
  plt.show(block=False)

def plot_pitch(waveform, sample_rate, pitch):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln2 = axis2.plot(
      time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

  axis2.legend(loc=0)
  plt.show(block=False)

def plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Kaldi Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label='Pitch', color='green')
  axis.set_ylim((-1.3, 1.3))

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, nfcc.shape[1])
  ln2 = axis2.plot(
      time_axis, nfcc[0], linewidth=2, label='NFCC', color='blue', linestyle='--')

  lns = ln1 + ln2
  labels = [l.get_label() for l in lns]
  axis.legend(lns, labels, loc=0)
  plt.show(block=False)

######################################################################
# Spectrogram
# -----------
#
# 시간에 따라 변하는 오디오 신호의 주파수 구성을 얻기 위해
# ``Spectrogram``을 사용할 수 있습니다.
#



waveform, sample_rate = get_speech_sample()

n_fft = 1024
win_length = None
hop_length = 512

# define transformation
spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)
# Perform transformation
spec = spectrogram(waveform)

print_stats(spec)
plot_spectrogram(spec[0], title='torchaudio')

######################################################################
# GriffinLim
# ----------
#
# Spectrogram에서 파형을 복구하기 위해 ``GriffinLim``을 사용할 수 있습니다.
#


torch.random.manual_seed(0)
waveform, sample_rate = get_speech_sample()
plot_waveform(waveform, sample_rate, title="Original")
play_audio(waveform, sample_rate)

n_fft = 1024
win_length = None
hop_length = 512

spec = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
)(waveform)

griffin_lim = T.GriffinLim(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
)
waveform = griffin_lim(spec)

plot_waveform(waveform, sample_rate, title="Reconstructed")
play_audio(waveform, sample_rate)

######################################################################
# Mel Filter Bank
# ---------------
#
# ``torchaudio.functional.create_fb_matrix``는 주파수 빈을 
# mel-scale 빈으로 변환하기 위한 Fiter Bank 를 생성합니다.
#
# 이 함수는 입력 오디오/특징을 필요로 하지 않기 때문에
# ``torchaudio.transforms``에는 동등한 변환이 없습니다..
#


n_fft = 256
n_mels = 64
sample_rate = 6000

mel_filters = F.create_fb_matrix(
    int(n_fft // 2 + 1),
    n_mels=n_mels,
    f_min=0.,
    f_max=sample_rate/2.,
    sample_rate=sample_rate,
    norm='slaney'
)
plot_mel_fbank(mel_filters, "Mel Filter Bank - torchaudio")

######################################################################
# librosa와 비교
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 참고로 여기 ``librosa`` 가 있는 mel filter bank를 얻는
# 동등한 방법이 있습니다.
#


mel_filters_librosa = librosa.filters.mel(
    sample_rate,
    n_fft,
    n_mels=n_mels,
    fmin=0.,
    fmax=sample_rate/2.,
    norm='slaney',
    htk=True,
).T

plot_mel_fbank(mel_filters_librosa, "Mel Filter Bank - librosa")

mse = torch.square(mel_filters - mel_filters_librosa).mean().item()
print('Mean Square Difference: ', mse)

######################################################################
# MelSpectrogram
# --------------
#
# mel-scale spectrogram을 생성하는 것은 spectrogram을 생성하고 mel-scale 변환을 수행하는 것을 포함합니다.
# ``torchaudio``, ``MelSpectrogram`` 에서 이러한 기능을 제공합니다.
# 
#


waveform, sample_rate = get_speech_sample()

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)

melspec = mel_spectrogram(waveform)
plot_spectrogram(
    melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')

######################################################################
# librosa와 비교
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 참고로 여기 ``librosa``로 mel-scale spectrograms를 생성하는
# 동등한 방법이 있습니다.
#


melspec_librosa = librosa.feature.melspectrogram(
    waveform.numpy()[0],
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    n_mels=n_mels,
    norm='slaney',
    htk=True,
)
plot_spectrogram(
    melspec_librosa, title="MelSpectrogram - librosa", ylabel='mel freq')

mse = torch.square(melspec - melspec_librosa).mean().item()
print('Mean Square Difference: ', mse)

######################################################################
# MFCC
# ----
#

waveform, sample_rate = get_speech_sample()

n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
      'n_fft': n_fft,
      'n_mels': n_mels,
      'hop_length': hop_length,
      'mel_scale': 'htk',
    }
)

mfcc = mfcc_transform(waveform)

plot_spectrogram(mfcc[0])

######################################################################
# librosa와 비교
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#


melspec = librosa.feature.melspectrogram(
  y=waveform.numpy()[0], sr=sample_rate, n_fft=n_fft,
  win_length=win_length, hop_length=hop_length,
  n_mels=n_mels, htk=True, norm=None)

mfcc_librosa = librosa.feature.mfcc(
  S=librosa.core.spectrum.power_to_db(melspec),
  n_mfcc=n_mfcc, dct_type=2, norm='ortho')

plot_spectrogram(mfcc_librosa)

mse = torch.square(mfcc - mfcc_librosa).mean().item()
print('Mean Square Difference: ', mse)

######################################################################
# Pitch
# -----
#


waveform, sample_rate = get_speech_sample()

pitch = F.detect_pitch_frequency(waveform, sample_rate)
plot_pitch(waveform, sample_rate, pitch)
play_audio(waveform, sample_rate)

######################################################################
# Kaldi Pitch (beta)
# ------------------
#
# Kaldi Pitch 기능 [1]은 자동 음성 인식 (ASR) 애플리케이션을 위해 조정된 피치 감지 매커니즘입니다.
# 이 기능은 ``torchaudio``의 베타 기능으로,
#  ``functional``에서만 사용할 수 있습니다..
#
# 1. 자동 음성 인식을 위해 조정된 피치 추출 알고리즘
#
#    Ghahremani, B. BabaAli, D. Povey, K. Riedhammer, J. Trmal and S.
#    Khudanpur
#
#    2014 IEEE International Conference on Acoustics, Speech and Signal
#    Processing (ICASSP), Florence, 2014, pp. 2494-2498, doi:
#    10.1109/ICASSP.2014.6854049.
#    [`abstract <https://ieeexplore.ieee.org/document/6854049>`__],
#    [`paper <https://danielpovey.com/files/2014_icassp_pitch.pdf>`__]
#


waveform, sample_rate = get_speech_sample(resample=16000)

pitch_feature = F.compute_kaldi_pitch(waveform, sample_rate)
pitch, nfcc = pitch_feature[..., 0], pitch_feature[..., 1]

plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc)
play_audio(waveform, sample_rate)
