# -*- coding: utf-8 -*-
"""
오디오 데이터 보강
=================

``torchaudio`` 는 오디오 데이터를 보강하는 다양한 방법을 제공합니다
"""

# Google Colab에서 이 튜토리얼을 실행할 때 다음을 사용하여 필수 패키지를 설치하십시오.
# !pip install torchaudio

import torch
import torchaudio
import torchaudio.functional as F

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# 데이터 및 유틸리티 기능 준비(이 섹션 건너뛰기)
# --------------------------------------------------------
#

#@title 데이터 및 유틸리티 함수를 준비합니다. {display-mode: "form"}
#@markdown
#@markdown 이 셀을 들여다볼 필요가 없습니다.
#@markdown 한 번만 실행하면 됩니다.
#@markdown
#@markdown 이 튜토리얼에서는 Creative Commos 4.0에 따라 라이선스가 부여된 [VOiCES dataset](https://iqtlabs.github.io/voices/)의 음성 데이터를 사용합니다.
#-------------------------------------------------------------------------------
# 데이터 및 도우미 기능 준비
#-------------------------------------------------------------------------------

import math
import os
import requests

import matplotlib.pyplot as plt
from IPython.display import Audio, display


_SAMPLE_DIR = "_sample_data"

SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "steam.wav")

SAMPLE_RIR_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/room-response/rm1/impulse/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo.wav"
SAMPLE_RIR_PATH = os.path.join(_SAMPLE_DIR, "rir.wav")

SAMPLE_WAV_SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SAMPLE_WAV_SPEECH_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")

SAMPLE_NOISE_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/distractors/rm1/babb/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"
SAMPLE_NOISE_PATH = os.path.join(_SAMPLE_DIR, "bg.wav")

os.makedirs(_SAMPLE_DIR, exist_ok=True)

def _fetch_data():
  uri = [
    (SAMPLE_WAV_URL, SAMPLE_WAV_PATH),
    (SAMPLE_RIR_URL, SAMPLE_RIR_PATH),
    (SAMPLE_WAV_SPEECH_URL, SAMPLE_WAV_SPEECH_PATH),
    (SAMPLE_NOISE_URL, SAMPLE_NOISE_PATH),
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

def get_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_PATH, resample=resample)

def get_speech_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)

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

def get_rir_sample(*, resample=None, processed=False):
  rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
  if not processed:
    return rir_raw, sample_rate
  rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
  rir = rir / torch.norm(rir, p=2)
  rir = torch.flip(rir, [1])
  return rir, sample_rate

def get_noise_sample(*, resample=None):
  return _get_sample(SAMPLE_NOISE_PATH, resample=resample)


######################################################################
# 효과 적용 및 필터링
# ------------------------------
#
# ``torchaudio.sox_effects`` 를 사용하면 ``sox`` 에서 사용 가능한 것과 유사한 필터를
# Tensor 개체 및 파일 개체 오디오 소스에 직접 적용할 수 있습니다.
#
# 이를 위한 두 가지 함수이 있습니다.
#
# -  Tensor에 효과를 적용하기 위한 ``torchaudio.sox_effects.apply_effects_tensor``.
# - 다른 오디오 소스에 효과를 적용하기 위한 ``torchaudio.sox_effects.apply_effects_file``.
#
# 두 함수 모두 ``List[List[str]]`` 형식의 효과 정의를 허용합니다.
# 이것은 대부분 ``sox`` 명령이 작동하는 방식과 일치하지만, 한 가지 주의할 점은 ``sox`` 는
# 일부 효과를 자동으로 추가하는 반면, ``torchaudio`` 의 구현은 그렇지 않다는 것입니다.
#
# 사용 가능한 효과 목록은 `sox 문서 <http://sox.sourceforge.net/sox.html>`__ 를 참조하십시오.
#
# **Tip** 즉석에서 오디오 데이터를 로드하고 다시 샘플링해야 하는 경우 ``"rate"`` 효과와
# 함께 ``torchaudio.sox_effects.apply_effects_file`` 을 사용할 수 있습니다.
#
# **Note** ``apply_effects_file`` 은 파일류 객체나 경로류 객체를
# 받아들입니다. ``torchaudio.load`` 와 유사하게
# 파일 확장자나 헤더에서 오디오 형식을 유추할 수 없는 경우
# 인수 ``형식`` 을 제공하여 오디오 소스의 형식을 지정할 수 있습니다.
#
# **Note** 이 과정은 미분할 수 없습니다.
#


# 데이터 로드 Load the data
waveform1, sample_rate1 = get_sample(resample=16000)

# 효과 정의 Define effects
effects = [
  ["lowpass", "-1", "300"], # 단극 저역 통과 필터(low pass) 적용
  ["speed", "0.8"],  # speed 줄이기
                     # 샘플 rate만 변경하므로 이후에 원래 샘플 레이트와
                     # 함께 'rate' 효과를 추가해야 합니다.
  ["rate", f"{sample_rate1}"],
  ["reverb", "-w"],  # Reverbration gives some dramatic feeling
]

# 효과 적용 Apply effects
waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
    waveform1, sample_rate1, effects)

plot_waveform(waveform1, sample_rate1, title="Original", xlim=(-.1, 3.2))
plot_waveform(waveform2, sample_rate2, title="Effects Applied", xlim=(-.1, 3.2))
print_stats(waveform1, sample_rate=sample_rate1, src="Original")
print_stats(waveform2, sample_rate=sample_rate2, src="Effects Applied")

######################################################################
# 효과를 적용한 후의 프레임 수와 채널 수는 원본과 다릅니다. 오디오를
# 들어봅시다. 더 드라마틱하지 않나요?
#

plot_specgram(waveform1, sample_rate1, title="Original", xlim=(0, 3.04))
play_audio(waveform1, sample_rate1)
plot_specgram(waveform2, sample_rate2, title="Effects Applied", xlim=(0, 3.04))
play_audio(waveform2, sample_rate2)


######################################################################
# 룸 잔향 시뮬레이션
# ----------------------------
#
# `컨볼루션
# 리버브 <https://en.wikipedia.org/wiki/Convolution_reverb>`__ 는 다른 환경에서
# 생성된 것처럼 깨끗한 오디오 사운드를 만드는 데 사용되는 기술입니다.
#
# 예를 들어 RIR(Room Impulse Response)을 사용하면 회의실에서
# 말하는 것처럼 깨끗한 음성을 만들 수 있습니다.
#
# 이 프로세스를 위해서는 RIR 데이터가 필요합니다. 다음 데이터는 VOiCES 데이터 세트에서
# 가져온 것이지만 직접 녹음할 수도 있습니다. 마이크를 켜고 박수를 치기만 하면 됩니다.
#


sample_rate = 8000

rir_raw, _ = get_rir_sample(resample=sample_rate)

plot_waveform(rir_raw, sample_rate, title="Room Impulse Response (raw)", ylim=None)
plot_specgram(rir_raw, sample_rate, title="Room Impulse Response (raw)")
play_audio(rir_raw, sample_rate)

######################################################################
# 먼저 RIR을 정리해야 합니다. 메인 임펄스를 추출하고 신호 전력을
# 정규화한 다음 시간 축을 따라 뒤집습니다.
#

rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
rir = rir / torch.norm(rir, p=2)
rir = torch.flip(rir, [1])

print_stats(rir)
plot_waveform(rir, sample_rate, title="Room Impulse Response", ylim=None)

######################################################################
# 그런 다음 RIR 필터를 사용하여 음성 신호를 합성합니다.
#

speech, _ = get_speech_sample(resample=sample_rate)

speech_ = torch.nn.functional.pad(speech, (rir.shape[1]-1, 0))
augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

plot_waveform(speech, sample_rate, title="Original", ylim=None)
plot_waveform(augmented, sample_rate, title="RIR Applied", ylim=None)

plot_specgram(speech, sample_rate, title="Original")
play_audio(speech, sample_rate)

plot_specgram(augmented, sample_rate, title="RIR Applied")
play_audio(augmented, sample_rate)


######################################################################
# 배경 소음 추가
# -----------------------
#
# 오디오 데이터에 배경 잡음을 추가하려면 오디오 데이터를 나타내는
# Tensor에 잡음 Tensor를 추가하기만 하면 됩니다. 노이즈 강도를 조정하는
# 일반적인 방법은 SNR(Signal-to-Noise Ratio)을 변경하는 것입니다.
# [`wikipedia <https://en.wikipedia.org/wiki/Signal-to-noise_ratio>`__]
#
# \begin{align}\mathrm{SNR} = \frac{P_\mathrm{signal}}{P_\mathrm{noise}}\end{align}
#
# \begin{align}{\mathrm  {SNR_{{dB}}}}=10\log _{{10}}\left({\mathrm  {SNR}}\right)\end{align}
#


sample_rate = 8000
speech, _ = get_speech_sample(resample=sample_rate)
noise, _ = get_noise_sample(resample=sample_rate)
noise = noise[:, :speech.shape[1]]

plot_waveform(noise, sample_rate, title="Background noise")
plot_specgram(noise, sample_rate, title="Background noise")
play_audio(noise, sample_rate)

speech_power = speech.norm(p=2)
noise_power = noise.norm(p=2)

for snr_db in [20, 10, 3]:
  snr = math.exp(snr_db / 10)
  scale = snr * noise_power / speech_power
  noisy_speech = (scale * speech + noise) / 2

  plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
  plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
  play_audio(noisy_speech, sample_rate)

######################################################################
# Tensor 객체에 코덱 적용
# -------------------------------
#
# ``torchaudio.functional.apply_codec`` 는 Tensor 객체에 코덱을 적용할 수 있습니다.
#
# **Note** 이 과정은 미분할 수 없습니다.
#


waveform, sample_rate = get_speech_sample(resample=8000)

plot_specgram(waveform, sample_rate, title="Original")
play_audio(waveform, sample_rate)

configs = [
    ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law"),
    ({"format": "gsm"}, "GSM-FR"),
    ({"format": "mp3", "compression": -9}, "MP3"),
    ({"format": "vorbis", "compression": -1}, "Vorbis"),
]
for param, title in configs:
  augmented = F.apply_codec(waveform, sample_rate, **param)
  plot_specgram(augmented, sample_rate, title=title)
  play_audio(augmented, sample_rate)

######################################################################
# 전화 녹음 시뮬레이션
# ---------------------------
#
# 이전 기술을 결합하여 사람들이 배경에서 말하는 소리가 들리는 방에서
# 전화 통화를 하는 것처럼 들리는 오디오를 시뮬레이션할 수 있습니다.
#

sample_rate = 16000
speech, _ = get_speech_sample(resample=sample_rate)

plot_specgram(speech, sample_rate, title="Original")
play_audio(speech, sample_rate)

# RIR 적용
rir, _ = get_rir_sample(resample=sample_rate, processed=True)
speech_ = torch.nn.functional.pad(speech, (rir.shape[1]-1, 0))
speech = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

plot_specgram(speech, sample_rate, title="RIR Applied")
play_audio(speech, sample_rate)

# 배경 소음 추가
# 소음은 실제 환경에서 녹음되기 때문에 소음에는 환경의 음향적 특성이
# 포함되어 있다고 간주합니다. 따라서 RIR 적용 후 노이즈를 추가합니다.
noise, _ = get_noise_sample(resample=sample_rate)
noise = noise[:, :speech.shape[1]]

snr_db = 8
scale = math.exp(snr_db / 10) * noise.norm(p=2) / speech.norm(p=2)
speech = (scale * speech + noise) / 2

plot_specgram(speech, sample_rate, title="BG noise added")
play_audio(speech, sample_rate)

# 필터링 적용 및 샘플 레이트 변경
speech, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
  speech,
  sample_rate,
  effects=[
      ["lowpass", "4000"],
      ["compand", "0.02,0.05", "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
      ["rate", "8000"],
  ],
)

plot_specgram(speech, sample_rate, title="Filtered")
play_audio(speech, sample_rate)

# 적용 telephony codec
speech = F.apply_codec(speech, sample_rate, format="gsm")

plot_specgram(speech, sample_rate, title="GSM Codec Applied")
play_audio(speech, sample_rate)
