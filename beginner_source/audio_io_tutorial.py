# -*- coding: utf-8 -*-
"""
오디오 I/O
=========

``torchaudio`` 는 ``libsox`` 를 통합하고 풍부한 오디오 I/O 세트를 제공합니다.
"""

# Google Colab에서 이 튜토리얼을 실행할 때 다음을 사용하여 
# 필수 패키지를 설치하십시오.
# !pip install torchaudio boto3

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# 데이터 및 유틸리티 기능 준비(이 섹션 건너뛰기)
# --------------------------------------------------------
#

#@title 데이터 및 유틸리티 기능을 준비합니다. {display-mode: "form"}
#@markdown
#@markdown 이 셀을 들여다볼 필요가 없습니다.
#@markdown 한 번만 실행하면 됩니다.
#@markdown
#@markdown 이 튜토리얼에서는 Creative Commos BY 4.0에 따라 라이선스가 부여된 [VOiCES dataset](https://iqtlabs.github.io/voices/)의 음성 데이터를 사용합니다.


import io
import os
import requests
import tarfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
from IPython.display import Audio, display


_SAMPLE_DIR = "_sample_data"
SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "steam.wav")

SAMPLE_MP3_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.mp3"
SAMPLE_MP3_PATH = os.path.join(_SAMPLE_DIR, "steam.mp3")

SAMPLE_GSM_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.gsm"
SAMPLE_GSM_PATH = os.path.join(_SAMPLE_DIR, "steam.gsm")

SAMPLE_WAV_SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SAMPLE_WAV_SPEECH_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")

SAMPLE_TAR_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit.tar.gz"
SAMPLE_TAR_PATH = os.path.join(_SAMPLE_DIR, "sample.tar.gz")
SAMPLE_TAR_ITEM = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"

S3_BUCKET = "pytorch-tutorial-assets"
S3_KEY = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"


def _fetch_data():
  os.makedirs(_SAMPLE_DIR, exist_ok=True)
  uri = [
    (SAMPLE_WAV_URL, SAMPLE_WAV_PATH),
    (SAMPLE_MP3_URL, SAMPLE_MP3_PATH),
    (SAMPLE_GSM_URL, SAMPLE_GSM_PATH),
    (SAMPLE_WAV_SPEECH_URL, SAMPLE_WAV_SPEECH_PATH),
    (SAMPLE_TAR_URL, SAMPLE_TAR_PATH),
  ]
  for url, path in uri:
    with open(path, 'wb') as file_:
      file_.write(requests.get(url).content)

_fetch_data()

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

def inspect_file(path):
  print("-" * 10)
  print("Source:", path)
  print("-" * 10)
  print(f" - File size: {os.path.getsize(path)} bytes")
  print(f" - {torchaudio.info(path)}")

######################################################################
# 오디오 메타데이터 쿼리
# ----------------------
#
# 함수 ``torchaudio.info`` 는 오디오 메타데이터를 가져옵니다. 
# 파일의 경로 또는 파일을 제공할 수 있습니다.
#

metadata = torchaudio.info(SAMPLE_WAV_PATH)
print(metadata)

######################################################################
# Where
#
# -  ``sample_rate`` 는 오디오의 샘플링 비율입니다
# -  ``num_channels`` 는 채널 수입니다
# -  ``num_frames`` 는 채널당 프레임 수입니다
# -  ``bits_per_sample`` 은 비트 심도입니다
# -  ``encoding`` 은 샘플 코딩 형식입니다
#
# ``encoding`` 은 다음 값 중 하나를 사용할 수 있습니다:
#
# -  ``"PCM_S"``: 부호 있는 정수 선형 PCM
# -  ``"PCM_U"``: 부호 없는 정수 선형 PCM
# -  ``"PCM_F"``: 부동 소수점 선형 PCM
# -  ``"FLAC"``: Flac, `Free Lossless Audio
#    Codec <https://xiph.org/flac/>`__
# -  ``"ULAW"``: Mu-law,
#    [`wikipedia <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`__]
# -  ``"ALAW"``: A-law
#    [`wikipedia <https://en.wikipedia.org/wiki/A-law_algorithm>`__]
# -  ``"MP3"`` : MP3, MPEG-1 Audio Layer III
# -  ``"VORBIS"``: OGG Vorbis [`xiph.org <https://xiph.org/vorbis/>`__]
# -  ``"AMR_NB"``: Adaptive Multi-Rate
#    [`wikipedia <https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_audio_codec>`__]
# -  ``"AMR_WB"``: Adaptive Multi-Rate Wideband
#    [`wikipedia <https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_Wideband>`__]
# -  ``"OPUS"``: Opus [`opus-codec.org <https://opus-codec.org/>`__]
# -  ``"GSM"``: GSM-FR
#    [`wikipedia <https://en.wikipedia.org/wiki/Full_Rate>`__]
# -  ``"UNKNOWN"`` 위에 없는것
#

######################################################################
# **Note**
#
# -  압축 및/또는 가변 비트 전송률(예: MP3)이 있는 형식의 경우
#     ``bits_per_sample`` 이 ``0`` 일 수 있습니다.
# -  ``num_frames`` 는 GSM-FR 형식의 경우 ``0`` 일 수 있습니다.
#

metadata = torchaudio.info(SAMPLE_MP3_PATH)
print(metadata)

metadata = torchaudio.info(SAMPLE_GSM_PATH)
print(metadata)


######################################################################
# 쿼리 파일 - 객체와 같은
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``info`` 는 파일류 객체에서 작동합니다.
#

print("Source:" , SAMPLE_WAV_URL)
with requests.get(SAMPLE_WAV_URL, stream=True) as response:
  metadata = torchaudio.info(response.raw)
print(metadata)

######################################################################
# **참고** 파일류 객체를 전달할 때, ``info`` 는 모든 기본 데이터를 읽지 않습니다. 
# 오히려 처음부터 데이터의 일부만 읽습니다. 
# 따라서 지정된 오디오 형식의 경우 형식 자체를 포함하여 올바른 메타데이터를 검색하지 못할 수 있습니다. 
# 다음 예는 이를 보여줍니다.
#
# -  인수 ``format`` 을 사용하여 입력의 오디오 형식을 지정합니다.
# -  반환된 메타데이터에는 ``num_frames = 0`` 이 있습니다.
#

print("Source:", SAMPLE_MP3_URL)
with requests.get(SAMPLE_MP3_URL, stream=True) as response:
  metadata = torchaudio.info(response.raw, format="mp3")

  print(f"Fetched {response.raw.tell()} bytes.")
print(metadata)

######################################################################
# Tensor에 오디오 데이터 로드
# ------------------------------
#
# 오디오 데이터를 로드하려면 ``torchaudio.load`` 를 사용할 수 있습니다.
#
# 이 함수는 경로류 객체 또는 파일류 객체를 입력으로 받습니다.
#
# 반환된 값은 파형(``tensor``)과 샘플 속도(``int``)의 튜플입니다.
#
# 기본적으로 결과 텐서 객체는 ``dtype=torch.float32`` 를 가지며
# 값 범위는 ``[-1.0, 1.0]`` 내에서 정규화됩니다.
#
# 지원되는 형식 목록은 다음을 참조하십시오. `the torchaudio
# documentation <https://pytorch.org/audio>`__.
#

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)

print_stats(waveform, sample_rate=sample_rate)
plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)
play_audio(waveform, sample_rate)


######################################################################
# 파일류 객체에서 로드
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``torchaudio`` 의 I/O 기능은 이제 파일과 유사한 객체를 지원합니다. 
# 이를 통해 로컬 파일 시스템 내부 및 외부 위치에서 오디오 데이터를 가져오고 디코딩할 수 있습니다. 
# 다음 예는 이를 보여줍니다.
#

# HTTP 요청으로 오디오 데이터 로드
with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
  waveform, sample_rate = torchaudio.load(response.raw)
plot_specgram(waveform, sample_rate, title="HTTP datasource")

# tar 파일에서 오디오 로드
with tarfile.open(SAMPLE_TAR_PATH, mode='r') as tarfile_:
  fileobj = tarfile_.extractfile(SAMPLE_TAR_ITEM)
  waveform, sample_rate = torchaudio.load(fileobj)
plot_specgram(waveform, sample_rate, title="TAR file")

# S3에서 오디오 로드
client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
response = client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
waveform, sample_rate = torchaudio.load(response['Body'])
plot_specgram(waveform, sample_rate, title="From S3")


######################################################################
# 슬라이싱 팁
# ~~~~~~~~~~~~~~~
#
# ``num_frames`` 및 ``frame_offset`` 인수를 제공하면 디코딩이 입력의 해당 세그먼트로 제한됩니다.
#
# vanilla Tensor 슬라이싱을 사용하여 동일한 결과를 얻을 수 있습니다.
# (i.e. ``waveform[:, frame_offset:frame_offset+num_frames]``). 그러나 
# ``num_frames`` 및 ``frame_offset`` 인수를 제공하는 것이 더 효율적입니다.
#
# 이는 요청된 프레임 디코딩이 완료되면 함수가 데이터 수집 및 디코딩을 종료하기 때문입니다. 
# 필요한 양의 데이터를 가져오는 즉시 데이터 전송이 중지되므로 
# 네트워크를 통해 오디오 데이터를 전송할 때 유리합니다.
#
# 다음 예는 이를 보여줍니다.
#

# 두 가지 다른 디코딩 방법의 일러스트레이션.
# 첫 번째 것은 모든 데이터를 가져와 디코딩하는 반면,
# 두 번째는 디코딩이 완료되면 데이터 가져오기를 중지합니다.
# 파형의 결과는 동일합니다.

frame_offset, num_frames = 16000, 16000  # Fetch and decode the 1 - 2 seconds

print("모든 데이터를 가져오는 중...")
with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
  waveform1, sample_rate1 = torchaudio.load(response.raw)
  waveform1 = waveform1[:, frame_offset:frame_offset+num_frames]
  print(f" - Fetched {response.raw.tell()} bytes")

print("요청한 프레임을 사용할 수 있을 때까지 가져오는 중...")
with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
  waveform2, sample_rate2 = torchaudio.load(
      response.raw, frame_offset=frame_offset, num_frames=num_frames)
  print(f" - Fetched {response.raw.tell()} bytes")

print("파형의 결과 확인 중... ", end="")
assert (waveform1 == waveform2).all()
print("일치!")


######################################################################
# 파일에 오디오 저장
# --------------------
#
# 일반 응용 프로그램에서 해석할 수 있는 형식으로 오디오 데이터를 저장하려면,
# ``torchaudio.save`` 를 사용할 수 있습니다.
#
# 이 함수는 path-like object 또는 file-like object를 받습니다.
#
# file-like object를 전달할 때 함수가 어떤 형식을 사용해야 하는지 알 수 있도록 인수 ``format`` 도 제공해야 합니다. 
# path-like object의 경우 함수는 확장자에서 형식을 유추합니다. 
# 확장자가 없는 파일에 저장하는 경우 ``format`` 인수를 제공해야 합니다.
#
# WAV 형식의 데이터를 저장할 때 ``float32`` Tensor의 기본 인코딩은 32비트 부동 소수점 PCM입니다. 
# 인수 ``encoding`` 및 ``bits_per_sample`` 을 제공하여 이 동작을 변경할 수 있습니다. 
# 예를 들어, 16비트 부호 있는 정수 PCM에 데이터를 저장하려면 다음을 수행할 수 있습니다.
#
# **참고** 낮은 비트 심도로 인코딩으로 데이터를 저장하면 
# 결과 파일 크기는 줄어들지만 정밀도도 줄어듭니다.
#


waveform, sample_rate = get_sample()
print_stats(waveform, sample_rate=sample_rate)

# 인코딩 옵션 없이 저장합니다.
# 함수는 제공된 데이터에 맞는 인코딩을 선택합니다.
path = "save_example_default.wav"
torchaudio.save(path, waveform, sample_rate)
inspect_file(path)

# 16비트 부호 있는 정수 Linear PCM으로 저장
# 결과 파일은 스토리지의 절반을 차지하지만 정밀도가 떨어집니다
path = "save_example_PCM_S16.wav"
torchaudio.save(
    path, waveform, sample_rate,
    encoding="PCM_S", bits_per_sample=16)
inspect_file(path)


######################################################################
# ``torchaudio.save`` 는 다른 형식도 처리할 수 있습니다. 몇 가지 예를 들면 다음과 같습니다:
#

waveform, sample_rate = get_sample(resample=8000)

formats = [
  "mp3",
  "flac",
  "vorbis",
  "sph",
  "amb",
  "amr-nb",
  "gsm",
]

for format in formats:
  path = f"save_example.{format}"
  torchaudio.save(path, waveform, sample_rate, format=format)
  inspect_file(path)


######################################################################
# file-like object에 저장
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 다른 I/O 기능과 마찬가지로 오디오를 파일과 같은 개체에 저장할 수 있습니다. 
# file-like object에 저장할 때 ``format`` 인수가 필요합니다.
#


waveform, sample_rate = get_sample()

# bytes buffer에 저장
buffer_ = io.BytesIO()
torchaudio.save(buffer_, waveform, sample_rate, format="wav")

buffer_.seek(0)
print(buffer_.read(16))

