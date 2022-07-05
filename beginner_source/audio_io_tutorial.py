# -*- coding: utf-8 -*-
"""
오디오 I/O
=========

``torchaudio`` 는 ``libsox`` 를 통합하고 풍부한 오디오 I/O 를 제공합니다.
"""

# 이번 튜토리얼을 구글 코랩(Google Colab)에서 실행할 때,
# 필요한 패키지들을 아래와 같이 설치해주세요.
# !pip install torchaudio boto3

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# 데이터 및 유용한 함수 준비하기 (이번 섹션 생략)
# --------------------------------------------------------
#

#@title 데이터 및 유용한 함수 준비하기 {display-mode: "form"}
#@markdown
#@markdown 이 부분을 자세히 보실 필요는 없습니다.
#@markdown 한번만 실행하면 충분합니다.
#@markdown
#@markdown 이번 튜토리얼에서, [VOiCES dataset](https://iqtlabs.github.io/voices/) 의 음성 데이터를 사용할 것이고, 이 데이터는 Creative Commos BY 4.0 에 의해 라이센스가 부여됩니다.


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
# 오디오 메타데이터 쿼리하기
# --------------------------
#
# 함수 ``torchaudio.info`` 는 오디오 메타데이터를 가져옵니다. 
# 경로 혹은 파일 형식의 객체를 파라미터로 넣을 수 있습니다.
#

metadata = torchaudio.info(SAMPLE_WAV_PATH)
print(metadata)

######################################################################
# 위 결과에서
#
# -  ``sample_rate`` 는 오디오의 샘플링 비율입니다.
# -  ``num_channels`` 는 채널의 개수입니다.
# -  ``num_frames`` 는 채널별 프레임의 개수입니다.
# -  ``bits_per_sample`` 은 샘플당 비트 수(bit depth)입니다.
# -  ``encoding`` 는 샘플 코딩 형식입니다.
#
# ``encoding`` 은 다음 값들 중 하나가 될 수 있습니다:
#
# -  ``"PCM_S"``: 부호가 있는 정수 선형 PCM
# -  ``"PCM_U"``: 부호가 없는 정수 선형 PCM
# -  ``"PCM_F"``: 부동소수점 선형 PCM
# -  ``"FLAC"``: Flac, `무료 무손실 오디오 코덱(Free Lossless Audio
#    Codec) <https://xiph.org/flac/>`__
# -  ``"ULAW"``: 뮤 법칙(Mu-law),
#    [`wikipedia <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`__]
# -  ``"ALAW"``: A 법칙(A-law)
#    [`wikipedia <https://en.wikipedia.org/wiki/A-law_algorithm>`__]
# -  ``"MP3"`` : MP3, MPEG-1 오디오 레이어 III
# -  ``"VORBIS"``: OGG Vorbis [`xiph.org <https://xiph.org/vorbis/>`__]
# -  ``"AMR_NB"``: 적응형 다중 속도(Adaptive Multi-Rate)
#    [`wikipedia <https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_audio_codec>`__]
# -  ``"AMR_WB"``: 적응형 다중 속도 광대역(Adaptive Multi-Rate Wideband)
#    [`wikipedia <https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_Wideband>`__]
# -  ``"OPUS"``: Opus [`opus-codec.org <https://opus-codec.org/>`__]
# -  ``"GSM"``: GSM-FR
#    [`wikipedia <https://en.wikipedia.org/wiki/Full_Rate>`__]
# -  ``"UNKNOWN"``: 위에 없음
#

######################################################################
# **참고**
#
# -  압축 및/또는 가변 비트 전송률 형식(예: MP3)의 경우 ``bits_per_sample`` 은 ``0`` 일 수 있습니다.
# -  GSM-FR 형식의 경우 ``num_frames`` 는 ``0`` 일 수 있습니다.
#

metadata = torchaudio.info(SAMPLE_MP3_PATH)
print(metadata)

metadata = torchaudio.info(SAMPLE_GSM_PATH)
print(metadata)


######################################################################
# 파일 형식의 객체 쿼리하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``info`` 는 파일 형식의 객체에서 동작합니다.
#

print("Source:", SAMPLE_WAV_URL)
with requests.get(SAMPLE_WAV_URL, stream=True) as response:
  metadata = torchaudio.info(response.raw)
print(metadata)

######################################################################
# **참고** 파일 형식의 객체를 넘길 때, ``info`` 는 모든 기본 데이터를 읽는 
# 것이 아니라 처음부터 데이터의 일부만 읽습니다.
# 따라서, 주어진 오디오 형식의 경우, 형식 자체를 포함하여, 
# 올바른 메타데이터를 검색하지 못할 수 있습니다.
# 다음 예시에서 이를 보여줍니다.
#
# -  ``format`` 인자를 사용하여 입력의 오디오 형식을 지정합니다.
# -  반환된 메타데이터에 ``num_frames = 0`` 가 있습니다.
#

print("Source:", SAMPLE_MP3_URL)
with requests.get(SAMPLE_MP3_URL, stream=True) as response:
  metadata = torchaudio.info(response.raw, format="mp3")

  print(f"Fetched {response.raw.tell()} bytes.")
print(metadata)

######################################################################
# 오디오 데이터를 텐서로 불러오기
# ------------------------------
#
# 오디오 데이터를 불러오기 위해, ``torchaudio.load`` 를 사용할 수 있습니다.
#
# 이 함수는 경로 혹은 파일 형식의 객체를 입력으로 받아들입니다.
#
# 반환되는 값은 파형 (``Tensor``) 과 샘플링 비율 (``int``) 의 튜플입니다.
#
# 기본적으로, 결과 텐서 객체는 ``dtype=torch.float32`` 이고,
# 값의 범위는 ``[-1.0, 1.0]`` 내에서 정규화됩니다.
#
# 지원되는 형식의 목록은 `torchaudio 문서 <https://pytorch.org/audio>`__
# 를 참고하세요.
#

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)

print_stats(waveform, sample_rate=sample_rate)
plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)
play_audio(waveform, sample_rate)


######################################################################
# 파일 형식의 객체 로드하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``torchaudio`` 의 I/O 함수는 이제 파일 형식의 객체를 지원합니다.
# 이를 통해 로컬 파일 시스템 내부 및 외부 위치에서 오디오 데이터를
# 가져오고 디코딩할 수 있습니다.
# 다음 예제는 이를 보여줍니다.
#

# HTTP 요청으로 오디오 데이터 로드
with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
  waveform, sample_rate = torchaudio.load(response.raw)
plot_specgram(waveform, sample_rate, title="HTTP datasource")

# 파일에서 오디오 로드
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
# 슬라이싱(slicing)을 위한 팁
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``num_frames`` 와 ``frame_offset`` 인자를 지정하면
# 디코딩이 입력의 해당 세그먼트로 제한됩니다.
#
# 평범한 텐서 슬라이싱(즉, ``waveform[:, frame_offset:frame_offset+num_frames]``)
# 을 사용하여 동일한 결과를 얻을 수 있습니다.
# 하지만 ``num_frames`` 와 ``frame_offset`` 인자를 지정하는 것이 더 효율적입니다.
#
# 이는 요청된 프레임의 디코딩이 완료되면 함수가 데이터 수집 및 디코딩을 
# 종료하기 때문입니다. 필요한 양의 데이터를 가져오는 즉시 데이터 전송이 중지되기 때문에,
# 오디오 데이터가 네트워크를 통해 전송될 때 유리합니다.
#
# 다음 예제에서 이를 보여줍니다.
#

# 두 가지 다른 디코딩 방법입니다.
# 첫번째 방법은 모든 데이터를 가져온 후 디코딩합니다.
# 두번째 방법은 디코딩이 완료되면 데이터를 가져오는 것을 중지합니다.
# 결과로 나오는 파형은 동일합니다.

frame_offset, num_frames = 16000, 16000  # 1-2 초 가져오기 및 디코딩

print("Fetching all the data...")
with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
  waveform1, sample_rate1 = torchaudio.load(response.raw)
  waveform1 = waveform1[:, frame_offset:frame_offset+num_frames]
  print(f" - Fetched {response.raw.tell()} bytes")

print("Fetching until the requested frames are available...")
with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
  waveform2, sample_rate2 = torchaudio.load(
      response.raw, frame_offset=frame_offset, num_frames=num_frames)
  print(f" - Fetched {response.raw.tell()} bytes")

print("Checking the resulting waveform ... ", end="")
assert (waveform1 == waveform2).all()
print("matched!")


######################################################################
# 오디오를 파일에 저장하기
# ------------------------
#
# 일반적인 응용 프로그램에서 해석 가능한 형식으로 오디오 데이터를 저장하려면
# ``torchaudio.save`` 를 사용할 수 있습니다.
#
# 이 함수는 경로 혹은 파일 형식의 객체를 입력으로 받아들입니다.
#
# 파일 형식의 객체를 전달할 때, 함수에서 사용할 형식을 알 수 있도록 
# ``format`` 인자를 넣어줘야 합니다. 경로 형식의 객체의 경우, 
# 함수에서 그 경로의 확장자로부터 형식을 추론하게 됩니다. 
# 확장자가 없는 파일에 저장하는 경우에는 ``format`` 인자를 넣어줘야 합니다.
#
# WAV 형식의 데이터를 저장할 때, ``float32`` 텐서의 기본 인코딩은 32비트 
# 부동소수점 PCM 입니다. ``encoding`` 과 ``bits_per_sample`` 인자를 넣어서
# 이 동작을 변경할 수 있습니다. 예를 들어, 데이터를 16비트 부호 있는 정수 PCM으로
# 저장하려면, 다음과 같이 작업합니다.
#
# **참고** 비트 깊이가 낮은 인코딩으로 데이터를 저장하면 결과 파일 크기가
# 줄어들 뿐만 아니라 정확도도 떨어집니다.
#


waveform, sample_rate = get_sample()
print_stats(waveform, sample_rate=sample_rate)

# 인코딩 옵션 없이 저장하기
# 이 함수는 제공된 데이터에 적합한 인코딩을 선택합니다.
path = "save_example_default.wav"
torchaudio.save(path, waveform, sample_rate)
inspect_file(path)

# 16비트 부호 있는 정수 선형 PCM으로 저장
# 결과 파일이 스토리지의 절반을 차지하지만 정확도가 떨어집니다.
path = "save_example_PCM_S16.wav"
torchaudio.save(
    path, waveform, sample_rate,
    encoding="PCM_S", bits_per_sample=16)
inspect_file(path)


######################################################################
# ``torchaudio.save`` 는 다른 형식도 처리할 수 있습니다. 몇 가지 예를 들면:
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
# 파일 형식의 객체에 저장하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 다른 I/O 기능과 마찬가지로, 오디오를 파일 형식의 객체로 저장할 수 있습니다.
# 파일 형식의 객체로 저장할 때는 ``format`` 인자가 필요합니다.
#


waveform, sample_rate = get_sample()

# 바이트 버퍼에 저장하기
buffer_ = io.BytesIO()
torchaudio.save(buffer_, waveform, sample_rate, format="wav")

buffer_.seek(0)
print(buffer_.read(16))

