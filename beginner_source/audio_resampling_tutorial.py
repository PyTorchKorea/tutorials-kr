# -*- coding: utf-8 -*-
"""
오디오 재샘플링
==========

여기서는 ``torchaudio`` 를 사용하여 오디오 파형의 재샘플링에 대해 살펴보겠습니다.

"""

# 이번 튜토리얼을 구글 코랩(Google Colab)에서 실행할 때,
# 필요한 패키지들을 아래와 같이 설치해주세요.
# !pip install torchaudio librosa

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

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

#-------------------------------------------------------------------------------
# 데이터 및 도움 기능 준비
#-------------------------------------------------------------------------------

import math
import time

import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import pandas as pd


DEFAULT_OFFSET = 201
SWEEP_MAX_SAMPLE_RATE = 48000
DEFAULT_LOWPASS_FILTER_WIDTH = 6
DEFAULT_ROLLOFF = 0.99
DEFAULT_RESAMPLING_METHOD = 'sinc_interpolation'


def _get_log_freq(sample_rate, max_sweep_rate, offset):
  """[0, max_sweep_rate // 2] 사이의 로그 스케일로 균등하게 간격을 두고 주파수를 가져옵니다.

  오프셋은 음의 무한대 `log(offset + x)`를 방지하기 위해 사용됩니다.

  """
  half = sample_rate // 2
  start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
  return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset

def _get_inverse_log_freq(freq, sample_rate, offset):
  """주어진 주파수가 _get_log_freq 로 주어지는 시간을 찾습니다."""
  half = sample_rate // 2
  return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))

def _get_freq_ticks(sample_rate, offset, f_max):
  # 스윕(sweep) 생성에 사용된 원래 샘플링 비율이 주어지면,
  # 로그 스케일의 주요 주파수 값이 속하는 x축 값을 찾습니다.
  time, freq = [], []
  for exp in range(2, 5):
    for v in range(1, 10):
      f = v * 10 ** exp
      if f < sample_rate // 2:
        t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
        time.append(t)
        freq.append(f)
  t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
  time.append(t_max)
  freq.append(f_max)
  return time, freq

def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
  max_sweep_rate = sample_rate
  freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
  delta = 2 * math.pi * freq / sample_rate
  cummulative = torch.cumsum(delta, dim=0)
  signal = torch.sin(cummulative).unsqueeze(dim=0)
  return signal

def plot_sweep(waveform, sample_rate, title, max_sweep_rate=SWEEP_MAX_SAMPLE_RATE, offset=DEFAULT_OFFSET):
  x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
  y_ticks = [1000, 5000, 10000, 20000, sample_rate//2]

  time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
  freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
  freq_y = [f for f in freq if f >= 1000 and f in y_ticks and f <= sample_rate // 2]

  figure, axis = plt.subplots(1, 1)
  axis.specgram(waveform[0].numpy(), Fs=sample_rate)
  plt.xticks(time, freq_x)
  plt.yticks(freq_y, freq_y)
  axis.set_xlabel('Original Signal Frequency (Hz, log scale)')
  axis.set_ylabel('Waveform Frequency (Hz)')
  axis.xaxis.grid(True, alpha=0.67)
  axis.yaxis.grid(True, alpha=0.67)
  figure.suptitle(f'{title} (sample rate: {sample_rate} Hz)')
  plt.show(block=True)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

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

def benchmark_resample(
    method,
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=DEFAULT_LOWPASS_FILTER_WIDTH,
    rolloff=DEFAULT_ROLLOFF,
    resampling_method=DEFAULT_RESAMPLING_METHOD,
    beta=None,
    librosa_type=None,
    iters=5
):
  if method == "functional":
    begin = time.time()
    for _ in range(iters):
      F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                 rolloff=rolloff, resampling_method=resampling_method)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "transforms":
    resampler = T.Resample(sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                           rolloff=rolloff, resampling_method=resampling_method, dtype=waveform.dtype)
    begin = time.time()
    for _ in range(iters):
      resampler(waveform)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "librosa":
    waveform_np = waveform.squeeze().numpy()
    begin = time.time()
    for _ in range(iters):
      librosa.resample(waveform_np, sample_rate, resample_rate, res_type=librosa_type)
    elapsed = time.time() - begin
    return elapsed / iters

######################################################################
# 오디오 파형을 한 주파수에서 다른 주파수로 재샘플링 하기 위해서
# ``transforms.Resample`` 또는 ``functional.resample`` 를 사용할 수 있습니다.
# ``transforms.Resample`` 는 재샘플링에 사용되는 커널을 미리 계산하고 캐시하는 반면
# ``functional.resample`` 는 그때그때 봐 가며 계산하므로, 동일한 매개변수를 사용하여 
# 여러 파형을 재샘플링할 때 ``transforms.Resample`` 를 사용하는 것이 속도가 
# 더 빠릅니다. (벤치마크 섹션 참조)
#
# 두 가지 재샘플링 방법 모두 임의의 시간 단계에서 신호 값을 계산하기 위해
# `대역 제한 싱크 보간(bandlimited sinc interpolation) <https://ccrma.stanford.edu/~jos/resample/>`__
# 을 사용합니다. 구현에는 합성곱(convolution)이 포함되므로 GPU/멀티 스레딩을 
# 활용하여 성능을 향상시킬 수 있습니다. 여러 작업자(worker) 프로세스로 데이터를
# 로딩하는 것과 같이 여러 하위 프로세스에서 재샘플링을 사용할 경우, 프로그램이
# 시스템에서 효율적으로 처리할 수 있는 것보다 더 많은 스레드를 생성할 수 있습니다.
# 이 경우 ``torch.set_num_threads(1)`` 를 설정하는 것이 도움이 될 수 있습니다.
#
# 제한된 수의 샘플은 제한된 수의 주파수만 나타낼 수 있기 때문에, 재샘플링은
# 완벽환 결과를 생성하지 않으며, 다양한 매개변수를 사용하여 결과의 품질과 계산 속도를
# 제어할 수 있습니다. 시간이 지남에 따라 주파수가 기하급수적으로 증가하는
# 사인 파형인 로그 사인 스윕(sweep)을 재샘플링하여 이러한 특성을 입증합니다.
#
# 아래 스펙트로그램은 신호의 주파수 표현을 보여주고, 여기서 x축은 원래 파형의
# 주파수(로그 스케일), y축은 표시된 파형의 주파수, 색의 강도는 진폭에 해당합니다.
#

sample_rate = 48000
resample_rate = 32000

waveform = get_sine_sweep(sample_rate)
plot_sweep(waveform, sample_rate, title="Original Waveform")
play_audio(waveform, sample_rate)

resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)
plot_sweep(resampled_waveform, resample_rate, title="Resampled Waveform")
play_audio(waveform, sample_rate)


######################################################################
# 매개변수를 사용한 재샘플링 품질 제어
# ---------------------------------------------
#
# 저역 통과 필터 폭
# ~~~~~~~~~~~~~~~~~~~~
#
# 보간에 사용되는 필터는 무한 확장되므로, ``lowpass_filter_width`` 매개변수는
# 보간 윈도우에 사용할 필터의 폭을 제어하기 위해 사용됩니다. 보간은 모든 시간
# 단위에서 0을 통과하기 때문에, 제로 크로싱(zero crossings)의 수라고도 합니다.
# 더 큰 ``lowpass_filter_width`` 를 사용하면 더 선명하고 정밀한 필터를 제공하지만
# 계산량이 늘어납니다.
#


sample_rate = 48000
resample_rate = 32000

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=6)
plot_sweep(resampled_waveform, resample_rate, title="lowpass_filter_width=6")

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)
plot_sweep(resampled_waveform, resample_rate, title="lowpass_filter_width=128")


######################################################################
# 롤오프(Rolloff)
# ~~~~~~~
#
# ``rolloff`` 매개변수는 주어진 제한된 샘플링 비율로 나타낼 수 있는 최대 주파수인
# 나이퀴스트(Nyquist) 주파수의 분수로 표현됩니다. ``rolloff`` 는 저역 통과 필터의
# 제한 범위(cutoff)를 결정하고 나이퀴스트보다 높은 주파수가 낮은 주파수로 맵핑될 때
# 발생하는 노이즈(aliasing)의 정도를 제어합니다. 따라서 낮은 롤오프는 노이즈의 양을
# 감소시키지만, 일부 높은 주파수 또한 감소시킬 수 있습니다.
#


sample_rate = 48000
resample_rate = 32000

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, rolloff=0.99)
plot_sweep(resampled_waveform, resample_rate, title="rolloff=0.99")

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, rolloff=0.8)
plot_sweep(resampled_waveform, resample_rate, title="rolloff=0.8")


######################################################################
# 윈도우 함수
# ~~~~~~~~~~~~~~~
#
# 기본적으로 ``torchaudio`` 의 재샘플은 가중 코사인 함수인 한(Hann) 윈도우
# 필터를 사용합니다. 또한 필터의 부드러움(smoothness)과 임펄스(impulse) 폭을
# 설계할 수 있는 추가 ``beta`` 매개변수를 포함하는 거의 최적의 윈도우 함수인
# 카이저(Kaiser) 윈도우를 추가로 지원합니다. 이는 ``resampling_method``
# 매개변수를 사용하여 제어할 수 있습니다.
#


sample_rate = 48000
resample_rate = 32000

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="sinc_interpolation")
plot_sweep(resampled_waveform, resample_rate, title="Hann Window Default")

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="kaiser_window")
plot_sweep(resampled_waveform, resample_rate, title="Kaiser Window Default")


######################################################################
# librosa와의 비교
# --------------------------
#
# ``torchaudio`` 의 재샘플 함수는 librosa (resampy)의 카이저 윈도우 재샘플링과
# 유사한 결과를 생성할 수 있으며, 약간의 노이즈가 있습니다.
#


sample_rate = 48000
resample_rate = 32000

### kaiser_best
resampled_waveform = F.resample(
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    resampling_method="kaiser_window",
    beta=14.769656459379492
)
plot_sweep(resampled_waveform, resample_rate, title="Kaiser Window Best (torchaudio)")

librosa_resampled_waveform = torch.from_numpy(
    librosa.resample(waveform.squeeze().numpy(), sample_rate, resample_rate, res_type='kaiser_best')).unsqueeze(0)
plot_sweep(librosa_resampled_waveform, resample_rate, title="Kaiser Window Best (librosa)")

mse = torch.square(resampled_waveform - librosa_resampled_waveform).mean().item()
print("torchaudio and librosa kaiser best MSE:", mse)

### kaiser_fast
resampled_waveform = F.resample(
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=16,
    rolloff=0.85,
    resampling_method="kaiser_window",
    beta=8.555504641634386
)
plot_specgram(resampled_waveform, resample_rate, title="Kaiser Window Fast (torchaudio)")

librosa_resampled_waveform = torch.from_numpy(
    librosa.resample(waveform.squeeze().numpy(), sample_rate, resample_rate, res_type='kaiser_fast')).unsqueeze(0)
plot_sweep(librosa_resampled_waveform, resample_rate, title="Kaiser Window Fast (librosa)")

mse = torch.square(resampled_waveform - librosa_resampled_waveform).mean().item()
print("torchaudio and librosa kaiser fast MSE:", mse)


######################################################################
# 성능 벤치마크
# ------------------------
#
# 아래는 두 쌍의 샘플링 비율 사이의 다운샘플링 및 업샘플링 파형에 대한
# 벤치마크입니다. ``lowpass_filter_wdith``, 윈도우 유형 및 샘플링 비율이
# 성능에 미칠 수 있는 영향을 보여줍니다. 또한, ``torchaudio`` 의 해당
# 매개변수를 사용하여 ``librosa`` 의 ``kaiser_best`` 및 ``kaiser_fast`` 와
# 비교합니다.
#
# 결과에 대해 자세히 설명하기 위해:
#
# - 더 큰 ``lowpass_filter_width`` 는 더 큰 재샘플링 커널을 생성하므로
#   커널 계산과 합성곱 모두에 대한 계산 시간을 증가시킵니다.
# - ``kaiser_window`` 를 사용하면 중간의 윈도우 값을 계산하는 것이 더
#   복잡하기 때문에 기본 ``sinc_interpolation`` 보다 계산 시간이 더 길어집니다.
# - 샘플과 재샘플 비율 사이의 큰 GCD는 더 작은 커널과 더 빠른 커널 계산을
#   가능하게 하도록 단순화할 수 있습니다.
#


configs = {
    "downsample (48 -> 44.1 kHz)": [48000, 44100],
    "downsample (16 -> 8 kHz)": [16000, 8000],
    "upsample (44.1 -> 48 kHz)": [44100, 48000],
    "upsample (8 -> 16 kHz)": [8000, 16000],
}

for label in configs:
  times, rows = [], []
  sample_rate = configs[label][0]
  resample_rate = configs[label][1]
  waveform = get_sine_sweep(sample_rate)

  # sinc 64 zero-crossings
  f_time = benchmark_resample("functional", waveform, sample_rate, resample_rate, lowpass_filter_width=64)
  t_time = benchmark_resample("transforms", waveform, sample_rate, resample_rate, lowpass_filter_width=64)
  times.append([None, 1000 * f_time, 1000 * t_time])
  rows.append(f"sinc (width 64)")

  # sinc 6 zero-crossings
  f_time = benchmark_resample("functional", waveform, sample_rate, resample_rate, lowpass_filter_width=16)
  t_time = benchmark_resample("transforms", waveform, sample_rate, resample_rate, lowpass_filter_width=16)
  times.append([None, 1000 * f_time, 1000 * t_time])
  rows.append(f"sinc (width 16)")

  # kaiser best
  lib_time = benchmark_resample("librosa", waveform, sample_rate, resample_rate, librosa_type="kaiser_best")
  f_time = benchmark_resample(
      "functional",
      waveform,
      sample_rate,
      resample_rate,
      lowpass_filter_width=64,
      rolloff=0.9475937167399596,
      resampling_method="kaiser_window",
      beta=14.769656459379492)
  t_time = benchmark_resample(
      "transforms",
      waveform,
      sample_rate,
      resample_rate,
      lowpass_filter_width=64,
      rolloff=0.9475937167399596,
      resampling_method="kaiser_window",
      beta=14.769656459379492)
  times.append([1000 * lib_time, 1000 * f_time, 1000 * t_time])
  rows.append(f"kaiser_best")

  # kaiser fast
  lib_time = benchmark_resample("librosa", waveform, sample_rate, resample_rate, librosa_type="kaiser_fast")
  f_time = benchmark_resample(
      "functional",
      waveform,
      sample_rate,
      resample_rate,
      lowpass_filter_width=16,
      rolloff=0.85,
      resampling_method="kaiser_window",
      beta=8.555504641634386)
  t_time = benchmark_resample(
      "transforms",
      waveform,
      sample_rate,
      resample_rate,
      lowpass_filter_width=16,
      rolloff=0.85,
      resampling_method="kaiser_window",
      beta=8.555504641634386)
  times.append([1000 * lib_time, 1000 * f_time, 1000 * t_time])
  rows.append(f"kaiser_fast")

  df = pd.DataFrame(times,
                    columns=["librosa", "functional", "transforms"],
                    index=rows)
  df.columns = pd.MultiIndex.from_product([[f"{label} time (ms)"],df.columns])
  display(df.round(2))
