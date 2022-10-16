# -*- coding: utf-8 -*-
"""
오디오 데이터 증강
=======================

*역자*: Lee Jong Bub <https://github.com/bub3690>

``torchaudio`` 는 오디오 데이터를 증강시키는 다양한 방법들을 제공합니다.

이 튜토리얼에서는 효과, 필터,
공간 임펄스 응답(RIR, Room Impulse Response)과 코덱을 적용하는 방법을 살펴보겠습니다.

하단부에서는, 깨끗한 음성으로 부터 휴대폰 너머의 잡음이 낀 음성을 합성하겠습니다. 
"""

import torch
import torchaudio
import torchaudio.functional as F

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# 준비
# -----------
#
# 먼저, 모듈을 불러오고 튜토리얼에 사용할 오디오 자료들을 다운로드합니다.
#

import math

from IPython.display import Audio
import matplotlib.pyplot as plt

from torchaudio.utils import download_asset

SAMPLE_WAV = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.wav")
SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")
SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")


######################################################################
# 효과와 필터링 적용하기
# ------------------------------
#
# :py:func:`torchaudio.sox_effects` 는 ``sox`` 와 유사한 필터들을 
# 텐서 객체들과 파일 객체 오디오 소스들에 직접 적용 해줍니다.
#
# 이를 위해 두가지 함수가 사용됩니다:
#
# -  :py:func:`torchaudio.sox_effects.apply_effects_tensor` 는 텐서에
#    효과를 적용합니다.
# -  :py:func:`torchaudio.sox_effects.apply_effects_file` 는 다른 오디오 소스들에
#    효과를 적용합니다.
#
# 두 함수들은 효과의 정의를  ``List[List[str]]`` 형태로 받아들입니다.
# ``sox`` 와 작동하는 방법이 거의 유사합니다. 하지만, 한가지 유의점은
# ``sox`` 는 자동으로 효과를 추가하지만, ``torchaudio`` 의 구현은 그렇지 않다는 점입니다.
#
# 사용 가능한 효과들의 목록을 알고싶다면, `the sox
# documentation <http://sox.sourceforge.net/sox.html>`__ 을 참조해주세요.
#
# **Tip** 즉석으로 오디오 데이터 로드와 다시 샘플링 하고싶다면, 
# 효과 ``"rate"`` 와 함께 :py:func:`torchaudio.sox_effects.apply_effects_file` 을 사용하세요.
#
# **Note** :py:func:`torchaudio.sox_effects.apply_effects_file` 는 파일 형태의 객체 또는 주소 형태의 객체를 받습니다.
# :py:func:`torchaudio.load` 와 유사하게, 오디오 포맷이
# 파일 확장자나 헤더를 통해 추론될 수 없으면,
# 전달인자 ``format`` 을 주어, 오디오 소스의 포맷을 구체화 해줄 수 있습니다.
#
# **Note** 이 과정은 미분 불가능합니다.
#

# 데이터를 불러옵니다.
waveform1, sample_rate1 = torchaudio.load(SAMPLE_WAV)

# 효과들을 정의합니다.
effects = [
    ["lowpass", "-1", "300"],  # 단극 저주파 통과 필터를 적용합니다.
    ["speed", "0.8"],  # 속도를 감소시킵니다.
    # 이 부분은 샘플 레이트만 변경하기에, 이후에
    # 필수적으로 `rate` 효과를 기존 샘플 레이트로 주어야합니다.
    ["rate", f"{sample_rate1}"],
    ["reverb", "-w"],  # 잔향은 약간의 극적인 느낌을 줍니다.
]

# 효과들을 적용합니다.
waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(waveform1, sample_rate1, effects)

print(waveform1.shape, sample_rate1)
print(waveform2.shape, sample_rate2)

######################################################################
# 효과가 적용되면, 프레임의 수와 채널의 수는 기존에 적용된 것들과 달라짐에 주의하세요.
# 이제 오디오를 들어봅시다.
#

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
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
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)

######################################################################
#

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)

######################################################################
# 기존:
# ~~~~~~~~~
#

plot_waveform(waveform1, sample_rate1, title="Original", xlim=(-0.1, 3.2))
plot_specgram(waveform1, sample_rate1, title="Original", xlim=(0, 3.04))
Audio(waveform1, rate=sample_rate1)

######################################################################
# 효과 적용 후:
# ~~~~~~~~~~~~~~~~
#

plot_waveform(waveform2, sample_rate2, title="Effects Applied", xlim=(-0.1, 3.2))
plot_specgram(waveform2, sample_rate2, title="Effects Applied", xlim=(0, 3.04))
Audio(waveform2, rate=sample_rate2)

######################################################################
# 좀 더 극적으로 들리지 않나요?
#

######################################################################
# 방 잔향 모의 실험하기
# -----------------------------
#
# `Convolution
# reverb <https://en.wikipedia.org/wiki/Convolution_reverb>`__ 는
# 깨끗한 오디오를 다른 환경에서 생성된 것처럼 만들어주는 기술입니다.
#
# 예를들어, 공간 임펄스 응답 (RIR)을 활용하여, 깨끗한 음성을
# 마치 회의실에서 발음된 것처럼 만들 수 있습니다.
#
# 이 과정을 위해서, RIR 데이터가 필요합니다. 다음 데이터들은 VOiCES 데이터셋에서 왔습니다.
# 하지만, 직접 녹음할 수도 있습니다. - 직접 마이크를 켜시고, 박수를 치세요!
#

rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
plot_waveform(rir_raw, sample_rate, title="Room Impulse Response (raw)")
plot_specgram(rir_raw, sample_rate, title="Room Impulse Response (raw)")
Audio(rir_raw, rate=sample_rate)

######################################################################
# 먼저, RIR을 깨끗하게 만들어줘야합니다. 주요한 임펄스를 추출하고,
# 신호 전력을 정규화 합니다. 그리고 나서 시간축을 뒤집어 줍니다.
#

rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.norm(rir, p=2)
RIR = torch.flip(rir, [1])

plot_waveform(rir, sample_rate, title="Room Impulse Response")

######################################################################
# 그 후, RIR 필터와 음성 신호를 합성곱 합니다.
#

speech, _ = torchaudio.load(SAMPLE_SPEECH)

speech_ = torch.nn.functional.pad(speech, (RIR.shape[1] - 1, 0))
augmented = torch.nn.functional.conv1d(speech_[None, ...], RIR[None, ...])[0]

######################################################################
# 기존:
# ~~~~~~~~~
#

plot_waveform(speech, sample_rate, title="Original")
plot_specgram(speech, sample_rate, title="Original")
Audio(speech, rate=sample_rate)

######################################################################
# RIR 적용 후:
# ~~~~~~~~~~~~
#

plot_waveform(augmented, sample_rate, title="RIR Applied")
plot_specgram(augmented, sample_rate, title="RIR Applied")
Audio(augmented, rate=sample_rate)


######################################################################
# 배경 소음 추가하기
# -----------------------
#
# 오디오 데이터에 소음을 추가하기 위해서, 간단히 소음 텐서를 오디오 데이터 텐서에 더할 수 있습니다.
# 소음의 정도를 조절하는 흔한 방법은 신호 대 잡음비 (SNR)를 바꾸는 것입니다.
# [`wikipedia <https://ko.wikipedia.org/wiki/%EC%8B%A0%ED%98%B8_%EB%8C%80_%EC%9E%A1%EC%9D%8C%EB%B9%84>`__]
#
# $$ \\mathrm{SNR} = \\frac{P_{signal}}{P_{noise}} $$
#
# $$ \\mathrm{SNR_{dB}} = 10 \\log _{{10}} \\mathrm {SNR} $$
#

speech, _ = torchaudio.load(SAMPLE_SPEECH)
noise, _ = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, : speech.shape[1]]

speech_power = speech.norm(p=2)
noise_power = noise.norm(p=2)

snr_dbs = [20, 10, 3]
noisy_speeches = []
for snr_db in snr_dbs:
    snr = 10 ** (snr_db / 20)
    scale = snr * noise_power / speech_power
    noisy_speeches.append((scale * speech + noise) / 2)

######################################################################
# 배경 잡음:
# ~~~~~~~~~~~~~~~~~
#

plot_waveform(noise, sample_rate, title="Background noise")
plot_specgram(noise, sample_rate, title="Background noise")
Audio(noise, rate=sample_rate)

######################################################################
# SNR 20 dB:
# ~~~~~~~~~~
#

snr_db, noisy_speech = snr_dbs[0], noisy_speeches[0]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
Audio(noisy_speech, rate=sample_rate)

######################################################################
# SNR 10 dB:
# ~~~~~~~~~~
#

snr_db, noisy_speech = snr_dbs[1], noisy_speeches[1]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
Audio(noisy_speech, rate=sample_rate)

######################################################################
# SNR 3 dB:
# ~~~~~~~~~
#

snr_db, noisy_speech = snr_dbs[2], noisy_speeches[2]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
Audio(noisy_speech, rate=sample_rate)


######################################################################
# 코덱을 텐서 객체에 적용하기
# -------------------------------
#
# :py:func:`torchaudio.functional.apply_codec` 는 텐서 오브젝트에 코덱을 적용합니다.
#
# **Note** 이 과정은 미분 불가능합니다.
#


waveform, sample_rate = torchaudio.load(SAMPLE_SPEECH)

configs = [
    {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
    {"format": "gsm"},
    {"format": "vorbis", "compression": -1},
]
waveforms = []
for param in configs:
    augmented = F.apply_codec(waveform, sample_rate, **param)
    waveforms.append(augmented)

######################################################################
# Original:
# ~~~~~~~~~
#

plot_waveform(waveform, sample_rate, title="Original")
plot_specgram(waveform, sample_rate, title="Original")
Audio(waveform, rate=sample_rate)

######################################################################
# 8 bit mu-law:
# ~~~~~~~~~~~~~
#

plot_waveform(waveforms[0], sample_rate, title="8 bit mu-law")
plot_specgram(waveforms[0], sample_rate, title="8 bit mu-law")
Audio(waveforms[0], rate=sample_rate)

######################################################################
# GSM-FR:
# ~~~~~~~
#

plot_waveform(waveforms[1], sample_rate, title="GSM-FR")
plot_specgram(waveforms[1], sample_rate, title="GSM-FR")
Audio(waveforms[1], rate=sample_rate)

######################################################################
# Vorbis:
# ~~~~~~~
#

plot_waveform(waveforms[2], sample_rate, title="Vorbis")
plot_specgram(waveforms[2], sample_rate, title="Vorbis")
Audio(waveforms[2], rate=sample_rate)

######################################################################
# 전화 녹음 모의 실험하기
# ---------------------------
#
# 이전 기술들을 혼합하여, 반향있는 방의 사람들이 이야기하는 배경에서 전화 통화하는 
# 것 처럼 들리는 오디오를 모의 실험할 수 있습니다.
#

sample_rate = 16000
original_speech, sample_rate = torchaudio.load(SAMPLE_SPEECH)

plot_specgram(original_speech, sample_rate, title="Original")

# RIR 적용하기
speech_ = torch.nn.functional.pad(original_speech, (RIR.shape[1] - 1, 0))
rir_applied = torch.nn.functional.conv1d(speech_[None, ...], RIR[None, ...])[0]

plot_specgram(rir_applied, sample_rate, title="RIR Applied")

# 배경 잡음 추가하기
# 잡음이 실제 환경에서 녹음되었기 때문에, 잡음이 환경의 음향 특징을 가지고 있다고 고려했습니다.
# 따라서, RIR 적용 후에 잡음을 추가했습니다
noise, _ = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, : rir_applied.shape[1]]

snr_db = 8
scale = math.exp(snr_db / 10) * noise.norm(p=2) / rir_applied.norm(p=2)
bg_added = (scale * rir_applied + noise) / 2

plot_specgram(bg_added, sample_rate, title="BG noise added")

# 필터링을 적용하고 샘플 레이트 수정하기
filtered, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
    bg_added,
    sample_rate,
    effects=[
        ["lowpass", "4000"],
        [
            "compand",
            "0.02,0.05",
            "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
            "-8",
            "-7",
            "0.05",
        ],
        ["rate", "8000"],
    ],
)

plot_specgram(filtered, sample_rate2, title="Filtered")

# 전화 코덱 적용하기
codec_applied = F.apply_codec(filtered, sample_rate2, format="gsm")

plot_specgram(codec_applied, sample_rate2, title="GSM Codec Applied")


######################################################################
# 기존 음성:
# ~~~~~~~~~~~~~~~~
#

Audio(original_speech, rate=sample_rate)

######################################################################
# RIR 적용 후:
# ~~~~~~~~~~~~
#

Audio(rir_applied, rate=sample_rate)

######################################################################
# 배경 잡음 추가 후:
# ~~~~~~~~~~~~~~~~~~~~~~~
#

Audio(bg_added, rate=sample_rate)

######################################################################
# 필터링 적용 후:
# ~~~~~~~~~
#

Audio(filtered, rate=sample_rate2)

######################################################################
# 코덱 적용 후:
# ~~~~~~~~~~~~~
#

Audio(codec_applied, rate=sample_rate2)
