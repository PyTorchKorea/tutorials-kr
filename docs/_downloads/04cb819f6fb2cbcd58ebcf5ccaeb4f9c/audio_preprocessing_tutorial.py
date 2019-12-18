"""
torchaudio 튜토리얼
===================


PyTorch는 GPU를 지원하는 연구 프로토타이핑에서
상품 배포까지 끊김없이 지원하는 오픈 소스 딥 러닝 플랫폼입니다.

머신 러닝 문제 해결에는 많은 노력을 데이터 준비에 씁니다.
``torchaudio`` 는 PyTorch의 GPU 지원을 활용하고, 데이터 로드를 더 쉽고 읽기 쉽게 해주는 많은 도구를 제공합니다.
이 튜토리얼에서는 간단한 데이터 세트에서 데이터를 로드하고 전처리하는 방법을 살펴 봅니다.

이 튜토리얼에서 더 쉬운 시각화를 위한 ``matplotlib`` 패키지가 설치되어 있는지 확인하십시오.

"""

import torch
import torchaudio
import matplotlib.pyplot as plt


######################################################################
# 데이터 세트 열기
# ---------------------
#


######################################################################
# torchaudio는 wav 및 mp3 형식의 사운드 파일 로드를 지원합니다.
# 우리는 결과 원시 오디오 신호(raw audio signal)를 파형(waveform)이라고 부릅니다.
#

filename = "../_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# 변환
# ---------------
#
# torchaudio 점점 더 많은
# `변환 <https://pytorch.org/audio/transforms.html>`_ 을 지원 합니다..
#
# -  **Resample**: 다른 샘플링 레이트로 파형을 리샘플.
# -  **Spectrogram**: 파형에서 스펙트로그램 생성.
# -  **MelScale**: 변환 행렬을 이용해 일반 STFT를 Mel-frequency STFT로 변환.
# -  **AmplitudeToDB**: 스펙트로그램을 파워/크기 단위에서 데시벨 단위로 변환.
# -  **MFCC**: 파형에서 Mel-frequency cepstrum coefficients 생성.
# -  **MelSpectrogram**: Pytorch의 STFT 함수를 이용해 파형에서 MEL Spectrograms 생성.
# -  **MuLawEncoding**: mu-law 압신 기반 파형 인코딩.
# -  **MuLawDecoding**: mu-law 인코딩된 파형 디코딩.
#
# 모든 변환은 nn.Modules 또는 jit.ScriptModules 이므로,
# 언제든지 신경망의 일부로 사용할 수 있습니다.
#


######################################################################
# 처음으로 로그 스케일 스펙트로그램을 볼 수 있습니다.
#

specgram = torchaudio.transforms.Spectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')


######################################################################
# 또는 로그 스케일의  Mel Spectrogram 을 볼 수 있습니다..
#

specgram = torchaudio.transforms.MelSpectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')


######################################################################
# 한번의 하나의 채널 씩 파형을 리샘플 할 수 있습니다.
#

new_sample_rate = sample_rate/10

# 리샘플이 단일 채널에 적용되기 때문에 우리는 여기서 첫번째 채널을 리샘플 합니다
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())


######################################################################
# 변형의 다른 예시로  Mu-Law 인코딩으로 신호를 인코딩 할 수 있습니다.
# 그러나 그렇게 하기 위해서, 신호가 -1과 1 사이가 되도록 하는 것이 필요합니다.
# 그 Tensor 는 일반적인 Pytorch Tensor이므로 표준 동작을 적용할 수 있습니다.
#

# Tensor가 [-1,1] 사이인지 확인해 봅시다.
print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))


######################################################################
# 파형이 이미 -1과 1 사이에 있기 때문에 정규화를 할 필요가 없습니다.
#

def normalize(tensor):
    # 평균을 빼고,  [-1,1] 간격으로 크기를 맞춥니다.
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

# 전체 간격 [-1,1]로 정규화 합니다.
# waveform = normalize(waveform)


######################################################################
# 파형을 인코팅해 봅시다.
#

transformed = torchaudio.transforms.MuLawEncoding()(waveform)

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())


######################################################################
# 이제 디코딩을 합니다.
#

reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)

print("Shape of recovered waveform: {}".format(reconstructed.size()))

plt.figure()
plt.plot(reconstructed[0,:].numpy())


######################################################################
# 마침내 원래 파형과 재구축된 버전을 비교할 수 있습니다.
#

# 중앙값 상대 차이(median relative difference) 계산
err = ((waveform-reconstructed).abs() / waveform.abs()).median()

print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))


######################################################################
# Kaldi에서 torchaudio로 변경
# ----------------------------------
#
# 음성인식 툴킷 `Kaldi <http://github.com/kaldi-asr/kaldi>`_ 에 익숙한 사용자를 위해,
# torchaudio는  ``torchaudio.kaldi_io`` 로 호환성을 제공합니다.
# 그것은 kaldi scp 또는 ark 파일 또는 streams 를 아래 함수로 읽을 수 있습니다:
#
# -  read_vec_int_ark
# -  read_vec_flt_scp
# -  read_vec_flt_arkfile/stream
# -  read_mat_scp
# -  read_mat_ark
#
# torchaudio 는 ``spectrogram`` 과 ``fbank`` 를 위해 GPU의 장점을 가진 Kaldi 호환 변형을 제공합니다.
# 더 많은 정보를 위해서 `여기 <compliance.kaldi.html>`__ 를 보십시오.
#

n_fft = 400.0
frame_length = n_fft / sample_rate * 1000.0
frame_shift = frame_length / 2.0

params = {
    "channel": 0,
    "dither": 0.0,
    "window_type": "hanning",
    "frame_length": frame_length,
    "frame_shift": frame_shift,
    "remove_dc_offset": False,
    "round_to_power_of_two": False,
    "sample_frequency": sample_rate,
}

specgram = torchaudio.compliance.kaldi.spectrogram(waveform, **params)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.t().numpy(), cmap='gray')


######################################################################
# 또한 Kaldi의 구현과 동일한 필터 뱅크 특징 계산을 지원합니다.
#

fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)

print("Shape of fbank: {}".format(fbank.size()))

plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')


######################################################################
# 결론
# ----------
#
# 원시 오디오 신호 또는 파형을 이용해서 torchaudio로 오디오 파일여는 방법과
# 그 파형을 전처리하고 변형하는 방법을 설명했습니다.
# Pytorch에 torchaudio 가 포함되어있기 때문에,
# 이 기술들은 GPU를 활용한 상태로 음성인식과 같은 더 발전된
# 오디오 어플리케이션의 블락을 구축하는데 사용될 수 있습니다.
#
