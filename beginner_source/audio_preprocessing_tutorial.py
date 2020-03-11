"""
torchaudio 튜토리얼
===================

PyTorch는 GPU를 지원을 포함하여 연구 프로토타이핑부터 프로덕션 배포까지 매끄럽게 연결되는 오픈소스 딥러닝 플랫폼 입니다.

머신 러닝 문제를 해결할때는 많은 노력이 데이터 준비과정에 할애됩니다.
``torchaudio`` 는 PyTorch의 GPU 지원을 활용하고, 더 쉽게 데이터 로드를 더 쉽게 하고 가독성을 높이는 여러 도구를 제공합니다.
이 튜토리얼에서는 간단한 데이터 세트에서 데이터를 로드하고 전처리하는 방법을 살펴 봅니다.

이 튜토리얼을 진행하기 위해서, 더 쉬운 시각화를 위한 ``matplotlib`` 패키지가 설치되어 있는지 확인하십시오.

"""

import torch
import torchaudio
import matplotlib.pyplot as plt

######################################################################
# 파일 열기
# -----------------
# 
# ``torchaudio`` 는 wav 및 mp3 형식의 사운드 파일 로드를 지원합니다.
# 우리는 결과 원시 오디오 신호(raw audio signal)를 파형(waveform)이라고 부릅니다.
# 

filename = "../_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())

######################################################################
# ``torchaudio`` 에 파일을 로드하면, ``torchaudio.set_audio_backend`` 를 통하여 `SoX <https://pypi.org/project/sox/>`_ 또는
# `SoundFile <https://pypi.org/project/SoundFile/>`_ 를 백엔드로 지정할 수 있습니다. 이 백엔드는 
# 필요할때 지연 로딩 됩니다.
# 
# ``torchaudio`` 는 함수를 위해 JIT compilation 을 옵션으로 만들며, 가능한 경우 ``nn.Module`` 을 사용합니다.

######################################################################
# 변환
# ---------------
# 
# ``torchaudio`` 는 점점 떠 많은 `변환 <https://pytorch.org/audio/transforms.html>`_ 을 지원합니다.
# 
# -  **Resample**: 다른 샘플링 레이트로 파형을 리샘플.
# -  **Spectrogram**: 파형에서 스펙트로그램 생성.
# -  **GriffinLim**: Compute waveform from a linear scale magnitude spectrogram using 
#    the Griffin-Lim transformation.
# -  **ComputeDeltas**: Compute delta coefficients of a tensor, usually a spectrogram.
# -  **ComplexNorm**: Compute the norm of a complex tensor.
# -  **MelScale**: 변환 행렬을 이용해 일반 STFT를 Mel-frequency STFT로 변환.
# -  **AmplitudeToDB**: 스펙트로그램을 파워/크기 단위에서 데시벨 단위로 변환.
# -  **MFCC**: 파형에서 Mel-frequency cepstrum coefficients 생성.
# -  **MelSpectrogram**: Pytorch의 STFT 함수를 이용해 파형에서 MEL Spectrograms 생성.
# -  **MuLawEncoding**: mu-law 압신 기반 파형 인코딩.
# -  **MuLawDecoding**: mu-law 인코딩된 파형 디코딩.
# -  **TimeStretch**: Stretch a spectrogram in time without modifying pitch for a given rate.
# -  **FrequencyMasking**: Apply masking to a spectrogram in the frequency domain.
# -  **TimeMasking**: Apply masking to a spectrogram in the time domain.
#
# Each transform supports batching: you can perform a transform on a single raw 
# audio signal or spectrogram, or many of the same shape.
# 
# 모든 변환은 nn.Modules 또는 jit.ScriptModules 이므로,
# 언제든지 신경망의 일부로 사용할 수 있습니다.
# 


######################################################################
# 처음으로, 로그 스케일 스펙트로그램을 볼 수 있습니다.
# 

specgram = torchaudio.transforms.Spectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')


######################################################################
# 또는 로그 스케일의  Mel Spectrogram 을 볼 수 있습니다.
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

# 인터벌 [-1,1] 에 Tensor가 있는지 확인해복겠습니다. 
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
# 파형을 인코딩해 봅시다.
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
# 함수
# ---------------
# 
# 위에서 본 변환들은 변환과정의 연산을 위하여 낮은 수준의 stateless 함수에 의존합니다. 
# ``torchaudio.functional`` 에서 함수들을 확인할 수 있으며, 아래 목록을 포함한 전체 함수는 
# `여기 <https://pytorch.org/audio/functional.html>`_ 에서 확인할 수 있습니다:
#
# -  **istft**: 단시간 푸리에 역변환.
# -  **gain**: 전체 파형에 증폭 또는 감쇠 적용.
# -  **dither**: 특정 비트 심도로 저장된 오디오의 동적 인식 구간 증폭.
# -  **compute_deltas**: 텐서의 계수 델타값 계산.
# -  **equalizer_biquad**: 바이쿼드 피크 이퀄라이저 필터 설계 및 필터 수행.
# -  **lowpass_biquad**: 바이쿼드 저대역 필터 설계 및 필터 수행.
# -  **highpass_biquad**: 바이쿼드 고대역 필터 설계 및 필터 수행.
# 
# 예시로 `mu_law_encoding` 함수를 사용해보겠습니다:

mu_law_encoding_waveform = torchaudio.functional.mu_law_encoding(waveform, quantization_channels=256)

print("Shape of transformed waveform: {}".format(mu_law_encoding_waveform.size()))

plt.figure()
plt.plot(mu_law_encoding_waveform[0,:].numpy())

######################################################################
# ``torchaudio.functional.mu_law_encoding`` 의 출력이 ``torchaudio.transforms.MuLawEncoding`` 의
# 출력과 어떻게 같은지 볼수 있습니다.
# 
# 이제 다른 함수들을 사용해보고 출력을 시각화 해보겠습니다.
# 스펙트로그램을 사용하여 델타를 구할 수 있습니다:

computed = torchaudio.functional.compute_deltas(specgram, win_length=3)
print("계산된 델타의 형태: {}".format(computed.shape))

plt.figure()
plt.imshow(computed.log2()[0,:,:].detach().numpy(), cmap='gray')

######################################################################
# 원본 파형을 가지고 다른 효과를 적용해볼 수 있습니다.
#

gain_waveform = torchaudio.functional.gain(waveform, gain_db=5.0)
print("gain_waveform 의 최소값: {}\ngain_waveform 의 최대값: {}\ngain_waveform 의 평균값: {}".format(gain_waveform.min(), gain_waveform.max(), gain_waveform.mean()))

dither_waveform = torchaudio.functional.dither(waveform)
print("dither_waveform 의 최소값: {}\ndither_waveform 의 최대값: {}\ndither_waveform 의 평균값: {}".format(dither_waveform.min(), dither_waveform.max(), dither_waveform.mean()))

######################################################################
# ``torchaudio.functional`` 에서 가능한 다른 예제는 필터를 파형에 적용하는 것입니다.
# 바이쿼드 저대역 필터를 파형에 적용하면 새로운 신호의 주파수가 바뀐 새로운 파형을 출력 합니다.

lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=3000)

print("lowpass_waveform 의 최소값: {}\nlowpass_waveform 의 최대값: {}\nlowpass_waveform 의 평균값: {}".format(lowpass_waveform.min(), lowpass_waveform.max(), lowpass_waveform.mean()))

plt.figure()
plt.plot(lowpass_waveform.t().numpy())

######################################################################
# 바이쿼드 고대역 필터를 적용해서 시각화 할수도 있습니다.
# 

highpass_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq=2000)

print("highpass_waveform의 최소값: {}\nhighpass_waveform의 최대값: {}\nhighpass_waveform 의 평균값: {}".format(highpass_waveform.min(), highpass_waveform.max(), highpass_waveform.mean()))

plt.figure()
plt.plot(highpass_waveform.t().numpy())


######################################################################
# Kaldi에서 torchaudio로 변경
# ----------------------------------
# 
# 음성인식 툴킷 `Kaldi <http://github.com/kaldi-asr/kaldi>`_ 에 익숙한 사용자를 위해,
# ``torchaudio`` 는  ``torchaudio.kaldi_io`` 로 호환성을 제공합니다.
# 그것은 kaldi scp 또는 ark 파일 또는 streams 를 아래 함수로 읽을 수 있습니다:
# 
# -  read_vec_int_ark
# -  read_vec_flt_scp
# -  read_vec_flt_arkfile/stream
# -  read_mat_scp
# -  read_mat_ark
# 
# ``torchaudio`` 는 GPU 지원의 장점과 함께 ``spectrogram``, ``fbank``, ``mfcc`` 와 
# ``resample_waveform`` 을 위한 Kaldi 호환 변형을 제공합니다.
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
# 또한 Kaldi의 구현에 맞는 필터 뱅크 특징 계산을 지원합니다.
# 

fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)

print("Shape of fbank: {}".format(fbank.size()))

plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')


######################################################################
# 오디오 신호에서 멜 주파수 cepstral 계수를 만들 수 있습니다
# 이것은 Kaldi의 compute-mfcc-feats의 입력 / 출력과 일치합니다.
# 

mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)

print("Shape of mfcc: {}".format(mfcc.size()))

plt.figure()
plt.imshow(mfcc.t().numpy(), cmap='gray')


######################################################################
# 사용 가능한 데이터셋
# -----------------
# 
# 모델을 훈련시키기 위한 데이터셋을 직접 만들고 싶지 않다면, ``torchaudio`` 는 통합된
# 데이터셋 인터페이스를 제공합니다. 이 인터페이스는 모델을 만들기 위한 기능(functions)과 
# 데이터셋들을 다운로드하고 파일을 메모리로 레이지 로딩 할 수 있도록 지원합니다.
# 
# ``torchaudio`` 가 현재 지원하는 데이터셋은 다음과 같습니다:
#
# -  **VCTK**: 다양한 억양을 가진 109명의 영어 원어민의 발성 데이터
#    (`더 읽어보기 <https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_).
# -  **Yesno**: 한 사람의 목소리로 히브리어로 네 또는 아니오를 60번 녹음한 데이터; 
#    각 녹음은 8 단어로 이루어져 있습니다. (`더 읽어보기 <https://www.openslr.org/1/>`_).
# -  **Common Voice**: 누구나 스피치 어플리케이션을 훈련하기 위해 사용할 수 있는  
#    오픈소스 다국어 음성 데이터셋 (`더 읽어보기 <https://voice.mozilla.org/en/datasets>`_).
# -  **LibriSpeech**: 대규모 (1,000 시간) 말뭉치로 구성된 영어 스피치 (`더 읽어보기 <http://www.openslr.org/12>`_).
# 

yesno_data = torchaudio.datasets.YESNO('./', download=True)

# Yesno 의 데이터 포인트는 튜플 (waveform, sample_rate, labels) 이며, 라벨은 정수 배열로 
# yes 면 1, no 면 0으로 되어 있습니다. 

# yesno_data 의 예를 보기 위해 데이터 포인트 3번을 확인해 보겠습니다:
n = 3
waveform, sample_rate, labels = yesno_data[n]

print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# 데이터셋에서 소리 파일을 찾을때만 소리 파일이 메모리에 로드 됩니다. 
# 사용하고자 하는 항목만 메모리에 불러와서 사용하므로 메모리를 절약할 수 있다는 의미입니다.
#

######################################################################
# 결론
# ----------
#
# 원시 오디오 신호 또는 파형을 이용해서 ``torchaudio`` 로 오디오 파일여는 방법과
# 그 파형을 전처리 및 변형 후 함수들(functions)을 적용하는 방법을 설명했습니다.
# 또한 잘 알려진 Kaldi 함수들을 사용하는 방법과, 우리의 모델을 만들기 위해 빌트인된 
# 데이터셋을 활용 하는것을 시연했습니다. 
# Pytorch에 ``torchaudio`` 가 포함되어있기 때문에, 이 기술들은 GPU를 활용한 
# 상태로 음성인식과 같은 더 발전된 오디오 어플리케이션의 블락을 구축하는데 사용될 수 있습니다.
# 
# 
