"""
torchaudio 튜토리얼
===================

PyTorch는 GPU를 지원하는 연구 프로토타이핑부터 상품 배포까지 매끄럽게 제공하는
오픈 소스 소프트웨어 딥 러닝 플랫폼입니다.

머신 러닝 문제 해결에 드는 많은 노력이 데이터 준비에 들어갑니다.
``torchaudio``는 PyTorch의 GPU 지원을 이용하고, 데이터 로드를 더 쉽고 읽기 좋게 해주는 많은 도구를 제공합니다.
이 튜토리얼에서는 우리는 간단한 데이터셋에서 데이터를 어떻게 로드하고 전처리하는지 살펴볼 겁니다.

이 튜토리얼에서 더 쉬운 시각화를 위한 ``matplotlib`` 패키지가 설치되어있는지 확인하십시오.

"""

import torch
import torchaudio
import matplotlib.pyplot as plt

######################################################################
# 파일 열기
# -----------------
# 
# ``torchaudio``는 wav와 mp3 형식의 사운드 파일 로드도 지원합니다.
# 우리는 파형(waveform)을 결과 원시 오디오 신호(raw audio signal)라고 부릅니다.
# 

filename = "../_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())

######################################################################
# ``torchaudio``에서 파일을 로드할 때, ``torchaudio.set_audio_backend``를 통해
# `SoX <https://pypi.org/project/sox/>`_ 또는 `SoundFile <https://pypi.org/project/SoundFile/>`_ 을
# 사용하기 위한 백엔드를 선택적으로 명시할 수 있습니다. 이 백엔드들은 우리가 필요할 때 느리게 로드됩니다.
# 
# ``torchaudio``는 함수에서 JIT 컴파일(just-in-time compilation)을 선택적이게 만들고, ``nn.Module``을 가능한 한 사용합니다.

######################################################################
# 변환
# ---------------
# 
# ``torchaudio``는 점점 더 많은
# `변환 <https://pytorch.org/audio/transforms.html>`_을 지원합니다.
# 
# -  **Resample**: 파형을 다른 샘플 레이트로 리샘플.
# -  **Spectrogram**: 파형에서 스펙트로그램 생성.
# -  **GriffinLim**: Griffin-Lim 변환을 사용해 선형 스케일 크기 스펙트로그램에서 파형을 계산.
# -  **ComputeDeltas**: tenser, 보통 스펙트로그램의 델타 계수(delta coefficients)를 계산.
# -  **ComplexNorm**: 복잡한 tensor의 norm을 계산.
# -  **MelScale**: 변환 행렬을 이용해 일반 STFT를 Mel-frequency STFT로 변환.
# -  **AmplitudeToDB**: 스펙트로그램을 힘/진폭 스케일부터 데시벨 스케일까지 변환.
# -  **MFCC**: 파형에서 Mel-frequency cepstrum coefficients 생성.
# -  **MelSpectrogram**: PyTorch의 STFT 함수를 이용해 파형에서 MEL Spectrograms 생성.
# -  **MuLawEncoding**: mu-law 압신(companding) 기반 파형을 인코딩.
# -  **MuLawDecoding**: mu-law 인코딩된 파형을 디코딩.
# -  **TimeStretch**: 주어진 레이트에서 pitch를 바꾸지 않고 제 시간에 스펙트로그램 확장.
# -  **FrequencyMasking**: 주파수 영역(frequency domain)에서 스펙트로그램에 Masking 적용.
# -  **TimeMasking**: 시간 영역(time domain)에서 스펙트로그램에 Making 적용.
#
# 각 변환은 batching을 지원: 단일 원시 오디오 신호 또는 스펙트로그램,
# 또는 같은 형태의 많은 것들에서 변환해볼 수 있습니다.
# 
# 모든 변환은 ``nn.Modules`` 또는 ``jit.ScriptModules``이므로,
# 언제든지 신경망의 일부로 사용할 수 있습니다.
# 


######################################################################
# 시작으로, 로그 스케일에서 스펙트로그램의 로그를 볼 수 있습니다.
# 

specgram = torchaudio.transforms.Spectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')


######################################################################
# 또는 로그 스케일에서 Mel Spectrogram을 볼 수 있습니다.
# 

specgram = torchaudio.transforms.MelSpectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')


######################################################################
# 한 번에 한 채널씩 파형을 리샘플 할 수 있습니다.
# 

new_sample_rate = sample_rate/10

# 리샘플은 단일 채널에 적용되기 때문에, 여기서 첫 번째 채널을 리샘플 합니다.
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())


######################################################################
# 변환의 다른 예시로, Mu-Law 인코딩 기반의 신호를 인코딩 할 수 있습니다.
# 그러나 이렇게 하기 위해서, 신호는 -1과 1 사이가 되도록 해야합니다.
# 그 tensor는 일반적인 PyTorch tensor이기 때문에 표준 동작을 적용할 수 있습니다.
# 

# tensor가 [-1,1] 사이에 있는지 확인해봅니다.
print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))


######################################################################
# 파형이 이미 -1과 1 사이에 있으므로, 이를 정규화할 필요는 없습니다.
# 

def normalize(tensor):
    # 평균을 빼고, [-1,1] 간격으로 크기를 맞춥니다.
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

# 전체 간격 [-1,1]로 정규화합니다.
# waveform = normalize(waveform)


######################################################################
# 파형을 인코딩해봅시다.
# 

transformed = torchaudio.transforms.MuLawEncoding()(waveform)

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())


######################################################################
# 이제 디코딩합니다.
# 

reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)

print("Shape of recovered waveform: {}".format(reconstructed.size()))

plt.figure()
plt.plot(reconstructed[0,:].numpy())


######################################################################
# 마침내 원래의 파형과 재구축된 버전을 비교할 수 있습니다.
# 

# 중앙값 상대 차이(median relative difference) 계산
err = ((waveform-reconstructed).abs() / waveform.abs()).median()

print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))


######################################################################
# 함수형
# ---------------
# 
# 위에서 본 변환은 계산에 대해 낮은 수준의 stateless functions에 의존합니다.
# 이 함수들은 ``torchaudio.functional`` 아래에서 사용 가능합니다.
# 완전한 리스트는 `여기<https://pytorch.org/audio/functional.html>`_에서 사용 가능하고 다음을 포함합니다:
#
# -  **istft**: 단시간 푸리에 변환의 역(Inverse short time Fourier Transform).
# -  **gain**: 전체 파형에 증폭 또는 감쇠를 적용.
# -  **dither**: 특정 bit-depth로 저장된 오디오의 감지된 동적 범위를 증가.
# -  **compute_deltas**: tensor의 델타 계수 계산
# -  **equalizer_biquad**: biquad peaking equalizer filter를 설계하고 필터링 수행.
# -  **lowpass_biquad**: biquad lowpass filter를 설계하고 필터링 수행. 
# -  **highpass_biquad**: biquad highpass filter를 설계하고 필터링 수행.
# 
# 예를 들어, `mu_law_encoding` 함수형을 써봅시다:

mu_law_encoding_waveform = torchaudio.functional.mu_law_encoding(waveform, quantization_channels=256)

print("Shape of transformed waveform: {}".format(mu_law_encoding_waveform.size()))

plt.figure()
plt.plot(mu_law_encoding_waveform[0,:].numpy())

######################################################################
# ``torchaudio.functional.mu_law_encoding``의 결과가 어떻게
# ``torchaudio.transforms.MuLawEncoding``의 결과와 같은지 볼 수 있습니다.
#
# 이제 몇 가지 다른 함수형들을 실험하고 그들의 결과를 시각화해봅시다.
# 스펙트로그램을 사용해 델타를 계산할 수 있습니다:

computed = torchaudio.functional.compute_deltas(specgram.contiguous(), win_length=3)
print("Shape of computed deltas: {}".format(computed.shape))

plt.figure()
plt.imshow(computed.log2()[0,:,:].detach().numpy(), cmap='gray')

######################################################################
# 우리는 원래의 파형을 가져오고, 이것에 다른 효과를 적용할 수 있습니다.
#

gain_waveform = torchaudio.functional.gain(waveform, gain_db=5.0)
print("Min of gain_waveform: {}\nMax of gain_waveform: {}\nMean of gain_waveform: {}".format(gain_waveform.min(), gain_waveform.max(), gain_waveform.mean()))

dither_waveform = torchaudio.functional.dither(waveform)
print("Min of dither_waveform: {}\nMax of dither_waveform: {}\nMean of dither_waveform: {}".format(dither_waveform.min(), dither_waveform.max(), dither_waveform.mean()))

######################################################################
# ``torchaudio.functional``의 능력에 대한 또 다른 예는 파형에 필터를 적용하는 것입니다.
# 파형에 lowpass biquad filter를 적용하면 주파수의 신호가 변형된 새 파형을 출력할겁니다. 
# 

lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=3000)

print("Min of lowpass_waveform: {}\nMax of lowpass_waveform: {}\nMean of lowpass_waveform: {}".format(lowpass_waveform.min(), lowpass_waveform.max(), lowpass_waveform.mean()))

plt.figure()
plt.plot(lowpass_waveform.t().numpy())

######################################################################
# highpass biquad filter를 적용한 파형 또한 시각화할 수 있습니다.
# 

highpass_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq=2000)

print("Min of highpass_waveform: {}\nMax of highpass_waveform: {}\nMean of highpass_waveform: {}".format(highpass_waveform.min(), highpass_waveform.max(), highpass_waveform.mean()))

plt.figure()
plt.plot(highpass_waveform.t().numpy())


######################################################################
# Kaldi에서 torchaudio로 변경
# ----------------------------------
# 
# 사용자들은 음성 인식 툴킷 `Kaldi <http://github.com/kaldi-asr/kaldi>`_ 가 익숙할 수 있습니다.
# 그들을 위해 ``torchaudio``는 ``torchaudio.kaldi_io``에서 호환성을 제공합니다.
# kaldi scp 또는 ark 파일, 또는 streams를 다음과 같은 함수로 읽을 수 있습니다:
# 
# -  read_vec_int_ark
# -  read_vec_flt_scp
# -  read_vec_flt_arkfile/stream
# -  read_mat_scp
# -  read_mat_ark
# 
# ``torchaudio``는 ``spectrogram``과 ``fbank``, ``mfcc``, 그리고
# ``resample_waveform``를 위해, GPU 지원의 장점을 가진 Kaldi 호환 변환을 제공합니다.
# 더 많은 정보는 `여기<compliance.kaldi.html>`__ 를 보십시오.
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
# 또한 kaldi의 실행과 같은 파형에서 필터뱅크 특징 계산을 지원합니다.
# 

fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)

print("Shape of fbank: {}".format(fbank.size()))

plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')


######################################################################
# 원시 오디오 신호에서 MFCC(mel frequency cepstral coefficient)를 생성할 수 있습니다.
# 이것은 Kaldi의 compute-mfcc-feats와 같습니다.
# 

mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)

print("Shape of mfcc: {}".format(mfcc.size()))

plt.figure()
plt.imshow(mfcc.t().numpy(), cmap='gray')


######################################################################
# 사용 가능한 데이터셋
# -----------------
# 
# 만약 당신의 모델을 훈련시키기 위한 당신만의 데이터셋을 생성하기를 원치 않는다면,
# ``torchaudio``는 통일된 데이터셋 인터페이스를 제공합니다.
# 이 인터페이스는 메모리에 파일 lazy-loading, 함수 다운로드와 추출, 모델 제작을 위한 데이터셋을 지원합니다.
# 
# 데이터셋 ``torchaudio``가 현재 지원하는 것들은 다음과 같습니다:
#
# -  **VCTK**: 다양한 억양을 가진 109명의 원어민들이 발음한 음성 데이터
#    (`더보기 <https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_).
# -  **Yesno**: 한 사람이 히브리어로 yes 또는 no 라고 말하는 60개의 녹음; 
#    각 녹음은 8 단어 길이 (`더보기 <https://www.openslr.org/1/>`_).
# -  **Common Voice**: 누구든 음성 지원 어플리케이션을 훈련시키는 데 사용할 수 있는 다국어 음성 데이터셋 오픈 소스
#    (`Read more here <https://voice.mozilla.org/en/datasets>`_).
# -  **LibriSpeech**: 영어 읽기 음성의 대규모(1000시간) 말뭉치 (`더보기 <http://www.openslr.org/12>`_).
# 

yesno_data = torchaudio.datasets.YESNO('./', download=True)

# Yesno의 데이터 포인트는 라벨들이 yes일 때 1, no일 때 0의 정수 목록인 tuple(waveform, sample_rate, labels)입니다.

# yesno_data의 예시를 보기 위해 데이터 포인트 넘버 3을 고릅니다:
n = 3
waveform, sample_rate, labels = yesno_data[n]

print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# 이제 당신이 데이터셋에서 음성 파일을 요청할 때면 언제든, 오직 당신이 요청할 때만 로드됩니다.
# 이는 데이터셋은 당신이 원하고 사용하는 아이템만 로드하고 메모리에 보관해서 메모리를 절약함을 의미합니다.
#

######################################################################
# 결론
# ----------
# 
# 우리는 원시 오디오 신호 또는 파형에 대한 예시를 사용해 ``torchaudio``로 오디오 파일을 여는 방법과 
# 그 파형을 전처리하고 변형하고 함수를 적용하는 방법을 설명했습니다.
# 또한 Kaldi 함수를 친숙하게 사용하는 방법에 더해 내장된 데이터셋을 활용해 모델을 구축하는 방법도 보여주었습니다.
# PyTorch에 ``torchaudio``가 포함되어 있기 때문에, 이 기술들은 GPU를 이용하면서
# 음성 인식과 같은 더욱 발전된 오디오 어플리케이션의 블럭을 구축하는 데 사용될 수 있습니다.
# 
