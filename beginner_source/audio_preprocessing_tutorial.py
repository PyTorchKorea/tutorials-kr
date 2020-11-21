"""
torchaudio Tutorial
===================

PyTorch는 연구 프로토타이핑에서 GPU 지원을 통한 상품 배포까지 원활한 경로를 제공하는 오픈 소스 딥 러닝 플랫폼입니다.

머신 러닝 문제를 해결하기 위한 상당한 노력이 데이터 준비에 들어갑니다.
tourchaudio는 PyTorch의 GPU 지원을 활용하고, 데이터 로드를 더 쉽고 읽기 편하게 하기 위한 많은 도구를 제공합니다.
이 튜토리얼에서는 간단한 데이터 세트에서 데이터를 로드하고 전처리하는 방법을 살펴 봅니다.

이 튜토리얼에서 보다 쉬운 시각화를 위해 matplotlib 패키지가 설치되어 있는지 확인하십시오.
"""

import torch
import torchaudio
import matplotlib.pyplot as plt

######################################################################
# 파일 열기
# -----------------
# 
# ``torchaudio``는 wav 및 mp3 형태의 사운드 파일 로드도 지원합니다.
# 우리는 파형(waveform)을 결과 원시 오디오 신호(resulting raw audio signal)라고 부릅니다.

filename = "../_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())

######################################################################
# ``torchaudio``에서 파일을 로드할 때, ``torchaudio.set_audio_backend``을 통해
# `SoX <https://pypi.org/project/sox/>`_ 또는 `SoundFile <https://pypi.org/project/SoundFile/>`_ 
# 을 사용하도록 선택적으로 백엔드를 지정할 수 있습니다.
# 
# 또한 ``torchaudio``는 함수에 대해 JIT 컴파일을 선택적이게 만들고, 가능한 경우 ``nn.Module``을 사용합니다.

######################################################################
# 변환
# ---------------
# 
# ``torchaudio``는 점점 더 많은
# `변환 <https://pytorch.org/audio/transforms.html>`_ 리스트를 지원합니다.
# 
# -  **Resample**: 파형을 다른 샘플 레이트로 리샘플.
# -  **Spectrogram**: 파형에서 스펙트로그램 생성.
# -  **GriffinLim**: Griffin-Lim 변환을 사용해 선형 척도 스펙트로그램에서 파형을 계산.
# -  **ComputeDeltas**: 일반적으로 스펙트로그램인 tensor의 델타 계수 계산.
# -  **ComplexNorm**: 복잡한 tensor의 norm을 계산.
# -  **MelScale**: 변환 행렬을 사용해 일반 STFT를 Mel-frequency STFT로 변환.
# -  **AmplitudeToDB**: 전력/진폭 스케일에서 데시벨 스케일로 스펙트로그램 전환.
# -  **MFCC**: 파형에서 Mel-frequency cepstrum coeffients를 생성.
# -  **MelSpectrogram**: PyTorch의 STFT 함수를 사용해 파형에서 Mel Spectrograms 생성.
# -  **MuLawEncoding**: mu-law 압신 기반 인코딩.
# -  **MuLawDecoding**: mu-law 인코딩된 파형을 디코딩.
# -  **TimeStretch**: 주어진 rate에 대해 pitch를 수정하지 않고 스펙트로그램을 제 시간에 늘림.
# -  **FrequencyMasking**: 주파수 도메인의 스펙트로그램에 마스킹을 적용.
# -  **TimeMasking**: 시간 도메인의 스펙트로그램에 마스킹을 적용.
#
# 각 변환은 일괄 처리를 지원: 단일 원시 오디오 신호 또는 스펙트로그램,
# 또다른 많은 동일한 모양에 대해 변환을 수행할 수 있습니다.
# 
# 모든 변환은 ``nn.Modules`` 또는 ``jit.ScriptModules``이므로,
# 언제든지 신경망의 일부로 사용할 수 있습니다.
# 


######################################################################
# 먼저 로그 스케일에서 스펙트로그램의 로그를 볼 수 있습니다.
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
# 파형을 한 번에 한 채널 씩 리샘플 할 수 있습니다.
# 

new_sample_rate = sample_rate/10

# Resample은 단일 채널에 적용되므로 여기서 첫 번째 채널을 리샘플합니다.
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())


######################################################################
# 변환의 또 다른 예로 Mu-Law 인코딩을 기반으로 신호를 인코딩 할 수 있습니다.
# 하지만 그렇게 하려면 신호가 -1과 1사이여야 합니다.
# tensor는 일반 PyTorch tensor이므로 표준 연산자를 적용할 수 있습니다.
# 

# tensor가 [-1, 1] 구간에 있는 지 확인해 봅시다.
print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))


######################################################################
# 파형이 이미 -1과 1 사이에 있기 때문에 정규화 할 필요는 없습니다.
# 

def normalize(tensor):
    # 평균을 빼고 [-1, 1] 간격으로 조정
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

# 전체 간격 [-1, 1]로 정규화
# waveform = normalize(waveform)


######################################################################
# 파형을 인코딩해봅시다. 
# 

transformed = torchaudio.transforms.MuLawEncoding()(waveform)

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())


######################################################################
# 그리고 디코딩합니다.
# 

reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)

print("Shape of recovered waveform: {}".format(reconstructed.size()))

plt.figure()
plt.plot(reconstructed[0,:].numpy())


######################################################################
# 마침내 원래 파형과 이것의 재구축된 버전을 비교할 수 있습니다.
# 

# 중앙값 상대 차이(median relative difference) 계산
err = ((waveform-reconstructed).abs() / waveform.abs()).median()

print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))


######################################################################
# 기능
# ---------------
# 
# 위에서 본 변환은 계산을 위해 하위 수준의 상태 비저장 함수(lower level stateless functions)에 의존합니다.
# 이러한 기능은 ``torchaudio.functional``에서 사용 가능합니다. 전체 목록은
# `여기 <https://pytorch.org/audio/functional.html>`_에서 확인 가능하며 다음을 포함합니다:
#
# -  **istft**: 단시간 푸리에 역변환.
# -  **gain**: 전체 파형에 증폭 또는 감쇠를 적용.
# -  **dither**: 특정 bit-depth로 저장된 오디오의 감지된 동적 범위를 증가.
# -  **compute_deltas**: tensor의 델타 계수 계산.
# -  **equalizer_biquad**: 바이쿼드 피킹 이퀄라이저 필터 설계와 필터링 수행
# -  **lowpass_biquad**: 바이쿼드 저역 통과(lowpass) 필터 설계와 필터링 수행
# -  **highpass_biquad**: 바이쿼드 고역 통과(highpass) 필터 설계와 필터링 수행
# 
# 예를 들어, `mu_law_encoding` 기능을 사용해봅시다:

mu_law_encoding_waveform = torchaudio.functional.mu_law_encoding(waveform, quantization_channels=256)

print("Shape of transformed waveform: {}".format(mu_law_encoding_waveform.size()))

plt.figure()
plt.plot(mu_law_encoding_waveform[0,:].numpy())

######################################################################
# ``torchaudio.functional.mu_law_encoding``의 출력이 
# ``torchaudio.transforms.MuLawEncoding``의 출력과 어떻게 동일한 지 확인할 수 있습니다.
#
# 이제 몇 가지 다른 기능을 실험하고 그 결과를 시각화 해보겠습니다.
# 스펙트로그램을 사용하여 델타를 계산할 수 있습니다:

computed = torchaudio.functional.compute_deltas(specgram.contiguous(), win_length=3)
print("Shape of computed deltas: {}".format(computed.shape))

plt.figure()
plt.imshow(computed.log2()[0,:,:].detach().numpy(), cmap='gray')

######################################################################
# 원래 파형을 가져와 다른 효과를 적용할 수 있습니다.
#

gain_waveform = torchaudio.functional.gain(waveform, gain_db=5.0)
print("Min of gain_waveform: {}\nMax of gain_waveform: {}\nMean of gain_waveform: {}".format(gain_waveform.min(), gain_waveform.max(), gain_waveform.mean()))

dither_waveform = torchaudio.functional.dither(waveform)
print("Min of dither_waveform: {}\nMax of dither_waveform: {}\nMean of dither_waveform: {}".format(dither_waveform.min(), dither_waveform.max(), dither_waveform.mean()))

######################################################################
# ``torchaudio.functional`` 기능의 또 다른 예는 파형에 필터를 적용하는 것입니다.
# lowpass biquad filter를 파형에 적용하면 주파수 신호가 수정된 새 파형이 출력됩니다.

lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=3000)

print("Min of lowpass_waveform: {}\nMax of lowpass_waveform: {}\nMean of lowpass_waveform: {}".format(lowpass_waveform.min(), lowpass_waveform.max(), lowpass_waveform.mean()))

plt.figure()
plt.plot(lowpass_waveform.t().numpy())

######################################################################
# highpass biquad filter를 사용해 파형을 시각화 할 수도 있습니다.
# 

highpass_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq=2000)

print("Min of highpass_waveform: {}\nMax of highpass_waveform: {}\nMean of highpass_waveform: {}".format(highpass_waveform.min(), highpass_waveform.max(), highpass_waveform.mean()))

plt.figure()
plt.plot(highpass_waveform.t().numpy())


######################################################################
# Kaldi에서 torchaudio로 이동
# ----------------------------------
# 
# 음성 인식 툴킷
# `Kaldi <http://github.com/kaldi-asr/kaldi>`_에 익숙한 사용자가 있을겁니다.
# ``torchaudio``에서 호환성을 제공합니다.
# kaldi scp 또는 ark 파일, 또는 streams에서 아래 함수로 읽을 수 있습니다.

# 
# -  read_vec_int_ark
# -  read_vec_flt_scp
# -  read_vec_flt_arkfile/stream
# -  read_mat_scp
# -  read_mat_ark
# 
# ``torchaudio``는, ``spectrogram``, ``fbank``, ``mfcc``, 
# 그리고 ``resample_waveform``에 대한 Kaldi 호환 변환을 GPU 지원의 이점으로 제공합니다.
# `여기 <compliance.kaldi.html>`__ 에서 더 많은 정보를 볼 수 있습니다.
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
# 또한 Kaldi의 구현과 일치하는 파형에서 필터뱅크 기능의 계산을 지원합니다.
# 

fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)

print("Shape of fbank: {}".format(fbank.size()))

plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')


######################################################################
# 원시 오디오 신호(raw audio signal)에서 mel 주파수 cepstral 계수를 생성할 수 있습니다.
# 이는 Kaldi의 compute-mfcc-feats의 입력/출력과 일치합니다.
# 

mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)

print("Shape of mfcc: {}".format(mfcc.size()))

plt.figure()
plt.imshow(mfcc.t().numpy(), cmap='gray')


######################################################################
# 사용 가능한 데이터세트
# -----------------
# 
# 모델 학습을 위해 자체 데이터세트를 생성하지 않길 원하는 경우, ``torchaudio``는
# 통합 데이터세트 인터페이스를 제공합니다. 이 인터페이스는 파일을 메모리에 지연로드,
# 함수를 다운로드 및 추출, 그리고 모델을 빌드하기 위한 데이터세트를 지원합니다.
# 
# 현재 ``torchaudio``가 지원하는 데이터세트:
#
# -  **VCTK**: 다양한 억양을 가진 109명의 영어 원어민의 음성 데이터
#    (`자세히 알아보기 <https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_).
# -  **Yesno**: 한 사람이 히브리어로 yes 또는 no라고 말하는 60개의 녹음. 각 녹음은 8 단어 길이입니다.
#    (`자세히 알아보기 <https://www.openslr.org/1/>`_).
# -  **Common Voice**: 누구나 음성 지원 어플리케이션을 훈련하는 데 사용할 수 있는 오픈 소스 다국어 음성 데이터 세트
#    (`자세히 알아보기 <https://voice.mozilla.org/en/datasets>`_).
# -  **LibriSpeech**: 읽기 영어 스피치의 대규모(1000시간) 말뭉치 (`자세히 알아보기 <http://www.openslr.org/12>`_).
# 

yesno_data = torchaudio.datasets.YESNO('./', download=True)

# Yesno 데이터 포인트는 레이블이 Yes일 경우 1, no일 경우 0인 정수 목록인 tuple (waveform, sample_rate, labels) 입니다.

# yesno_data의 예를 보려면 데이터 포인트 번호 3을 선택합니다:
n = 3
waveform, sample_rate, labels = yesno_data[n]

print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# 이제 데이터세트에서 사운드 파일을 요청할 때마다, 이를 요청할 때만 메모리에 로드됩니다.
# 즉, 데이터세트는 원하고 사용하는 항목만 로드하고 메모리에 보관하여 메모리를 절약합니다.
#

######################################################################
# 결론
# ----------
# 
# 원시 오디오 신호(raw audio signal) 또는 파형의 예제를 통해
# ``torchaudio``를 사용하여 오디오 파일을 여는 방법과 그 파형을 전처리, 변환 및 적용하는 방법을 설명했습니다.
# 또한 익숙한 Kaldi 함수를 사용하는 방법과 내장된 데이터세트를 사용하여 모델을 구성하는 방법도 시연했습니다.
# ``torchaudio``가 PyTorch를 기반으로 구축되었기 때문에 이러한 기술은 GPU를 활용하면서
# 음성 인식과 같은 더 발전된 오디오 애플리케이션을 위한 블록을 구축하는 데 사용할 수 있습니다.
# 
