"""
Wav2Vec2를 사용한 음성 인식
================================

**Author**: `Moto Hira <moto@fb.com>`__

이 튜토리얼은 wav2vec 2.0의 사전 훈련된 모델을 사용하여
음성 인식을 수행하는 방법을 보여줍니다
[`paper <https://arxiv.org/abs/2006.11477>`__].

"""


######################################################################
# 개요
# --------
# 
# 음성 인식의 과정은 다음과 같습니다.
# 
# 1. 오디오 파형에서 음향 특징 추출
# 
# 2. 프레임별 음향 특징의 클래스 추정
# 
# 3. 클래스 확률의 수열에서 가설 생성
# 
# Torchaudio는 사전 훈련된 가중치와 예상 샘플링 속도 및 클래스 레이블과
# 같은 관련 정보에 쉽게 접근할 수 있도록 하며
# ``torchaudio.pipelines`` 모듈에서 함께 사용할 수 있습니다.
# 


######################################################################
# 준비
# -----------
# 
# 먼저 필요한 패키지를 가져오고 작업하는 데이터를 가져옵니다.
# 

# %matplotlib inline

import os

import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
import IPython

matplotlib.rcParams['figure.figsize'] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(torchaudio.__version__)
print(device)

SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SPEECH_FILE = "_assets/speech.wav"

if not os.path.exists(SPEECH_FILE):
  os.makedirs('_assets', exist_ok=True)
  with open(SPEECH_FILE, 'wb') as file:
    file.write(requests.get(SPEECH_URL).content)


######################################################################
# pipeline 작성
# -------------------
# 
# 먼저 형상 추출과 분류를 수행하는
# Wav2Vec2 모델을 만들 것입니다.
# 
# Torchaudio에서 사용할 수 있는 두 가지 유형의 Wav2Vec2 사전 훈련 웨이트가 있습니다.
# ARS 작업을 위해 미세 조정된 것과 미세 조정되지 않은 것입니다.
# 
# Wav2Vec2 (및 HuBERT) 모델은 자체 감독 방식으로 훈련됩니다. 그들은 먼저
# 표현 학습을 위한 오디오로만 훈련한 다음 추가 레이블이 있는
# 특정 작업에 맞게 미세 조정됩니다.
# 
# 미세 조정 없이 사전 훈련된 가중치는 다른 다운스트림 작업에도
# 미세 조정할 수 있지만, 이 튜토리얼에서는 이를 다루지 않습니다.
# 
# 여기서는 :py:func:`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` 를 사용합니다.
# 
# :py:mod:`torchaudio.pipelines` 로 사용할 수 있는 여러 모델이 있습니다.
# 교육 방법에 대한 자세한 내용은 문서를 참조하십시오.
# 
# Bundle 객체는 모델 및 기타 정보를 인스턴스화할 수 있는 인터페이스를
# 제공합니다. 샘플링 속도 및 클래스 라벨은 다음과 같습니다.
# 

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())


######################################################################
# 모델은 다음과 같이 구성할 수 있습니다. 이 프로세스는 사전 훈련된
# 가중치를 자동으로 가져와 모델에 로드합니다.
# 

model = bundle.get_model().to(device)

print(model.__class__)


######################################################################
# 데이터 로딩
# ------------
# 
# Creative Commos BY 4.0에 따라 라이선스가 부여된 `VOiCES
# dataset <https://iqtlabs.github.io/voices/>`__ 의 음성 데이터를
# 사용할 것입니다.
# 

IPython.display.Audio(SPEECH_FILE)


######################################################################
# 데이터를 로드하기 위해 :py:func:`torchaudio.load` 를 사용합니다.
# 
# 샘플링 속도가 파이프라인에서 예상하는 것과 다르다면 
# :py:func:`torchaudio.functional.resample` 을 사용하여 다시 샘플링할 수 있습니다.
# 
# .. note::
#
#    - :py:func:`torchaudio.functional.resample` 은 CUDA 텐더에서도 작동합니다.
#    - 동일한 샘플링 속도 집합에서 여러 번 재샘플링을 수행하는 경우
#      :py:func:`torchaudio.transforms.Resample` 을 사용하면 성능이 향상될 수 있습니다.
# 

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
  waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


######################################################################
# 음향 특징 추출
# ----------------------------
# 
# 다음 단계는 오디오에서 음향 특징을 추출하는 것입니다.
# 
# .. note::
#    ARS 작업에 맞게 미세 조정된 Wav2Vec2 모델은 형상 추출 및 분류를 한번에 
#    수행할 수 있지만, 튜토리얼을 위해 여기서는 형상 추출을
#    수행하는 방법도 보여줍니다.
# 

with torch.inference_mode():
  features, _ = model.extract_features(waveform)


######################################################################
# 반환된 기능은 텐서 목록입니다. 
# 각 텐서는 변압기 층의 출력입니다.
# 

fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
  ax[i].imshow(feats[0].cpu())
  ax[i].set_title(f"Feature from transformer layer {i+1}")
  ax[i].set_xlabel("Feature dimension")
  ax[i].set_ylabel("Frame (time-axis)")
plt.tight_layout()
plt.show()


######################################################################
# 특징 분류
# ----------------------
# 
# 음향 특징을 추출한 후 다음 단계는 
# 이들을 일련의 범주로 분류하는 것입니다.
# 
# Wav2Vec2 모델은 형상 추출 및 분류를 한번에 수행할 수 있는
# 방법을 제공합니다.
# 

with torch.inference_mode():
  emission, _ = model(waveform)


######################################################################
# 출력은 logits의 형태입니다. 
# 이것은 확률의 형태가 아닙니다.
# 
# 시각화 해 봅시다.
# 

plt.imshow(emission[0].cpu().T)
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.show()
print("Class labels:", bundle.get_labels())


######################################################################
# 시간대에 걸쳐 특정 라벨에 강한 징후가 있음을 알 수 있습니다.
# 
# 클래스 1 부터 3까지, (``<pad>``, ``</s>`` and ``<unk>``) 대부분
# 큰 음수 값을 가지고 있으며, 이러한 라벨이 기본적으로 추가되지만
# 훈련중에는 사용되지 않는 원래의 ``fairseq`` 구현에서
# 비롯된 인공물입니다.
# 

######################################################################
# Transcripts 생성
# ----------------------
# 
# 레이블 확률의 순서로부터, 이제 transcript를 생성하려고 합니다.
# 가설을 생성하는 과정을 보통 “decoding” 이라고 합니다.
# 
# 특정 시간 단계의 디코딩은 주변 관측치에 의해 영향을 받을 수 있기 때문에
# 디코딩은 단순한 분류보다 더 정교합니다.
# 
# 예를 들어 ``night`` 와 ``knight`` 같은 단어를 생각해 보세요. 비록 그들의
# 사전 확률 분포가 다르더라도 (일반적인 대화에서는, ``night`` 이 ``knight`` 보다
# 훨씬 더 자주 발생함), "칼을 든 기사"와 같은 "knight"로 정확한 대화록을
# 생성하기 위해서는 해독 과정이 충분한 맥락을 볼 때까지
# 최종 결정을 미뤄야 합니다.
# 
# 제안된 많은 디코딩 기법이 있으며, 단어 사전 및 언어 모델과 같은
# 외부 자원이 필요합니다.
# 
# 이 튜토리얼에서는 단순성을 위해 이러한 외부 구성 요소에 의존하지 않는
# Greedy decoding을 수행하고 각 시간 단계에서 가장 좋은 가설을 간단하게 선택할 것입니다.
# 따라서 컨텍스트 정보는 사용되지 않으며 하나의 스크립트만 생성할 수 있습니다.
# 
# Greedy decoding 알고리즘을 정의하는 것으로 시작합니다.
# 

class GreedyCTCDecoder(torch.nn.Module):
  def __init__(self, labels, ignore):
    super().__init__()
    self.labels = labels
    self.ignore = ignore

  def forward(self, emission: torch.Tensor) -> str:
    """Given a sequence emission over labels, get the best path string
    Args:
      emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

    Returns:
      str: The resulting transcript
    """
    indices = torch.argmax(emission, dim=-1)  # [num_seq,]
    indices = torch.unique_consecutive(indices, dim=-1)
    indices = [i for i in indices if i not in self.ignore]
    return ''.join([self.labels[i] for i in indices])


######################################################################
# 이제 디코더 객체를 만들고 스크립트를 디코딩하십시오.
# 

decoder = GreedyCTCDecoder(
    labels=bundle.get_labels(),
    ignore=(0, 1, 2, 3),
)
transcript = decoder(emission[0])


######################################################################
# 결과를 확인하고 다시 오디오를 들어봅니다.
# 

print(transcript)
IPython.display.Audio(SPEECH_FILE)


######################################################################
# ASR 모델은 Connectionist Temporal Classification (CTC)라는 손실 함수를 사용하여 미세 조정됩니다.
# CTC 손실에 대한 자세한 내용은 
# `<https://distill.pub/2017/ctc/>`__에 설명되어 있습니다. CTC에서 빈 토큰(ϵ)은
# 이전 기호의 반복을 나타내는 특별한 토큰입니다.
# 디코딩에서는 이러한 사항이 단순하게 무시됩니다.
# 
# 두 번째로, 형상 추출 섹션에서 설명했듯이, ``fairseq`` 에서 유래한
# Wav2Vec2 모델은 사용되지 않는 라벨을 가지고 있습니다.
# 이것들 또한 무시되어야 합니다.
# 

######################################################################
# 결론
# ----------
# 
# 본 튜토리얼에서는 :py:mod:`torchaudio.pipelines` 를 사용하여 음향 기능 추출 및
# 음성 인식을 수행하는 방법에 대해 알아보았습니다.
# 모델을 구성하고 emission을 얻는 것은 두 줄로 짧습니다.
# 
# ::
# 
#    model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
#    emission = model(waveforms, ...)
# 
