"""
토치 오디오를 사용한 텍스트 음성 변환
==============================

**Author**: `Yao-Yuan Yang <https://github.com/yangarbiter>`__, `Moto
Hira <moto@fb.com>`__

"""

# %matplotlib inline


######################################################################
# 개요
# --------
# 
# 이 튜토리얼은 토치오디오에서 사전 훈련된 Tacotron2를 사용하여 
# 텍스트 음성 변환 파이프라인을 구축하는 방법을 보여줍니다.
# 
# TTS(텍스트 음성 변환) 파이프라인은 다음과 같이 진행됩니다: 1. 텍스트 전처리
# 
# 첫째, 입력 텍스트는 기호 목록으로 인코딩됩니다. 
# 이 튜토리얼에서는 영어 문자와 음소를 기호로 사용합니다.
# 
# 2. 스펙트로그램 생
# 
# 인코딩된 텍스트에서 스펙트로그램이 생성됩니다. 
# 이를 위해 ``Tacotron2`` 모델을 사용합니다.
# 
# 3. 시간 영역 변환
# 
# 마지막 단계는 스펙트로그램을 파형으로 변환하는 것입니다. 
# 스펙트로그램에서 음성을 생성하는 프로세스를 보코더라고도 합니다. 
# 이 튜토리얼에서는 세 가지 다른 보코더를 사용합니다.
# ```WaveRNN`` <https://pytorch.org/audio/stable/models/wavernn.html>`__,
# ```Griffin-Lim`` <https://pytorch.org/audio/stable/transforms.html#griffinlim>`__,
# and
# ```Nvidia's WaveGlow`` <https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/>`__.
# 
# 다음 그림은 전체 프로세스를 보여줍니다.
# 
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/tacotron2_tts_pipeline.png
# 


######################################################################
# 준비
# -----------
# 
# 먼저 필요한 종속성을 설치합니다. 
# 음소 기반 인코딩을 수행하려면 ``torchaudio`` 외에 ``DeepPhonemizer`` 가 필요합니다.
# 

# 노트북에서 이 예제를 실행할 때 DeepPhonemizer를 설치하십시오.
# !pip3 install deep_phonemizer

import torch
import torchaudio
import matplotlib.pyplot as plt

import IPython

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"



######################################################################
# 텍스트 프로세싱
# ---------------
# 


######################################################################
# 문자 기반 인코딩
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 이 섹션에서는 문자 기반 인코딩이 작동하는 방식을 살펴보겠습니다.
# 
# 사전 훈련된 Tacotron2 모델은 특정 기호 테이블 세트를 기대하기 때문에 
# ``torchaudio`` 에서 동일한 기능을 사용할 수 있습니다. 
# 이 섹션은 인코딩의 기초에 대한 설명입니다.
# 
# 먼저 기호 집합을 정의합니다. 예를 들어 ``'_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz'`` 를 사용할 수 있습니다. 
# 그런 다음 입력 텍스트의 각 문자를 테이블의 해당 기호 인덱스에 매핑합니다.
# 
# 다음은 그러한 처리의 예입니다. 
# 예에서 테이블에 없는 기호는 무시됩니다.
# 

symbols = '_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz'
look_up = {s: i for i, s in enumerate(symbols)}
symbols = set(symbols)

def text_to_sequence(text):
  text = text.lower()
  return [look_up[s] for s in text if s in symbols]

text = "Hello world! Text to speech!"
print(text_to_sequence(text))


######################################################################
# 위에서 언급했듯이 기호 테이블과 인덱스는 사전 훈련된 Tacotron2 모델이 기대하는 것과 일치해야 합니다. 
# ``torchaudio`` 는 사전 훈련된 모델과 함께 변환을 제공합니다. 
# 예를 들어 다음과 같은 변환을 인스턴스화하고 사용할 수 있습니다.
# 

processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()

text = "Hello world! Text to speech!"
processed, lengths = processor(text)

print(processed)
print(lengths)


######################################################################
# ``processor`` 객체는 텍스트 또는 텍스트 목록을 입력으로 사용합니다
# 텍스트 목록이 제공되면 반환된 ``lengths`` 변수는 출력 배치에서 
# 처리된 각 토큰의 유효한 길이를 나타냅니다.
# 
# 중간 표현은 다음과 같이 검색할 수 있습니다.
# 

print([processor.tokens[i] for i in processed[0, :lengths[0]]])


######################################################################
# 음소 기반 인코딩
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# 음소 기반 인코딩은 문자 기반 인코딩과 유사하지만 
# 음소 기반의 기호 테이블과 G2P(Grapheme-to-Phoneme) 모델을 사용합니다.
# 
# G2P 모델의 세부 사항은 이 듀토리얼의 범위를 벗어납니다. 
# 변환이 어떻게 보이는지 살펴보겠습니다.
# 
# 문자 기반 인코딩의 경우와 유사하게 
# 인코딩 프로세스는 사전 훈련된 Tacotron2 모델이 훈련된 것과 일치할 것으로 예상됩니다.
# ``torchaudio`` 에는 프로세스를 생성하는 인터페이스가 있습니다.
# 
# 다음 코드는 프로세스를 만들고 사용하는 방법을 보여줍니다. 
# 무대 뒤에서 ``DeepPhonemizer`` 패키지를 사용하여 G2P 모델을 만들고 
# ``DeepPhonemizer`` 의 작성자가 게시한 사전 훈련된 가중치를 가져옵니다.
# 

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()

text = "Hello world! Text to speech!"
with torch.inference_mode():
  processed, lengths = processor(text)

print(processed)
print(lengths)


######################################################################
# 인코딩된 값은 문자 기반 인코딩의 예와 다릅니다.
# 
# 중간 표현은 다음과 같습니다.
# 

print([processor.tokens[i] for i in processed[0, :lengths[0]]])


######################################################################
# 스펙트로그램 생성
# ----------------------
# 
# ``Tacotron2`` 는 인코딩된 텍스트에서 스펙트로그램을 생성하는 데 사용하는 모델입니다. 
# 모델에 대한 자세한 내용은 다음을 참조하십시오. `the
# paper <https://arxiv.org/abs/1712.05884>`__.
# 
# 사전 훈련된 가중치로 Tacotron2 모델을 인스턴스화하는 것은 쉽지만 
# Tacotron2 모델에 대한 입력은 일치하는 텍스트 프로세서에 의해 처리됩니다.
# 
# ``torchaudio`` 는 파이프라인을 쉽게 생성할 수 있도록 일치하는 모델과 프로세서를 함께 묶습니다
# 
# (사용 가능한 번들 및 사용법은 다음을 참조하십시오 : `the
# documentation <https://pytorch.org/audio/stable/pipelines.html#tacotron2-text-to-speech>`__.)
# 

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)

text = "Hello world! Text to speech!"

with torch.inference_mode():
  processed, lengths = processor(text)
  processed = processed.to(device)
  lengths = lengths.to(device)
  spec, _, _ = tacotron2.infer(processed, lengths)


plt.imshow(spec[0].cpu().detach())


######################################################################
# ``Tacotron2.infer`` 방법은 다항 샘플링을 수행하므로 
# 스펙트로그램을 생성하는 과정에서 임의성이 발생합니다.
# 

for _ in range(3):
  with torch.inference_mode():
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
  plt.imshow(spec[0].cpu().detach())
  plt.show()


######################################################################
# 파형 생성
# -------------------
# 
# 스펙트로그램이 생성되면 마지막 프로세스는 스펙트로그램에서 파형을 복구하는 것입니다.
# 
# ``torchaudio`` 는 ``GriffinLim`` 및 ``WaveRNN`` 기반의 보코더를 제공합니다.
# 


######################################################################
# WaveRNN
# ~~~~~~~
# 
# 이전 섹션에 계속해서 동일한 번들에서 일치하는 WaveRNN 모델을 인스턴스화할 수 있습니다.
# 

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

text = "Hello world! Text to speech!"

with torch.inference_mode():
  processed, lengths = processor(text)
  processed = processed.to(device)
  lengths = lengths.to(device)
  spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
  waveforms, lengths = vocoder(spec, spec_lengths)

torchaudio.save("output_wavernn.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)
IPython.display.display(IPython.display.Audio("output_wavernn.wav"))


######################################################################
# Griffin-Lim
# ~~~~~~~~~~~
# 
# Griffin-Lim 보코더를 사용하는 것은 WaveRNN과 동일합니다. 
# ``get_vocoder`` 메소드를 사용하여 vocode 객체를 인스턴스화하고 스펙트로그램을 전달할 수 있습니다.
# 

bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

with torch.inference_mode():
  processed, lengths = processor(text)
  processed = processed.to(device)
  lengths = lengths.to(device)
  spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
waveforms, lengths = vocoder(spec, spec_lengths)

torchaudio.save("output_griffinlim.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)
IPython.display.display(IPython.display.Audio("output_griffinlim.wav"))


######################################################################
# Waveglow
# ~~~~~~~~
# 
# Waveglow는 Nvidia에서 출시한 보코더입니다. 사전 훈련된 가중치는 Torch Hub에 게시됩니다. 
# ``torch.hub`` 모듈을 사용하여 모델을 인스턴스화할 수 있습니다.
# 

waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to(device)
waveglow.eval()

with torch.no_grad():
  waveforms = waveglow.infer(spec)

torchaudio.save("output_waveglow.wav", waveforms[0:1].cpu(), sample_rate=22050)
IPython.display.display(IPython.display.Audio("output_waveglow.wav"))
