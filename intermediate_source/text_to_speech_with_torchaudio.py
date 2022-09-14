"""
torchaudio를 사용하여 텍스트에서 음성으로 변환(text-to-speech)
==============================================================
**Author**: `Yao-Yuan Yang <https://github.com/yangarbiter>`__, `Moto Hira <moto@fb.com>`__
**번역자**: `이가람 <https://github.com/garam24>`__

"""

# %matplotlib inline


######################################################################
# 개요
# --------
# 
# 이번 튜토리얼에서는 torchaudio에서 사전학습된 Tacotron2를 사용하여 텍스트에서 음성으로 변환하는 
# 파이프라인을 소개합니다. 
# 
# 텍스트에서 음성으로 변환하는 파이프라인은 다음의 단계를 따릅니다: 1. 텍스트 전처리
# 
# 먼저, 입력 텍스트를 기호 리스트로 인코딩(encoding)합니다. 이 튜토리얼에서는 영문자를 사용하고
# 기호로는 음소(phonene)를 사용하고자 합니다.
# 
# 2. 스펙트로그램(spectrogram) 생성
# 
# 인코딩된 텍스트로부터 스펙트로그램을 생성합니다. 이를 위해 ``Tacotron2`` 모델을 사용할 예정입니다.
# 
# 3. 시간-도메인(time-domain) 변환
# 
# 마지막 단계에서 스펙트로그램을 파형(waveform)으로 변환합니다.
# 스펙트로그램으로부터 음성을 생성하는 이 과정을 보코더(vocoder)라고 부르기도 합니다.
# 이 튜토리얼에서는 세 가지 종류의 보코더가 사용됩니다.
# ```WaveRNN`` <https://pytorch.org/audio/stable/models/wavernn.html>`__,
# ```Griffin-Lim`` <https://pytorch.org/audio/stable/transforms.html#griffinlim>`__,
# and
# ```Nvidia's WaveGlow`` <https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/>`__.
# 
# 다음 그림은 전체 과정을 보여줍니다.
# 
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/tacotron2_tts_pipeline.png
# 


######################################################################
# 준비 단계
# -----------
# 
# 먼저, 필요한 라이브러라를 설치합니다. 음소 단위 인코딩을 하기 위해서는 ``torchaudio`` 를 비롯하여, ``DeepPhonemizer`` 가 필요합니다.
# 
# 주피터 노트북에서 이 예제를 실행할 때, DeepPhonemizer를 설치해주세요.
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
# 텍스트 처리
# ---------------
# 


######################################################################
# 문자 기반 인코딩
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 이번 섹션에서는 문자 기반 인코딩이 어떻게 이루어지는지 다룰 예정입니다.
# 
# 사전학습된 Tacotron2 모델은 기호 테이블들의 집합을 구체적으로 필요로 하기 때문에,
# ``torchaudio`` 는 해당 기능을 제공하고 있습니다. 이번 섹션에서는 인코딩 기초에 대한 설명보다 조금 더 나아가고자 합니다.
# 
# 먼저 기호들의 집합을 정의합니다. 예를 들어, ``'_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz'`` 와 같은 것들을 사용할 수 있습니다. 
# 그리고 나서 입력 텍스트의 각각의 문자를 테이블 상에서 대응하는 기호의 인덱스에 맵핑(mapping)합니다.
# 
# 아래는 이러한 과정의 예시입니다. 테이블에 포함되어있지 않은 기호들은 이 예제에서 제외하였습니다.
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
# 위에서 언급한 것과 같이, 기호 테이블과 인덱스는 사전학습된 Tacotron2 모델에서 요구하는 형태와
# 일치해야합니다. ``torchaudio`` 는 사전학습된 모델에 맞추어 변환시키는 기능을 제공합니다.
# 이 예제에서는 이러한 변환 기능을 아래와 같이 인스턴스화하여 사용할 수 있습니다.
# 

processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()

text = "Hello world! Text to speech!"
processed, lengths = processor(text)

print(processed)
print(lengths)


######################################################################
# ``processor`` 객체는 텍스트 또는 텍스트 리스트를 입력으로 받아들입니다.
# 텍스트 리스트가 주어질 때, 반환되는 ``lenghts`` 변수는 출력 배치(batch)에서 
# 처리된 각 토큰의 유효 길이를 나타냅니다.
# 
# 중간 단계의 형태는 다음과 같이 검색할 수 있습니다.
# 

print([processor.tokens[i] for i in processed[0, :lengths[0]]])


######################################################################
# 음소 기반 인코딩
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 음소 기반 인코딩은 문자 기반 인코딩과 유사하지만, 
# 음소에 기반한 기호 테이블을 사용하고 G2P (Grapheme-to-Phoneme) 모델을 사용한다는 점에서 다릅니다.
# 
# G2P 모델에 대한 상세한 내용은 이번 튜토리얼의 범위를 벗어나기 때문에 
# 해당 변환이 어떻게 이루어지는지를 중심으로 살펴보겠습니다.
# 
# 문자 기반 인코딩의 경우와 비슷하게, 인코딩 과정은 사전학습된 Tacotron2가 학습된 형태에 매칭되어야 합니다.
# ``torchaudio`` 는 이러한 과정을 위한 인터페이스(interface)를 제공합니다.
# 
# 다음의 코드는 이러한 과정을 만들고 사용하는 방법을 보여줍니다. 
# 뒤 편에서는, ``DeepPhonemizer`` 패키지를 사용하여 G2P 모델이 생성되고 ``DeepPhonemizer`` 의 저자가
# 공개한 사전학습된 가중치가 불러들여지게 됩니다.
# 

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()

text = "Hello world! Text to speech!"
with torch.inference_mode():
  processed, lengths = processor(text)

print(processed)
print(lengths)


######################################################################
# 인코딩된 값들이 문자 기반 인코딩의 예제와는 다르다는 점에 유의하세요.
# 
# 중간 과정은 다음과 같은 모습을 보입니다.
# 

print([processor.tokens[i] for i in processed[0, :lengths[0]]])


######################################################################
# 스펙트로그램 생성
# ------------------------------
# 
# ``Tacotron2`` 는 인코딩된 텍스트로부터 스펙트로그램을 생성하는 데 사용되는 모델입니다. 
# 모델에 대한 자세한 내용은 다음의 `논문<https://arxiv.org/abs/1712.05884>`__ 을 참고해주세요.
# 
# 사전학습된 가중치로 Tacotron2 모델을 인스턴스화 하는 것은 간단합니다. 
# 하지만 Tacotron2 모델의 입력은 매칭되는 텍스트 프로세서(text processor)로 처리되어야 한다는 것을
# 유의해주세요.
# 
# ``torchaudio`` 는 매칭되는 모델과 프로세서를 함께 묶어서 파이프라인을 만들기 쉽도록 해줍니다.
# 
# (사용할 수 있는 번들의 종류와 사용법이 궁금하다면, `이 문서 <https://pytorch.org/audio/stable/pipelines.html#tacotron2-text-to-speech>`__ 를 참고하세요.)
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
# ``Tacotron2.infer`` 메소드(method)는 다항 샘플링(multinomial sampling)을 한다는 점을 유의하세요,
# 따라서 스펙트로그램을 생성하는 이 과정에서 무작위성이 발생합니다.
# 

for _ in range(3):
  with torch.inference_mode():
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
  plt.imshow(spec[0].cpu().detach())
  plt.show()


######################################################################
# 파형 생성
# ---------
# 
# 스펙트로그램이 일단 생성되면, 마지막 단계는 스펙트로그램으로부터 파형을 복원하는 것입니다.
# 
# ``torchaudio`` 는 그리핀-림(``GriffinLim``)과 웨이브 RNN(``WaveRNN``)에 기반한 보코더를 제공합니다.
# 


######################################################################
# 웨이브 RNN
# ~~~~~~~~~~~
# 
# 이전 섹션에 이어서, 같은 번들에서 일치하는 웨이브 RNN 모델을 인스턴스화할 수 있습니다.
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
# 그리핀-림
# ~~~~~~~~~
# 
# 그리핀-림 보코더는 웨이브 RNN과 사용하는 방식이 같습니다. 
# 보코드 객체를 ``get_vocoder`` 메소드로 인스턴스화하여 스펙트로그램을 통과할 수 있습니다.
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
# 웨이브 글로우(Waveglow)
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# 웨이브 글로우는 엔비디아(Nvidia)가 공개한 보코더입니다. 사전학습된 가중치가 토치 허브(Torch Hub)에 공개되어 있습니다.
# ``torch.hub`` 모듈을 사용하여 모델을 인스턴스화 할 수 있습니다.
# 

waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to(device)
waveglow.eval()

with torch.no_grad():
  waveforms = waveglow.infer(spec)

torchaudio.save("output_waveglow.wav", waveforms[0:1].cpu(), sample_rate=22050)
IPython.display.display(IPython.display.Audio("output_waveglow.wav"))
