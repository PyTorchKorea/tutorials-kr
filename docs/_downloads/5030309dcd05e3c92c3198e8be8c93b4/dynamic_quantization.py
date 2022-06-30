"""
동적 양자화
===========

이 레시피에서는 동적 양자화(dynamic quantization)를 활용하여, LSTM과
유사한 형태의 순환 신경망이 좀 더 빠르게 추론하도록 만드는 방법을 살펴
봅니다. 이를 통해 모델에서 사용하는 가중치의 규모를 줄이고 수행 속도를
빠르게 만들 것입니다.

도입
----

우리는 신경망을 설계할 때 여러 트레이드오프(trade-off)를 마주하게
됩니다. 모델을 개발하고 학습할 때 순환 신경망의 레이어나 매개변수의
수를 바꿔볼 수 있을 텐데, 그럴 때면 정확도와 모델의 규모나 응답
속도(또는 처리량) 사이에 트레이드오프가 생기게 됩니다. 그러한 변화를 줄
때면 시간과 컴퓨터 자원이 많이 소모되는데, 이는 모델 학습 과정에 대해
반복 작업을 수행하기 때문입니다. 양자화 기법을 기법을 사용하면 알려진
모델의 학습이 끝난 후 성능과 모델의 정확도 사이에 비슷한 트레이드오프를
줄 수 있게 될 것입니다.

여러분이 이를 한 번 시도해 본다면 정확도가 별로 손실되지 않으면서도
모델의 규모를 상당히 줄이면서 응답 시간도 감소시킬 수 있을 것입니다.

동적 양자화란 무엇인가?
-----------------------

신경망을 양자화한다는 말의 의미는 가중치나 활성화 함수에서 정밀도가
낮은 정수 표현을 사용하도록 바꾼다는 것입니다. 이를 통해 모델의 규모를
줄일 수 있으며, CPU나 GPU에서 수행하는 수치 연산의 처리량도 높일 수
있습니다.

부동소수점 실수 값을 정수로 바꾸는 것은 본질적으로 실수 값에 어떤
배율을 곱하여 그 결괏값을 정수로 반올림하는 것과 같습니다. 이 배율을
어떻게 정할 것이냐에 따라 양자화하는 방법도 여러 가지로 나뉩니다.

여기서 살펴 볼 동적 양자화의 핵심은 모델을 수행할 때 데이터의 범위를
살펴 보고, 그에 따라 활성화 함수에 곱할 배율을 동적으로 결정하는 데에
있습니다. 이를 통해 배율이 튜닝될 수 있도록, 즉 살펴보고 있는
데이터셋에 포함된 정보가 최대한 유지되도록 할 수 있습니다.

반면에 모델 매개변수는 모델을 변환하는 시점에 이미 알고 있는 상태이며,
따라서 사전에 INT8 형태로 바꿔놓을 수 있습니다.

양자화된 모델에서의 수치 연산은 벡터화된 INT8 연산을 통해 이뤄집니다.
값을 누적하는 연산은 보통 INT16이나 INT32로 수행하게 되는데, 이는
오버플로를 방지하기 위합니다. 이처럼 높은 정밀도로 표현한 값은, 그
다음 레이어가 양자화되어 있다면 다시 INT8로 맞추고, 출력이라면 FP32로
바꿉니다.

동적 양자화는 매개변수 튜닝, 즉 모델을 제품 파이프라인에 넣기 적합한
형태로 만들어야 한다는 부담이 적은 편입니다. 이러한 작업은 LSTM 모델을
배포용으로 변환할 때면 표준적으로 거치는 단계입니다.



.. note::
   여기서 소개할 접근법의 한계


   이 레시피에서는 PyTorch의 동적 양자화 기능과 이를 사용하기 위한
   작업 흐름에 대해 간단히 살펴보려 합니다. 우리는 모델을 변환할 때
   사용할 특정 함수에 초점을 맞춰 설명하려 합니다. 그리고 간결하고
   명료한 설명을 위해 상당한 부분을 단순화할 것입니다.


1. 아주 작은 LSTM 네트워크를 가지고 시작합니다.
2. 네트워크를 랜덤한 은닉 상태로 초기화합니다.
3. 네트워크를 랜덤한 입력으로 테스트합니다.
4. 이 튜토리얼에서는 네트워크를 학습하지 않을 것입니다.
5. 우리가 가지고 시작한 부동소수점 실수를 사용하는 네트워크를 양자화
   했을 때, 규모가 줄어들고 수행 속도가 빨라짐을 살펴볼 것입니다.
6. 네트워크의 출력값이 FP32 네트워크와 크게 다르지 않음을 살펴보겠지만,
   실제로 학습된 네트워크의 정확도 손실 기댓값이 어떻게 되는지는
   살펴보지 않을 것입니다.


여러분은 동적 양자화가 어떻게 진행되는지 살펴보고, 이를 통해 메모리
사용량과 응답 시간이 줄어든다는 점을 살펴볼 것입니다. 이 기법을
학습된 LSTM에 적용하더라도 정확도를 높은 수준으로 유지할 수 있음을
살펴보는 것은 고급 튜토리얼의 내용으로 남겨두겠습니다. 만약 여러분이
좀 더 엄밀한 내용으로 넘어가고 싶다면 `고급 동적 양자화 튜토리얼
<https://tutorials.pytorch.kr/advanced/dynamic_quantization_tutorial.html>`__
을 참고하시기 바랍니다.


단계
----

이 레시피는 다섯 단계로 구성되어 있습니다.

1. 준비 - 이 단계에서는 아주 간단한 LSTM을 정의하고, 필요한 모듈을
   불러 오고, 몇 개의 랜덤 입력 텐서를 준비합니다.

2. 양자화 수행 - 이 단계에서는 부동소수점 실수를 사용하는 모델을
   만들고 이를 양자화한 버전을 생성합니다.

3. 모델의 규모 살펴보기 - 이 단계에서는 모델의 규모가 줄어들었음을
   살펴봅니다.

4. 응답 시간 살펴보기 - 이 단계에서는 두 모델을 구동시키고 실행
   속도(응답 시간)를 비교합니다.

5. 정확도 살펴보기 - 이 단계에서는 두 모델을 구동시키고 출력을
   비교합니다.


1: 준비
~~~~~~~
이 단계에서는 이 레시피에서 계속 사용할 몇 줄의 간단한 코드를
준비합니다.

우리가 여기서 불러올 유일한 모듈은 torch.quantization 뿐이며, 이
모듈에는 PyTorch의 양자화 관련 연산자 및 변환 함수가 포함되어
있습니다. 우리는 또 아주 간단한 LSTM 모델을 정의하고 몇 개의 입력을
준비합니다.

"""

# 이 레시피에서 사용할 모듈을 여기서 불러옵니다
import torch
import torch.quantization
import torch.nn as nn
import copy
import os
import time


# 설명을 위해 아주 아주 간단한 LSTM을 정의합니다
# 여기서는 레이어가 하나 뿐이고 사전 작업이나 사후 작업이 없는
# nn.LSTM을 감싸서 사용합니다
# 이는 Robert Guthrie 의
# https://tutorials.pytorch.kr/beginner/nlp/sequence_models_tutorial.html 과
# https://tutorials.pytorch.kr/advanced/dynamic_quantization_tutorial.html 에서
# 영감을 받은 부분입니다

class lstm_for_demonstration(nn.Module):
  """기초적인 LSTM모델로, 단순히 nn.LSTM 를 감싼 것입니다.
     설명용 예시 이외의 용도로 사용하기에는 적합하지 않습니다.
  """
  def __init__(self,in_dim,out_dim,depth):
     super(lstm_for_demonstration,self).__init__()
     self.lstm = nn.LSTM(in_dim,out_dim,depth)

  def forward(self,inputs,hidden):
     out,hidden = self.lstm(inputs,hidden)
     return out, hidden


torch.manual_seed(29592)  # 재현을 위한 설정

# 매개변수 다듬기(네트워크의 모양(shape) 정하기)
model_dimension=8
sequence_length=20
batch_size=1
lstm_depth=1

# 입력용 랜덤 데이터
inputs = torch.randn(sequence_length,batch_size,model_dimension)
# hidden 은 사실 초기 은닉 상태(hidden state)와 초기 셀 상태(cell state)로 구성된 튜플입니다
hidden = (torch.randn(lstm_depth,batch_size,model_dimension), torch.randn(lstm_depth,batch_size,model_dimension))


######################################################################
# 2: 양자화 수행
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이제 재밌는 부분을 살펴보려 합니다. 우선은 양자화 할 모델 객체를 하나
# 만들고 그 이름을 float\_lstm 으로 둡니다. 우리가 여기서 사용할 함수는
#
# ::
#
#     torch.quantization.quantize_dynamic()
#
# 입니다 (`관련 문서 참고
# <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`__).
# 이 함수는 모델과, 만약 등장한다면 양자화하고 싶은 서브모듈의 목록,
# 그리고 우리가 사용하려 하는 자료형을 입력으로 받습니다. 이 함수는
# 원본 모델을 양자화한 버전을 새로운 모듈의 형태로 반환합니다.
#
# 이게 내용의 전부입니다.
#

# 부동소수점 실수를 사용하는 객체입니다
float_lstm = lstm_for_demonstration(model_dimension, model_dimension,lstm_depth)

# 이 함수 호출이 작업을 수행하는 부분입니다
quantized_lstm = torch.quantization.quantize_dynamic(
    float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# 어떤 차이가 있는지 살펴봅니다
print('Here is the floating point version of this module:')
print(float_lstm)
print('')
print('and now the quantized version:')
print(quantized_lstm)


######################################################################
# 3. 모델의 규모 살펴보기
# ~~~~~~~~~~~~~~~~~~~~~~~
# 자, 이제 모델을 양자화 했습니다. 그러면 어떤 이득이 있을까요? 우선 첫
# 번째는 FP32 모델 매개변수를 INT8 값으로 변환했다는 (그리고 배율 값도
# 구했다는) 점입니다. 이는 우리가 값을 저장하고 다루는 데에 필요한 데이터의
# 양이 약 75% 감소했다는 의미입니다. 기본적인 값이 있기 때문에 아래처럼
# 감소량이 75% 보다는 적지만, 만약 앞에서 모델의 규모를 더 크게 잡았다면
# (가령 모델의 차원을 80 같은 값으로 두었다면) 감소율이 4분의 1로 수렴할
# 것입니다. 이는 저장된 모델의 규모가 매개변수의 값에 훨씬 더 의존하게 되기
# 때문입니다.
#

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# 규모 비교하기
f=print_size_of_model(float_lstm,"fp32")
q=print_size_of_model(quantized_lstm,"int8")
print("{0:.2f} times smaller".format(f/q))


######################################################################
# 4. 응답 시간 살펴보기
# ~~~~~~~~~~~~~~~~~~~~~
# 좋은 점 두 번째는 통상적으로 양자화된 모델의 수행 속도가 좀 더
# 빠르다는 점입니다. 이는
#
# 1. 매개변수 데이터를 처리하는 데 시간이 덜 들기 때문
# 2. INT8 연산이 빠르기 때문
#
# 등의 이유 때문입니다.
#
# 이제 살펴보겠지만, 이 아주 간단한 네트워크의 양자화된 버전은 그 수행
# 속도가 더 빠릅니다. 이는 좀 더 복잡한 네트워크에 대해서도 대체로
# 성립하는 특징이지만, 모델의 구조나 작업을 수행할 하드웨어의 특성 등
# 여러 가지 요소에 따라 그때 그때 다를 수 있습니다.
#

# 성능 비교하기
print("Floating point FP32")
# %timeit float_lstm.forward(inputs, hidden)

print("Quantized INT8")
# %timeit quantized_lstm.forward(inputs,hidden)


######################################################################
# 5: 정확도 살펴보기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 우리는 여기서 정확도를 자세히 살펴보진 않을 것입니다. 이는 우리가
# 제대로 학습된 네트워크가 아니라 랜덤하게 초기화된 네트워크를 사용하기
# 때문입니다. 그럼에도 불구하고 양자화된 네트워크의 출력 텐서가 원본과
# '크게 다르지 않다'는 점을 살펴보는 것은 의미가 있다고 봅니다.
#
# 좀 더 자세한 분석은 이 레시피의 끝부분에 참고 자료로 올려둔 고급
# 튜토리얼을 참고하시기 바랍니다.
#

# 부동소수점 모델 구동하기
out1, hidden1 = float_lstm(inputs, hidden)
mag1 = torch.mean(abs(out1)).item()
print('mean absolute value of output tensor values in the FP32 model is {0:.5f} '.format(mag1))

# 양자화된 모델 구동하기
out2, hidden2 = quantized_lstm(inputs, hidden)
mag2 = torch.mean(abs(out2)).item()
print('mean absolute value of output tensor values in the INT8 model is {0:.5f}'.format(mag2))

# 둘의 결과 비교하기
mag3 = torch.mean(abs(out1-out2)).item()
print('mean absolute value of the difference between the output tensors is {0:.5f} or {1:.2f} percent'.format(mag3,mag3/mag1*100))


######################################################################
# 좀 더 알아보기
# --------------
# 우리는 동적 양자화가 무엇이며 어떤 이점이 있는지 살펴보았고, 간단한
# LSTM 모델을 빠르게 양자화하기 위해 ``torch.quantization.quantize_dynamic()``
# 함수를 사용했습니다.
#
# 이 문서는 빠르고 고수준의 내용입니다. 좀 더 자세하게 보시려면,
# `(beta) Dynamic Quantization on an LSTM Word Language Model Tutorial <https://tutorials.pytorch.kr/advanced/dynamic\_quantization\_tutorial.html>`_
# 방문하여 보시기 바랍니다
#
# 이 레시피에서는 이러한 내용을 빠르게, 그리고 고수준에서 살펴 보았습니다.
# 좀 더 자세한 내용을 알아보고 싶다면 `(베타) LSTM 언어 모델 동적 양자화
# 튜토리얼 <https://tutorials.pytorch.kr/advanced/dynamic\_quantization\_tutorial.html>`_
# 을 계속 공부해 보시기 바랍니다.
#
# 참고 자료
# =========
# 문서
# ~~~~
#
# `양자화 API 문서 <https://pytorch.org/docs/stable/quantization.html>`_
#
# 튜토리얼
# ~~~~~~~~
#
# `(베타) BERT 동적 양자화 <https://tutorials.pytorch.kr/intermediate/dynamic\_quantization\_bert\_tutorial.html>`_
#
# `(베타) LSTM 언어 모델 동적 양자화 <https://tutorials.pytorch.kr/advanced/dynamic\_quantization\_tutorial.html>`_
#
# 블로그 글
# ~~~~~~~~~
# `PyTorch에서 양자화 수행하기 입문서 <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/>`_
#
