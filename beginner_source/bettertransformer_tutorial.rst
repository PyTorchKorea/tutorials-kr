Better Transformer를 이용한 고속 트랜스포머 추론
===============================================================

**저자**: `마이클 그쉬빈드 <https://github.com/mikekgfb>`__

 
이 튜토리얼에서는 PyTorch 1.12 릴리스의 일부로 Better Transformer (BT)를 소개합니다.
여기서는 torchtext를 사용한 Production 추론에 Better Transformer를 적용하는 방법을 보여줍니다.
Better Transformer는 CPU와 GPU에서 고성능으로 트랜스포머 모델의 배포를 가속화하기 위한 Production으로 바로 적용가능한 fastpath입니다.
이 fastpath 기능은 PyTorch 코어 nn.module을 직접 기반으로 하거나 torchtext를 사용하는 모델에 대해 이해하기 쉽고 명확하게 작동합니다.

Better Transformer fastpath로 가속화될 수 있는 모델은 PyTorch 코어 torch.nn.module 클래스인 TransformerEncoder, TransformerEncoderLayer, 
그리고 MultiHeadAttention을 사용하는 모델입니다. 
또한, torchtext는 fastpath 가속화의 이점을 얻기 위해 코어 라이브러리 모듈들을 사용하도록 업데이트되었습니다. 
(추후 더 많은 모듈이 fastpath 실행을 지원할 수 있습니다.)


Better Transformer는 두 가지 유형의 가속화를 제공합니다:

* CPU와 GPU에 대한 Native multihead attention(MHA) 구현으로 전반적인 실행 효율성을 향상시킵니다.
* NLP 추론에서의 sparsity를 활용합니다. 가변 길이 입력(variable input lengths)으로 인해 입력 토큰에 많은 수의
  패딩 토큰이 포함될 수 있는데, 이러한 토큰들의 처리를 건너뛰어 상당한 속도 향상을 제공합니다.

Fastpath 실행은 몇 가지 기준을 충족해야 합니다. 가장 중요한 건, 모델이 추론 모드에서 실행되어야 하며 
gradient tape 정보를 수집하지 않는 입력 텐서에 대해 작동해야 한다는 것입니다(예: torch.no_grad를 사용하여 실행).

이 예제를 Google Colab에서 따라하려면, `여기를 클릭
<https://colab.research.google.com/drive/1KZnMJYhYkOMYtNIX5S3AGIYnjyG0AojN?usp=sharing>`__.



이 튜토리얼에서 Better Transformer의 기능들
--------------------------------------------

* 사전 훈련된 모델 로드 (Better Transformer 없이 PyTorch 버전 1.12 이전에 생성된 모델)
* CPU에서 BT fastpath를 사용한 경우와 사용하지 않은 경우의 추론의 실행 및 벤치마크 (네이티브 MHA만 해당)
* (구성 가능한)디바이스에서 BT fastpath를 사용한 경우와 사용하지 않은 경우의 추론의 실행 및 벤치마크 (네이티브 MHA만 해당)
* sparsity 지원 활성화
* (구성 가능한) 디바이스에서 BT fastpath를 사용한 경우와 사용하지 않은 경우의 추론의 실행 및 벤치마크 (네이티브 MHA + 희소성)



추가적인 정보들
-----------------------
더 나은 트랜스포머에 대한 추가 정보는 PyTorch.Org 블로그에서 확인할 수 있습니다.  
`고속 트랜스포머 추론을 위한 Better Transformer 
<https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference//>`__.



1. 설정

1.1 사전 훈련된 모델 로드

`torchtext.models <https://pytorch.org/text/main/models.html>`__ 의 지침에 따라 미리 정의된 torchtext 모델에서 XLM-R 모델을 다운로드합니다.
또한 가속기 상에서의 테스트를 실행하기 위해 DEVICE를 설정합니다. (필요에 따라 사용 환경에 맞게 GPU 실행을 활성화면 됩니다.)

.. code-block:: python 

    import torch
    import torch.nn as nn

    print(f"torch version: {torch.__version__}")

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"torch cuda available: {torch.cuda.is_available()}")

    import torch, torchtext
    from torchtext.models import RobertaClassificationHead
    from torchtext.functional import to_tensor
    xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
    classifier_head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim = 1024)
    model = xlmr_large.get_model(head=classifier_head)
    transform = xlmr_large.transform()

1.2 데이터셋 설정

두 가지 유형의 입력을 설정하겠습니다. 작은 입력 배치와 sparsity를 가진 큰 입력 배치입니다.

.. code-block:: python

    small_input_batch = [
                   "Hello world", 
                   "How are you!"
    ]
    big_input_batch = [
                   "Hello world", 
                   "How are you!", 
                   """`Well, Prince, so Genoa and Lucca are now just family estates of the
    Buonapartes. But I warn you, if you don't tell me that this means war,
    if you still try to defend the infamies and horrors perpetrated by
    that Antichrist- I really believe he is Antichrist- I will have
    nothing more to do with you and you are no longer my friend, no longer
    my 'faithful slave,' as you call yourself! But how do you do? I see
    I have frightened you- sit down and tell me all the news.`

    It was in July, 1805, and the speaker was the well-known Anna
    Pavlovna Scherer, maid of honor and favorite of the Empress Marya
    Fedorovna. With these words she greeted Prince Vasili Kuragin, a man
    of high rank and importance, who was the first to arrive at her
    reception. Anna Pavlovna had had a cough for some days. She was, as
    she said, suffering from la grippe; grippe being then a new word in
    St. Petersburg, used only by the elite."""
    ]

다음으로, 작은 입력 배치 또는 큰 입력 배치 중 하나를 선택하고, 입력을 전처리한 후 모델을 테스트합니다.

.. code-block:: python

    input_batch=big_input_batch

    model_input = to_tensor(transform(input_batch), padding_value=1)
    output = model(model_input)
    output.shape

마지막으로, 벤치마크 반복 횟수를 설정합니다.

.. code-block:: python

    ITERATIONS=10

2. 실행

2.1   CPU에서 BT fastpath를 사용한 경우와 사용하지 않은 경우의 추론의 실행 및 벤치마크 (네이티브 MHA만 해당)

CPU에서 모델을 실행하고 프로파일 정보를 수집합니다:

* 첫 번째 실행은 전통적인 실행('slow path')을 사용합니다.
* 두 번째 실행은 model.eval()을 사용하여 모델을 추론 모드로 설정하고 torch.no_grad()로 변화도(gradient) 수집을 비활성화하여 BT fastpath 실행을 활성화합니다.

CPU에서 모델을 실행할 때 성능이 향상된 것을 볼 수 있을 겁니다.(향상 정도는 CPU 모델에 따라 다릅니다)
fastpath 프로파일에서 대부분의 실행 시간이 네이티브 `TransformerEncoderLayer`의 저수준 연산을 구현한 `aten::_transformer_encoder_layer_fwd`에 소요되는 것을 주목하세요:

.. code-block:: python

    print("slow path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
      for i in range(ITERATIONS):  
        output = model(model_input)
    print(prof)

    model.eval()

    print("fast path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
      with torch.no_grad():
        for i in range(ITERATIONS):
          output = model(model_input)
    print(prof)


2.2 (구성 가능한)디바이스에서 BT fastpath를 사용한 경우와 사용하지 않은 경우의 추론의 실행 및 벤치마크 (네이티브 MHA만 해당)

BT sparsity 설정을 확인해보겠습니다.

.. code-block:: python

    model.encoder.transformer.layers.enable_nested_tensor
    

이번엔 BT sparsity을 비활성화합니다.

.. code-block:: python

    model.encoder.transformer.layers.enable_nested_tensor=False    
    
 
DEVICE에서 모델을 실행하고, DEVICE에서의 네이티브 MHA 실행에 대한 프로파일 정보를 수집합니다:

* 첫 번째 실행은 전통적인 ('slow path') 실행을 사용합니다.
* 두 번째 실행은 model.eval()을 사용하여 모델을 추론 모드로 설정하고 torch.no_grad()로 변화도(gradient) 수집을 비활성화하여 BT fastpath 실행을 활성화합니다.

GPU에서 실행할 때, 특히 작은 입력 배치로 설정한 경우 속도가 크게 향상되는 것을 볼 수 있을 겁니다.

.. code-block:: python

    model.to(DEVICE)
    model_input = model_input.to(DEVICE)

    print("slow path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
      for i in range(ITERATIONS):  
        output = model(model_input)
    print(prof)

    model.eval()

    print("fast path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
      with torch.no_grad():
        for i in range(ITERATIONS):
          output = model(model_input)
    print(prof)
    

2.3 (구성 가능한) 디바이스에서 BT fastpath를 사용한 경우와 사용하지 않은 경우의 추론의 실행 및 벤치마크 (네이티브 MHA + 희소성)

sparsity 지원을 활성화합니다.

.. code-block:: python

    model.encoder.transformer.layers.enable_nested_tensor = True

DEVICE에서 모델을 실행하고, DEVICE에서의 네이티브 MHA와 sparsity 지원 실행에 대한 프로파일 정보를 수집합니다:

* 첫 번째 실행은 전통적인 ('slow path') 실행을 사용합니다.
* 두 번째 실행은 model.eval()을 사용하여 모델을 추론 모드로 설정하고 torch.no_grad()로 변화도(gradient) 수집을 비활성화하여 BT fastpath 실행을 활성화합니다.

GPU에서 실행할 때, 특히 sparsity를 포함하는 큰 입력 배치 설정에서 상당한 속도 향상을 볼 수 있을 겁니다.

.. code-block:: python

    model.to(DEVICE)
    model_input = model_input.to(DEVICE)

    print("slow path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
      for i in range(ITERATIONS):  
        output = model(model_input)
    print(prof)

    model.eval()

    print("fast path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
      with torch.no_grad():
        for i in range(ITERATIONS):
          output = model(model_input)
    print(prof)


요약
-------
 
이 튜토리얼에서는 torchtext에서 PyTorch 코어의 트랜스포머 인코더 모델을 위한 Better Transformer 지원을 활용하여, 
Better Transformer를 이용한 고속 트랜스포머 추론을 소개했습니다. 
BT fastpath 실행이 가능해지기 이전에 훈련된 모델에서 Better Transformer의 사용을 시연했습니다. 
또한 BT fastpath 실행의 두 가지 모드인 네이티브 MHA 실행과 BT sparsity 가속화의 사용을 시연 및 벤치마크를 해보았습니다.