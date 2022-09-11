양자화 레시피
============

이 레시피는 Pytorch 모델을 양자화하는 방법을 설명합니다. 양자화된 모델은 원본 모델과 거의 같은 정확도를 내면서, 사이즈가 줄어들고 추론 속도가 빨라집니다. 양자화 작업은 서버 모델과 모바일 모델 배포에 모두 적용될 수 있지만, 모바일 환경에서 특히 중요하고 매우 필요합니다. 그 이유는 양자화를 적용하지 않은 모델의 크기가 iOS나 Android 앱이 허용하는 크기 한도를 초과하고, 그로 인해 모델의 배포나 OTA 업데이트가 너무 오래 걸리며, 또한 추론 속도가 너무 느려서 사용자의 쾌적함을 방해하기 때문입니다.

소개
----

양자화는 모델 매개변수를 구성하는 32비트 크기의 실수 자료형의 숫자를 8비트 크기의 정수 자료형의 숫자로 전환하는 기법입니다. 양자화 기법을 적용하면, 정확도는 거의 같게 유지하면서, 모델의 크기와 메모리 전체 사용량을 원본 모델의 4분의 1까지 감소시킬 수 있고, 추론은 2~4배 정도 빠르게 만들 수 있습니다. 

모델을 양자화하는 데는 전부 세 가지의 접근법 및 작업방식이 있습니다. 학습 후 동적 양자화(post training dynamic quantization), 학습 후 정적 양자화(post training static quantization), 그리고 양자화를 고려한 학습(quantization aware training)이 있습니다. 하지만 사용하려는 모델이 이미 양자화된 버전이 있다면, 위의 세 가지 방식을 거치지 않고 그 버전을 바로 사용하면 됩니다. 예를 들어, `torchvision` 라이브러리에는 이미 MobileNet v2, ResNet 18, ResNet 50, Inception v3, GoogleNet을 포함한 모델의 양자화된 버전이 존재합니다. 따라서 비록 단순한 작업이겠지만, 사전 학습 및 양자화된 모델 사용(use pretrained quantized model)을 또 다른 작업 방식 중 하나로 포함하려 합니다.

.. note::
    양자화는 일부 제한된 범위의 연산자에만 지원됩니다. 더 많은 정보는 `여기 <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#device-and-operator-support>`_ 를 참고하세요.

요구 사항
--------

PyTorch 1.6.0 or 1.7.0

torchvision 0.6.0 or 0.7.0

작업 흐름
---------

모델을 양자화하려면 다음 4가지 방식 중 하나를 사용하세요.

1. 사전 학습 및 양자화된 MobileNet v2 사용하기 (Use Pretrained Quantized MobileNet v2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

사전 학습된 MobileNet v2 모델을 불러오려면, 다음을 입력하세요.

::

    import torchvision
    model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)


양자화 전의 MobileNet v2 모델과 양자화된 버전의 모델의 크기를 비교합니다.

::

    model = torchvision.models.mobilenet_v2(pretrained=True)

    import os
    import torch

    def print_model_size(mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
        os.remove('tmp.pt')

    print_model_size(model)
    print_model_size(model_quantized)


출력은 다음과 같습니다.

::

    14.27 MB
    3.63 MB

2. 학습 후 동적 양자화 (Post Training Dynamic Quantization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

동적 양자화를 적용하면, 모델의 모든 가중치는 32비트 크기의 실수 자료형에서 8비트 크기의 정수 자료형으로 전환되지만, 활성화에 대한 계산을 진행하기 직전까지는 활성 함수는 8비트 정수형으로 전환하지 않게 됩니다. 동적 양자화를 적용하려면, `torch.quantization.quantize_dynamic` 을 사용하면 됩니다. 

::

    model_dynamic_quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )

여기서 `qconfig_spec` 으로 `model` 내에서 양자화 적용 대상인 내부 모듈(submodules)을 지정합니다.

.. warning:: 동적 양자화는 사전 학습된 양자화 적용 모델이 준비되지 않았을 때 사용하기 가장 쉬운 방식이지만, 이 방식의 주요 한계는 `qconfig_spec` 옵션이 현재는 `nn.Linear` 과 `nn.LSTM` 만 지원한다는 것입니다. 이는 `nn.Conv2d` 같은 다른 모듈을 양자화할 때, 나중에 논의될 정적 양자화나 양자화를 고려한 학습을 사용해야 한다는 걸 의미합니다.

`quantize_dynamic` API call 관련 전체 문서는 `여기 <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_ 를 참고하세요. 학습 후 동적 양자화를 사용하는 세 가지 예제에는 `the Bert example <https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html>`_, `an LSTM model example <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html#test-dynamic-quantization>`_, `demo LSTM example <https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html#do-the-quantization>`_ 이 있습니다.

3. 학습 후 정적 양자화 (Post Training Static Quantization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이 방식은 모델의 가중치와 활성 함수 모두를 8비트 크기의 정수 자료형으로 사전에 바꾸기 때문에, 동적 양자화처럼 추론 과정 중에 활성 함수를 전환하지는 않습니다. 따라서 이 방식은 성능이 뛰어납니다.

정적 양자화를 모델에 적용하는 코드는 다음과 같습니다.

::

    backend = "qnnpack"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

이다음에 `print_model_size(model_static_quantized)` 를 실행하면 정적 양자화가 적용된 모델이 `3.98MB` 라 표시됩니다.

모델의 전체 정의와 정적 양자화의 예제는 `여기 <https://pytorch.org/docs/stable/quantization.html#quantization-api-summary>`_ 에서 확인하세요. 특수한 정적 양자화 튜토리얼은 `여기 <https://tutorials.pytorch.kr/advanced/static_quantization_tutorial.html>`_ 에서 확인하세요.

.. note::
   모바일 장비는 일반적으로 ARM 아키텍처를 탑재하는데 여기서 모델이 작동하게 하려면, `qnnpack` 을 `backend` 로 사용해야 합니다. 이와 달리 x86 아키텍처를 탑재한 컴퓨터에서 모델이 작동하게 하려면, `fbgemm` 을 `backend` 로 사용하세요.

4. 양자화를 고려한 학습 (Quantization Aware Training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

양자화를 고려한 학습은 모델 학습 과정에서 모든 가중치와 활성 함수에 가짜 양자화를 삽입하게 되고, 학습 후 양자화하는 방법보다 높은 추론 정확도를 가집니다. 이는 주로 CNN 모델에 사용됩니다.

모델을 양자화를 고려한 학습을 가능하게 하려면, 모델 정의 부분의 `__init__` 메소드에서 `QuantStub` 과 `DeQuantStub` 을 정의해야 합니다. 이들은 각각 tensor를 실수형에서 양자화된 자료형으로 전환하거나 반대로 전환하는 역할입니다.

::

    self.quant = torch.quantization.QuantStub()
    self.dequant = torch.quantization.DeQuantStub()

그다음, 모델 정의 부분의 `forward` 메소드의 시작 부분과 끝부분에서, `x = self.quant(x)` 와 `x = self.dequant(x)` 를 호출하세요.

양자화를 고려한 학습을 진행하려면, 다음의 코드 조각을 사용하십시오.

::

    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    model_qat = torch.quantization.prepare_qat(model, inplace=False)
    # 양자화를 고려한 학습이 여기서 진행됩니다.
    model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)

양자화를 고려한 학습의 더 자세한 예시는 `여기 <https://pytorch.org/docs/master/quantization.html#quantization-aware-training>`_ 와 `여기 <https://tutorials.pytorch.kr/advanced/static_quantization_tutorial.html#quantization-aware-training>`_ 를 참고하세요.

사전 학습된 양자화 적용 모델도 양자화를 고려한 전이 학습에 사용될 수 있습니다. 이때도 위에서 사용한 `quant` 와 `dequant` 를 똑같이 사용합니다. 전체 예제는 `여기 <https://tutorials.pytorch.kr/intermediate/quantized_transfer_learning_tutorial.html#part-1-training-a-custom-classifier-based-on-a-quantized-feature-extractor>`_ 를 확인하세요.

위의 단계 중 하나를 이용해 양자화된 모델이 생성된 후에, 모바일 장치에서 작동되게 하려면 추가로 `TorchScript` 형식으로 전환하고 모바일 app에 최적화를 진행해야 합니다. 자세한 내용은 `Script and Optimize for Mobile recipe <script_optimized.html>`_ 를 확인하세요.

더 알아보기
----------

다른 양자화 적용법에 대한 추가 정보는 `여기 <https://pytorch.org/docs/stable/quantization.html#quantization-workflows>`_ 와 `여기 <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#post-training-static-quantization>`_ 를 참고하세요.
