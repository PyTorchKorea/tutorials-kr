
ONNX Live 튜토리얼
==================

이 튜토리얼은 PyTorch에서 나온 neural style transfer 모델을 Apple CoreML 형식으로 ONNX을 이용하여 변환하는 방법을 소개합니다.
또한, 이 튜토리얼은 어떻게 Apple 장치들에 딥 러닝 모델들을 사용하는지 알려주고 특히나 이번 튜토리얼에서는 카메라로부터 실시간 스트림을 이용합니다.

ONNX란 무엇인가?
-------------

ONNX (Open Neural Network Exchange) 는 딥 러닝 모델들을 대표하는 공개되어있는 형식입니다.
ONNX는 AI 개발자들이 최첨단의 도구들과 모델들의 조합을 그들에게 가장 잘 맞는 것을 쉽게 선택하도록 해줍니다.
ONNX는 커뮤니티의 파트너들이 개발하며 또한 지원됩니다.
ONNX와 어떤 도구들이 지원되는지는 `onnx.ai <https://onnx.ai/>`_ 를 방문하여 더 알아볼 수 있습니다.


튜토리얼 개요
-----------------

이번 튜토리얼에서는 4가지 중심 사항들을 소개해드리겠습니다:


#. `Download (or train) PyTorch style transfer models`_
#. `PyTorch style transfer 모델 다운로드 (또는 학습시키기)`_
#. `PyTorch 모델을 ONNX 모델로 변환하기`_
#. `ONNX 모델을 CoreML 모델로 변환하기`_
#. `style transfer iOS App 에서 CoreML 모델 돌리기`_

개발 환경 준비하기
-------------------------

현재 로컬에 설치되어있는 패키지들과의 충돌을 피하기 위해 virtualenv를 사용하겠습니다.
튜토리얼에서 Python 3.6 버전을 사용하지만 다른 버전도 작동하는데 무리가 없습니다.

.. code-block:: python

   python3.6 -m venv venv
   source ./venv/bin/activate


PyTorch와 onnx-coreml 변환해주는 라이브러리를 설치해야합니다.

.. code-block:: bash

   pip install torchvision onnx-coreml


iPhone 에서 iOS style transfer 어플리케이션을 실행하려면 XCode 가 필요합니다.
Linux 에서도 모델을 변환할 수 있지만, iOS 어플리케이션을 실행 하려면 Mac 이 필요합니다.

PyTorch style transfer 모델 다운로드 (또는 학습시키기)
-------------------------------------------------

이번 튜토리얼에서 우리는 https://github.com/pytorch/examples/tree/master/fast_neural_style 에  PyTorch 로 만들어진 style transfer 을 이용하겠습니다.
다른 PyTorch 나 ONNX 모델을 사용하길 원한다면 이번 단계를 생략해도 좋습니다.

이 모델들은 이미지에 style transfer 가 적용되도록 만들어져서 비디오에 적용하기에는 최적화가 되어있지 않아 충분히 빠르지 않습니다.
하지만 해상도를 어느 정도 낮춘다면 비디오에서 충분한 속도에 작동됩니다.


모델들을 다운받아봅시다.

.. code-block:: bash

   git clone https://github.com/pytorch/examples
   cd examples/fast_neural_style


직접 모델을 훈련시키기 원한다면 방금 클론한 저장소의 pytorch/examples 를 방문하면 어떻게 해야하는지에 대한 더 많은 정보가 있습니다.
지금은 저장소에서 제공해주는 스크립트를 사용하여 미리 훈련된 모델을 다운로드하겠습니다:

.. code-block:: bash

   python download_saved_models.py


이 스크립트는 미리 훈련된 PyTorch 모델들을 다운받고 ``saved_models`` 폴더에 저장합니다.
폴더에는 4가지 파일 ``candy.pth``\ , ``mosaic.pth``\ , ``rain_princess.pth`` 그리고 ``udnie.pth`` 가 폴더에 다운 받아졌습니다.

PyTorch 모델을 ONNX 모델로 변환하기
-----------------------------------------

이제 우리는 미리 훈련된 PyTorch 모델들을  ``saved_models`` 폴더에 ``.pth`` 파일로 가지고 있어서 나중에 저 모델들을 ONNX 형식으로 변환 할 수 있습니다.
모델 정의는 방금 전 클론 했던 저장소의 pytorch/examples 에 있으며 단 몇 줄의 파이썬 코드를 이용해서 ONNX 로 변환 할 수 있습니다.
인공 신경망을 실제로 실행하는 대신 이러한 경우에는 우리는 ``torch.onnx._export``\ 를 호출하는데 이것은 PyTorch 가 제공하는 API 로 PyTorch 에서 직접적으로 ONNX 형식의 모델로 변환합니다.
하지만 이번 경우에는 우리는 저것을 호출 안해도 되는데 그 이유는 저런 기능을 우리를 위해 대신해주는 스크립트가 ``neural_style/neural_style.py`` 에 있기 때문입니다.
You can also take a look at that script if you would like to apply it to other models.
다른 모델에 저 스크립트를 적용하길 원하시면 스크립트를 보는것도 좋습니다.

PyTorch 에서 ONNX 형식으로 변환하는것은 결과적으로 당신의 신경망을 추적해 나가는것과 같습니다. 이 API 호출은 내부적으로 그래프를 생성해내기 위해 '더미 데이터' 를 이용하여 신경망을 실행합니다.
이렇기 때문에 입력값으로 style transfer 를 적용할 이미지가 입력값으로 필요하며 기본적인 비어있는 이미지여도 됩니다.
그러나 이미지의 픽셀 크기는 중요한데 그 이유는 픽셀 크기는 변환될 style transfer 모델의 크기로도 이용되기 때문입니다.
좋은 퍼포먼스를 얻기 위해 우리는 250x540 의 해상도를 사용할 예정입니다. FPS보다 style transfer의 품질에 더 많이 신경 쓰신다면 더 큰 해상도를 사용하셔도 됩니다.

`ImageMagick <https://www.imagemagick.org/>`_ 을 사용하여 우리가 원하는 크기의 비어있는 이미지를 생성해봅시다.
.. code-block:: bash

   convert -size 250x540 xc:white png24:dummy.jpg


그리고 PyTorch 모델을 변환하는데 사용해봅니다.

.. code-block:: bash

   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/candy.pth --cuda 0 --export_onnx ./saved_models/candy.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/udnie.pth --cuda 0 --export_onnx ./saved_models/udnie.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/rain_princess.pth --cuda 0 --export_onnx ./saved_models/rain_princess.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/mosaic.pth --cuda 0 --export_onnx ./saved_models/mosaic.onnx


위의 과정을 거치면 4개의 파일 ``candy.onnx``\ , ``mosaic.onnx``\ , ``rain_princess.onnx`` 그리고 ``udnie.onnx``\ ,
가 ``.pth`` 파일들에 각각 대응되어 생성되어있어야 합니다.

ONNX 모델을 CoreML 모델로 변환하기
----------------------------------------

이제 우리는 ONNX 모델들이 있기 때문에 Apple 장치에서 실행시키기 위해 모델들을 CoreML 모델로 변환시킬 수 있습니다.
이렇게 하기 위해, 우리는 전에 설치한 onnx-coreml 컨버터를 사용합니다.
이 컨버터는 위의 설치 단계에서 추가된 ``convert-onnx-to-coreml`` 스크립트를 포함하고 있습니다.
아쉽게도 이 스크립트를 우리가 원하는 기능인 신경망의 입력값과 출력값을 이미지로서 표시해주지 못하고
우리가 원하는 기능은 파이썬에서 컨버터를 호출해야만 지원됩니다.

style transfer 모델을 보면 (예를 들면 .onnx 파일을 `Netron <https://github.com/lutzroeder/Netron>`_\ 과 같은 어플리케이션에서 여는것)
우리는 입력값이 '0' 으로 되어있고 출력값이 '186' 으로 되어있는걸 확인할 수 있습니다. 이것들은 PyTorch 에 의해 할당된 숫자 id 들 입니다.
우리는 이러한 값들을 이미지로써 표시해야합니다.

자 그래서 이제 ``onnx_to_coreml.py`` 라고 하는 작은 파이썬 파일을 생성합니다. 이것은 touch 명령어나 가장 좋아하는 편집기로 다음의 몇줄의 코드만 추가하면 만들수 있습니다.

.. code-block:: python

   import sys
   from onnx import onnx_pb
   from onnx_coreml import convert

   model_in = sys.argv[1]
   model_out = sys.argv[2]

   model_file = open(model_in, 'rb')
   model_proto = onnx_pb.ModelProto()
   model_proto.ParseFromString(model_file.read())
   coreml_model = convert(model_proto, image_input_names=['0'], image_output_names=['186'])
   coreml_model.save(model_out)


이제 실행시켜봅시다.

.. code-block:: bash

   python onnx_to_coreml.py ./saved_models/candy.onnx ./saved_models/candy.mlmodel
   python onnx_to_coreml.py ./saved_models/udnie.onnx ./saved_models/udnie.mlmodel
   python onnx_to_coreml.py ./saved_models/rain_princess.onnx ./saved_models/rain_princess.mlmodel
   python onnx_to_coreml.py ./saved_models/mosaic.onnx ./saved_models/mosaic.mlmodel


이제 ``saved_models`` 폴더에 CoreML 모델이 4개 , ``candy.mlmodel``\ , ``mosaic.mlmodel``\ , ``rain_princess.mlmodel`` and ``udnie.mlmodel``, 가 있습니다.

style transfer iOS앱에서 CoreML 모델들 실행하기
-------------------------------------------------

이 저장소(README.md를 읽고 계시는 현재의 저장소)는 CoreML style transfer 모델들을 핸드폰 카메라로 실시간 카메라 스트림을 이용하여 실행할 수 있는 iOS 앱을 포함하고 있습니다. 이제 저장소를 클론해봅시다.

.. code-block:: bash

   git clone https://github.com/onnx/tutorials


그리고 ``tutorials/examples/CoreML/ONNXLive/ONNXLive.xcodeproj`` 프로젝트를 XCode 에서 열어봅시다.
우리는 XCode 9.3 버전과 iPhone X를 사용하는것을 추천합니다. 오래된 기기나 오래된 버전의 XCode 를 사용해 실행하는데는 이슈가 있을 수도 있습니다.

``Models/`` 폴더에 프로젝트는 .mlmodel 파일들이 포함되어 있습니다. 우리는 저 파일들을 우리가 방금 생성해낸 모델들로 교체해야합니다.

그리고 iPhone을 이용해서 앱을 실행시키면 모든 준비가 끝났습니다. 스크린을 탭하면서 모델들을 바꿔보도록 합시다.

결론
----------

우리는 이 튜토리얼이 ONNX 가 무엇이고 어떻게 신경망들을 프레임워크 사이에서 변환하면서 사용하는지 충분히 설명이 되었기를 바랍니다.
이번에는 PyTorch에서 CoreML 로 style transfer 모델들을 변환하였습니다.

앞서 소개해드린 단계들을 활용하며 자유롭게 자신만의 모델을 실험해 보시기를 바랍니다.
피드백이나 오류가 있다면 언제든지 알려주세요.
