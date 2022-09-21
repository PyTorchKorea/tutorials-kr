"""
**파이토치(PyTorch) 기본 익히기** ||
`빠른 시작 <quickstart_tutorial.html>`_ ||
`텐서(Tensor) <tensorqs_tutorial.html>`_ ||
`Dataset과 Dataloader <data_tutorial.html>`_ ||
`변형(Transform) <transforms_tutorial.html>`_ ||
`신경망 모델 구성하기 <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`최적화(Optimization) <optimization_tutorial.html>`_ ||
`모델 저장하고 불러오기 <saveloadrun_tutorial.html>`_

파이토치(PyTorch) 기본 익히기
==========================================================================

Authors:
`Suraj Subramanian <https://github.com/suraj813>`_,
`Seth Juarez <https://github.com/sethjuarez/>`_,
`Cassie Breviu <https://github.com/cassieview/>`_,
`Dmitry Soshnikov <https://soshnikov.com/>`_,
`Ari Bornstein <https://github.com/aribornstein/>`_

번역:
`박정환 <https://github.com/9bow>`_

대부분의 머신러닝 워크플로우는 데이터 작업과 모델 생성, 모델 매개변수 최적화, 학습된 모델 저장이 포함됩니다.
이 튜토리얼에서는 이러한 개념들에 대해 더 자세히 알아볼 수 있는 바로가기와 함께 PyTorch로 구현된 전체 ML 워크플로우를 소개합니다.

FashionMNIST 데이터셋을 사용하여 입력 이미지가 다음 분류(class) 중 하나에 속하는지를 예측하는 신경망을 학습합니다:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot.

`이 튜토리얼은 Python과 딥러닝 개념에 대해 기본적인 지식이 있다고 가정합니다.`


튜토리얼 코드 실행하기
------------------------------------------------------------------------------------------

다음의 두 가지 방법으로 이 튜토리얼을 실행해볼 수 있습니다:

- **클라우드**: 시작하기 가장 쉬운 방법입니다! 각 섹션의 맨 위에는 "Run in Microsoft Learn" 링크가 있으며, 이 링크는 완전히 호스팅되는 환경에서 Microsoft Learn의 노트북을 엽니다.
- **로컬**: 먼저 로컬 컴퓨터에 PyTorch와 TorchVision을 설치해야 합니다 (`설치 방법 <https://pytorch.kr/get-started/locally/>`_). 노트북을 내려받거나 코드를 원하는 IDE에 복사하세요.


튜토리얼 사용 방법
------------------------------------------------------------------------------------------
다른 딥러닝 프레임워크에 익숙하다면, `0. 빠른 시작 <quickstart_tutorial.html>`_ 을 보고 PyTorch의 API들을 빠르게 익히세요.

딥러닝 프레임워크가 처음이라면, 단계별(step-by-step) 가이드의 첫번째인 `1. 텐서(Tensor) <tensorqs_tutorial.html>`_ 로 이동하세요.


.. include:: /beginner_source/basics/qs_toc.txt

.. toctree::
   :hidden:

"""
