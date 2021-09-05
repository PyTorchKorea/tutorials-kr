PyTorch로 딥러닝하기: 60분만에 끝장내기
---------------------------------------------------
**Author**: `Soumith Chintala <http://soumith.ch>`_
  **번역**: `박정환 <http://github.com/9bow>`_

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/u7x8RXwLKcA" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

파이토치(PyTorch)가 무엇인가요?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyTorch는 Python 기반의 과학 연산 패키지로 다음 두 가지 목적으로 제공됩니다:

- GPU 및 다른 가속기의 성능을 사용하기 위한 NumPy의 대체제 제공
- 신경망 구현에 유용한 자동 미분(automatic differntiation) 라이브러리 제공

이 튜토리얼의 목표
~~~~~~~~~~~~~~~~~~~~~~~~
-  높은 수준에서 PyTorch의 Tensor library와 신경망(Neural Network)를 이해합니다.
-  이미지를 분류하는 작은 신경망을 학습시킵니다.

.. Note::
    `torch`_ 와 `torchvision`_ 패키지를 설치했는지 확인하십시오.

.. _torch: https://github.com/pytorch/pytorch
.. _torchvision: https://github.com/pytorch/vision


.. toctree::
   :hidden:

   /beginner/blitz/tensor_tutorial
   /beginner/blitz/autograd_tutorial
   /beginner/blitz/neural_networks_tutorial
   /beginner/blitz/cifar10_tutorial

.. galleryitem:: /beginner/blitz/tensor_tutorial.py
    :figure: /_static/img/tensor_illustration_flat.png

.. galleryitem:: /beginner/blitz/autograd_tutorial.py
    :figure: /_static/img/autodiff.png

.. galleryitem:: /beginner/blitz/neural_networks_tutorial.py
    :figure: /_static/img/mnist.png

.. galleryitem:: /beginner/blitz/cifar10_tutorial.py
    :figure: /_static/img/cifar10.png


.. raw:: html

    <div style='clear:both'></div>
