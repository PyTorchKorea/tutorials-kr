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

아래 튜토리얼을 실행하기 전에, `torch`_ 와 `torchvision`_ 패키지가 설치되어 있는지 확인하세요.

.. _torch: https://github.com/pytorch/pytorch
.. _torchvision: https://github.com/pytorch/vision


.. toctree::
   :hidden:

   /beginner/blitz/tensor_tutorial
   /beginner/blitz/autograd_tutorial
   /beginner/blitz/neural_networks_tutorial
   /beginner/blitz/cifar10_tutorial

.. grid:: 4

   .. grid-item-card::  :octicon:`file-code;1em` Tensors
      :link: blitz/tensor_tutorial.html

      In this tutorial, you will learn the basics of PyTorch tensors.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` A Gentle Introduction to torch.autograd
      :link: blitz/autograd_tutorial.html

      Learn about autograd.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` Neural Networks
      :link: blitz/neural_networks_tutorial.html

      This tutorial demonstrates how you can train neural networks in PyTorch.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` Training a Classifier
      :link: blitz/cifar10_tutorial.html

      Learn how to train an image classifier in PyTorch by using the
      CIFAR10 dataset.
      +++
      :octicon:`code;1em` Code
