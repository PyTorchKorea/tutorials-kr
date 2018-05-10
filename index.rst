PyTorch 튜토리얼에 오신 것을 환영합니다
========================================

PyTorch 학습을 시작하시려면 초급(Beginner) 튜토리얼로 시작하세요.
일반적으로 :doc:`/beginner/deep_learning_60min_blitz` 부터 시작하시면 PyTorch의
개요를 빠르게 학습할 수 있습니다. 예제를 보고 학습하는걸 좋아하신다면
:doc:`/beginner/pytorch_with_examples` 을 추천합니다.

튜토리얼을 IPython / Jupyter를 이용하여 대화식으로(interactively) 진행하길 원한다면,
각각의 튜토리얼의 Jupyter 노트북과 Python 소스코드를 다운로드받으실 수 있습니다.

또한, 이미지 분류, 비지도 학습(unsupervised learning), 강화학습(reinforcement learning),
기계 번역과 같은 다양한 고품질의 예제들을 https://github.com/pytorch/examples/ 에서
제공하고 있습니다.

PyTorch의 API와 계층(Layer)에 대한 참고 문서는 http://docs.pytorch.org 이나
인라인(inline) 도움말을 참고해주세요. 만약 튜토리얼을 개선하고 싶으시다면,
GitHub 이슈를 통해 의견을 주시기 바랍니다: https://github.com/pytorch/tutorials

(역자 주: 한국어 번역에 대한 오타나 오역을 발견하시면
`번역 저장소 <https://github.com/9bow/PyTorch-tutorials-kr>`__ 에
`이슈 <https://github.com/9bow/PyTorch-tutorials-kr/issues/new>`__ 또는
`PR <https://github.com/9bow/PyTorch-tutorials-kr/pulls>`__ 을 남겨주세요.)

초급 튜토리얼
------------------

.. customgalleryitem::
   :figure: /_static/img/thumbnails/pytorch-logo-flat.png
   :tooltip: Understand PyTorch’s Tensor library and neural networks at a high level.
   :description: :doc:`/beginner/deep_learning_60min_blitz`

.. customgalleryitem::
   :tooltip: Understand similarities and differences between torch and pytorch.
   :figure: /_static/img/thumbnails/torch-logo.png
   :description: :doc:`/beginner/former_torchies_tutorial`

.. customgalleryitem::
   :tooltip: This tutorial introduces the fundamental concepts of PyTorch through self-contained examples.
   :figure: /_static/img/thumbnails/examples.png
   :description: :doc:`/beginner/pytorch_with_examples`

.. galleryitem:: beginner/transfer_learning_tutorial.py

.. galleryitem:: beginner/data_loading_tutorial.py

.. customgalleryitem::
    :tooltip: I am writing this tutorial to focus specifically on NLP for people who have never written code in any deep learning framework
    :figure: /_static/img/thumbnails/babel.jpg
    :description: :doc:`/beginner/deep_learning_nlp_tutorial`

.. raw:: html

    <div style='clear:both'></div>


.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: Beginner Tutorials

   beginner/deep_learning_60min_blitz
   beginner/former_torchies_tutorial
   beginner/pytorch_with_examples
   beginner/transfer_learning_tutorial
   beginner/data_loading_tutorial
   beginner/deep_learning_nlp_tutorial

중급 튜토리얼
----------------------

.. galleryitem:: intermediate/char_rnn_classification_tutorial.py

.. galleryitem:: intermediate/char_rnn_generation_tutorial.py
  :figure: _static/img/char_rnn_generation.png

.. galleryitem:: intermediate/seq2seq_translation_tutorial.py
  :figure: _static/img/seq2seq_flat.png

.. galleryitem:: intermediate/reinforcement_q_learning.py
    :figure: _static/img/cartpole.gif

.. customgalleryitem::
   :tooltip: Writing Distributed Applications with PyTorch.
   :description: :doc:`/intermediate/dist_tuto`
   :figure: _static/img/distributed/DistPyTorch.jpg


.. galleryitem:: intermediate/spatial_transformer_tutorial.py


.. raw:: html

    <div style='clear:both'></div>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Intermediate Tutorials

   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   intermediate/reinforcement_q_learning
   intermediate/dist_tuto
   intermediate/spatial_transformer_tutorial


고급 튜토리얼
------------------

.. galleryitem:: advanced/neural_style_tutorial.py
    :intro: This tutorial explains how to implement the Neural-Style algorithm developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.

.. galleryitem:: advanced/numpy_extensions_tutorial.py

.. galleryitem:: advanced/super_resolution_with_caffe2.py

.. customgalleryitem::
   :tooltip: Implement custom extensions in C++ or CUDA.
   :description: :doc:`/advanced/cpp_extension`
   :figure: _static/img/cpp_logo.png


.. raw:: html

    <div style='clear:both'></div>


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Advanced Tutorials

   advanced/neural_style_tutorial
   advanced/numpy_extensions_tutorial
   advanced/super_resolution_with_caffe2
   advanced/cpp_extension
