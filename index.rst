파이토치(PyTorch) 튜토리얼에 오신 것을 환영합니다
===================================================

PyTorch를 어떻게 사용하는지 알고 싶다면 시작하기(Getting Started) 튜토리얼부터 시작해보세요.
:doc:`/beginner/deep_learning_60min_blitz` 가 가장 일반적인 출발점으로, 심층
신경망(deep neural network)을 구축할 때 PyTorch를 어떻게 사용하는지에 대한
전반적인 내용을 기본부터 제공합니다.

아래 내용도 한 번 살펴보시면 좋습니다:

* 사용자들이 튜토리얼과 연관된 노트북을 Google Colab에서 열어볼 수 있는 기능이 추가되었습니다.
  `이 페이지 <https://tutorials.pytorch.kr/beginner/colab.html>`_ 을 방문하셔서 자세히 알아보세요!
* 각 튜토리얼에는 Jupyter 노트북과 Python 소스코드를 다운로드 링크가 있습니다.
  IPython 또는 Jupyter를 이용하여 대화식으로 실행해보시려면 이용해보세요.
* 이미지 분류, 비지도 학습, 강화 학습, 기계 번역을 비롯한 다양한 고품질의 예제가
  `PyTorch Examples <https://github.com/pytorch/examples/>`_ 에 준비되어 있습니다.
* PyTorch API, 계층(layer)에 대한 참고 문서는 `PyTorch Docs <https://pytorch.org/docs>`_ 를
  참고해주세요.
* 튜토리얼 개선에 참여하시려면 `이곳 <https://github.com/pytorch/tutorials>`_ 에 의견과 함께
  이슈를 남겨주세요.
  (역자 주: 한국어 번역과 관련한 이슈는
  `한국어 번역 저장소 <https://github.com/9bow/PyTorch-tutorials-kr>`_ 에 부탁드립니다.)
* `PyTorch 치트 시트(Cheat Sheet) <https://tutorials.pytorch.kr/beginner/ptcheat.html>`_ 를
  참고하시면 유용한 정보들을 얻으실 수 있습니다.
* 마지막으로 `PyTorch 릴리즈 노트(Release Notes) <https://github.com/pytorch/pytorch/releases>`_
  도 참고해보세요.

시작하기 (Getting Started)
---------------------------

.. customgalleryitem::
   :figure: /_static/img/thumbnails/pytorch-logo-flat.png
   :tooltip: Understand PyTorch’s Tensor library and neural networks at a high level
   :description: :doc:`/beginner/deep_learning_60min_blitz`

.. customgalleryitem::
   :figure: /_static/img/thumbnails/landmarked_face2.png
   :tooltip: Learn how to load and preprocess/augment data from a non trivial dataset
   :description: :doc:`/beginner/data_loading_tutorial`

.. customgalleryitem::
   :figure: /_static/img/thumbnails/pytorch_tensorboard.png
   :tooltip: Learn to use TensorBoard to visualize data and model training
   :description: :doc:`intermediate/tensorboard_tutorial`

.. raw:: html

    <div style='clear:both'></div>


이미지 (Image)
----------------------

.. customgalleryitem::
   :figure: /_static/img/thumbnails/tv-img.png
   :tooltip: Finetuning a pre-trained Mask R-CNN model
   :description: :doc:`intermediate/torchvision_tutorial`

.. customgalleryitem::
   :figure: /_static/img/thumbnails/sphx_glr_transfer_learning_tutorial_001.png
   :tooltip: In transfer learning, a model created from one task is used in another
   :description: :doc:`beginner/transfer_learning_tutorial`

.. customgalleryitem::
   :figure: /_static/img/stn/Five.gif
   :tooltip: Learn how to augment your network using a visual attention mechanism called spatial transformer networks
   :description: :doc:`intermediate/spatial_transformer_tutorial`

.. customgalleryitem::
   :figure: /_static/img/neural-style/sphx_glr_neural_style_tutorial_004.png
   :tooltip: How to implement the Neural-Style algorithm developed by Gatys, Ecker, and Bethge
   :description: :doc:`advanced/neural_style_tutorial`

.. customgalleryitem::
   :figure: /_static/img/panda.png
   :tooltip: Raise your awareness to the security vulnerabilities of ML models, and get insight into the hot topic of adversarial machine learning
   :description: :doc:`beginner/fgsm_tutorial`

.. customgalleryitem::
   :tooltip: Train a generative adversarial network (GAN) to generate new celebrities
   :figure: /_static/img/dcgan_generator.png
   :description: :doc:`beginner/dcgan_faces_tutorial`

.. raw:: html

    <div style='clear:both'></div>

Named Tensor (experimental)
----------------------

.. customgalleryitem::
   :figure: /_static/img/named_tensor.png
   :tooltip: Named Tensor
   :description: :doc:`intermediate/named_tensor_tutorial`

.. raw:: html

    <div style='clear:both'></div>

오디오 (Audio)
----------------------

.. customgalleryitem::
   :figure: /_static/img/audio_preprocessing_tutorial_waveform.png
   :tooltip: Preprocessing with torchaudio Tutorial
   :description: :doc:`beginner/audio_preprocessing_tutorial`

.. raw:: html

    <div style='clear:both'></div>


텍스트 (Text)
----------------------

.. customgalleryitem::
   :figure: /_static/img/rnnclass.png
   :tooltip: Build and train a basic character-level RNN to classify words
   :description: :doc:`intermediate/char_rnn_classification_tutorial`

.. customgalleryitem::
   :figure: /_static/img/char_rnn_generation.png
   :tooltip: Generate names from languages
   :description: :doc:`intermediate/char_rnn_generation_tutorial`

.. galleryitem:: intermediate/seq2seq_translation_tutorial.py
  :figure: _static/img/seq2seq_flat.png

.. customgalleryitem::
    :tooltip: Sentiment Ngrams with Torchtext
    :figure: /_static/img/text_sentiment_ngrams_model.png
    :description: :doc:`/beginner/text_sentiment_ngrams_tutorial`

.. customgalleryitem::
    :tooltip: Language Translation with Torchtext
    :figure: /_static/img/thumbnails/german_to_english_translation.png
    :description: :doc:`/beginner/torchtext_translation_tutorial`

.. customgalleryitem::
    :tooltip: Transformer Tutorial
    :figure: /_static/img/transformer_architecture.jpg
    :description: :doc:`/beginner/transformer_tutorial`

.. raw:: html

    <div style='clear:both'></div>


강화 학습 (Reinforcement Learning)
------------------------------------------------------------

.. customgalleryitem::
    :tooltip: Use PyTorch to train a Deep Q Learning (DQN) agent
    :figure: /_static/img/cartpole.gif
    :description: :doc:`intermediate/reinforcement_q_learning`

.. raw:: html

    <div style='clear:both'></div>

PyTorch 모델을 운영환경(Production)에 배포하기
------------------------------------------------------------

.. customgalleryitem::
   :tooltip: Deploying PyTorch and Building a REST API using Flask
   :description: :doc:`/intermediate/flask_rest_api_tutorial`
   :figure: _static/img/flask.png

.. customgalleryitem::
   :tooltip: Introduction to TorchScript
   :description: :doc:`beginner/Intro_to_TorchScript_tutorial`
   :figure: _static/img/torchscript.png

.. customgalleryitem::
   :tooltip: Loading a PyTorch model in C++
   :description: :doc:`advanced/cpp_export`
   :figure: _static/img/torchscript_to_cpp.png

.. customgalleryitem::
   :figure: /_static/img/cat.jpg
   :tooltip: Exporting a Model from PyTorch to ONNX and Running it using ONNXRuntime
   :description: :doc:`advanced/super_resolution_with_onnxruntime`

.. raw:: html

    <div style='clear:both'></div>

병렬 & 분산 학습 (Parallel and Distributed Training)
------------------------------------------------------------

.. customgalleryitem::
  :tooltip: Model parallel training on multiple GPUs
  :description: :doc:`/intermediate/model_parallel_tutorial`
  :figure: _static/img/distributed/DistPyTorch.jpg

.. customgalleryitem::
  :tooltip: Getting started with DistributedDataParallel
  :description: :doc:`/intermediate/ddp_tutorial`
  :figure: _static/img/distributed/DistPyTorch.jpg

.. customgalleryitem::
   :tooltip: Parallelize computations across processes and clusters of machines
   :description: :doc:`/intermediate/dist_tuto`
   :figure: _static/img/distributed/DistPyTorch.jpg

.. customgalleryitem::
   :tooltip: PyTorch distributed trainer with Amazon AWS
   :description: :doc:`/beginner/aws_distributed_training_tutorial`
   :figure: _static/img/distributed/DistPyTorch.jpg

.. raw:: html

    <div style='clear:both'></div>

PyTorch 확장하기
------------------------------------------------------------

.. customgalleryitem::
   :tooltip: Implement custom operators in C++ or CUDA for TorchScript
   :description: :doc:`/advanced/torch_script_custom_ops`
   :figure: _static/img/cpp_logo.png

.. customgalleryitem::
    :tooltip: Create extensions using numpy and scipy
    :figure: /_static/img/scipynumpy.png
    :description: :doc:`advanced/numpy_extensions_tutorial`

.. customgalleryitem::
   :tooltip: Implement custom extensions in C++ or CUDA for eager PyTorch
   :description: :doc:`/advanced/cpp_extension`
   :figure: _static/img/cpp_logo.png

.. raw:: html

    <div style='clear:both'></div>

Quantization (experimental)
---------------------------

.. customgalleryitem::
   :tooltip: Perform dynamic quantization on a pre-trained PyTorch model
   :description: :doc:`/advanced/dynamic_quantization_tutorial`
   :figure: _static/img/quant_asym.png

.. customgalleryitem::
    :tooltip: (experimental) Static Quantization with Eager Mode in PyTorch
    :figure: /_static/img/qat.png
    :description: :doc:`advanced/static_quantization_tutorial`

.. raw:: html

    <div style='clear:both'></div>


다른 언어에서의 PyTorch (PyTorch in Other Languages)
------------------------------------------------------------

.. customgalleryitem::
    :tooltip: Using the PyTorch C++ Frontend
    :figure: /_static/img/cpp-pytorch.png
    :description: :doc:`advanced/cpp_frontend`

.. raw:: html

    <div style='clear:both'></div>

PyTorch Fundamentals In-Depth
-----------------------------

.. customgalleryitem::
   :tooltip: This tutorial introduces the fundamental concepts of PyTorch through self-contained examples
   :figure: /_static/img/thumbnails/examples.png
   :description: :doc:`/beginner/pytorch_with_examples`

.. customgalleryitem::
   :figure: /_static/img/torch.nn.png
   :tooltip: Use torch.nn to create and train a neural network
   :description: :doc:`beginner/nn_tutorial`

.. raw:: html

    <div style='clear:both'></div>


.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: 시작하기 (Getting Started)

   beginner/deep_learning_60min_blitz
   beginner/data_loading_tutorial
   intermediate/tensorboard_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 이미지 (Image)

   intermediate/torchvision_tutorial
   beginner/transfer_learning_tutorial
   intermediate/spatial_transformer_tutorial
   advanced/neural_style_tutorial
   beginner/fgsm_tutorial
   advanced/dcgan_faces_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 오디오 (Audio)

   beginner/audio_preprocessing_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 텍스트 (Text)

   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   beginner/text_sentiment_ngrams_tutorial
   beginner/torchtext_translation_tutorial
   beginner/transformer_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Named Tensor (experimental)

   intermediate/named_tensor_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 강화 학습

   intermediate/reinforcement_q_learning

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: PyTorch 모델을 운영환경에 배포하기

   intermediate/flask_rest_api_tutorial
   beginner/Intro_to_TorchScript_tutorial
   advanced/cpp_export
   advanced/super_resolution_with_onnxruntime

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 병렬 & 분산 학습

   intermediate/model_parallel_tutorial
   intermediate/ddp_tutorial
   intermediate/dist_tuto
   beginner/aws_distributed_training_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: PyTorch 확장하기

   advanced/torch_script_custom_ops
   advanced/numpy_extensions_tutorial
   advanced/cpp_extension

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Quantization (experimental)

   advanced/dynamic_quantization_tutorial
   advanced/static_quantization_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 다른 언어에서의 PyTorch

   advanced/cpp_frontend

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: PyTorch Fundamentals In-Depth

   beginner/pytorch_with_examples
   beginner/nn_tutorial