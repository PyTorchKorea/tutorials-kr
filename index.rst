파이토치(PyTorch) 튜토리얼에 오신 것을 환영합니다
==================================================

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: 60분 만에 끝장내기는 PyTorch를 어떻게 사용하는지 대략적으로 알아볼 수 있는 일반적인 시작점입니다. 심층신경망 모델 구성에 대한 기본적인 내용을 다룹니다.
   :header: PyTorch가 처음이신가요?
   :button_link: beginner/deep_learning_60min_blitz.html
   :button_text: 60분 만에 끝장내기 시작

.. customcalloutitem::
   :description: 한 입 크기의, 바로 사용할 수 있는 PyTorch 코드 예제들을 확인해보세요.
   :header: 파이토치(PyTorch) 레시피
   :button_link: recipes/recipes_index.html
   :button_text: 레시피 찾아보기

.. End of callout item section

.. raw:: html

        </div>
    </div>

    <div id="tutorial-cards-container">

    <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">
        <div class="tutorial-tags-container">
            <div id="dropdown-filter-tags">
                <div class="tutorial-filter-menu">
                    <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">All</div>
                </div>
            </div>
        </div>
    </nav>

    <hr class="tutorials-hr">

    <div class="row">

    <div id="tutorial-cards">
    <div class="list">

.. Add tutorial cards below this line

.. Learning PyTorch

.. customcarditem::
   :header: 파이토치(PyTorch)로 딥러닝하기: 60분만에 끝장내기
   :card_description: 높은 수준에서 PyTorch의 텐서 라이브러리와 신경망을 이해합니다.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: beginner/deep_learning_60min_blitz.html
   :tags: Getting-Started

.. customcarditem::
   :header: 예제로 배우는 파이토치(PyTorch)
   :card_description: 튜토리얼에 포함된 예제들로 PyTorch의 기본 개념을 이해합니다.
   :image: _static/img/thumbnails/cropped/learning-pytorch-with-examples.png
   :link: beginner/pytorch_with_examples.html
   :tags: Getting-Started

.. customcarditem::
   :header: What is torch.nn really?
   :card_description: Use torch.nn to create and train a neural network.
   :image: _static/img/thumbnails/cropped/torch-nn.png
   :link: beginner/nn_tutorial.html
   :tags: Getting-Started

.. customcarditem::
   :header: TensorBoard로 모델, 데이터, 학습 시각화하기
   :card_description: TensorBoard로 데이터 및 모델 교육을 시각화하는 방법을 배웁니다.
   :image: _static/img/thumbnails/cropped/visualizing-with-tensorboard.png
   :link: intermediate/tensorboard_tutorial.html
   :tags: Interpretability,Getting-Started,Tensorboard

.. Image/Video

.. customcarditem::
   :header: TorchVision 객체 검출 미세조정(Finetuning) 튜토리얼
   :card_description: 이미 훈련된 Mask R-CNN 모델을 미세조정합니다.
   :image: _static/img/thumbnails/cropped/TorchVision-Object-Detection-Finetuning-Tutorial.png
   :link: intermediate/torchvision_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: 컴퓨터 비전을 위한 전이학습(TRANSFER LEARNING) 튜토리얼
   :card_description: 전이학습으로 이미지 분류를 위한 합성곱 신경망을 학습합니다.
   :image: _static/img/thumbnails/cropped/Transfer-Learning-for-Computer-Vision-Tutorial.png
   :link: beginner/transfer_learning_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: 적대적 예제 생성(Adversarial Example Generation)
   :card_description: 가장 많이 사용되는 공격 방법 중 하나인 FGSM (Fast Gradient Sign Attack)을 이용해 MNIST 분류기를 속이는 방법을 배웁니다.
   :image: _static/img/thumbnails/cropped/Adversarial-Example-Generation.png
   :link: beginner/fgsm_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: DCGAN Tutorial
   :card_description: Train a generative adversarial network (GAN) to generate new celebrities.
   :image: _static/img/thumbnails/cropped/DCGAN-Tutorial.png
   :link: beginner/dcgan_faces_tutorial.html
   :tags: Image/Video

.. Audio

.. customcarditem::
   :header: torchaudio Tutorial
   :card_description: Learn to load and preprocess data from a simple dataset with PyTorch's torchaudio library.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_preprocessing_tutorial.html
   :tags: Audio

.. Text

.. customcarditem::
   :header: nn.Transformer 와 TorchText 로 시퀀스-투-시퀀스 모델링하기
   :card_description: nn.Transformer 모듈을 사용하여 어떻게 시퀀스-투-시퀀스(Seq-to-Seq) 모델을 학습하는지 배웁니다.
   :image: _static/img/thumbnails/cropped/Sequence-to-Sequence-Modeling-with-nnTransformer-andTorchText.png
   :link: beginner/transformer_tutorial.html
   :tags: Text

.. customcarditem::
   :header: 기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 분류하기
   :card_description:
   torchtext를 사용하지 않고 기본적인 문자-단위 RNN을 사용하여 단어를 분류하는 모델을 기초부터 만들고 학습합니다. 총 3개로 이뤄진 튜토리얼 시리즈의 첫번째 편입니다.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Classifying-Names-with-a-Character-Level-RNN.png
   :link: intermediate/char_rnn_classification_tutorial
   :tags: Text

.. customcarditem::
   :header: 기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 생성하기
   :card_description: 문자-단위 RNN을 사용하여 이름을 분류해봤으니, 이름을 생성하는 방법을 학습합니다. 총 3개로 이뤄진 튜토리얼 시리즈 중 두번째 편입니다.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Generating-Names-with-a-Character-Level-RNN.png
   :link: intermediate/char_rnn_generation_tutorial.html
   :tags: Text

.. customcarditem::
   :header: 기초부터 시작하는 NLP: 시퀀스-투-시퀀스 네트워크와 어텐션을 이용한 번역
   :card_description: “기초부터 시작하는 NLP”의 세번째이자 마지막 편으로, NLP 모델링 작업을 위한 데이터 전처리에 사용할 자체 클래스와 함수들을 작성해보겠습니다.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Translation-with-a-Sequence-to-Sequence-Network-and-Attention.png
   :link: intermediate/seq2seq_translation_tutorial.html
   :tags: Text

.. customcarditem::
   :header: Text Classification with Torchtext
   :card_description: This is the third and final tutorial on doing “NLP From Scratch”, where we write our own classes and functions to preprocess the data to do our NLP modeling tasks.
   :image: _static/img/thumbnails/cropped/Text-Classification-with-TorchText.png
   :link: beginner/text_sentiment_ngrams_tutorial.html
   :tags: Text

.. customcarditem::
   :header: TorchText로 언어 번역하기
   :card_description: 영어와 독어가 포함된 잘 알려진 데이터셋을 torchtext를 사용하여 전처리한 뒤, 시퀀스-투-시퀀스(Seq-to-Seq) 모델을 사용하여 학습합니다.
   :image: _static/img/thumbnails/cropped/Language-Translation-with-TorchText.png
   :link: beginner/torchtext_translation_tutorial.html
   :tags: Text

.. Reinforcement Learning

.. customcarditem::
   :header: 강화 학습(DQN) 튜토리얼
   :card_description: PyTorch를 사용하여 OpenAI Gym의 CartPole-v0 태스크에서 DQN(Deep Q Learning) 에이전트를 학습하는 방법을 살펴봅니다.
   :image: _static/img/cartpole.gif
   :link: intermediate/reinforcement_q_learning.html
   :tags: Reinforcement-Learning

.. Deploying PyTorch Models in Production

.. customcarditem::
   :header: Flask를 사용하여 Python에서 PyTorch를 REST API로 배포하기
   :card_description: Flask를 사용하여 PyTorch 모델을 배포하고, 미리 학습된 DenseNet 121 모델을 예제로 활용하여 모델 추론(inference)을 위한 REST API를 만들어보겠습니다.
   :image: _static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png
   :link: intermediate/flask_rest_api_tutorial.html
   :tags: Production

.. customcarditem::
   :header: TorchScript 소개
   :card_description: C++과 같은 고성능 환경에서 실행할 수 있도록 (nn.Module의 하위 클래스인) PyTorch 모델의 중간 표현(intermediate representation)을 제공하는 TorchScript를 소개합니다.
   :image: _static/img/thumbnails/cropped/Introduction-to-TorchScript.png
   :link: beginner/Intro_to_TorchScript_tutorial.html
   :tags: Production

.. customcarditem::
   :header: C++에서 TorchScript 모델 로딩하기
   :card_description: PyTorch가 어떻게 기존의 Python 모델을 직렬화된 표현으로 변환하여 Python 의존성 없이 순수하게 C++에서 불러올 수 있는지 배웁니다.
   :image: _static/img/thumbnails/cropped/Loading-a-TorchScript-Model-in-Cpp.png
   :link: advanced/cpp_export.html
   :tags: Production

.. customcarditem::
   :header: (optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime
   :card_description:  Convert a model defined in PyTorch into the ONNX format and then run it with ONNX Runtime.
   :image: _static/img/thumbnails/cropped/optional-Exporting-a-Model-from-PyTorch-to-ONNX-and-Running-it-using-ONNX-Runtime.png
   :link: advanced/super_resolution_with_onnxruntime.html
   :tags: Production

.. Frontend APIs

.. customcarditem::
   :header: (prototype) Introduction to Named Tensors in PyTorch
   :card_description: Learn how to use PyTorch to train a Deep Q Learning (DQN) agent on the CartPole-v0 task from the OpenAI Gym.
   :image: _static/img/thumbnails/cropped/experimental-Introduction-to-Named-Tensors-in-PyTorch.png
   :link: intermediate/memory_format_tutorial.html
   :tags: Frontend-APIs,Named-Tensor,Best-Practice

.. customcarditem::
   :header: (beta) Channels Last Memory Format in PyTorch
   :card_description: Get an overview of Channels Last memory format and understand how it is used to order NCHW tensors in memory preserving dimensions.
   :image: _static/img/thumbnails/cropped/experimental-Channels-Last-Memory-Format-in-PyTorch.png
   :link: intermediate/memory_format_tutorial.html
   :tags: Memory-Format,Best-Practice

.. customcarditem::
   :header: Using the PyTorch C++ Frontend
   :card_description: Walk through an end-to-end example of training a model with the C++ frontend by training a DCGAN – a kind of generative model – to generate images of MNIST digits.
   :image: _static/img/thumbnails/cropped/Using-the-PyTorch-Cpp-Frontend.png
   :link: advanced/cpp_frontend.html
   :tags: Frontend-APIs,C++

.. customcarditem::
   :header: Custom C++ and CUDA Extensions
   :card_description:  Create a neural network layer with no parameters using numpy. Then use scipy to create a neural network layer that has learnable weights.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/cpp_extension.html
   :tags: Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Operators
   :card_description:  Implement a custom TorchScript operator in C++, how to build it into a shared library, how to use it in Python to define TorchScript models and lastly how to load it into a C++ application for inference workloads.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Operators.png
   :link: advanced/torch_script_custom_ops.html
   :tags: Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Classes
   :card_description: This is a continuation of the custom operator tutorial, and introduces the API we’ve built for binding C++ classes into TorchScript and Python simultaneously.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Classes.png
   :link: advanced/torch_script_custom_classes.html
   :tags: Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Dynamic Parallelism in TorchScript
   :card_description: This tutorial introduces the syntax for doing *dynamic inter-op parallelism* in TorchScript.
   :image: _static/img/thumbnails/cropped/TorchScript-Parallelism.jpg
   :link: advanced/torch-script-parallelism.html
   :tags: Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Autograd in C++ Frontend
   :card_description: The autograd package helps build flexible and dynamic nerural netorks. In this tutorial, exploreseveral examples of doing autograd in PyTorch C++ frontend
   :image: _static/img/thumbnails/cropped/Autograd-in-Cpp-Frontend.png
   :link: advanced/cpp_autograd.html
   :tags: Frontend-APIs,C++

.. Model Optimization

.. customcarditem::
   :header: Pruning Tutorial
   :card_description: Learn how to use torch.nn.utils.prune to sparsify your neural networks, and how to extend it to implement your own custom pruning technique.
   :image: _static/img/thumbnails/cropped/Pruning-Tutorial.png
   :link: intermediate/pruning_tutorial.html
   :tags: Model-Optimization,Best-Practice

.. customcarditem::
   :header: (beta) Dynamic Quantization on an LSTM Word Language Model
   :card_description: Apply dynamic quantization, the easiest form of quantization, to a LSTM-based next word prediction model.
   :image: _static/img/thumbnails/cropped/experimental-Dynamic-Quantization-on-an-LSTM-Word-Language-Model.png
   :link: advanced/dynamic_quantization_tutorial.html
   :tags: Text,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Dynamic Quantization on BERT
   :card_description: Apply the dynamic quantization on a BERT (Bidirectional Embedding Representations from Transformers) model.
   :image: _static/img/thumbnails/cropped/experimental-Dynamic-Quantization-on-BERT.png
   :link: intermediate/dynamic_quantization_bert_tutorial.html
   :tags: Text,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Static Quantization with Eager Mode in PyTorch
   :card_description: Learn techniques to impove a model's accuracy =  post-training static quantization, per-channel quantization, and quantization-aware training.
   :image: _static/img/thumbnails/cropped/experimental-Static-Quantization-with-Eager-Mode-in-PyTorch.png
   :link: advanced/static_quantization_tutorial.html
   :tags: Image/Video,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Quantized Transfer Learning for Computer Vision Tutorial
   :card_description: Learn techniques to impove a model's accuracy -  post-training static quantization, per-channel quantization, and quantization-aware training.
   :image: _static/img/thumbnails/cropped/experimental-Quantized-Transfer-Learning-for-Computer-Vision-Tutorial.png
   :link: advanced/static_quantization_tutorial.html
   :tags: Image/Video,Quantization,Model-Optimization

.. Parallel-and-Distributed-Training

.. customcarditem::
   :header: PyTorch Distributed Overview
   :card_description: Briefly go over all concepts and features in the distributed package. Use this document to find the distributed training technology that can best serve your application.
   :image: _static/img/thumbnails/cropped/PyTorch-Distributed-Overview.png
   :link: beginner/dist_overview.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Single-Machine Model Parallel Best Practices
   :card_description:  Learn how to implement model parallel, a distributed training technique which splits a single model onto different GPUs, rather than replicating the entire model on each GPU
   :image: _static/img/thumbnails/cropped/Model-Parallel-Best-Practices.png
   :link: intermediate/model_parallel_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Distributed Data Parallel
   :card_description: Learn the basics of when to use distributed data paralle versus data parallel and work through an example to set it up.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-Distributed-Data-Parallel.png
   :link: intermediate/ddp_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: (advanced) PyTorch 1.0 Distributed Trainer with Amazon AWS
   :card_description: Set up the distributed package of PyTorch, use the different communication strategies, and go over some the internals of the package.
   :image: _static/img/thumbnails/cropped/advanced-PyTorch-1point0-Distributed-Trainer-with-Amazon-AWS.png
   :link: beginner/aws_distributed_training_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: PyTorch로 분산 어플리케이션 개발하기
   :card_description: PyTorch의 분산 패키지를 설정하고, 서로 다른 통신 전략을 사용하고, 내부를 살펴봅니다.
   :image: _static/img/thumbnails/cropped/Writing-Distributed-Applications-with-PyTorch.png
   :link: intermediate/dist_tuto.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Distributed RPC Framework
   :card_description: Learn how to build distributed training using the torch.distributed.rpc package.
   :image: _static/img/thumbnails/cropped/Getting Started with Distributed-RPC-Framework.png
   :link: intermediate/rpc_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Implementing a Parameter Server Using Distributed RPC Framework
   :card_description: Walk through a through a simple example of implementing a parameter server using PyTorch’s Distributed RPC framework.
   :image: _static/img/thumbnails/cropped/Implementing-a-Parameter-Server-Using-Distributed-RPC-Framework.png
   :link: intermediate/rpc_param_server_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Distributed Pipeline Parallelism Using RPC
   :card_description: Demonstrate how to implement distributed pipeline parallelism using RPC
   :image: _static/img/thumbnails/cropped/Distributed-Pipeline-Parallelism-Using-RPC.png
   :link: intermediate/dist_pipeline_parallel_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Implementing Batch RPC Processing Using Asynchronous Executions
   :card_description: Learn how to use rpc.functions.async_execution to implement batch RPC
   :image: _static/img/thumbnails/cropped/Implementing-Batch-RPC-Processing-Using-Asynchronous-Executions.png
   :link: intermediate/rpc_async_execution.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Combining Distributed DataParallel with Distributed RPC Framework
   :card_description: Walk through a through a simple example of how to combine distributed data parallelism with distributed model parallelism.
   :image: _static/img/thumbnails/cropped/Combining-Distributed-DataParallel-with-Distributed-RPC-Framework.png
   :link: advanced/rpc_ddp_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. End of tutorial card section

.. raw:: html

    </div>

    <div class="pagination d-flex justify-content-center"></div>

    </div>

    </div>
    <br>
    <br>


추가 자료
============================

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :header: 파이토치(PyTorch) 예제
   :description: 비전, 텍스트, Reinforcement-Learning 등의 분야에서의 PyTorch 예제 모음
   :button_link: https://github.com/pytorch/examples
   :button_text: Checkout Examples

.. customcalloutitem::
   :header: PyTorch Cheat Sheet
   :description: Quick overview to essential PyTorch elements.
   :button_link: beginner/ptcheat.html
   :button_text: Download

.. customcalloutitem::
   :header: 공식 튜토리얼 저장소(GitHub)
   :description: GitHub에서 공식 튜토리얼을 만나보세요.
   :button_link: https://github.com/pytorch/tutorials
   :button_text: Go To GitHub

.. customcalloutitem::
   :header: (비공식) 한국어 튜토리얼 저장소(GitHub)
   :description: GitHub에서 (비공식) 한국어 튜토리얼을 만나보세요.
   :button_link: https://github.com/9bow/PyTorch-tutorials-kr
   :button_text: Go To GitHub


.. End of callout section

.. raw:: html

        </div>
    </div>

    <div style='clear:both'></div>

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: 파이토치(PyTorch) 레시피

   모든 레시피 보기 <recipes/recipes_index>

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: 파이토치(PyTorch) 배우기

   beginner/deep_learning_60min_blitz
   beginner/pytorch_with_examples
   beginner/nn_tutorial
   intermediate/tensorboard_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 이미지/비디오

   intermediate/torchvision_tutorial
   beginner/transfer_learning_tutorial
   beginner/fgsm_tutorial
   beginner/dcgan_faces_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 오디오

   beginner/audio_preprocessing_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 텍스트

   beginner/transformer_tutorial
   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   beginner/text_sentiment_ngrams_tutorial
   beginner/torchtext_translation_tutorial


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 강화학습

   intermediate/reinforcement_q_learning

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: PyTorch 모델을 프로덕션 환경에 배포하기

   intermediate/flask_rest_api_tutorial
   beginner/Intro_to_TorchScript_tutorial
   advanced/cpp_export
   advanced/super_resolution_with_onnxruntime

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 프론트엔드 API

   intermediate/named_tensor_tutorial
   intermediate/memory_format_tutorial
   advanced/cpp_frontend
   advanced/cpp_extension
   advanced/torch_script_custom_ops
   advanced/torch_script_custom_classes
   advanced/torch-script-parallelism
   advanced/cpp_autograd
   advanced/dispatcher

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 모델 최적화

   intermediate/pruning_tutorial
   advanced/dynamic_quantization_tutorial
   intermediate/dynamic_quantization_bert_tutorial
   advanced/static_quantization_tutorial
   intermediate/quantized_transfer_learning_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 병렬 및 분산 학습

   beginner/dist_overview
   intermediate/model_parallel_tutorial
   intermediate/ddp_tutorial
   intermediate/dist_tuto
   intermediate/rpc_tutorial
   beginner/aws_distributed_training_tutorial
   intermediate/rpc_param_server_tutorial
   intermediate/dist_pipeline_parallel_tutorial
   intermediate/rpc_async_execution
   advanced/rpc_ddp_tutorial