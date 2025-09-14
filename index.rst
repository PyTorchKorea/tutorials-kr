:og:description: 파이토치(PyTorch) 한국어 튜토리얼에 오신 것을 환영합니다. 파이토치 한국 사용자 모임은 한국어를 사용하시는 많은 분들께 PyTorch를 소개하고 함께 배우며 성장하는 것을 목표로 하고 있습니다.

파이토치(PyTorch) 한국어 튜토리얼에 오신 것을 환영합니다!
=============================================================

**아래 튜토리얼들이 새로 추가되었습니다:**

* `Integrating Custom Operators with SYCL for Intel GPU <https://tutorials.pytorch.kr/advanced/cpp_custom_ops_sycl.html>`__
* `Supporting Custom C++ Classes in torch.compile/torch.export <https://docs.tutorials.pytorch.kr/advanced/custom_class_pt2.html>`__
* `Accelerating torch.save and torch.load with GPUDirect Storage <https://docs.tutorials.pytorch.kr/unstable/gpu_direct_storage.html>`__
* `Getting Started with Fully Sharded Data Parallel (FSDP2) <https://docs.tutorials.pytorch.kr/intermediate/FSDP_tutorial.html>`__

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: PyTorch 개념과 모듈을 익힙니다. 데이터를 불러오고, 심층 신경망을 구성하고, 모델을 학습하고 저장하는 방법을 배웁니다.
   :header: PyTorch 기본 익히기
   :button_link: beginner/basics/intro.html
   :button_text: PyTorch 시작하기

.. customcalloutitem::
   :description: 한 입 크기의, 바로 사용할 수 있는 PyTorch 코드 예제들을 확인해보세요.
   :header: 파이토치(PyTorch) 레시피
   :button_link: recipes_index.html
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
   :header: PyTorch 기본 익히기
   :card_description: PyTorch로 전체 ML워크플로우를 구축하기 위한 단계별 학습 가이드입니다.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: beginner/basics/intro.html
   :tags: Getting-Started

.. customcarditem::
   :header: Introduction to PyTorch on YouTube
   :card_description: An introduction to building a complete ML workflow with PyTorch. Follows the PyTorch Beginner Series on YouTube.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: beginner/introyt/introyt_index.html
   :tags: Getting-Started

.. customcarditem::
   :header: 예제로 배우는 파이토치(PyTorch)
   :card_description: 튜토리얼에 포함된 예제들로 PyTorch의 기본 개념을 이해합니다.
   :image: _static/img/thumbnails/cropped/learning-pytorch-with-examples.png
   :link: beginner/pytorch_with_examples.html
   :tags: Getting-Started

.. customcarditem::
   :header: torch.nn이 실제로 무엇인가요?
   :card_description: torch.nn을 사용하여 신경망을 생성하고 학습합니다.
   :image: _static/img/thumbnails/cropped/torch-nn.png
   :link: beginner/nn_tutorial.html
   :tags: Getting-Started

.. customcarditem::
   :header: TensorBoard로 모델, 데이터, 학습 시각화하기
   :card_description: TensorBoard로 데이터 및 모델 교육을 시각화하는 방법을 배웁니다.
   :image: _static/img/thumbnails/cropped/visualizing-with-tensorboard.png
   :link: intermediate/tensorboard_tutorial.html
   :tags: Interpretability,Getting-Started,Tensorboard

.. customcarditem::
   :header: Good usage of `non_blocking` and `pin_memory()` in PyTorch
   :card_description: A guide on best practices to copy data from CPU to GPU.
   :image: _static/img/pinmem.png
   :link: intermediate/pinmem_nonblock.html
   :tags: Getting-Started

.. customcarditem::
   :header: Understanding requires_grad, retain_grad, Leaf, and Non-leaf Tensors
   :card_description: Learn the subtleties of requires_grad, retain_grad, leaf, and non-leaf tensors
   :image: _static/img/thumbnails/cropped/understanding_leaf_vs_nonleaf.png
   :link: beginner/understanding_leaf_vs_nonleaf_tutorial.html
   :tags: Getting-Started

.. customcarditem::
   :header: Visualizing Gradients in PyTorch
   :card_description: Visualize the gradient flow of a network.
   :image: _static/img/thumbnails/cropped/visualizing_gradients_tutorial.png
   :link: intermediate/visualizing_gradients_tutorial.html
   :tags: Getting-Started

.. Image/Video

.. customcarditem::
   :header: TorchVision 객체 검출 미세조정(Finetuning) 튜토리얼
   :card_description: 이미 훈련된 Mask R-CNN 모델을 미세조정합니다.
   :image: _static/img/thumbnails/cropped/TorchVision-Object-Detection-Finetuning-Tutorial.png
   :link: intermediate/torchvision_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: 컴퓨터 비전을 위한 전이학습(Transfer Learning) 튜토리얼
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

.. customcarditem::
   :header: Spatial Transformer Networks Tutorial
   :card_description: Learn how to augment your network using a visual attention mechanism.
   :image: _static/img/stn/Five.gif
   :link: intermediate/spatial_transformer_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Inference on Whole Slide Images with TIAToolbox
   :card_description: Learn how to use the TIAToolbox to perform inference on whole slide images.
   :image: _static/img/thumbnails/cropped/TIAToolbox-Tutorial.png
   :link: intermediate/tiatoolbox_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Semi-Supervised Learning Tutorial Based on USB
   :card_description: Learn how to train semi-supervised learning algorithms (on custom data) using USB and PyTorch.
   :image: _static/img/usb_semisup_learn/code.png
   :link: advanced/usb_semisup_learn.html
   :tags: Image/Video

.. Audio

.. customcarditem::
   :header: Audio IO
   :card_description: Learn to load data with torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_io_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Resampling
   :card_description: Learn to resample audio waveforms using torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_resampling_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Data Augmentation
   :card_description: Learn to apply data augmentations using torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_data_augmentation_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Feature Extractions
   :card_description: Learn to extract features using torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_feature_extractions_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Feature Augmentation
   :card_description: Learn to augment features using torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_feature_augmentation_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Datasets
   :card_description: Learn to use torchaudio datasets.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_datasets_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Automatic Speech Recognition with Wav2Vec2 in torchaudio
   :card_description: Learn how to use torchaudio's pretrained models for building a speech recognition application.
   :image: _static/img/thumbnails/cropped/torchaudio-asr.png
   :link: intermediate/speech_recognition_pipeline_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Speech Command Classification
   :card_description: Learn how to correctly format an audio dataset and then train/test an audio classifier network on the dataset.
   :image: _static/img/thumbnails/cropped/torchaudio-speech.png
   :link: intermediate/speech_command_classification_with_torchaudio_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Text-to-Speech with torchaudio
   :card_description: Learn how to use torchaudio's pretrained models for building a text-to-speech application.
   :image: _static/img/thumbnails/cropped/torchaudio-speech.png
   :link: intermediate/text_to_speech_with_torchaudio.html
   :tags: Audio

.. customcarditem::
   :header: Forced Alignment with Wav2Vec2 in torchaudio
   :card_description: Learn how to use torchaudio's Wav2Vec2 pretrained models for aligning text to speech
   :image: _static/img/thumbnails/cropped/torchaudio-alignment.png
   :link: intermediate/forced_alignment_with_torchaudio_tutorial.html
   :tags: Audio

.. NLP

.. customcarditem::
   :header: 기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 분류하기
   :card_description: torchtext를 사용하지 않고 기본적인 문자-단위 RNN을 사용하여 단어를 분류하는 모델을 기초부터 만들고 학습합니다. 총 3개로 이뤄진 튜토리얼 시리즈의 첫번째 편입니다.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Classifying-Names-with-a-Character-Level-RNN.png
   :link: intermediate/char_rnn_classification_tutorial
   :tags: NLP

.. customcarditem::
   :header: 기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 생성하기
   :card_description: 문자-단위 RNN을 사용하여 이름을 분류해봤으니, 이름을 생성하는 방법을 학습합니다. 총 3개로 이뤄진 튜토리얼 시리즈 중 두번째 편입니다.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Generating-Names-with-a-Character-Level-RNN.png
   :link: intermediate/char_rnn_generation_tutorial.html
   :tags: NLP

.. customcarditem::
   :header: 기초부터 시작하는 NLP: 시퀀스-투-시퀀스 네트워크와 어텐션을 이용한 번역
   :card_description: “기초부터 시작하는 NLP”의 세번째이자 마지막 편으로, NLP 모델링 작업을 위한 데이터 전처리에 사용할 자체 클래스와 함수들을 작성해보겠습니다.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Translation-with-a-Sequence-to-Sequence-Network-and-Attention.png
   :link: intermediate/seq2seq_translation_tutorial.html
   :tags: NLP

.. ONNX

.. customcarditem::
   :header: Exporting a PyTorch model to ONNX using TorchDynamo backend and Running it using ONNX Runtime
   :card_description: Build a image classifier model in PyTorch and convert it to ONNX before deploying it with ONNX Runtime.
   :image: _static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png
   :link: beginner/onnx/export_simple_model_to_onnx_tutorial.html
   :tags: Production,ONNX,Backends

.. customcarditem::
   :header: Extending the ONNX exporter operator support
   :card_description: Demonstrate end-to-end how to address unsupported operators in ONNX.
   :image: _static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png
   :link: beginner/onnx/onnx_registry_tutorial.html
   :tags: Production,ONNX,Backends

.. customcarditem::
   :header: Exporting a model with control flow to ONNX
   :card_description: Demonstrate how to handle control flow logic while exporting a PyTorch model to ONNX.
   :image: _static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png
   :link: beginner/onnx/export_control_flow_model_to_onnx_tutorial.html
   :tags: Production,ONNX,Backends

.. Reinforcement Learning

.. customcarditem::
   :header: 강화 학습(DQN) 튜토리얼
   :card_description: PyTorch를 사용하여 OpenAI Gym의 CartPole-v0 태스크에서 DQN(Deep Q Learning) 에이전트를 학습하는 방법을 살펴봅니다.
   :image: _static/img/cartpole.gif
   :link: intermediate/reinforcement_q_learning.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Reinforcement Learning (PPO) with TorchRL
   :card_description: Learn how to use PyTorch and TorchRL to train a Proximal Policy Optimization agent on the Inverted Pendulum task from Gym.
   :image: _static/img/invpendulum.gif
   :link: intermediate/reinforcement_ppo.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Train a Mario-playing RL Agent
   :card_description: Use PyTorch to train a Double Q-learning agent to play Mario.
   :image: _static/img/mario.gif
   :link: intermediate/mario_rl_tutorial.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Recurrent DQN
   :card_description: Use TorchRL to train recurrent policies
   :image: _static/img/rollout_recurrent.png
   :link: intermediate/dqn_with_rnn_tutorial.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Code a DDPG Loss
   :card_description: Use TorchRL to code a DDPG Loss
   :image: _static/img/half_cheetah.gif
   :link: advanced/coding_ddpg.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Writing your environment and transforms
   :card_description: Use TorchRL to code a Pendulum
   :image: _static/img/pendulum.gif
   :link: advanced/pendulum.html
   :tags: Reinforcement-Learning


.. Deploying PyTorch Models in Production

.. customcarditem::
   :header: Profiling PyTorch
   :card_description: Learn how to profile a PyTorch application
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: beginner/profiler.html
   :tags: Profiling

.. customcarditem::
   :header: Profiling PyTorch
   :card_description: Introduction to Holistic Trace Analysis
   :link: beginner/hta_intro_tutorial.html
   :tags: Profiling

.. customcarditem::
   :header: Profiling PyTorch
   :card_description: Trace Diff using Holistic Trace Analysis
   :link: beginner/hta_trace_diff_tutorial.html
   :tags: Profiling


.. Code Transformations with FX

.. customcarditem::
   :header: Building a Simple Performance Profiler with FX
   :card_description: Build a simple FX interpreter to record the runtime of op, module, and function calls and report statistics
   :image: _static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png
   :link: intermediate/fx_profiling_tutorial.html
   :tags: FX

.. Frontend APIs

.. customcarditem::
   :header: (베타) PyTorch의 Channels Last 메모리 형식
   :card_description: Channels Last 메모리 형식에 대한 개요를 확인하고 차원 순서를 유지하며 메모리 상의 NCHW 텐서를 정렬하는 방법을 이해합니다.
   :image: _static/img/thumbnails/cropped/experimental-Channels-Last-Memory-Format-in-PyTorch.png
   :link: intermediate/memory_format_tutorial.html
   :tags: Memory-Format,Best-Practice,Frontend-APIs

.. customcarditem::
   :header: Using the PyTorch C++ Frontend
   :card_description: Walk through an end-to-end example of training a model with the C++ frontend by training a DCGAN – a kind of generative model – to generate images of MNIST digits.
   :image: _static/img/thumbnails/cropped/Using-the-PyTorch-Cpp-Frontend.png
   :link: advanced/cpp_frontend.html
   :tags: Frontend-APIs,C++

.. customcarditem::
   :header: PyTorch Custom Operators Landing Page
   :card_description: This is the landing page for all things related to custom operators in PyTorch.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/cpp_extension.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Custom Python Operators
   :card_description: Create Custom Operators in Python. Useful for black-boxing a Python function for use with torch.compile.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/python_custom_ops.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Compiled Autograd: Capturing a larger backward graph for ``torch.compile``
   :card_description: Learn how to use compiled autograd to capture a larger backward graph.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/compiled_autograd_tutorial
   :tags: Model-Optimization,CUDA

.. customcarditem::
   :header: Custom C++ and CUDA Operators
   :card_description: How to extend PyTorch with custom C++ and CUDA operators.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/cpp_custom_ops.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Autograd in C++ Frontend
   :card_description: The autograd package helps build flexible and dynamic neural netorks. In this tutorial, explore several examples of doing autograd in PyTorch C++ frontend
   :image: _static/img/thumbnails/cropped/Autograd-in-Cpp-Frontend.png
   :link: advanced/cpp_autograd.html
   :tags: Frontend-APIs,C++

.. customcarditem::
   :header: Registering a Dispatched Operator in C++
   :card_description: The dispatcher is an internal component of PyTorch which is responsible for figuring out what code should actually get run when you call a function like torch::add.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/dispatcher.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Extending Dispatcher For a New Backend in C++
   :card_description: Learn how to extend the dispatcher to add a new device living outside of the pytorch/pytorch repo and maintain it to keep in sync with native PyTorch devices.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/extend_dispatcher.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Facilitating New Backend Integration by PrivateUse1
   :card_description: Learn how to integrate a new backend living outside of the pytorch/pytorch repo and maintain it to keep in sync with the native PyTorch backend.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/privateuseone.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Custom Function Tutorial: Double Backward
   :card_description: Learn how to write a custom autograd Function that supports double backward.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/custom_function_double_backward_tutorial.html
   :tags: Extending-PyTorch,Frontend-APIs

.. customcarditem::
   :header: Custom Function Tutorial: Fusing Convolution and Batch Norm
   :card_description: Learn how to create a custom autograd Function that fuses batch norm into a convolution to improve memory usage.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/custom_function_conv_bn_tutorial.html
   :tags: Extending-PyTorch,Frontend-APIs

.. customcarditem::
   :header: Forward-mode Automatic Differentiation
   :card_description: Learn how to use forward-mode automatic differentiation.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/forward_ad_usage.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Jacobians, Hessians, hvp, vhp, and more
   :card_description: Learn how to compute advanced autodiff quantities using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/jacobians_hessians.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Model Ensembling
   :card_description: Learn how to ensemble models using torch.vmap
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/ensembling.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Per-Sample-Gradients
   :card_description: Learn how to compute per-sample-gradients using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/per_sample_grads.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Neural Tangent Kernels
   :card_description: Learn how to compute neural tangent kernels using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/neural_tangent_kernels.html
   :tags: Frontend-APIs

.. Model Optimization

.. customcarditem::
   :header: Performance Profiling in PyTorch
   :card_description: Learn how to use the PyTorch Profiler to benchmark your module's performance.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: beginner/profiler.html
   :tags: Model-Optimization,Best-Practice,Profiling

.. customcarditem::
   :header: Performance Profiling in TensorBoard
   :card_description: Learn how to use the TensorBoard plugin to profile and analyze your model's performance.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: intermediate/tensorboard_profiler_tutorial.html
   :tags: Model-Optimization,Best-Practice,Profiling,TensorBoard

.. customcarditem::
   :header: Hyperparameter Tuning Tutorial
   :card_description: Learn how to use Ray Tune to find the best performing set of hyperparameters for your model.
   :image: _static/img/ray-tune.png
   :link: beginner/hyperparameter_tuning_tutorial.html
   :tags: Model-Optimization,Best-Practice

.. customcarditem::
   :header: Parametrizations Tutorial
   :card_description: Learn how to use torch.nn.utils.parametrize to put constraints on your parameters (e.g. make them orthogonal, symmetric positive definite, low-rank...)
   :image: _static/img/thumbnails/cropped/parametrizations.png
   :link: intermediate/parametrizations.html
   :tags: Model-Optimization,Best-Practice

.. customcarditem::
   :header: 가지치기 기법(pruning) 튜토리얼
   :card_description: torch.nn.utils.prune을 사용하여 신경망을 희소화(sparsify)하는 방법과, 이를 확장하여 사용자 정의 가지치기 기법을 구현하는 방법을 알아봅니다.
   :image: _static/img/thumbnails/cropped/Pruning-Tutorial.png
   :link: intermediate/pruning_tutorial.html
   :tags: Model-Optimization,Best-Practice

.. customcarditem::
   :header: How to save memory by fusing the optimizer step into the backward pass
   :card_description: Learn a memory-saving technique through fusing the optimizer step into the backward pass using memory snapshots.
   :image: _static/img/thumbnails/cropped/pytorch-logo.png
   :link: intermediate/optimizer_step_in_backward_tutorial.html
   :tags: Model-Optimization,Best-Practice,CUDA,Frontend-APIs

.. customcarditem::
   :header: (beta) Accelerating BERT with semi-structured sparsity
   :card_description: Train BERT, prune it to be 2:4 sparse, and then accelerate it to achieve 2x inference speedups with semi-structured sparsity and torch.compile.
   :image: _static/img/thumbnails/cropped/Pruning-Tutorial.png
   :link: advanced/semi_structured_sparse.html
   :tags: NLP,Model-Optimization

.. customcarditem::
   :header: Multi-Objective Neural Architecture Search with Ax
   :card_description: Learn how to use Ax to search over architectures find optimal tradeoffs between accuracy and latency.
   :image: _static/img/ax_logo.png
   :link: intermediate/ax_multiobjective_nas_tutorial.html
   :tags: Model-Optimization,Best-Practice,Ax,TorchX

.. customcarditem::
   :header: torch.compile Tutorial
   :card_description: Speed up your models with minimal code changes using torch.compile, the latest PyTorch compiler solution.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/torch_compile_tutorial.html
   :tags: Model-Optimization

.. customcarditem::
   :header: Building a Convolution/Batch Norm fuser in torch.compile
   :card_description: Build a simple pattern matcher pass that fuses batch norm into convolution to improve performance during inference.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/torch_compile_conv_bn_fuser.html
   :tags: Model-Optimization

.. customcarditem::
   :header: Inductor CPU Backend Debugging and Profiling
   :card_description: Learn the usage, debugging and performance profiling for ``torch.compile`` with Inductor CPU backend.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/inductor_debug_cpu.html
   :tags: Model-Optimization

.. customcarditem::
   :header: (beta) Implementing High-Performance Transformers with SCALED DOT PRODUCT ATTENTION
   :card_description: This tutorial explores the new torch.nn.functional.scaled_dot_product_attention and how it can be used to construct Transformer components.
   :image: _static/img/thumbnails/cropped/pytorch-logo.png
   :link: intermediate/scaled_dot_product_attention_tutorial.html
   :tags: Model-Optimization,Attention,Transformer

.. customcarditem::
   :header: Knowledge Distillation in Convolutional Neural Networks
   :card_description:  Learn how to improve the accuracy of lightweight models using more powerful models as teachers.
   :image: _static/img/thumbnails/cropped/knowledge_distillation_pytorch_logo.png
   :link: beginner/knowledge_distillation_tutorial.html
   :tags: Model-Optimization,Image/Video

.. customcarditem::
   :header: Accelerating PyTorch Transformers by replacing nn.Transformer with Nested Tensors and torch.compile()
   :card_description: This tutorial goes over recommended best practices for implementing Transformers with native PyTorch.
   :image: _static/img/thumbnails/cropped/pytorch-logo.png
   :link: intermediate/transformer_building_blocks.html
   :tags: Transformer


.. Parallel-and-Distributed-Training

.. customcarditem::
   :header: PyTorch Distributed Overview
   :card_description: Briefly go over all concepts and features in the distributed package. Use this document to find the distributed training technology that can best serve your application.
   :image: _static/img/thumbnails/cropped/PyTorch-Distributed-Overview.png
   :link: beginner/dist_overview.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Distributed Data Parallel in PyTorch - Video Tutorials
   :card_description: This series of video tutorials walks you through distributed training in PyTorch via DDP.
   :image: _static/img/thumbnails/cropped/PyTorch-Distributed-Overview.png
   :link: beginner/ddp_series_intro.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: 단일 머신을 사용한 모델 병렬화 모범 사례
   :card_description: 개별 GPU들에 전체 모델을 복제하는 대신, 하나의 모델을 여러 GPU에 분할하여 분산학습을 하는 모델 병렬 처리를 구현하는 방법을 배웁니다.
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
   :header: PyTorch로 분산 어플리케이션 개발하기
   :card_description: PyTorch의 분산 패키지를 설정하고, 서로 다른 통신 전략을 사용하고, 내부를 살펴봅니다.
   :image: _static/img/thumbnails/cropped/Writing-Distributed-Applications-with-PyTorch.png
   :link: intermediate/dist_tuto.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Large Scale Transformer model training with Tensor Parallel
   :card_description: Learn how to train large models with Tensor Parallel package.
   :image: _static/img/thumbnails/cropped/Large-Scale-Transformer-model-training-with-Tensor-Parallel.png
   :link: intermediate/TP_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Customize Process Group Backends Using Cpp Extensions
   :card_description: Extend ProcessGroup with custom collective communication implementations.
   :image: _static/img/thumbnails/cropped/Customize-Process-Group-Backends-Using-Cpp-Extensions.png
   :link: intermediate/process_group_cpp_extension_tutorial.html
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
   :header: Introduction to Distributed Pipeline Parallelism
   :card_description: Demonstrate how to implement pipeline parallelism using torch.distributed.pipelining
   :image: _static/img/thumbnails/cropped/Introduction-to-Distributed-Pipeline-Parallelism.png
   :link: intermediate/pipelining_tutorial.html
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

.. customcarditem::
   :header: Getting Started with Fully Sharded Data Parallel (FSDP2)
   :card_description: Learn how to train models with Fully Sharded Data Parallel (fully_shard) package.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-FSDP.png
   :link: intermediate/FSDP_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Introduction to Libuv TCPStore Backend
   :card_description: TCPStore now uses a new server backend for faster connection and better scalability.
   :image: _static/img/thumbnails/cropped/Introduction-to-Libuv-Backend-TCPStore.png
   :link: intermediate/TCPStore_libuv_backend.html
   :tags: Parallel-and-Distributed-Training


.. Edge

.. customcarditem::
   :header: Exporting to ExecuTorch Tutorial
   :card_description: Learn about how to use ExecuTorch, a unified ML stack for lowering PyTorch models to edge devices.
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html
   :tags: Edge

.. customcarditem::
   :header: Running an ExecuTorch Model in C++ Tutorial
   :card_description: Learn how to load and execute an ExecuTorch model in C++
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/running-a-model-cpp-tutorial.html
   :tags: Edge

.. customcarditem::
   :header: Using the ExecuTorch SDK to Profile a Model
   :card_description: Explore how to use the ExecuTorch SDK to profile, debug, and visualize ExecuTorch models
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://docs.pytorch.org/executorch/main/tutorials/devtools-integration-tutorial.html
   :tags: Edge

.. customcarditem::
   :header: Building an ExecuTorch iOS Demo App
   :card_description: Explore how to set up the ExecuTorch iOS Demo App, which uses the MobileNet v3 model to process live camera images leveraging three different backends: XNNPACK, Core ML, and Metal Performance Shaders (MPS).
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://github.com/meta-pytorch/executorch-examples/tree/main/mv3/apple/ExecuTorchDemo
   :tags: Edge

.. customcarditem::
   :header: Building an ExecuTorch Android Demo App
   :card_description: Learn how to set up the ExecuTorch Android Demo App for image segmentation tasks using the DeepLab v3 model and XNNPACK FP32 backend.
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo#executorch-android-demo-app
   :tags: Edge

.. customcarditem::
   :header: Lowering a Model as a Delegate
   :card_description: Learn to accelerate your program using ExecuTorch by applying delegates through three methods: lowering the whole module, composing it with another module, and partitioning parts of a module.
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/examples-end-to-end-to-lower-model-to-delegate.html
   :tags: Edge


.. Recommendation Systems

.. customcarditem::
   :header: Introduction to TorchRec
   :card_description: TorchRec is a PyTorch domain library built to provide common sparsity & parallelism primitives needed for large-scale recommender systems.
   :image: _static/img/thumbnails/torchrec.png
   :link: intermediate/torchrec_intro_tutorial.html
   :tags: TorchRec,Recommender

.. customcarditem::
   :header: Exploring TorchRec sharding
   :card_description: This tutorial covers the sharding schemes of embedding tables by using <code>EmbeddingPlanner</code> and <code>DistributedModelParallel</code> API.
   :image: _static/img/thumbnails/torchrec.png
   :link: advanced/sharding.html
   :tags: TorchRec,Recommender


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
   :description: 비전, 텍스트, 강화학습 등의 분야에서 기존 코드에 통합하여 사용할 수 있는 PyTorch 예제 모음
   :button_link: https://pytorch.org/examples?utm_source=examples&utm_medium=examples-landing
   :button_text: Checkout Examples

.. customcalloutitem::
   :header: 공식 튜토리얼 저장소(GitHub)
   :description: GitHub에서 공식 튜토리얼을 만나보세요.
   :button_link: https://github.com/pytorchkorea/tutorials-kr
   :button_text: Go To GitHub

.. customcalloutitem::
   :header: 튜토리얼을 Google Colab에서 실행하기
   :description: Google Colab에서 튜토리얼을 실행하기 위해 튜토리얼 데이터를 Google Drive로 복사하는 방법을 배웁니다.
   :button_link: beginner/colab.html
   :button_text: Open

.. customcalloutitem::
   :header: (비공식) 한국어 튜토리얼 저장소(GitHub)
   :description: GitHub에서 (비공식) 한국어 튜토리얼을 만나보세요.
   :button_link: https://github.com/PyTorchKorea/tutorials-kr
   :button_text: Go To GitHub

.. customcalloutitem::
   :header: 파이토치 한국어 커뮤니티
   :description: 파이토치를 사용하는 다른 사용자들과 의견을 나눠보세요.
   :button_link: https://discuss.pytorch.kr
   :button_text: Open

.. End of callout section

.. raw:: html

        </div>
    </div>

    <div style='clear:both'></div>

.. ----------------------------------------------------------------------------------------
.. Page TOC
.. ----------------------------------------------------------------------------------------
.. toctree::
   :maxdepth: 1
   :caption: 파이토치(PyTorch) 시작하기

   beginner/basics/intro
   beginner/basics/quickstart_tutorial
   beginner/basics/tensorqs_tutorial
   beginner/basics/data_tutorial
   beginner/basics/transforms_tutorial
   beginner/basics/buildmodel_tutorial
   beginner/basics/autogradqs_tutorial
   beginner/basics/optimization_tutorial
   beginner/basics/saveloadrun_tutorial

.. toctree::
   :maxdepth: 1
   :caption: Introduction to PyTorch on YouTube

   beginner/introyt/introyt1_tutorial
   beginner/introyt/tensors_deeper_tutorial
   beginner/introyt/autogradyt_tutorial
   beginner/introyt/modelsyt_tutorial
   beginner/introyt/tensorboardyt_tutorial
   beginner/introyt/trainingyt
   beginner/introyt/captumyt

.. toctree::
   :maxdepth: 1
   :caption: 파이토치(PyTorch) 배우기

   beginner/deep_learning_60min_blitz
   beginner/pytorch_with_examples
   beginner/nn_tutorial
   beginner/understanding_leaf_vs_nonleaf_tutorial
   intermediate/tensorboard_tutorial

.. toctree::
   :maxdepth: 1
   :caption: 이미지/비디오

   intermediate/torchvision_tutorial
   beginner/transfer_learning_tutorial
   beginner/fgsm_tutorial
   beginner/dcgan_faces_tutorial
   intermediate/tiatoolbox_tutorial

.. toctree::
   :maxdepth: 1
   :caption: 오디오

   beginner/audio_io_tutorial
   beginner/audio_resampling_tutorial
   beginner/audio_data_augmentation_tutorial
   beginner/audio_feature_extractions_tutorial
   beginner/audio_feature_augmentation_tutorial
   beginner/audio_datasets_tutorial
   intermediate/speech_recognition_pipeline_tutorial
   intermediate/speech_command_classification_with_torchaudio_tutorial
   intermediate/text_to_speech_with_torchaudio
   intermediate/forced_alignment_with_torchaudio_tutorial

.. toctree::
   :maxdepth: 1
   :caption: 텍스트

   beginner/bettertransformer_tutorial
   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   intermediate/transformer_building_blocks

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 백엔드

   beginner/onnx/intro_onnx

.. toctree::
   :maxdepth: 1
   :caption: 강화학습

   intermediate/reinforcement_q_learning
   intermediate/reinforcement_ppo
   intermediate/mario_rl_tutorial
   advanced/pendulum

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: PyTorch 모델을 프로덕션 환경에 배포하기

   beginner/onnx/intro_onnx
   intermediate/realtime_rpi

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: PyTorch 프로파일링

   beginner/profiler
   beginner/hta_intro_tutorial
   beginner/hta_trace_diff_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Code Transforms with FX

   intermediate/fx_profiling_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 프론트엔드 API

   intermediate/memory_format_tutorial
   intermediate/forward_ad_usage
   intermediate/jacobians_hessians
   intermediate/ensembling
   intermediate/per_sample_grads
   intermediate/neural_tangent_kernels
   intermediate/visualizing_gradients_tutorial
   advanced/cpp_frontend
   advanced/torch-script-parallelism
   advanced/cpp_autograd

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: PyTorch 확장하기

   intermediate/custom_function_double_backward_tutorial
   intermediate/custom_function_conv_bn_tutorial
   advanced/cpp_extension
   advanced/dispatcher
   advanced/extend_dispatcher
   advanced/privateuseone

.. toctree::
   :maxdepth: 1
   :caption: 모델 최적화

   beginner/profiler
   intermediate/tensorboard_profiler_tutorial
   beginner/hyperparameter_tuning_tutorial
   intermediate/parametrizations
   intermediate/pruning_tutorial
   advanced/static_quantization_tutorial
   intermediate/ax_multiobjective_nas_tutorial
   intermediate/torch_compile_tutorial
   intermediate/compiled_autograd_tutorial
   intermediate/inductor_debug_cpu
   intermediate/scaled_dot_product_attention_tutorial
   beginner/knowledge_distillation_tutorial

.. toctree::
   :maxdepth: 1
   :caption: 병렬 및 분산 학습

   beginner/dist_overview
   beginner/ddp_series_intro
   intermediate/model_parallel_tutorial
   intermediate/ddp_tutorial
   intermediate/dist_tuto
   intermediate/FSDP_tutorial
   intermediate/FSDP1_tutorial
   intermediate/FSDP_advanced_tutorial
   intermediate/TP_tutorial
   intermediate/process_group_cpp_extension_tutorial
   intermediate/rpc_tutorial
   intermediate/rpc_param_server_tutorial
   intermediate/dist_pipeline_parallel_tutorial
   intermediate/rpc_async_execution
   advanced/rpc_ddp_tutorial
   advanced/ddp_pipeline
   advanced/generic_join

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Edge with ExecuTorch

   Exporting to ExecuTorch Tutorial <https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html>
   Running an ExecuTorch Model in C++ Tutorial < https://pytorch.org/executorch/stable/running-a-model-cpp-tutorial.html>
   Using the ExecuTorch SDK to Profile a Model <https://pytorch.org/executorch/stable/tutorials/sdk-integration-tutorial.html>
   Building an ExecuTorch iOS Demo App <https://pytorch.org/executorch/stable/demo-apps-ios.html>
   Building an ExecuTorch Android Demo App <https://pytorch.org/executorch/stable/demo-apps-android.html>
   Lowering a Model as a Delegate <https://pytorch.org/executorch/stable/examples-end-to-end-to-lower-model-to-delegate.html>
