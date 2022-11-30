PyTorch Recipes
---------------------------------------------
Recipes are bite-sized bite-sized, actionable examples of how to use specific PyTorch features, different from our full-length tutorials.

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

.. Add recipe cards below this line

.. Basics

.. customcarditem::
   :header: PyTorch에서 데이터 불러오기
   :card_description: PyTorch 패키지를 이용해서 공용 데이터셋을 불러오고 모델에 적용하는 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/loading-data.PNG
   :link: ../recipes/recipes/loading_data_recipe.html
   :tags: Basics

.. customcarditem::
   :header: 신경망 정의하기
   :card_description: MNIST dataset을 사용한 신경망을 만들고 정의하기 위해 PyTorch의 torch.nn 패키지를 어떻게 사용하는 지 알아봅시다.
   :image: ../_static/img/thumbnails/cropped/defining-a-network.PNG
   :link: ../recipes/recipes/defining_a_neural_network.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch에서 state_dict란 무엇인가요?
   :card_description: PyTorch에서 모델을 저장하거나 불러올 때 Python 사전인 state_dict 객체가 어떻게 사용되는지 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/what-is-a-state-dict.PNG
   :link: ../recipes/recipes/what_is_state_dict.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch에서 추론(inference)을 위해 모델 저장하기 & 불러오기
   :card_description: PyTorch에서 추론을 위해 모델을 저장하고 불러오는 두 가지 접근 방식(state_dict 및 전체 모델)을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-models-for-inference.PNG
   :link: ../recipes/recipes/saving_and_loading_models_for_inference.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch에서 일반적인 체크포인트(checkpoint) 저장하기 & 불러오기
   :card_description: 추론 또는 학습을 재개하기 위해 일반적인 체크포인트를 저장하고 불러오는 것은 마지막으로 중단한 부분을 고르는데 도움이 됩니다. 이 레시피에서는 어떻게 여러개의 체크포인트를 저장하고 불러오는지 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-general-checkpoint.PNG
   :link: ../recipes/recipes/saving_and_loading_a_general_checkpoint.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch에서 여러 모델을 하나의 파일에 저장하기 & 불러오기
   :card_description: 이전에 학습했던 여러 모델들을 저장하고 불러와 모델을 재사용하는 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/saving-multiple-models.PNG
   :link: ../recipes/recipes/saving_multiple_models_in_one_file.html
   :tags: Basics

.. customcarditem::
   :header: Warmstarting model using parameters from a different model in PyTorch
   :card_description: Learn how warmstarting the training process by partially loading a model or loading a partial model can help your model converge much faster than training from scratch.
   :image: ../_static/img/thumbnails/cropped/warmstarting-models.PNG
   :link: ../recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch에서 다양한 장치 간 모델을 저장하고 불러오기
   :card_description: PyTorch를 사용하여 다양한 장치(CPU와 GPU) 간의 모델을 저장하고 불러오는 비교적 간단한 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-models-across-devices.PNG
   :link: ../recipes/recipes/save_load_across_devices.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch에서 변화도를 0으로 만들기
   :card_description: 변화도를 언제 0으로 만들어야 하며, 그렇게 하는 것이 모델의 정확도를 높이는 데에 어떻게 도움이 되는지 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/zeroing-out-gradients.PNG
   :link: ../recipes/recipes/zeroing_out_gradients.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch 벤치마크
   :card_description: PyTorch의 벤치마크 모듈을 사용하여 코드의 성능을 측정하고 비교하는 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/benchmark.html
   :tags: Basics

.. customcarditem::
   :header: Timer 빠른 시작
   :card_description: 코드 조각(snippet)의 실행 시간을 측정하고 명령어들을 확인하는 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/timer_quick_start.html
   :tags: Basics

.. customcarditem::
   :header: Pytorch 프로파일러
   :card_description: PyTorch의 프로파일러를 사용하여 연산 시간과 메모리 소비량을 측정하는 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/profiler_recipe.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Profiler with Instrumentation and Tracing Technology API (ITT API) support
   :card_description: Learn how to use PyTorch's profiler with Instrumentation and Tracing Technology API (ITT API) to visualize operators labeling in Intel® VTune™ Profiler GUI
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/profile_with_itt.html
   :tags: Basics

.. Customization

.. customcarditem::
   :header: 사용자 정의 데이터셋, Transforms & Dataloader
   :card_description: PyTorch 데이터셋 API를 이용하여 어떻게 쉽게 사용자 정의 데이터셋과 dataloader를 만드는지 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/custom-datasets-transforms-and-dataloaders.png
   :link: ../recipes/recipes/custom_dataset_transforms_loader.html
   :tags: Data-Customization

.. Interpretability

.. customcarditem::
   :header: Captum을 사용하여 모델 해석하기
   :card_description: Captum을 사용하여 이미지 분류기의 예측을 해당 이미지의 특징(features)에 사용하고 속성(attribution) 결과를 시각화 하는데 사용하는 방법을 학습합니다.
   :image: ../_static/img/thumbnails/cropped/model-interpretability-using-captum.png
   :link: ../recipes/recipes/Captum_Recipe.html
   :tags: Interpretability,Captum

.. customcarditem::
   :header: PyTorch로 TensorBoard 사용하기
   :card_description: PyTorch로 TensorBoard를 사용하는 기본 방법과 TensorBoard UI에서 데이터를 시각화하는 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/tensorboard_scalars.png
   :link: ../recipes/recipes/tensorboard_with_pytorch.html
   :tags: Visualization,TensorBoard

.. Quantization

.. customcarditem::
   :header: Dynamic Quantization
   :card_description:  Apply dynamic quantization to a simple LSTM model.
   :image: ../_static/img/thumbnails/cropped/using-dynamic-post-training-quantization.png
   :link: ../recipes/recipes/dynamic_quantization.html
   :tags: Quantization,Text,Model-Optimization


.. Production Development

.. customcarditem::
   :header: TorchScript로 배포하기
   :card_description: 학습된 모델을 TorchScript 형식으로 내보내는 방법과 TorchScript 모델을 C++로 불러오고 추론하는 방법에 대해 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/torchscript_overview.png
   :link: ../recipes/torchscript_inference.html
   :tags: TorchScript

.. customcarditem::
   :header: Flask로 배포하기
   :card_description: 경량 웹서버 Flask를 사용하여 학습된 PyTorch Model을 Web API로 빠르게 만드는 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/using-flask-create-restful-api.png
   :link: ../recipes/deployment_with_flask.html
   :tags: Production,TorchScript

.. customcarditem::
   :header: PyTorch 모바일 성능 레시피
   :card_description: 모바일(Android와 iOS) 상에서 PyTorch를 사용하기 위한 성능 최적화 레시피 목록들.
   :image: ../_static/img/thumbnails/cropped/mobile.png
   :link: ../recipes/mobile_perf.html
   :tags: Mobile,Model-Optimization

.. customcarditem::
   :header: PyTorch 사전 빌드된 라이브러리를 사용하는 네이티브 Android 애플리케이션 만들기
   :card_description: LibTorch C++ API와 사용자 지정 C++ 연산자를 가지는 TorchScript를 사용해서 Android 애플리케이션을 처음부터 만드는 방법을 알아봅니다.
   :image: ../_static/img/thumbnails/cropped/android.png
   :link: ../recipes/android_native_app_with_custom_op.html
   :tags: Mobile

.. customcarditem::
  :header: Fuse Modules recipe
  :card_description: Learn how to fuse a list of PyTorch modules into a single module to reduce the model size before quantization.
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/fuse.html
  :tags: Mobile

.. customcarditem::
  :header: Quantization for Mobile Recipe
  :card_description: Learn how to reduce the model size and make it run faster without losing much on accuracy.
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/quantization.html
  :tags: Mobile,Quantization

.. customcarditem::
  :header: Script and Optimize for Mobile
  :card_description: Learn how to convert the model to TorchScipt and (optional) optimize it for mobile apps.
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/script_optimized.html
  :tags: Mobile

.. customcarditem::
  :header: Model Preparation for iOS Recipe
  :card_description: Learn how to add the model in an iOS project and use PyTorch pod for iOS.
  :image: ../_static/img/thumbnails/cropped/ios.png
  :link: ../recipes/model_preparation_ios.html
  :tags: Mobile

.. customcarditem::
  :header: Model Preparation for Android Recipe
  :card_description: Learn how to add the model in an Android project and use the PyTorch library for Android.
  :image: ../_static/img/thumbnails/cropped/android.png
  :link: ../recipes/model_preparation_android.html
  :tags: Mobile

.. customcarditem::
   :header: Mobile Interpreter Workflow in Android and iOS
   :card_description: Learn how to use the mobile interpreter on iOS and Andriod devices.
   :image: ../_static/img/thumbnails/cropped/mobile.png
   :link: ../recipes/mobile_interpreter.html
   :tags: Mobile

.. customcarditem::
   :header: Profiling PyTorch RPC-Based Workloads
   :card_description: How to use the PyTorch profiler to profile RPC-based workloads.
   :image: ../_static/img/thumbnails/cropped/profile.png
   :link: ../recipes/distributed_rpc_profiling.html
   :tags: Production

.. Automatic Mixed Precision
.. customcarditem::
   :header: Automatic Mixed Precision
   :card_description: Use torch.cuda.amp to reduce runtime and save memory on NVIDIA GPUs.
   :image: ../_static/img/thumbnails/cropped/amp.png
   :link: ../recipes/recipes/amp_recipe.html
   :tags: Model-Optimization

.. Performance
.. customcarditem::
   :header: Performance Tuning Guide
   :card_description: Tips for achieving optimal performance.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/tuning_guide.html
   :tags: Model-Optimization

.. Intel(R) Extension for PyTorch*

.. customcarditem::
   :header: Intel® Extension for PyTorch*
   :card_description: Introduction of Intel® Extension for PyTorch*
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/intel_extension_for_pytorch.html
   :tags: Model-Optimization

.. Intel(R) Neural Compressor for PyTorch*
.. customcarditem::
   :header: Intel® Neural Compressor for PyTorch
   :card_description: Ease-of-use quantization for PyTorch with Intel® Neural Compressor.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/intel_neural_compressor_for_pytorch.html
   :tags: Quantization,Model-Optimization

.. Distributed Training
.. customcarditem::
   :header: Shard Optimizer States with ZeroRedundancyOptimizer
   :card_description: How to use ZeroRedundancyOptimizer to reduce memory consumption.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/zero_redundancy_optimizer.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Direct Device-to-Device Communication with TensorPipe RPC
   :card_description: How to use RPC with direct GPU-to-GPU communication.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/cuda_rpc.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Distributed Optimizer with TorchScript support
   :card_description: How to enable TorchScript support for Distributed Optimizer.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/distributed_optim_torchscript.html
   :tags: Distributed-Training,TorchScript

.. End of tutorial card section

.. raw:: html

    </div>

    <div class="pagination d-flex justify-content-center"></div>

    </div>

    </div>

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :hidden:

   /recipes/recipes/loading_data_recipe
   /recipes/recipes/defining_a_neural_network
   /recipes/recipes/what_is_state_dict
   /recipes/recipes/saving_and_loading_models_for_inference
   /recipes/recipes/saving_and_loading_a_general_checkpoint
   /recipes/recipes/saving_multiple_models_in_one_file
   /recipes/recipes/warmstarting_model_using_parameters_from_a_different_model
   /recipes/recipes/save_load_across_devices
   /recipes/recipes/zeroing_out_gradients
   /recipes/recipes/profiler_recipe
   /recipes/recipes/profile_with_itt
   /recipes/recipes/Captum_Recipe
   /recipes/recipes/tensorboard_with_pytorch
   /recipes/recipes/dynamic_quantization
   /recipes/recipes/amp_recipe
   /recipes/recipes/tuning_guide
   /recipes/recipes/intel_extension_for_pytorch
   /recipes/torchscript_inference
   /recipes/deployment_with_flask
   /recipes/distributed_rpc_profiling
   /recipes/zero_redundancy_optimizer
   /recipes/cuda_rpc
   /recipes/distributed_optim_torchscript
   /recipes/mobile_interpreter
