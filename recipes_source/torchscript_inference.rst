TorchScript로 배포하기
==========================

이 레시피에서는 다음 내용을 알아봅니다:

-  TorchScript란?
-  학습된 모델을 TorchScript 형식으로 내보내기
-  TorchScript 모델을 C++로 불러오고 추론하기

요구 사항
------------

-  PyTorch 1.5
-  TorchVision 0.6.0
-  libtorch 1.5
-  C++ 컴파일러

3가지 PyTorch 컴포넌트를 설치하는 방법은 `pytorch.org`_에서 확인할 수 있습니다.
C++ 컴파일러는 당신의 플랫폼에 따라 달라집니다. 

TorchScript란?
--------------------

**TorchScript**는 C++ 같은 고성능 환경에서 실행할 수 있는 PyTorch 모델의 중간 표현(``nn.Module``의 하위 클래스)입니다. Python의 고성능 하위 집합이며 모델 연산의 런타임 최적화를 수행하는 **PyTorch JIT Compiler,** 에서 사용됩니다. TorchScript는 PyTorch 모델에서 스케일 추론을 수행할 때 권장되는 모델 형식입니다. 자세한 내용은 `pytorch.org`_에 있는 `Introduction to TorchScript
tutorial`_, `Loading A TorchScript Model in C++ tutorial`_, `full TorchScript documentation`_ 에서 확인하세요.

모델 내보내기
------------------------

사전 학습된 시각 모델을 살펴봅시다. TorchVision의 모든 사전 학습 모델은 TorchScript와 호환됩니다. 

스크립트나 REPL에서 다음의 Python 3 코드를 실행하세요:

.. code:: python3

   import torch
   import torch.nn.functional as F
   import torchvision.models as models

   r18 = models.resnet18(pretrained=True)       # 이제 사전 학습된 모델의 인스턴스가 있습니다. 
   r18_scripted = torch.jit.script(r18)         # *** 여기가 TorchScript로 내보내는 부분입니다. 
   dummy_input = torch.rand(1, 3, 224, 224)     # 빠르게 테스트 해봅니다.

두 모델이 정말 같은지에 대해 정밀 검사를 해보겠습니다. 

::

   unscripted_output = r18(dummy_input)         # 스크립트화 되지 않은 모델의 예측을 얻고...
   scripted_output = r18_scripted(dummy_input)  # ...스크립트화 된 모델도 똑같이 반복합니다.

   unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
   scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

   print('Python model top 5 results:\n  {}'.format(unscripted_top5))
   print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))

두 모델의 결과가 동일함을 확인할 수 있습니다:

::

   Python model top 5 results:
     tensor([[463, 600, 731, 899, 898]])
   TorchScript model top 5 results:
     tensor([[463, 600, 731, 899, 898]])

검사가 끝났으면 모델을 저장합니다:

::

   r18_scripted.save('r18_scripted.pt')

C++로 TorchScript 모델 불러오기
---------------------------------

다음과 같은 C++ 파일을 만들고 파일명을 ``ts-infer.cpp`` 라고 짓습니다.

.. code:: cpp

   #include <torch/script.h>
   #include <torch/nn/functional/activation.h>


   int main(int argc, const char* argv[]) {
       if (argc != 2) {
           std::cerr << "usage: ts-infer <path-to-exported-model>\n";
           return -1;
       }

       std::cout << "Loading model...\n";

       // ScriptModule을 역직렬화(deserialize) 합니다.
       torch::jit::script::Module module;
       try {
           module = torch::jit::load(argv[1]);
       } catch (const c10::Error& e) {
           std::cerr << "Error loading model\n";
           std::cerr << e.msg_without_backtrace();
           return -1;
       }

       std::cout << "Model loaded successfully\n";

       torch::NoGradGuard no_grad; // autograd가 꺼져있는지 확인합니다.
       module.eval(); // dropout과 학습 단의 레이어 및 함수들을 끕니다. 

       // 입력 "이미지"를 생성합니다.
       std::vector<torch::jit::IValue> inputs;
       inputs.push_back(torch::rand({1, 3, 224, 224}));

       // 모델을 실행하고 출력 값을 tensor로 뽑아냅니다.
       at::Tensor output = module.forward(inputs).toTensor();

       namespace F = torch::nn::functional;
       at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
       std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
       at::Tensor top5 = std::get<1>(top5_tensor);

       std::cout << top5[0] << "\n";

       std::cout << "\nDONE\n";
       return 0;
   }

이런 것들을 알아보았습니다:

- 명령 줄에서 지정한 모델 불러오기
- 더미 입력 "이미지" tensor 생성하기
- 입력에 대한 추론 수행하기

또한, 이 코드에는 TorchVision에 대한 종속성이 없다는 것에 유의하세요. 저장된 TorchScript 모델에는 학습 가중치와 연산 그래프가 있으며 다른 것은 필요하지 않습니다.

C++ 추론 엔진 빌드하고 실행하기 
----------------------------------------------

다음과 같은 ``CMakeLists.txt`` 파일을 생성합니다:

::

   cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
   project(custom_ops)

   find_package(Torch REQUIRED)

   add_executable(ts-infer ts-infer.cpp)
   target_link_libraries(ts-infer "${TORCH_LIBRARIES}")
   set_property(TARGET ts-infer PROPERTY CXX_STANDARD 11)

프로그램을 실행합니다:

::

   cmake -DCMAKE_PREFIX_PATH=<path to your libtorch installation>
   make

이제 C++에서 추론을 수행하고 결과를 확인할 수 있습니다.

::

   $ ./ts-infer r18_scripted.pt
   Loading model...
   Model loaded successfully
    418
    845
    111
    892
    644
   [ CPULongType{5} ]

   DONE

중요 참고자료
-------------------

-  `pytorch.org`_ 에서 설치 방법과 추가 문서 및 튜토리얼들을 확인할 수 있습니다. 
-  `Introduction to TorchScript tutorial`_ 에서 더 심화된 TorchScript 기초 설명을 확인할 수 있습니다.
-  `Full TorchScript documentation`_ 에서 전체 TorchScript 언어 및 API를 참조할 수 있습니다.

.. _pytorch.org: https://pytorch.org/
.. _Introduction to TorchScript tutorial: https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html
.. _Full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
.. _Loading A TorchScript Model in C++ tutorial: https://tutorials.pytorch.kr/advanced/cpp_export.html
.. _full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
