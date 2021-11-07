C++에서 TorchScript 모델 로딩하기
=====================================

PyTorch의 이름에서 알 수 있듯이 PyTorch는 Python 프로그래밍 언어를 기본 인터페이스로 하고 있습니다.
Python은 동적성과 신속한 이터레이션이 필요한 상황에 적합하고 선호되는 언어입니다. 하지만 마찬가지로
이러한 Python의 특징들이 Python을 사용하기 적합하지 않게 만드는 상황도 많이 발생합니다. Python을 사용하기
적합하지 않은 대표적인 예로 상용 환경이 있습니다. 상용 환경에서는 짧은 지연시간이 중요하고
배포하는 데에도 많은 제약이 따릅니다. 이로 인해 상용 환경에서는 많은 사람들이 C++를 개발언어로 채택하게
됩니다. 단지 Java, Rust, 또는 Go와 같은 다른 언어들을 바인딩하기 위한 목적일 뿐일지라도 말이죠.
앞으로 이 튜토리얼에서 저희는 어떻게 PyTorch에서 Python으로 작성된 모델들을 Python 의존성이 전혀
없는 C++환경에서도 읽고 실행할 수 있는 방식으로 직렬화할 수 있는지 알아보겠습니다.

단계 1. PyTorch 모델을 TorchScript 모델로 변환하기
-----------------------------------------------------

`Torch Script
<https://pytorch.org/docs/master/jit.html>`_ 는 PyTorch 모델을 Python에서
C++로 변환하는 것을 가능하게 해줍니다. TorchScript는 TorchScript 컴파일러가 이해하고, 컴파일하고,
직렬화할 수 있는 PyTorch 모델의 한 표현방식입니다. 만약 기본적인 "즉시 실행"[역자 주: eager execution]
API를 사용해 작성된 PyTorch 모델이 있다면, 처음으로 해야할 일은 이 모델을 TorchScript 모델로 변환하는
것입니다. 아래에 설명되어있듯이, 대부분의 경우에 이 과정은 매우 간단합니다. 이미 TorchScript 모듈을 가지고 있다면,
이 섹션을 건너뛰어도 좋습니다.

PyTorch 모델을 TorchScript로 변환하는 방법에는 두가지가 있습니다. 첫번째는 트레이싱(tracing)이라는 방법으로
어떤 입력값을 사용하여 모델의 구조를 파악하고 이 입력값의 모델 안에서의 흐름을 통해 모델을 기록하는 방식입니다.
이 방법은 조건문을 많이 사용하지 않는 모델의 경우에 적합합니다. PyTorch 모델을 TorchScript로 변환하는
두번째 방법은 모델에 명시적인 어노테이션(annotation)을 추가하여 TorchScript 컴파일러로
하여금 직접 모델 코드를 분석하고 컴파일하게하는 방식입니다. 이 방식을 사용할 때는 TorchScript 언어
자체에 제약이 있을 수 있습니다.

.. tip::

  위 두 방식에 관련된 정보와 둘 중 어떤 방법을 사용해야할지 등에 대한 가이드는 공식 기술문서인 `Torch Script
  reference <https://pytorch.org/docs/master/jit.html>`_ 에서 확인하실 수 있습니다.

트레이싱(tracing)을 통해 TorchScript로 변환하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch 모델을 트레이싱을 통해 TorchScript로 변환하기 위해서는, 여러분이 구현한 모델의 인스턴스를
예제 입력값과 함께 ``torch.jit.trace`` 함수에 넘겨주어야 합니다. 그러면 이 함수는 ``torch.jit.ScriptModule``
객체를 생성하게 됩니다. 이렇게 생성된 객체에는 모듈의 ``forward`` 메소드의 모델 실행시 런타임을 trace한
결과가 포함되게 됩니다::

  import torch
  import torchvision

  # 모델 인스턴스 생성
  model = torchvision.models.resnet18()

  # 일반적으로 모델의 forward() 메소드에 넘겨주는 입력값
  example = torch.rand(1, 3, 224, 224)

  # torch.jit.trace를 사용하여 트레이싱을 이용해 torch.jit.ScriptModule 생성
  traced_script_module = torch.jit.trace(model, example)

이렇게 trace된 ``ScriptModule`` 은 일반적인 PyTorch 모듈과 같은 방식으로 입력값을 받아
처리할 수 있습니다::

  In[1]: output = traced_script_module(torch.ones(1, 3, 224, 224))
  In[2]: output[0, :5]
  Out[2]: tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)

어노테이션(annotation)을 통해 TorchScript로 변환하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

특정한 환경(가령 모델이 어떤 제어흐름을 사용하고 있는 경우)에서는 여러분의 모델을 어노테이트(annotate)하여
TorchScript로 바로 작성하는 것이 바람직한 경우가 있습니다. 예를 들어, 아래와 같은 PyTorch 모델이
있다고 가정하겠습니다::

  import torch

  class MyModule(torch.nn.Module):
      def __init__(self, N, M):
          super(MyModule, self).__init__()
          self.weight = torch.nn.Parameter(torch.rand(N, M))

      def forward(self, input):
          if input.sum() > 0:
            output = self.weight.mv(input)
          else:
            output = self.weight + input
          return output


이 모듈의 ``forward`` 메소드는 입력값에 영향을 받는 제어흐름을 사용하고 있기 때문에, 이 모듈은
트레이싱에는 적합하지 않습니다. 대신 우리는 이 모듈을 ``ScriptModule`` 로 변환할 수 있습니다.
모듈을 ``ScriptModule`` 로 변환하기 위해서는, 아래와 같이 ``torch.jit.script`` 함수를 사용해
모듈을 컴파일해야 합니다::


    class MyModule(torch.nn.Module):
        def __init__(self, N, M):
            super(MyModule, self).__init__()
            self.weight = torch.nn.Parameter(torch.rand(N, M))

        def forward(self, input):
            if input.sum() > 0:
              output = self.weight.mv(input)
            else:
              output = self.weight + input
            return output

    my_module = MyModule(10,20)
    sm = torch.jit.script(my_module)

아직 TorchScript에서 지원하지 않는 Python 기능을 사용하고 있는 메소드들을 여러분의 ``nn.Module``
에서 제외하고 싶다면, 그 메소드들을 ``@torch.jit.ignore`` 로 어노테이트하면 됩니다.

``sm`` 은 직렬화(serialization) 준비가 된 ``ScriptModule`` 의 인스턴스입니다.

단계 2. Script 모듈을 파일로 직렬화하기
-------------------------------------------------

모델을 트레이싱이나 어노테이팅을 통해 ``ScriptModule`` 로 변환하였다면, 이제 그것을 파일로 직렬화할
수도 있습니다. 나중에 C++를 이용해 파일로부터 모듈을 읽어올 수 있고 Python에 어떤 의존성도 없이
그 모듈을 실행할 수 있습니다. 예를 들어 트레이싱 예시에서 들었던 ``ResNet18`` 모델을
직렬화하고 싶다고 가정합시다. 직렬화를 하기 위해서는, `save <https://pytorch.org/docs/master/jit.html#torch.jit.ScriptModule.save>`_
함수를 호출하고 모듈과 파일명만 넘겨주면 됩니다::

  traced_script_module.save("traced_resnet_model.pt")

이 함수는 ``traced_resnet_model.pt`` 파일을 작업 디렉토리에 생성할 것입니다. 만약 어노테이션 예시의
``sm`` 을 직렬화하고 싶다면, ``sm.save("my_module_model.pt")`` 를
호출하면 됩니다. 이로써 우리는 이제 Python의 세계에서 벗어나 C++ 환경에서 작업할 준비를 마쳤습니다.

단계 3. C++에서 Script 모듈 로딩하기
------------------------------------------

직렬화된 PyTorch 모델을 C++에서 로드하기 위해서는, 어플리케이션이 반드시 *LibTorch* 라고 불리는
PyTorch C++ API를 사용해야합니다. LibTorch는 여러 공유 라이브러리들, 헤더 파일들, 그리고 CMake
빌드 설정파일들을 포함하고 있습니다. CMake는 LibTorch를 쓰기위한 필수 요구사항은 아니지만, 권장되는
방식이고 향후에도 계속 지원될 예정입니다. 이 튜토리얼에서는 CMake와 LibTorch를 사용하여 직렬화된
PyTorch 모델을 읽고 실행하는 아주 간단한 C++ 어플리케이션을 만들어보도록 하겠습니다.

간단한 C++ 어플리케이션
^^^^^^^^^^^^^^^^^^^^^^^^^

우선 모듈을 로드하는 코드에 대해 살펴보도록 하겠습니다. 아래의 간단한 코드로 모듈을 쉽게 읽어올 수 있습니다:

.. code-block:: cpp

    #include <torch/script.h> // 필요한 단 하나의 헤더파일.

    #include <iostream>
    #include <memory>

    int main(int argc, const char* argv[]) {
      if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
      }


      torch::jit::script::Module module;
      try {
        // torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
        module = torch::jit::load(argv[1]);
      }
      catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
      }

      std::cout << "ok\n";
    }


``<torch/script.h>`` 헤더는 예시를 실행하기 위한 모든 LibTorch 라이브러리를 포함하고 있습니다.
우리의 어플리케이션은 직렬화된 PyTorch ``ScriptModule`` 의 경로를 유일한 명령행 인자로 입력받고
이 파일경로를 인자로 받는 ``torch::jit::load()`` 를 사용해 모듈을 역직렬화합니다. 그 결과로
``torch::jit::script::Module`` 를 돌려받습니다. 이 리턴받은 모듈을 어떻게 사용하는지에 대해서는 곧 살펴보겠습니다.

LibTorch 사용 및 어플리케이션 빌드 방법
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

위의 코드를 ``example-app.cpp`` 이라는 파일에 저장하였다고 가정합니다. 위 코드를 빌드하기 위한
간단한 ``CMakeLists.txt`` 입니다:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(custom_ops)

  find_package(Torch REQUIRED)

  add_executable(example-app example-app.cpp)
  target_link_libraries(example-app "${TORCH_LIBRARIES}")
  set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

예시 어플리케이션을 빌드하기 위해 마지막으로 필요한 것은 LibTorch 배포판입니다. 언제나 가장 최신의
안정 버전을 PyTorch 웹사이트의 `download
page <https://pytorch.org/>`_ 로부터 받으실 수 있습니다. 가장 최신 버전을 다운로드 받아 압축을
푸시면, 아래와 같은 디렉토리 구조의 폴더를 확인하실 수 있습니다:

.. code-block:: sh

  libtorch/
    bin/
    include/
    lib/
    share/

- ``lib/`` 폴더는 링크해야할 공유 라이브러리를 포함하고 있습니다.
- ``include/`` 폴더는 여러분의 프로그램이 include해야할 헤더파일들을 담고 있습니다.
- ``share/`` 폴더는 위에서 실행한 간단한 명령어인 ``find_package(Torch)`` 를 실행하게해주는 CMake 설정을 담고있습니다.

.. tip::
  윈도우에서는 디버그 빌드와 릴리즈 빌드가 ABI-compatible하지 않습니다. 만약 프로젝트를
  debug 모드에서 빌드하고 싶다면, LibTorch의 debug 버전을 사용해야합니다. 그리고 `cmake --build .``
  에 알맞은 설정을 명시해주어야 합니다.

마지막 단계는 어플리케이션을 빌드하는 것입니다. 이를 위해서 디렉토리 구조가 아래와 같이 같다고
가정하겠습니다.

.. code-block:: sh

  example-app/
    CMakeLists.txt
    example-app.cpp

이제 아래 명령어들을 사용해 ``example-app/`` 폴더 안에서 어플리케이션을 빌드할 수 있습니다.

.. code-block:: sh

  mkdir build
  cd build
  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
  cmake --build . --config Release

여기서 ``/path/to/libtorch`` 는 LibTorch 배포판의 압축을 푼 전체 경로입니다. 모든 것이 잘 되었다면,
아래와 같은 것이 나타날 것입니다:

.. code-block:: sh

  root@4b5a67132e81:/example-app# mkdir build
  root@4b5a67132e81:/example-app# cd build
  root@4b5a67132e81:/example-app/build# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
  -- The C compiler identification is GNU 5.4.0
  -- The CXX compiler identification is GNU 5.4.0
  -- Check for working C compiler: /usr/bin/cc
  -- Check for working C compiler: /usr/bin/cc -- works
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Check for working CXX compiler: /usr/bin/c++
  -- Check for working CXX compiler: /usr/bin/c++ -- works
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Looking for pthread.h
  -- Looking for pthread.h - found
  -- Looking for pthread_create
  -- Looking for pthread_create - not found
  -- Looking for pthread_create in pthreads
  -- Looking for pthread_create in pthreads - not found
  -- Looking for pthread_create in pthread
  -- Looking for pthread_create in pthread - found
  -- Found Threads: TRUE
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /example-app/build
  root@4b5a67132e81:/example-app/build# make
  Scanning dependencies of target example-app
  [ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
  [100%] Linking CXX executable example-app
  [100%] Built target example-app

이제 trace된 ``ResNet18`` 모델인 ``traced_resnet_model.pt`` 경로를 ``example-app`` 바이너리에
입력했다면, 우리는 "ok" 메시지를 확인할 수 있을 것입니다. 만약이 예제에 ``my_module_model.pt`` 를
인자로 넘겼다면, 입력값이 호환되지 않는 모양이라는 에러메시지가 출력됩니다. ``my_module_model.pt`` 는
4D가 아닌 1D 텐서를 받도록 되어있기 때문입니다.

.. code-block:: sh

  root@4b5a67132e81:/example-app/build# ./example-app <path_to_model>/traced_resnet_model.pt
  ok

단계 4. Script 모듈을 C++에서 실행하기
------------------------------------------

``ResNet18`` 을 C++에서 성공적으로 로딩한 뒤, 이제 몇 줄의 코드만 더 추가하면 모듈을 실행할 수 있습니다.
C++ 어플리케이션의 ``main()`` 함수에 아래의 코드를 추가하겠습니다.

.. code-block:: cpp

    // 입력값 벡터를 생성합니다.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // 모델을 실행한 뒤 리턴값을 텐서로 변환합니다.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

첫 두줄은 모델의 입력값을 생성합니다. ``torch::jit::IValue`` (``script::Module`` 메소드들이
입력받고 또 리턴할 수 있는 타입이 소거된 자료형)의 벡터를 만들고 그 벡터에 하나의 입력값을 추가합니다.
입력값 텐서를 만들기 위해서 우리는 ``torch::ones()`` 을 사용합니다. 이 함수는 ``torch.ones`` 의 C++ API 버전입니다.
이제 ``script::Module`` 의 ``forward`` 메소드에 입력값 벡터를 넘겨주어 실행하면, 우리는 새로운
``IValue`` 를 리턴받게되고, 이 값을 ``toTensor()`` 를 통해 텐서로 변환할 수 있습니다.

.. tip::

  ``torch::ones`` 를 비롯한 PyTorch C++ API에 대해 더 알고 싶다면 https://pytorch.org/cppdocs에 있는
  문서를 참고하시면 됩니다. PyTorch C++ API는 Python API와 거의 동일한 기능을 제공하여 사용자들이
  텐서를 다루고 사용하는 것을 Python과 동일하게 할 수 있도록 합니다.

마지막 줄에서 출력값의 첫 다섯 값들을 프린트합니다. 이번 튜토리얼의 앞부분에서 Python 모델에 동일한
입력값을 넘겨주었기 때문에, 이 부분에서도 출력값은 같을 것이라고 예상할 수 있습니다. 그럼 어플리케이션을
다시 컴파일하고 같은 직렬화된 모델에 대해 실행해보겠습니다:

.. code-block:: sh

  root@4b5a67132e81:/example-app/build# make
  Scanning dependencies of target example-app
  [ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
  [100%] Linking CXX executable example-app
  [100%] Built target example-app
  root@4b5a67132e81:/example-app/build# ./example-app traced_resnet_model.pt
  -0.2698 -0.0381  0.4023 -0.3010 -0.0448
  [ Variable[CPUFloatType]{1,5} ]


참고로, 이전의 Python에서의 출력값은 아래와 같았습니다::

  tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)

두 출력값이 일치하는 걸 확인하실 수 있습니다!

.. tip::

  모델을 GPU 메모리에 올리기 위해서는, ``model.to(at::kCUDA);`` 를 사용하면 됩니다.
  모델에 넘겨주는 입력값들에 대해서도 ``tensor.to(at::kCUDA)`` 를 통해 CUDA 메모리에 올린 뒤
  사용해야합니다. ``tensor.to(at::kCUDA)`` 는 CUDA 메모리에 있는 새로운 텐서를 리턴합니다.

단계 5. API 더 알아보기
------------------------------------------

이 튜토리얼이 PyTorch 모델을 Python에서부터 C++로 변환하는 과정을 이해하는데 도움이 되었길 바랍니다.
본 튜토리얼에서 다룬 개념들로, 여러분은 이제 "즉시 실행" 버전의 PyTorch 모델에서부터 Python에서 컴파일된 ``ScriptModule`` 로,
더 나아가 디스크 상의 직렬화된 파일로, 그리고 마지막으로 C++에서 실행가능한 ``script::Module`` 까지 만들
수 있게 되었습니다.

물론 이 튜토리얼에서 다루지못한 개념들도 많습니다. 예를 들어 여러분의 ``ScriptModule`` 이 C++나 CUDA로
정의된 커스텀 연산자를 사용할 수 있게하는 방법 또는 이러한 커스텀 연산자를 C++ 상용 환경의 ``ScriptModule`` 에서
사용할 수 있게하는 방법에 대해서는 본 튜토리얼에서 다루지 않았습니다. 좋은 소식은 이러한 것들이 가능하다는 것이고 지원되고
있다는 점입니다! 저희가 곧 이것에 관한 튜토리얼을 업로드할 때까지 `이 폴더 <https://github.com/pytorch/pytorch/tree/master/test/custom_operator>`_
를 예시로 삼아 참고하시면 되겠습니다. 또 아래 링크들이 도움이 될 것입니다:

- The Torch Script reference: https://pytorch.org/docs/master/jit.html
- The PyTorch C++ API documentation: https://pytorch.org/cppdocs/
- The PyTorch Python API documentation: https://pytorch.org/docs/

언제나 그렇듯이, 문제를 맞닥뜨리시거나 질문이 있으시면 저희 `forum <https://discuss.pytorch.org/>`_ 또는
`GitHub issues
<https://github.com/pytorch/pytorch/issues>`_ 에 올려주시면 되겠습니다.
