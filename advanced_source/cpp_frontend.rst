PyTorch C++ 프론트엔드 사용하기
=============================

PyTorch C++ 프론트엔드는 PyTorch 머신러닝 프레임워크의 순수 C++ 인터페이스입니다.
PyTorch의 주된 인터페이스는 물론 파이썬이지만 이 파이썬 API는 텐서(tensor)나 자동
미분과 같은 기초적인 자료구조 및 기능을 제공하는 C++ 코드베이스 위에 구현돼있습니다.
C++ 프론트엔드는 이 기초 C++ 코드베이스를 비롯해 머신러닝 학습과 추론을 위해 필요한
도구들을 상속하는 순수 C++11 API를 노출합니다. 여기에는 신경망 모델링을 위해 필요한
공동의 빌트인 컴포넌트의 모음, 그것을 상속하기 위한 커스텀 모듈, 경사 하강법과 같은
유명한 최적화 알고리즘 라이브러리, 병렬 데이터 로더 및 데이터셋을 정의하고 불러오기
위한 API, 직렬화 루틴 등이 포합됩니다.

이 튜토리얼은 C++ 프론트엔드로 모델을 학습하는 엔드 투 엔드 예제를 안내합니다.
구체적으로, 우리는 생성 모델 중 하나인 `DCGAN
<https://arxiv.org/abs/1511.06434>`_
을 학습시켜 MNIST 숫자 이미지들을 생성할
것입니다. 개념적으로 쉬운 예시이지만, 여러분이 PyTorch C++ 프론트엔드에 대한 대략적인
개요를 파악하고 더 복잡한 모델을 학습시키고싶은 욕구를 불러일으키기에 충분할 것입니다.
먼저 C++ 프론트엔드를 사용에 대한 동기 부여가 될 만한 논의로 시작하고, 곧바로 모델을
정의하고 학습해보도록 하겠습니다.

.. tip::

  C++ 프론트엔드에 대한 짧고 재미있는 발표를 보려면 `이 CppCon 2018 라이트닝 토크
  <https://www.youtube.com/watch?v=auRPXMMHJzc>`_ 를 시청하세요


.. tip::

  `이 노트 <https://pytorch.org/cppdocs/frontend.html>`_ 는 C++ 프론트엔드의 컴포넌트와
  디자인 철학의 전면적인 개요를 제공합니다.

.. tip::

  PyTorch C++ 생태계에 대한 문서는 https://pytorch.org/cppdocs에서 확인할 수 있습니다.
  API 레벨의 문서뿐만 아니라 개괄적인 설명도 찾을 수 있을 것입니다.

동기 부여를 위해
---------------

GAN과 MNIST 숫자로의 설레는 여정을 시작하기에 앞서, 먼저 파이썬 대신 C++ 프론트엔드를
사용하는 이유에 대해 한 발 물러서서 설명하겠습니다. 우리(PyTorch 팀)는 파이썬을 사용할
수 없거나 사용하기에 적합하지 않은 환경에서 연구를 가능하게 하기 위해 C++ 프론트엔드를
만들었습니다. 예를 들면 다음과 같습니다.

- **저지연 시스템**: 초당 프레임 수가 높고 지연 시간이 짧은 순수 C++ 게임 엔진에서
  강화 학습 연구를 수행할 수 있습니다. 그러한 환경에서는 파이썬 라이브러리보다 순수 C++
  라이브러리를 사용하는 것이 훨씬 더 적합합니다. 파이썬은 느린 인터프리터 때문에 매우
- **극심한 멀티쓰레딩 환경**: 글로벌 인터프리터 락(GIL)으로 인해 파이썬은 동시에 둘
  이상의 시스템 쓰레드를 실행할 수 없습니다. 대안으로 멀티프로세싱을 사용하면 확장성이
  떨어지며 심각한 한계가 있습니다. C++는 이러한 제약 조건이 없으며 쓰레드를 쉽게 만들고
  사용할 수 있습니다. `Deep Neuroevolution <https://eng.uber.com/deep-neuroevolution/>`_에 사용된 것과 같이 고도의 병렬화가
  필요한 모델도 이를 활용할 수 있습니다.
- **기존의 C++ 코드베이스**: 백엔드 서버의 웹 페이지 서비스부터 사진 편집 소프트웨어의
  3D 그래픽 렌더링에 이르기까지 모든 작업을 수행하는 기존 C++ 애플리케이션의 소유자가
  머신러닝 방법론을 시스템에 통합하고자 합니다. C++ 프론트엔드는 PyTorch (파이썬) 경험
  본연의 높은 유연성과 직관성을 유지하면서, 파이썬과 C++를 앞뒤로 바인딩하는 번거로움 없이
  C++를 사용할 수 있게 해줍니다.

C++ 프론트엔드의 목적은 파이썬 프론트엔드와 경쟁하는 것이 아닌 보완하는 것입니다. 연구자와
엔지니어 모두가 PyTorch의 단순성, 유연성 및 직관적인 API를 매우 좋아합니다. 우리의 목표는
여러분이 위의 예시를 비롯한 모든 가능한 환경에서 이 핵심 디자인 원칙을 이용할 수 있도록 하는
것입니다. 이러한 시나리오 중 하나가 여러분의 사례에 해당하거나 단순히 관심이 있거나 궁금하다면
아래 내용을 통해 C++ 프론트엔드에 대해 자세히 살펴보세요.

.. tip::

	C++ 프론트엔드는 파이썬 프론트엔드와 최대한 유사한 API를 제공하고자 합니다. 만일 파이썬
  프론트엔드에 익숙한 사람이 "C++ 프론트엔드로 X를 어떻게 해야 하는가?" 의문을 갖는다면, 많은
  경우에 파이썬에서와 같은 방식으로 코드를 작성해 파이썬에서와 동일한 함수와 메서드를 사용할 수
  있을 것입니다. (다만, 온점을 더블 콜론으로 바꾸는 것에 유의하세요.)

기본 애플리케이션 작성
--------------------

먼저 최소한의 C++ 애플리케이션을 작성해 우리의 설정 및 빌드 환경이 동일한지 확인하겠습니다.
먼저, C++ 프론트엔드를 사용하는 데 필요한 모든 관련 헤더, 라이브러리 및 CMake 빌드 파일을
패키징하는 *LibTorch* 배포판의 사본이 필요합니다. 리눅스, 맥OS, 윈도우용 LibTorch 배포판은
`PyTorch website <https://pytorch.org/get-started/locally/>`_ 에서 다운로드할 수 있습니다. 이 튜토리얼의 나머지 부분은 기본 우분투 리눅스
환경을 가정하지만 맥OS나 윈도우를 사용하셔도 괜찮습니다.

.. tip::

  `PyTorch C++ 배포판 설치 <https://pytorch.org/cppdocs/installing.html>`_ 의 설명에 다음의 과정이 더 자세히 안내되어 있습니다.

.. tip::

  윈도우에서는 디버그 및 릴리스 빌드가 ABI와 호환되지 않습니다. 프로젝트를 디버그 모드로 빌드하려면
  LibTorch의 디버그 버전을 사용해보세요. 아래의 ``cmake --build .`` 에 올바른 설정을 지정하는 것도
  잊지 마세요.

가장 먼저 할 것은 PyTorch 웹사이트에서 검색된 링크를 통해 LibTorch 배포판을 로컬에 다운로드하는
것입니다. 바닐라 Ubuntu Linux 환경의 경우 다음 명령어를 실행합니다.

.. code-block:: shell

  # If you need e.g. CUDA 9.0 support, please replace "cpu" with "cu90" in the URL below.
  wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  unzip libtorch-shared-with-deps-latest.zip

다음으로 ``torch/torch.h`` 를 호출하는 ``dcgan.cpp`` 라는 이름의 C++ 파일 하나를 작성합시다. 우선은
아래와 같이 3x3 항등 행렬을 출력하기만 하면 됩니다:

.. code-block:: cpp

  #include <torch/torch.h>
  #include <iostream>

  int main() {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
  }

이 작은 애플리케이션과 이후 완성할 학습용 스크립트를 빌드하기 위해 우리는 아래의 ``CMakeLists.txt`` 를
사용할 것입니다:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(dcgan)

  find_package(Torch REQUIRED)

  add_executable(dcgan dcgan.cpp)
  target_link_libraries(dcgan "${TORCH_LIBRARIES}")
  set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)

.. note::

  CMake는 LibTorch에 권장되는 빌드 시스템이지만 필수 요구 사항은 아닙니다. Visual Studio 프로젝트 파일,
  QMake, 일반 Make 파일 등 다른 빌드 환경을 사용해도 됩니다. 하지만 이에 대한 즉각적인 지원은 제공하지
  않습니다.

위 CMake 파일 4번째 줄의 ``find_package(Torch REQUIRED)`` 는 CMake가 LibTorch 라이브러리 빌드 설정을
찾도록 안내합니다. CMake가 해당 파일의 *위치*를 찾을 수 있도록 하려면 ``cmake`` 호출 시 ``CMAKE_PREFIX_PATH`` 를
설정해야 합니다. 이에 앞서 ``dcgan`` 애플리케이션에 대해 다음의 디렉터리 구조를 다음과 같이 통일하도록
하겠습니다:

.. code-block:: shell

  dcgan/
    CMakeLists.txt
    dcgan.cpp

또한 앞으로 압축 해제된 LibTorch 배포판의 경로를 ``/path/to/libtorch`` 로 부르도록 하겠습니다. 이는 **반드시**
**절대 경로여야** 합니다. 특히 ``CMAKE_PREFIX_PATH`` 를 ``../../libtorch`` 와 같이 설정하면 예상치 못한
오류가 발생할 수 있습니다. 그보다는 해당 절대 경로를 가져오기 위해 "$PWD/../../libtorch"를 입력하세요.
이제 애플리케이션을 빌드할 준비가 되었습니다.

.. code-block:: shell

  root@fa350df05ecf:/home# mkdir build
  root@fa350df05ecf:/home# cd build
  root@fa350df05ecf:/home/build# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
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
  -- Found torch: /path/to/libtorch/lib/libtorch.so
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /home/build
  root@fa350df05ecf:/home/build# cmake --build . --config Release
  Scanning dependencies of target dcgan
  [ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
  [100%] Linking CXX executable dcgan
  [100%] Built target dcgan

위에서 우리는 먼저 ``dcgan`` 디렉터리 안에 ``build`` 폴더를 만들고 이 폴더에 들어가서 필요한 빌드(Make) 파일을
생성하는 ``cmake`` 명령어를 실행한 후 ``cmake --build . --config Release`` 를 실행하여 프로젝트를 성공적으로
컴파일했습니다. 이제 우리의 작은 바이너리를 실행하고 기본 프로젝트 설정에 대한 이 섹션을 완료할 준비가 됐습니다.

.. code-block:: shell

  root@fa350df05ecf:/home/build# ./dcgan
  1  0  0
  0  1  0
  0  0  1
  [ Variable[CPUFloatType]{3,3} ]

제가 보기엔 항등 행렬인 것 같군요!

신경망 모델 정의하기
-------------------

이제 기본적인 환경을 설정했으니, 이번 튜토리얼을 통해 훨씬 더 흥미로운 부분을 살펴봅시다. 먼저 C++ 프론트엔드에서
모듈을 정의하고 상호 작용하는 방법에 대해 논의하겠습니다. 기본적인 소규모 예제 모듈부터 시작하여 C++ 프론트엔드가
제공하는 다양한 내장 모듈 라이브러리를 사용하여 완성도 있는 GAN을 구현하겠습니다.

모듈 API 기본기
^^^^^^^^^^^^^^

파이썬 인터페이스와 마찬가지로, C++ 프론트엔드에 기반을 둔 신경망도 *모듈*이라 불리는 재사용 가능한 빌딩 블록으로
구성되어 있습니다. 파이썬에 다른 모든 모듈이 파생되는 ``torch.nn.Module`` 라는 기본 모듈 클래스가 있듯이 C++에는
``torch::nn::Module`` 클래스가 있습니다. 모듈에는 캡슐화하는 알고리즘을 구현하는 ``forward()`` 메서드를 비롯해
일반적으로 매개 변수, 버퍼 및 하위 모듈의 세 가지 하위 개체가 포함됩니다.

매개 변수와 버퍼는 텐서의 형태로 상태를 저장합니다. 매개 변수는 그래디언트를 기록하지만 버퍼는 기록하지 않습니다.
매개 변수는 일반적으로 신경망의 학습 가능한 가중치입니다. 버퍼의 예로는 배치 정규화를 위한 평균 및 분산이 있습니다.
특정 논리 및 상태 블록을 재사용하기 위해, PyTorch API는 모듈들이 중첩되는 것을 허용합니다. 중첩된 모듈은 *하위*
*모듈*이라고 합니다.

매개 변수, 버퍼 및 하위 모듈은 명시적으로 등록(register)을 해야 합니다. 등록이 되면 ``parameters()`` 나 ``buffers()``
같은 메서드를 사용하여 (중첩을 포함한) 전체 모듈 계층 구조에서 모든 매개 변수 묶음을 검색할 수 있습니다. 마찬가지로,
``to(...)`` 와 같은 메서드는 모듈 계층 구조 전체에 대한 메서드입니다. 예를 들어, ``to(torch::kCUDA)`` 는 모든 매개
변수와 버퍼를 CPU에서 CUDA 메모리로 이동시킵니다.

모듈 정의 및 매개변수 등록
*************************

이 개념들을 코드로 구현하기 위해, 파이썬 인터페이스로 작성된 아래 모듈을 살펴봅시다.

.. code-block:: python

  import torch

  class Net(torch.nn.Module):
    def __init__(self, N, M):
      super(Net, self).__init__()
      self.W = torch.nn.Parameter(torch.randn(N, M))
      self.b = torch.nn.Parameter(torch.randn(M))

    def forward(self, input):
      return torch.addmm(self.b, input, self.W)


이를 C++로 작성하면 다음과 같습니다.

.. code-block:: cpp

  #include <torch/torch.h>

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M) {
      W = register_parameter("W", torch::randn({N, M}));
      b = register_parameter("b", torch::randn(M));
    }
    torch::Tensor forward(torch::Tensor input) {
      return torch::addmm(b, input, W);
    }
    torch::Tensor W, b;
  };

파이썬에서와 마찬가지로 모듈 기본 클래스에서 파생한 ``Net`` 이라는 클래스를 정의합니다. (쉬운 설명을 위해 ``class``
대신 ``struct``을 사용했습니다.) 파이썬에서 torch.randn을 사용하는 것처럼 생성자에서는 ``torch::randn`` 을 사용해
텐서를 만듭니다. 한 가지 흥미로운 차이점은 매개변수를 등록하는 방법입니다. 파이썬에서는 텐서를 ``torch.nn``으로
감싸는 것과 달리, C++에서는 ``register_parameter`` 메서드를 통해 텐서를 전달해야 합니다. 이러한 차이의 원인은 파이썬
API의 경우, 어떤 속성(attirbute)이 ''torch.nn.Parameter`` 타입인지 감지해 그러한 텐서를 자동으로 등록할 수 있기
때문에 나타납니다. C++에서는 리플렉션(reflection)이 매우 제한적이므로 보다 전통적인 (그리하여 덜 마법적인) 방식이
제공됩니다.

서브모듈 등록 및 모듈 계층 구조 탐색
**********************************

매개 변수 등록과 마찬가지 방법으로 서브모듈을 등록할 수 있습니다. 파이썬에서 서브모듈은 어떤 모듈의 속성으로 지정될 때
자동으로 감지되고 등록됩니다.

.. code-block:: python

  class Net(torch.nn.Module):
    def __init__(self, N, M):
        super(Net, self).__init__()
        # Registered as a submodule behind the scenes
        self.linear = torch.nn.Linear(N, M)
        self.another_bias = torch.nn.Parameter(torch.rand(M))

    def forward(self, input):
      return self.linear(input) + self.another_bias

예를 들어, ``parameters()`` 메서드를 사용하면 모듈 계층의 모든 매개 변수에 재귀적으로 액세스할 수 있습니다.

.. code-block:: python

  >>> net = Net(4, 5)
  >>> print(list(net.parameters()))
  [Parameter containing:
  tensor([0.0808, 0.8613, 0.2017, 0.5206, 0.5353], requires_grad=True), Parameter containing:
  tensor([[-0.3740, -0.0976, -0.4786, -0.4928],
          [-0.1434,  0.4713,  0.1735, -0.3293],
          [-0.3467, -0.3858,  0.1980,  0.1986],
          [-0.1975,  0.4278, -0.1831, -0.2709],
          [ 0.3730,  0.4307,  0.3236, -0.0629]], requires_grad=True), Parameter containing:
  tensor([ 0.2038,  0.4638, -0.2023,  0.1230, -0.0516], requires_grad=True)]

C++에서 ``torch::nn::Linear`` 등의 모듈을 서브모듈로 등록하려면 이름이 말해주듯이 ``register_module()`` 메서드를
사용합니다.

.. code-block:: cpp

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M)
        : linear(register_module("linear", torch::nn::Linear(N, M))) {
      another_bias = register_parameter("b", torch::randn(M));
    }
    torch::Tensor forward(torch::Tensor input) {
      return linear(input) + another_bias;
    }
    torch::nn::Linear linear;
    torch::Tensor another_bias;
  };

.. tip::

  ``torch::nn``에 대한 `이 문서 <https://pytorch.org/cppdocs/api/namespace_torch__nn.html>`_ 에서 ``torch::nn::Linear``, ``torch::nn::Dropout``, ``torch::nn::Conv2d`` 등 사용 가능한
  전체 빌트인 모듈 목록을 확인할 수 있습니다.

위 코드에서 한 가지 미묘한 사실은 서브모듈은 생성자의 이니셜라이저 목록에 작성되고 매개 변수는 생성자의 바디(body)에
작성되었다는 것입니다. 여기에는 충분한 이유가 있으며 아래 C++ 프론트엔드의 *오너십 모델* 섹션에서 더 다룰 예정입니다.
그렇지만 최종 결론은 파이썬에서처럼 모듈 트리의 매개 변수에 재귀적으로 액세스할 수 있다는 것입니다. ``parameters()``를
호출하면 순회가 가능한 ``std::vector<torch::Tensor>``가 반환됩니다.

.. code-block:: cpp

  int main() {
    Net net(4, 5);
    for (const auto& p : net.parameters()) {
      std::cout << p << std::endl;
    }
  }

이를 실행한 결과는 다음과 같습니다.

.. code-block:: shell

  root@fa350df05ecf:/home/build# ./dcgan
  0.0345
  1.4456
  -0.6313
  -0.3585
  -0.4008
  [ Variable[CPUFloatType]{5} ]
  -0.1647  0.2891  0.0527 -0.0354
  0.3084  0.2025  0.0343  0.1824
  -0.4630 -0.2862  0.2500 -0.0420
  0.3679 -0.1482 -0.0460  0.1967
  0.2132 -0.1992  0.4257  0.0739
  [ Variable[CPUFloatType]{5,4} ]
  0.01 *
  3.6861
  -10.1166
  -45.0333
  7.9983
  -20.0705
  [ Variable[CPUFloatType]{5} ]

파이썬에서처럼 세 개의 매개변수가 출력됐습니다. 이 매개 변수들의 이름을 확인할 수 있도록 C++ API는 ``named_parameters()``
메서드를 제공하며, 이는 파이썬에서와 같이 ``Orderdict``를 반환합니다.

.. code-block:: cpp

  Net net(4, 5);
  for (const auto& pair : net.named_parameters()) {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }

마찬가지로 코드를 실행하면 결과는 아래와 같습니다.

.. code-block:: shell

  root@fa350df05ecf:/home/build# make && ./dcgan                                                                                                                                            11:13:48
  Scanning dependencies of target dcgan
  [ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
  [100%] Linking CXX executable dcgan
  [100%] Built target dcgan
  b: -0.1863
  -0.8611
  -0.1228
  1.3269
  0.9858
  [ Variable[CPUFloatType]{5} ]
  linear.weight:  0.0339  0.2484  0.2035 -0.2103
  -0.0715 -0.2975 -0.4350 -0.1878
  -0.3616  0.1050 -0.4982  0.0335
  -0.1605  0.4963  0.4099 -0.2883
  0.1818 -0.3447 -0.1501 -0.0215
  [ Variable[CPUFloatType]{5,4} ]
  linear.bias: -0.0250
  0.0408
  0.3756
  -0.2149
  -0.3636
  [ Variable[CPUFloatType]{5} ]

.. note::

  ``torch::nn::Module``에 대한 `문서 <https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#exhale-class-classtorch-1-1nn-1-1-module>`_ 는 모듈 계층 구조에 대한 메서드 목록 전체가 포함되어 있습니다.

Running the Network in Forward Mode
***********************************

To execute the network in C++, we simply call the ``forward()`` method we
defined ourselves:

.. code-block:: cpp

  int main() {
    Net net(4, 5);
    std::cout << net.forward(torch::ones({2, 4})) << std::endl;
  }

which prints something like:

.. code-block:: shell

  root@fa350df05ecf:/home/build# ./dcgan
  0.8559  1.1572  2.1069 -0.1247  0.8060
  0.8559  1.1572  2.1069 -0.1247  0.8060
  [ Variable[CPUFloatType]{2,5} ]

Module Ownership
****************

At this point, we know how to define a module in C++, register parameters,
register submodules, traverse the module hierarchy via methods like
``parameters()`` and finally run the module's ``forward()`` method. While there
are many more methods, classes and topics to devour in the C++ API, I will refer
you to `docs <https://pytorch.org/cppdocs/api/namespace_torch__nn.html>`_ for
the full menu. We'll also touch upon some more concepts as we implement the
DCGAN model and end-to-end training pipeline in just a second. Before we do so,
let me briefly touch upon the *ownership model* the C++ frontend provides for
subclasses of ``torch::nn::Module``.

For this discussion, the ownership model refers to the way modules are stored
and passed around -- which determines who or what *owns* a particular module
instance. In Python, objects are always allocated dynamically (on the heap) and
have reference semantics. This is very easy to work with and straightforward to
understand. In fact, in Python, you can largely forget about where objects live
and how they get referenced, and focus on getting things done.

C++, being a lower level language, provides more options in this realm. This
increases complexity and heavily influences the design and ergonomics of the C++
frontend. In particular, for modules in the C++ frontend, we have the option of
using *either* value semantics *or* reference semantics. The first case is the
simplest and was shown in the examples thus far: module objects are allocated on
the stack and when passed to a function, can be either copied, moved (with
``std::move``) or taken by reference or by pointer:

.. code-block:: cpp

  struct Net : torch::nn::Module { };

  void a(Net net) { }
  void b(Net& net) { }
  void c(Net* net) { }

  int main() {
    Net net;
    a(net);
    a(std::move(net));
    b(net);
    c(&net);
  }

For the second case -- reference semantics -- we can use ``std::shared_ptr``.
The advantage of reference semantics is that, like in Python, it reduces the
cognitive overhead of thinking about how modules must be passed to functions and
how arguments must be declared (assuming you use ``shared_ptr`` everywhere).

.. code-block:: cpp

  struct Net : torch::nn::Module {};

  void a(std::shared_ptr<Net> net) { }

  int main() {
    auto net = std::make_shared<Net>();
    a(net);
  }

In our experience, researchers coming from dynamic languages greatly prefer
reference semantics over value semantics, even though the latter is more
"native" to C++. It is also important to note that ``torch::nn::Module``'s
design, in order to stay close to the ergonomics of the Python API, relies on
shared ownership. For example, take our earlier (here shortened) definition of
``Net``:

.. code-block:: cpp

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M)))
    { }
    torch::nn::Linear linear;
  };

In order to use the ``linear`` submodule, we want to store it directly in our
class. However, we also want the module base class to know about and have access
to this submodule. For this, it must store a reference to this submodule. At
this point, we have already arrived at the need for shared ownership. Both the
``torch::nn::Module`` class and concrete ``Net`` class require a reference to
the submodule. For this reason, the base class stores modules as
``shared_ptr``\s, and therefore the concrete class must too.

But wait! I don't see any mention of ``shared_ptr`` in the above code! Why is
that? Well, because ``std::shared_ptr<MyModule>`` is a hell of a lot to type. To
keep our researchers productive, we came up with an elaborate scheme to hide the
mention of ``shared_ptr`` -- a benefit usually reserved for value semantics --
while retaining reference semantics. To understand how this works, we can take a
look at a simplified definition of the ``torch::nn::Linear`` module in the core
library (the full definition is `here
<https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/modules/linear.h>`_):

.. code-block:: cpp

  struct LinearImpl : torch::nn::Module {
    LinearImpl(int64_t in, int64_t out);

    Tensor forward(const Tensor& input);

    Tensor weight, bias;
  };

  TORCH_MODULE(Linear);

In brief: the module is not called ``Linear``, but ``LinearImpl``. A macro,
``TORCH_MODULE`` then defines the actual ``Linear`` class. This "generated"
class is effectively a wrapper over a ``std::shared_ptr<LinearImpl>``. It is a
wrapper instead of a simple typedef so that, among other things, constructors
still work as expected, i.e. you can still write ``torch::nn::Linear(3, 4)``
instead of ``std::make_shared<LinearImpl>(3, 4)``. We call the class created by
the macro the module *holder*. Like with (shared) pointers, you access the
underlying object using the arrow operator (like ``model->forward(...)``). The
end result is an ownership model that resembles that of the Python API quite
closely. Reference semantics become the default, but without the extra typing of
``std::shared_ptr`` or ``std::make_shared``. For our ``Net``, using the module
holder API looks like this:

.. code-block:: cpp

  struct NetImpl : torch::nn::Module {};
  TORCH_MODULE(Net);

  void a(Net net) { }

  int main() {
    Net net;
    a(net);
  }

There is one subtle issue that deserves mention here. A default constructed
``std::shared_ptr`` is "empty", i.e. contains a null pointer. What is a default
constructed ``Linear`` or ``Net``? Well, it's a tricky choice. We could say it
should be an empty (null) ``std::shared_ptr<LinearImpl>``. However, recall that
``Linear(3, 4)`` is the same as ``std::make_shared<LinearImpl>(3, 4)``. This
means that if we had decided that ``Linear linear;`` should be a null pointer,
then there would be no way to construct a module that does not take any
constructor arguments, or defaults all of them. For this reason, in the current
API, a default constructed module holder (like ``Linear()``) invokes the
default constructor of the underlying module (``LinearImpl()``). If the
underlying module does not have a default constructor, you get a compiler error.
To instead construct the empty holder, you can pass ``nullptr`` to the
constructor of the holder.

In practice, this means you can use submodules either like shown earlier, where
the module is registered and constructed in the *initializer list*:

.. code-block:: cpp

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M)))
    { }
    torch::nn::Linear linear;
  };

or you can first construct the holder with a null pointer and then assign to it
in the constructor (more familiar for Pythonistas):

.. code-block:: cpp

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M) {
      linear = register_module("linear", torch::nn::Linear(N, M));
    }
    torch::nn::Linear linear{nullptr}; // construct an empty holder
  };

In conclusion: Which ownership model -- which semantics -- should you use? The
C++ frontend's API best supports the ownership model provided by module holders.
The only disadvantage of this mechanism is one extra line of boilerplate below
the module declaration. That said, the simplest model is still the value
semantics model shown in the introduction to C++ modules. For small, simple
scripts, you may get away with it too. But you'll find sooner or later that, for
technical reasons, it is not always supported. For example, the serialization
API (``torch::save`` and ``torch::load``) only supports module holders (or plain
``shared_ptr``). As such, the module holder API is the recommended way of
defining modules with the C++ frontend, and we will use this API in this
tutorial henceforth.

Defining the DCGAN Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^

We now have the necessary background and introduction to define the modules for
the machine learning task we want to solve in this post. To recap: our task is
to generate images of digits from the `MNIST dataset
<http://yann.lecun.com/exdb/mnist/>`_. We want to use a `generative adversarial
network (GAN)
<https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`_ to solve
this task. In particular, we'll use a `DCGAN architecture
<https://arxiv.org/abs/1511.06434>`_ -- one of the first and simplest of its
kind, but entirely sufficient for this task.

.. tip::

  You can find the full source code presented in this tutorial `in this
  repository <https://github.com/pytorch/examples/tree/master/cpp/dcgan>`_.

What was a GAN aGAN?
********************

A GAN consists of two distinct neural network models: a *generator* and a
*discriminator*. The generator receives samples from a noise distribution, and
its aim is to transform each noise sample into an image that resembles those of
a target distribution -- in our case the MNIST dataset. The discriminator in
turn receives either *real* images from the MNIST dataset, or *fake* images from
the generator. It is asked to emit a probability judging how real (closer to
``1``) or fake (closer to ``0``) a particular image is. Feedback from the
discriminator on how real the images produced by the generator are is used to
train the generator. Feedback on how good of an eye for authenticity the
discriminator has is used to optimize the discriminator. In theory, a delicate
balance between the generator and discriminator makes them improve in tandem,
leading to the generator producing images indistinguishable from the target
distribution, fooling the discriminator's (by then) excellent eye into emitting
a probability of ``0.5`` for both real and fake images. For us, the end result
is a machine that receives noise as input and generates realistic images of
digits as its output.

The Generator Module
********************

We begin by defining the generator module, which consists of a series of
transposed 2D convolutions, batch normalizations and ReLU activation units.
We explicitly pass inputs (in a functional way) between modules in the
``forward()`` method of a module we define ourselves:

.. code-block:: cpp

  struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl(int kNoiseSize)
        : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
                    .bias(false)),
          batch_norm1(256),
          conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                    .stride(2)
                    .padding(1)
                    .bias(false)),
          batch_norm2(128),
          conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                    .stride(2)
                    .padding(1)
                    .bias(false)),
          batch_norm3(64),
          conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                    .stride(2)
                    .padding(1)
                    .bias(false))
   {
     // register_module() is needed if we want to use the parameters() method later on
     register_module("conv1", conv1);
     register_module("conv2", conv2);
     register_module("conv3", conv3);
     register_module("conv4", conv4);
     register_module("batch_norm1", batch_norm1);
     register_module("batch_norm2", batch_norm2);
     register_module("batch_norm3", batch_norm3);
   }

   torch::Tensor forward(torch::Tensor x) {
     x = torch::relu(batch_norm1(conv1(x)));
     x = torch::relu(batch_norm2(conv2(x)));
     x = torch::relu(batch_norm3(conv3(x)));
     x = torch::tanh(conv4(x));
     return x;
   }

   nn::ConvTranspose2d conv1, conv2, conv3, conv4;
   nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
  };
  TORCH_MODULE(DCGANGenerator);

  DCGANGenerator generator(kNoiseSize);

We can now invoke ``forward()`` on the ``DCGANGenerator`` to map a noise sample to an image.

The particular modules chosen, like ``nn::ConvTranspose2d`` and ``nn::BatchNorm2d``,
follows the structure outlined earlier. The ``kNoiseSize`` constant determines
the size of the input noise vector and is set to ``100``. Hyperparameters were,
of course, found via grad student descent.

.. attention::

	No grad students were harmed in the discovery of hyperparameters. They were
	fed Soylent regularly.

.. note::

	A brief word on the way options are passed to built-in modules like ``Conv2d``
	in the C++ frontend: Every module has some required options, like the number
	of features for ``BatchNorm2d``. If you only need to configure the required
	options, you can pass them directly to the module's constructor, like
	``BatchNorm2d(128)`` or ``Dropout(0.5)`` or ``Conv2d(8, 4, 2)`` (for input
	channel count, output channel count, and kernel size). If, however, you need
	to modify other options, which are normally defaulted, such as ``bias``
	for ``Conv2d``, you need to construct and pass an *options* object. Every
	module in the C++ frontend has an associated options struct, called
	``ModuleOptions`` where ``Module`` is the name of the module, like
	``LinearOptions`` for ``Linear``. This is what we do for the ``Conv2d``
	modules above.

The Discriminator Module
************************

The discriminator is similarly a sequence of convolutions, batch normalizations
and activations. However, the convolutions are now regular ones instead of
transposed, and we use a leaky ReLU with an alpha value of 0.2 instead of a
vanilla ReLU. Also, the final activation becomes a Sigmoid, which squashes
values into a range between 0 and 1. We can then interpret these squashed values
as the probabilities the discriminator assigns to images being real.

To build the discriminator, we will try something different: a `Sequential` module.
Like in Python, PyTorch here provides two APIs for model definition: a functional one
where inputs are passed through successive functions (e.g. the generator module example),
and a more object-oriented one where we build a `Sequential` module containing the
entire model as submodules. Using `Sequential`, the discriminator would look like:

.. code-block:: cpp

  nn::Sequential discriminator(
    // Layer 1
    nn::Conv2d(
        nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
    nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
    // Layer 2
    nn::Conv2d(
        nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
    nn::BatchNorm2d(128),
    nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
    // Layer 3
    nn::Conv2d(
        nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
    nn::BatchNorm2d(256),
    nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
    // Layer 4
    nn::Conv2d(
        nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
    nn::Sigmoid());

.. tip::

  A ``Sequential`` module simply performs function composition. The output of
  the first submodule becomes the input of the second, the output of the third
  becomes the input of the fourth and so on.


Loading Data
------------

Now that we have defined the generator and discriminator model, we need some
data we can train these models with. The C++ frontend, like the Python one,
comes with a powerful parallel data loader. This data loader can read batches of
data from a dataset (which you can define yourself) and provides many
configuration knobs.

.. note::

	While the Python data loader uses multi-processing, the C++ data loader is truly
	multi-threaded and does not launch any new processes.

The data loader is part of the C++ frontend's ``data`` api, contained in the
``torch::data::`` namespace. This API consists of a few different components:

- The data loader class,
- An API for defining datasets,
- An API for defining *transforms*, which can be applied to datasets,
- An API for defining *samplers*, which produce the indices with which datasets are indexed,
- A library of existing datasets, transforms and samplers.

For this tutorial, we can use the ``MNIST`` dataset that comes with the C++
frontend. Let's instantiate a ``torch::data::datasets::MNIST`` for this, and
apply two transformations: First, we normalize the images so that they are in
the range of ``-1`` to ``+1`` (from an original range of ``0`` to ``1``).
Second, we apply the ``Stack`` *collation*, which takes a batch of tensors and
stacks them into a single tensor along the first dimension:

.. code-block:: cpp

  auto dataset = torch::data::datasets::MNIST("./mnist")
      .map(torch::data::transforms::Normalize<>(0.5, 0.5))
      .map(torch::data::transforms::Stack<>());

Note that the MNIST dataset should be located in the ``./mnist`` directory
relative to wherever you execute the training binary from. You can use `this
script <https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03>`_
to download the MNIST dataset.

Next, we create a data loader and pass it this dataset. To make a new data
loader, we use ``torch::data::make_data_loader``, which returns a
``std::unique_ptr`` of the correct type (which depends on the type of the
dataset, the type of the sampler and some other implementation details):

.. code-block:: cpp

  auto data_loader = torch::data::make_data_loader(std::move(dataset));

The data loader does come with a lot of options. You can inspect the full set
`here
<https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/dataloader_options.h>`_.
For example, to speed up the data loading, we can increase the number of
workers. The default number is zero, which means the main thread will be used.
If we set ``workers`` to ``2``, two threads will be spawned that load data
concurrently. We should also increase the batch size from its default of ``1``
to something more reasonable, like ``64`` (the value of ``kBatchSize``). So
let's create a ``DataLoaderOptions`` object and set the appropriate properties:

.. code-block:: cpp

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));


We can now write a loop to load batches of data, which we'll only print to the
console for now:

.. code-block:: cpp

  for (torch::data::Example<>& batch : *data_loader) {
    std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    for (int64_t i = 0; i < batch.data.size(0); ++i) {
      std::cout << batch.target[i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
  }

The type returned by the data loader in this case is a ``torch::data::Example``.
This type is a simple struct with a ``data`` field for the data and a ``target``
field for the label. Because we applied the ``Stack`` collation earlier, the
data loader returns only a single such example. If we had not applied the
collation, the data loader would yield ``std::vector<torch::data::Example<>>``
instead, with one element per example in the batch.

If you rebuild and run this code, you should see something like this:

.. code-block:: shell

  root@fa350df05ecf:/home/build# make
  Scanning dependencies of target dcgan
  [ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
  [100%] Linking CXX executable dcgan
  [100%] Built target dcgan
  root@fa350df05ecf:/home/build# make
  [100%] Built target dcgan
  root@fa350df05ecf:/home/build# ./dcgan
  Batch size: 64 | Labels: 5 2 6 7 2 1 6 7 0 1 6 2 3 6 9 1 8 4 0 6 5 3 3 0 4 6 6 6 4 0 8 6 0 6 9 2 4 0 2 8 6 3 3 2 9 2 0 1 4 2 3 4 8 2 9 9 3 5 8 0 0 7 9 9
  Batch size: 64 | Labels: 2 2 4 7 1 2 8 8 6 9 0 2 2 9 3 6 1 3 8 0 4 4 8 8 8 9 2 6 4 7 1 5 0 9 7 5 4 3 5 4 1 2 8 0 7 1 9 6 1 6 5 3 4 4 1 2 3 2 3 5 0 1 6 2
  Batch size: 64 | Labels: 4 5 4 2 1 4 8 3 8 3 6 1 5 4 3 6 2 2 5 1 3 1 5 0 8 2 1 5 3 2 4 4 5 9 7 2 8 9 2 0 6 7 4 3 8 3 5 8 8 3 0 5 8 0 8 7 8 5 5 6 1 7 8 0
  Batch size: 64 | Labels: 3 3 7 1 4 1 6 1 0 3 6 4 0 2 5 4 0 4 2 8 1 9 6 5 1 6 3 2 8 9 2 3 8 7 4 5 9 6 0 8 3 0 0 6 4 8 2 5 4 1 8 3 7 8 0 0 8 9 6 7 2 1 4 7
  Batch size: 64 | Labels: 3 0 5 5 9 8 3 9 8 9 5 9 5 0 4 1 2 7 7 2 0 0 5 4 8 7 7 6 1 0 7 9 3 0 6 3 2 6 2 7 6 3 3 4 0 5 8 8 9 1 9 2 1 9 4 4 9 2 4 6 2 9 4 0
  Batch size: 64 | Labels: 9 6 7 5 3 5 9 0 8 6 6 7 8 2 1 9 8 8 1 1 8 2 0 7 1 4 1 6 7 5 1 7 7 4 0 3 2 9 0 6 6 3 4 4 8 1 2 8 6 9 2 0 3 1 2 8 5 6 4 8 5 8 6 2
  Batch size: 64 | Labels: 9 3 0 3 6 5 1 8 6 0 1 9 9 1 6 1 7 7 4 4 4 7 8 8 6 7 8 2 6 0 4 6 8 2 5 3 9 8 4 0 9 9 3 7 0 5 8 2 4 5 6 2 8 2 5 3 7 1 9 1 8 2 2 7
  Batch size: 64 | Labels: 9 1 9 2 7 2 6 0 8 6 8 7 7 4 8 6 1 1 6 8 5 7 9 1 3 2 0 5 1 7 3 1 6 1 0 8 6 0 8 1 0 5 4 9 3 8 5 8 4 8 0 1 2 6 2 4 2 7 7 3 7 4 5 3
  Batch size: 64 | Labels: 8 8 3 1 8 6 4 2 9 5 8 0 2 8 6 6 7 0 9 8 3 8 7 1 6 6 2 7 7 4 5 5 2 1 7 9 5 4 9 1 0 3 1 9 3 9 8 8 5 3 7 5 3 6 8 9 4 2 0 1 2 5 4 7
  Batch size: 64 | Labels: 9 2 7 0 8 4 4 2 7 5 0 0 6 2 0 5 9 5 9 8 8 9 3 5 7 5 4 7 3 0 5 7 6 5 7 1 6 2 8 7 6 3 2 6 5 6 1 2 7 7 0 0 5 9 0 0 9 1 7 8 3 2 9 4
  Batch size: 64 | Labels: 7 6 5 7 7 5 2 2 4 9 9 4 8 7 4 8 9 4 5 7 1 2 6 9 8 5 1 2 3 6 7 8 1 1 3 9 8 7 9 5 0 8 5 1 8 7 2 6 5 1 2 0 9 7 4 0 9 0 4 6 0 0 8 6
  ...

Which means we are successfully able to load data from the MNIST dataset.

Writing the Training Loop
-------------------------

Let's now finish the algorithmic part of our example and implement the delicate
dance between the generator and discriminator. First, we'll create two
optimizers, one for the generator and one for the discriminator. The optimizers
we use implement the `Adam <https://arxiv.org/pdf/1412.6980.pdf>`_ algorithm:

.. code-block:: cpp

  torch::optim::Adam generator_optimizer(
      generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
  torch::optim::Adam discriminator_optimizer(
      discriminator->parameters(), torch::optim::AdamOptions(5e-4).beta1(0.5));

.. note::

	As of this writing, the C++ frontend provides optimizers implementing Adagrad,
	Adam, LBFGS, RMSprop and SGD. The `docs
	<https://pytorch.org/cppdocs/api/namespace_torch__optim.html>`_ have the
	up-to-date list.

Next, we need to update our training loop. We'll add an outer loop to exhaust
the data loader every epoch and then write the GAN training code:

.. code-block:: cpp

  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data;
      torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images.
      torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1});
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();

      std::printf(
          "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
          epoch,
          kNumberOfEpochs,
          ++batch_index,
          batches_per_epoch,
          d_loss.item<float>(),
          g_loss.item<float>());
    }
  }

Above, we first evaluate the discriminator on real images, for which it should
assign a high probability. For this, we use
``torch::empty(batch.data.size(0)).uniform_(0.8, 1.0)`` as the target
probabilities.

.. note::

	We pick random values uniformly distributed between 0.8 and 1.0 instead of 1.0
	everywhere in order to make the discriminator training more robust. This trick
	is called *label smoothing*.

Before evaluating the discriminator, we zero out the gradients of its
parameters. After computing the loss, we back-propagate through the network by
calling ``d_loss.backward()`` to compute new gradients. We repeat this spiel for
the fake images. Instead of using images from the dataset, we let the generator
create fake images for this by feeding it a batch of random noise. We then
forward those fake images to the discriminator. This time, we want the
discriminator to emit low probabilities, ideally all zeros. Once we have
computed the discriminator loss for both the batch of real and the batch of fake
images, we can progress the discriminator's optimizer by one step in order to
update its parameters.

To train the generator, we again first zero its gradients, and then re-evaluate
the discriminator on the fake images. However, this time we want the
discriminator to assign probabilities very close to one, which would indicate
that the generator can produce images that fool the discriminator into thinking
they are actually real (from the dataset). For this, we fill the ``fake_labels``
tensor with all ones. We finally step the generator's optimizer to also update
its parameters.

We should now be ready to train our model on the CPU. We don't have any code yet
to capture state or sample outputs, but we'll add this in just a moment. For
now, let's just observe that our model is doing *something* -- we'll later
verify based on the generated images whether this something is meaningful.
Re-building and running should print something like:

.. code-block:: shell

  root@3c0711f20896:/home/build# make && ./dcgan
  Scanning dependencies of target dcgan
  [ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
  [100%] Linking CXX executable dcgan
  [100%] Built target dcga
  [ 1/10][100/938] D_loss: 0.6876 | G_loss: 4.1304
  [ 1/10][200/938] D_loss: 0.3776 | G_loss: 4.3101
  [ 1/10][300/938] D_loss: 0.3652 | G_loss: 4.6626
  [ 1/10][400/938] D_loss: 0.8057 | G_loss: 2.2795
  [ 1/10][500/938] D_loss: 0.3531 | G_loss: 4.4452
  [ 1/10][600/938] D_loss: 0.3501 | G_loss: 5.0811
  [ 1/10][700/938] D_loss: 0.3581 | G_loss: 4.5623
  [ 1/10][800/938] D_loss: 0.6423 | G_loss: 1.7385
  [ 1/10][900/938] D_loss: 0.3592 | G_loss: 4.7333
  [ 2/10][100/938] D_loss: 0.4660 | G_loss: 2.5242
  [ 2/10][200/938] D_loss: 0.6364 | G_loss: 2.0886
  [ 2/10][300/938] D_loss: 0.3717 | G_loss: 3.8103
  [ 2/10][400/938] D_loss: 1.0201 | G_loss: 1.3544
  [ 2/10][500/938] D_loss: 0.4522 | G_loss: 2.6545
  ...

Moving to the GPU
-----------------

While our current script can run just fine on the CPU, we all know convolutions
are a lot faster on GPU. Let's quickly discuss how we can move our training onto
the GPU. We'll need to do two things for this: pass a GPU device specification
to tensors we allocate ourselves, and explicitly copy any other tensors onto the
GPU via the ``to()`` method all tensors and modules in the C++ frontend have.
The simplest way to achieve both is to create an instance of ``torch::Device``
at the top level of our training script, and then pass that device to tensor
factory functions like ``torch::zeros`` as well as the ``to()`` method. We can
start by doing this with a CPU device:

.. code-block:: cpp

  // Place this somewhere at the top of your training script.
  torch::Device device(torch::kCPU);

New tensor allocations like

.. code-block:: cpp

  torch::Tensor fake_labels = torch::zeros(batch.data.size(0));

should be updated to take the ``device`` as the last argument:

.. code-block:: cpp

  torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);

For tensors whose creation is not in our hands, like those coming from the MNIST
dataset, we must insert explicit ``to()`` calls. This means

.. code-block:: cpp

  torch::Tensor real_images = batch.data;

becomes

.. code-block:: cpp

  torch::Tensor real_images = batch.data.to(device);

and also our model parameters should be moved to the correct device:

.. code-block:: cpp

  generator->to(device);
  discriminator->to(device);

.. note::

	If a tensor already lives on the device supplied to ``to()``, the call is a
	no-op. No extra copy is made.

At this point, we've just made our previous CPU-residing code more explicit.
However, it is now also very easy to change the device to a CUDA device:

.. code-block:: cpp

  torch::Device device(torch::kCUDA)

And now all tensors will live on the GPU, calling into fast CUDA kernels for all
operations, without us having to change any downstream code. If we wanted to
specify a particular device index, it could be passed as the second argument to
the ``Device`` constructor. If we wanted different tensors to live on different
devices, we could pass separate device instances (for example one on CUDA device
0 and the other on CUDA device 1). We can even do this configuration
dynamically, which is often useful to make our training scripts more portable:

.. code-block:: cpp

  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  }

or even

.. code-block:: cpp

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

학습 상태 저장 및 복원
---------------------

마지막으로 학습 스크립트에 추가해야 할 내용은 모델 매개 변수 및 옵티마이저의 상태, 그리고 생성된 몇 개의 이미지 샘플을
주기적으로 저장하는 것입니다. 학습 과정 도중에 컴퓨터가 다운되면 이렇게 저장된 상태로부터 학습 상태를 복원할 수 있습니다.
이는 장시간 지속되는 학습을 위해 필수로 요구됩니다. 다행히도 C++ 프론트엔드는 개별 텐서뿐만 아니라 모델 및 옵티마이저
상태를 직렬화하고 역직렬화할 수 있는 API를 제공합니다.

이를 위한 핵심 API는 ``torch::save(thing,filename)`` 와 ``torch::load(thing,filename)`` 로, 여기서 ``thing`` 은 ``torch::nn::Module``
의 하위 클래스 혹은 우리의 학습 스크립트의 ``Adam`` 객체와 같은 옵티마이저 인스턴스가 될 수 있습니다. 모델 및 옵티마이저
상태를 특정 주기마다 저장하도록 학습 루프를 수정해보겠습니다.

.. code-block:: cpp

  if (batch_index % kCheckpointEvery == 0) {
    // Checkpoint the model and optimizer state.
    torch::save(generator, "generator-checkpoint.pt");
    torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::save(discriminator, "discriminator-checkpoint.pt");
    torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
    // Sample the generator and save the images.
    torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
    torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
    std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
  }

여기서 ``100`` 배치마다 상태를 저장하려면 ``kCheckpointEvery``를 ``100`` 과 같은 정수로 설정할 수 있으며, ``checkpoint_counter``는
상태를 저장할 때마다 증가하는 카운터입니다.

학습 상태를 복원하기 위해 모델 및 옵티마이저를 모두 생성한 후 학습 루프 앞에 다음 코드를 추가할 수 있습니다.

.. code-block:: cpp

  torch::optim::Adam generator_optimizer(
      generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
  torch::optim::Adam discriminator_optimizer(
      discriminator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

  if (kRestoreFromCheckpoint) {
    torch::load(generator, "generator-checkpoint.pt");
    torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::load(discriminator, "discriminator-checkpoint.pt");
    torch::load(
        discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
  }

  int64_t checkpoint_counter = 0;
  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {


생성된 이미지 검사하기
---------------------

학습 스크립트가 완성되어 CPU에서든 GPU에서든 GAN을 훈련시킬 준비가 됐습니다. 학습 과정의 중간 출력을 검사하기 위해
``"dcgan-sample-xxx.pt"``에 주기적으로 이미지 샘플을 저장하는 코드를 추가했으니, 텐서들을 불러와 matplotlib로
시각화하는 간단한 파이썬 스크립트를 작성해보겠습니다.

.. code-block:: python

  from __future__ import print_function
  from __future__ import unicode_literals

  import argparse

  import matplotlib.pyplot as plt
  import torch


  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--sample-file", required=True)
  parser.add_argument("-o", "--out-file", default="out.png")
  parser.add_argument("-d", "--dimension", type=int, default=3)
  options = parser.parse_args()

  module = torch.jit.load(options.sample_file)
  images = list(module.parameters())[0]

  for index in range(options.dimension * options.dimension):
    image = images[index].detach().cpu().reshape(28, 28).mul(255).to(torch.uint8)
    array = image.numpy()
    axis = plt.subplot(options.dimension, options.dimension, 1 + index)
    plt.imshow(array, cmap="gray")
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

  plt.savefig(options.out_file)
  print("Saved ", options.out_file)

이제 약 30 에폭 정도 모델을 학습시킵시다.

.. code-block:: shell

  root@3c0711f20896:/home/build# make && ./dcgan                                                                                                                                10:17:57
  Scanning dependencies of target dcgan
  [ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
  [100%] Linking CXX executable dcgan
  [100%] Built target dcgan
  CUDA is available! Training on GPU.
  [ 1/30][200/938] D_loss: 0.4953 | G_loss: 4.0195
  -> checkpoint 1
  [ 1/30][400/938] D_loss: 0.3610 | G_loss: 4.8148
  -> checkpoint 2
  [ 1/30][600/938] D_loss: 0.4072 | G_loss: 4.36760
  -> checkpoint 3
  [ 1/30][800/938] D_loss: 0.4444 | G_loss: 4.0250
  -> checkpoint 4
  [ 2/30][200/938] D_loss: 0.3761 | G_loss: 3.8790
  -> checkpoint 5
  [ 2/30][400/938] D_loss: 0.3977 | G_loss: 3.3315
  ...
  -> checkpoint 120
  [30/30][938/938] D_loss: 0.3610 | G_loss: 3.8084

그리고 이미지들을 플롯에 시각화합니다.

.. code-block:: shell

  root@3c0711f20896:/home/build# python display.py -i dcgan-sample-100.pt
  Saved out.png

그 결과는 아래와 같을 것입니다.

.. figure:: /_static/img/cpp-frontend/digits.png
   :alt: digits

숫자네요! 만세! 이제 여러분 차례입니다. 숫자가 보다 나아 보이도록 모델을 개선할 수 있나요?

결론
----

이 튜토리얼을 통해 PyTorch C++ 프론트엔드에 대한 어느 정도 이해도가 생기셨기 바랍니다. 필연적으로 PyTorch
같은 머신러닝 라이브러리는 매우 다양하고 광범위한 API를 가지고 있습니다. 따라서, 여기서 논의하기에 시간과
공간이 부족했던 개념들이 많습니다. 그러나 직접 API를 사용해보고, 특히 `라이브러리 API <https://pytorch.org/cppdocs/api/library_root.html>`_ 섹션을 참조해보는
것을 권장드립니다. 또한, C++ 프론트엔드가 파이썬 프론트엔드의 디자인과 시맨틱을 따른다는 사실을 잘 기억하면
보다 빠르게 학습할 수 있을 것입니다.

.. tip::

  본 튜토리얼에 대한 전체 소스코드는 `이 저장소 <https://github.com/pytorch/examples/tree/master/cpp/dcgan>`_ 에 제공되어 있습니다.

언제나 그렇듯이 어떤 문제가 생기거나 질문이 있으면 저희 `포럼 <https://discuss.pytorch.org/>`_ 을 이용하거나 `Github 이슈 <https://github.com/pytorch/pytorch/issues>`_ 로 연락주세요.