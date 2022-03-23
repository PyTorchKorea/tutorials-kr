PyTorch C++ 프론트엔드 사용하기
=============================

**번역**: `유용환 <https://github.com/yoosful>`_

PyTorch C++ 프론트엔드는 PyTorch 머신러닝 프레임워크의 순수 C++
인터페이스입니다. PyTorch의 주된 인터페이스는 물론 파이썬이지만
이 곳의 API는 텐서(tensor)나 자동 미분과 같은 기초적인 자료구조
및 기능을 제공하는 C++ 코드베이스 위에 구현되었습니다. C++
프론트엔드는 이러한 기초적인 C++ 코드베이스를 비롯해 머신러닝 학습과 추론을
위해 필요한 도구들을 상속하는 순수 C++11 API를 제공합니다. 여기에는
신경망 모델링을 위해 필요한 공용 컴포넌트들의 빌트인 모음, 그것을
상속하기 위한 커스텀 모듈, 확률적 경사 하강법과 같은 유명한 최적화 알고리즘
라이브러리, 병렬 데이터 로더 및 데이터셋을 정의하고 불러오기 위한
API, 직렬화 루틴 등이 포합됩니다.

이 튜토리얼은 C++ 프론트엔드로 모델을 학습하는 엔드 투 엔드
예제를 안내합니다. 구체적으로, 우리는 생성 모델 중 하나인
`DCGAN <https://arxiv.org/abs/1511.06434>`_ 을 학습시켜
MNIST 숫자 이미지들을 생성할 것입니다. 개념적으로 쉬운 예시이지만,
여러분이 PyTorch C++ 프론트엔드에 대한 대략적인 개요를 파악하고 더
복잡한 모델을 학습시키고 싶은 욕구를 불러일으키기에 충분할 것입니다.
먼저 C++ 프론트엔드 사용에 대한 동기부여가 될 만한 이야기로 시작하고,
곧바로 모델을 정의하고 학습해보도록 하겠습니다.

.. tip::

  C++ 프론트엔드에 대한 짧고 재미있는 발표를 보려면 `CppCon 2018 라이트닝 토크
  <https://www.youtube.com/watch?v=auRPXMMHJzc>`_ 를 시청하세요.


.. tip::

  `이 노트 <https://pytorch.org/cppdocs/frontend.html>`_ 는 C++
  프론트엔드의 컴포넌트와 디자인 철학의 전반적인 개요를 제공합니다.

.. tip::

  PyTorch C++ 생태계에 대한 문서는 https://pytorch.org/cppdocs에서
  확인할 수 있습니다. API 레벨의 문서뿐만 아니라 개괄적인 설명도
  찾을 수 있을 것입니다.

동기부여
--------

GAN과 MNIST 숫자로의 설레는 여정을 시작하기에 앞서, 먼저
파이썬 대신 C++ 프론트엔드를 사용하는 이유에 대해
설명하겠습니다. 우리(PyTorch 팀)는 파이썬을 사용할 수 없거나
사용하기에 적합하지 않은 환경에서 연구를 가능하게 하기 위해
C++ 프론트엔드를 만들었습니다. 예를 들면 다음과 같습니다.

- **저지연 시스템**: 초당 프레임 수가 높고 지연 시간이 짧은
  순수 C++ 게임 엔진에서 강화 학습 연구를 수행할 수 있습니다.
  그러한 환경에서는 파이썬 라이브러리보다 순수 C++ 라이브러리를
  사용하는 것이 훨씬 더 적합합니다. 파이썬은 느린 인터프리터
  때문에 다루기가 쉽지 않습니다.
- **고도의 멀티쓰레딩 환경**: 글로벌 인터프리터 락(GIL)으로 인해
  파이썬은 동시에 둘 이상의 시스템 쓰레드를 실행할 수 없습니다.
  대안으로 멀티프로세싱을 사용하면 확장성이 떨어지며 심각한 한계가
  있습니다. C++는 이러한 제약 조건이 없으며 쓰레드를 쉽게 만들고
  사용할 수 있습니다. `Deep Neuroevolution <https://eng.uber.com/deep-neuroevolution/>`_ 에
  사용된 것과 같이 고도의 병렬화가 필요한 모델도 이를 활용할 수
  있습니다.
- **기존의 C++ 코드베이스**: 백엔드 서버의 웹 페이지 서비스부터
  사진 편집 소프트웨어의 3D 그래픽 렌더링에 이르기까지 어떠한
  작업이라도 수행하는 기존 C++ 애플리케이션 소유자로서, 머신러닝
  방법론을 시스템에 통합하고 싶을 수 있습니다. C++ 프론트엔드는
  PyTorch (파이썬) 경험 본연의 높은 유연성과 직관성을 유지하면서,
  파이썬과 C++를 앞뒤로 바인딩하는 번거로움 없이 C++를 사용할 수
  있게 해줍니다.

C++ 프론트엔드의 목적은 파이썬 프론트엔드와 경쟁하는 것이 아닌
보완하는 것입니다. 연구자와 엔지니어 모두가 PyTorch의 단순성,
유연성 및 직관적인 API를 매우 좋아합니다. 우리의 목표는 여러분이
위의 예시를 비롯한 모든 가능한 환경에서 이 핵심 디자인 원칙을
이용할 수 있도록 하는 것입니다. 이러한 시나리오 중 하나가 여러분의
사례에 해당하거나, 단순히 관심이 있거나 궁금하다면 아래 내용을 통해
C++ 프론트엔드에 대해 자세히 살펴보세요.

.. tip::

	C++ 프론트엔드는 파이썬 프론트엔드와 최대한 유사한 API를
  제공하고자 합니다. 만일 파이썬 프론트엔드에 익숙한 사람이 "C++
  프론트엔드로 X를 어떻게 해야 하는가?" 의문을 갖는다면, 많은 경우에
  파이썬에서와 같은 방식으로 코드를 작성해 파이썬에서와 동일한 함수와
  메서드를 사용할 수 있을 것입니다. (다만, 온점을 더블 콜론으로 바꾸는
  것에 유의하세요.)

기본 애플리케이션 작성하기
------------------------

먼저 최소한의 C++ 애플리케이션을 작성해 우리의 설정과
빌드 환경이 동일한지 확인하겠습니다. 먼저, C++
프론트엔드를 사용하는 데 필요한 모든 관련 헤더, 라이브러리 및
CMake 빌드 파일을 패키징하는 *LibTorch* 배포판의 사본이
필요합니다. 리눅스, 맥OS, 윈도우용 LibTorch 배포판은
`PyTorch website <https://pytorch.org/get-started/locally/>`_ 에서
다운로드할 수 있습니다. 이 튜토리얼의 나머지 부분은 기본 우분투 리눅스
환경을 가정하지만 맥OS나 윈도우를 사용하셔도 괜찮습니다.

.. tip::

  `PyTorch C++ 배포판 설치 <https://pytorch.org/cppdocs/installing.html>`_
  의 설명에 다음의 과정이 더 자세히 안내되어
  있습니다.

.. tip::
  윈도우에서는 디버그 및 릴리스 빌드가 ABI와 호환되지 않습니다. 프로젝트를
  디버그 모드로 빌드하려면 LibTorch의 디버그 버전을 사용해보세요.
  아래의 ``cmake --build .`` 에 올바른 설정을 지정하는 것도 잊지
  마세요.

가장 먼저 할 것은 PyTorch 웹사이트에서 검색된 링크를 통해 LibTorch
배포판을 로컬에 다운로드하는 것입니다. 일반적 Ubuntu Linux 환경의 경우
다음 명령어를 실행합니다.

.. code-block:: shell

  # CUDA 9.0 등에 대한 지원이 필요한 경우 아래 URL에서 "cpu"를 "cu90"로 바꾸세요.
  wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  unzip libtorch-shared-with-deps-latest.zip

다음으로 ``torch/torch.h`` 를 호출하는 ``dcgan.cpp`` 라는 이름의 C++
파일 하나를 작성합시다. 우선은 아래와 같이 3x3 항등 행렬을 출력하기만 하면
됩니다.

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

  CMake는 LibTorch에 권장되는 빌드 시스템이지만 필수 요구
  사항은 아닙니다. Visual Studio 프로젝트 파일, QMake, 일반
  Make 파일 등 다른 빌드 환경을 사용해도 됩니다. 하지만
  이에 대한 즉각적인 지원은 제공하지 않습니다.

위 CMake 파일 4번째 줄의 ``find_package(Torch REQUIRED)`` 는
CMake가 LibTorch 라이브러리 빌드 설정을 찾도록 안내합니다.
CMake가 해당 파일의 *위치* 를 찾을 수 있도록 하려면 ``cmake`` 호출 시
``CMAKE_PREFIX_PATH`` 를 설정해야 합니다. 이에 앞서 ``dcgan`` 애플리케이션에
대해 디렉터리 구조를 다음과 같이 통일하도록 하겠습니다.

.. code-block:: shell

  dcgan/
    CMakeLists.txt
    dcgan.cpp

또한 앞으로 압축 해제된 LibTorch 배포판의 경로를 ``/path/to/libtorch``
로 부르도록 하겠습니다. 이는 **반드시 절대 경로여야** 합니다. 특히
``CMAKE_PREFIX_PATH`` 를 ``../../libtorch`` 와 같이 설정하면 예상치 못한
오류가 발생할 수 있습니다. 그보다는 ``$PWD/../../libtorch`` 와 같이 해당
절대 경로를 입력하세요. 이제 애플리케이션을 빌드할 준비가 되었습니다.

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

위에서 우리는 먼저 ``dcgan`` 디렉토리 안에 ``build`` 폴더를 만들고
이 폴더에 들어가서 필요한 빌드(Make) 파일을 생성하는 ``cmake`` 명령어를
실행한 후 ``cmake --build . --config Release`` 를 실행하여 프로젝트를
성공적으로 컴파일했습니다. 이제 우리의 작은 바이너리를 실행하고 기본
프로젝트 설정에 대한 이 섹션을 완료할 준비가 됐습니다.

.. code-block:: shell

  root@fa350df05ecf:/home/build# ./dcgan
  1  0  0
  0  1  0
  0  0  1
  [ Variable[CPUFloatType]{3,3} ]

제가 보기엔 항등 행렬인 것 같군요!

신경망 모델 정의하기
-------------------

이제 기본적인 환경을 설정했으니, 이번 튜토리얼에서 훨씬
더 흥미로운 부분을 살펴봅시다. 먼저 C++ 프론트엔드에서 모듈을
정의하고 상호 작용하는 방법에 대해 논의하겠습니다. 기본적인
소규모 예제 모듈부터 시작하여 C++ 프론트엔드가 제공하는 다양한
내장 모듈 라이브러리를 사용하여 완성도 있는 GAN을 구현하겠습니다.

모듈 API 기초
^^^^^^^^^^^^^

파이썬 인터페이스와 마찬가지로, C++ 프론트엔드에 기반을 둔 신경망도
*모듈* 이라 불리는 재사용 가능한 빌딩 블록으로 구성되어 있습니다. 파이썬에
다른 모든 모듈이 파생되는 ``torch.nn.Module`` 라는 기본 모듈 클래스가
있듯이 C++에는 ``torch::nn::Module`` 클래스가 있습니다.
일반적으로 모듈에는 캡슐화된 알고리즘을 구현하는 ``forward()``
메서드를 비롯해 매개변수, 버퍼 및 하위 모듈 세 가지 하위 객체가
포함됩니다.

매개변수와 버퍼는 텐서의 형태로 상태를 저장합니다. 매개변수는 그래디언트를
기록하지만 버퍼는 기록하지 않습니다. 매개변수는 일반적으로 신경망의 학습
가능한 가중치입니다. 버퍼의 예로는 배치 정규화를 위한 평균 및 분산이
있습니다. 특정 논리 및 상태 블록을 재사용하기 위해, PyTorch API는
모듈들이 중첩되는 것을 허용합니다. 중첩된 모듈은 *하위 모듈* 이라고
합니다.

매개변수, 버퍼 및 하위 모듈은 명시적으로 등록(register)을 해야 합니다.
등록이 되면 ``parameters()`` 나 ``buffers()`` 같은 메서드를 사용하여 (중첩을
포함한) 전체 모듈 계층 구조에서 모든 매개변수 묶음을 검색할 수 있습니다.
마찬가지로, ``to(...)`` 와 같은 메서드는 모듈 계층 구조 전체에 대한 메서드입니다.
예를 들어, ``to(torch::kCUDA)`` 는 모든 매개변수와 버퍼를 CPU에서 CUDA 메모리로
이동시킵니다.

모듈 정의 및 매개변수 등록
*************************

이 내용을 코드로 구현하기 위해, 파이썬 인터페이스로 작성된 간단한 모듈 하나를
생각해 봅시다.

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

파이썬에서와 마찬가지로 모듈 기본 클래스에서 파생한 ``Net`` 이라는 클래스를
정의합니다. (쉬운 설명을 위해 ``class`` 대신 ``struct`` 을 사용했습니다.)
파이썬에서 torch.randn을 사용하는 것처럼 생성자에서는 ``torch::randn`` 을
사용해 텐서를 만듭니다. 한 가지 흥미로운 차이점은 매개변수를 등록하는
방법입니다. 파이썬에서는 텐서를 ``torch.nn`` 으로 감싸는 것과 달리,
C++에서는 ``register_parameter`` 메서드를 통해 텐서를 전달해야
합니다. 이러한 차이의 원인은 파이썬 API의 경우, 어떤 속성(attirbute)이
``torch.nn.Parameter`` 타입인지 감지해 그러한 텐서를 자동으로 등록할 수 있기
때문에 나타납니다. C++에서는 리플렉션(reflection)이 매우 제한적이므로 보다
전통적인 (그리하여 덜 마법적인) 방식이 제공됩니다.

서브모듈 등록 및 모듈 계층 구조 탐색
**********************************

매개변수 등록과 마찬가지 방법으로 서브모듈을 등록할 수 있습니다.
파이썬에서 서브모듈은 어떤 모듈의 속성으로 지정될 때 자동으로
감지되고 등록됩니다.

.. code-block:: python

  class Net(torch.nn.Module):
    def __init__(self, N, M):
        super(Net, self).__init__()
        # Registered as a submodule behind the scenes
        self.linear = torch.nn.Linear(N, M)
        self.another_bias = torch.nn.Parameter(torch.rand(M))

    def forward(self, input):
      return self.linear(input) + self.another_bias

예를 들어, ``parameters()`` 메서드를 사용하면 모듈 계층의 모든 매개변수에
재귀적으로 액세스할 수 있습니다.

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

C++에서 ``torch::nn::Linear`` 등의 모듈을 서브모듈로 등록하려면 이름에서
유추할 수 있듯이 ``register_module()`` 메서드를 사용합니다.

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

  ``torch::nn`` 에 대한 `이 문서 <https://pytorch.org/cppdocs/api/namespace_torch__nn.html>`_
  에서 ``torch::nn::Linear``, ``torch::nn::Dropout``, ``torch::nn::Conv2d``
  등 사용 가능한 전체 빌트인 모듈 목록을 확인할 수
  있습니다.

위 코드에서 한 가지 미묘한 사실은 서브모듈은 생성자의 이니셜라이저
목록에 작성되고 매개변수는 생성자의 바디(body)에 작성되었다는
것입니다. 여기에는 충분한 이유가 있으며 아래 C++ 프론트엔드의
*오너십 모델* 섹션에서 더 다룰 예정입니다. 그렇지만 최종 결론은
파이썬에서처럼 모듈 트리의 매개변수에 재귀적으로 액세스할 수
있다는 것입니다. ``parameters()`` 를 호출하면 순회가 가능한
``std::vector<torch::Tensor>`` 가 반환됩니다.

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

파이썬에서와 같이 세 개의 매개변수가 출력됐습니다. 이 매개변수들의 이름을
확인할 수 있도록 C++ API는 ``named_parameters()`` 메서드를 제공하며, 이는
파이썬에서와 같이 ``Orderdict`` 를 반환합니다.

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

  ``torch::nn::Module`` 에 대한 `문서
  <https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#exhale-class-classtorch-1-1nn-1-1-module>`_ 는
  모듈 계층 구조에 대한 메서드 목록 전체가 포함되어
  있습니다.

순전파(forward) 모드로 네트워크 실행
**********************************

네트워크를 C++로 실행하기 위해서는, 우리가 정의한 ``forward()`` 메서드를
호출하기만 하면 됩니다.

.. code-block:: cpp

  int main() {
    Net net(4, 5);
    std::cout << net.forward(torch::ones({2, 4})) << std::endl;
  }

출력은 대략 아래와 같을 것입니다

.. code-block:: shell

  root@fa350df05ecf:/home/build# ./dcgan
  0.8559  1.1572  2.1069 -0.1247  0.8060
  0.8559  1.1572  2.1069 -0.1247  0.8060
  [ Variable[CPUFloatType]{2,5} ]

모듈 오너십 (Ownership)
**********************

이제 우리는 C++에서 모듈을 정의하고, 매개변수를 등록하고, 하위 모듈을
등록하고, ``parameters()`` 등의 메서드를 통해 모듈 계층을 탐색하고,
모듈의 ``forward()`` 메서드를 실행하는 방법을 배웠습니다. C++ API에는
다른 메서드, 클래스, 그리고 주제가 많지만 전체 목록은 `문서
<https://pytorch.org/cppdocs/api/namespace_torch__nn.html>`_ 를
참조하시기 바랍니다. 잠시 후에 DCGAN 모델과 엔드 투 엔드 학습
파이프라인을 구현하면서도 몇 가지 개념을 더 다룰 예정입니다. 그에 앞서
C++ 프론트엔드에서 ``torch::nn::Module`` 의 하위 클래스들에 대해 제공하는
*오너십 모델* 에 대해 간단히 설명하겠습니다.

이 논의에서 오너십 모델이란 모듈을 저장하고 전달하는 방식
(누가 혹은 무엇이 특정 모듈 인스턴스를 소유하는지)을 지칭합니다.
파이썬에서 객체는 항상 힙에 동적으로 할당되며 레퍼런스 시맨틱을
가지는데, 이는 다루고 이해하기가 매우 쉽습니다. 실제로 파이썬에서는
객체가 어디에 존재하고 어떻게 레퍼런스되는지 신경 쓰지 않고 하려는
일에만 집중할 수 있습니다.

저급 언어인 C++는 이 부분에서 더 많은 옵션을 제공합니다. 이는
C++ 프론트엔드의 복잡성을 증가시키며 그 설계와 인체공학적 요소에도
큰 영향을 줍니다. 특히, C++ 프론트엔드 모듈에서는 밸류 시맨틱
*또는* 레퍼런스 시맨틱을 사용할 수 있습니다. 전자가 지금까지의
사례에서 살펴본 가장 단순한 경우로, 모듈 객체가 스택에 할당되고
함수에 전달될 때 레퍼런스 혹은 포인터로 복사 및 이동(``std:move``)
시키거나 가져올 수 있습니다.

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

후자(레퍼런스 시맨틱)의 경우, ``std::shared_ptr`` 를 사용할 수 있습니다.
모든 곳에서 ``shared_ptr`` 를 사용한다는 가정 하에, 레퍼런스 시맨틱의
장점은 파이썬에서와 같이 모듈이 함수에 전달되고 인자가 선언되는 방식에
대해 생각할 부담을 덜어준다는 것입니다.

.. code-block:: cpp

  struct Net : torch::nn::Module {};

  void a(std::shared_ptr<Net> net) { }

  int main() {
    auto net = std::make_shared<Net>();
    a(net);
  }

경험적으로, 동적 언어를 사용하던 연구자들은 비록 밸류 시맨틱이
더 C++에 "네이티브"함에도 불구하고 레퍼런스 시맨틱을 훨씬
선호합니다. 또한 ``torch::nn::Module`` 의 설계는
사용자 친화적인 파이썬 API를 유사하게 따르기 위해 shared 오너십에
의존합니다. 앞서 예시로 들었던 ``Net`` 의 정의를 축약해서 다시
살펴봅시다.

.. code-block:: cpp

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M)))
    { }
    torch::nn::Linear linear;
  };

하위 모듈인 ``linear`` 를 사용하기 위해 이를 클래스에 직접 저장하고자
합니다. 그러나 동시에 모듈의 기초 클래스가 이 하위 모듈에 대해 알고 접근할
수 있기를 원합니다. 이를 위해서는 해당 하위 모듈에 대한 참조를 저장해야 합니다.
이 순간 이미 우리는 shared 오너십을 필요로 합니다. ``torch::nn::Module``
클래스와 구상 클래스인 ``Net`` 모두에서 하위 모듈에 대한 레퍼런스가
필요합니다. 따라서 기초 클래스는 모듈을 ``shared_ptr`` 로 저장하며 이에
따라 구상 클래스 또한 마찬가지일 것입니다.

하지만 잠깐! 위의 코드에는 ``shared_ptr`` 에 대한 언급이 없습니다! 왜 그런
것일까요? 왜냐하면 ``std::shared_ptr<MyModule>`` 는 타이핑하기에 너무 길기 때문입니다.
연구원들의 생산성을 유지하기 위해, 우리는 레퍼런스 시맨틱을 유지하면서 밸류
시맨틱만의 장점인 ``shared_ptr`` 에 대한 언급을 숨기기 위한 정교한 계획을
세웠습니다. 그 작동 방식을 이해하기 위해 코어 라이브러리에 있는 ``torch::nn::Linear``
모듈의 단순화된 정의를 살펴보겠습니다. (전체 정의는
`여기 <https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/modules/linear.h>`_ 에서
확인할 수 있습니다.)

.. code-block:: cpp

  struct LinearImpl : torch::nn::Module {
    LinearImpl(int64_t in, int64_t out);

    Tensor forward(const Tensor& input);

    Tensor weight, bias;
  };

  TORCH_MODULE(Linear);

요약하자면 이 모듈은 ``Linear`` 가 아닌 ``LinearImpl`` 이라고 불립니다. 그리고
``TORCH_MODULE`` 라는 매크로가 실제 ``Linear`` 클래스를 정의합니다. 이렇게 "생성된"
클래스는 ``std::shared_ptr<LinearImpl>`` 를 감싸는 래퍼(wrapper)입니다.
단순한 typedef가 아닌 래퍼이므로 생성자도 여전히 예상하는 대로 작동합니다.
즉, ``std::make_shared<LinearImpl>(3, 4)`` 가 아닌 ``torch::nn::Linear(3, 4)``
라고 쓸 수 있습니다. 이렇게 매크로에 의해 생성된 클래스는 *holder* 모듈이라고
부릅니다. (shared) 포인터와 마찬가지로 화살표 연산자(즉,
``model->forward(...)``)를 사용해 기저 객체에 액세스합니다.
결론적으로 파이썬 API와 매우 유사한 오너십 모델을 얻었습니다.
기본적으로 레퍼런스 시맨틱을 따르지만, ``std:shared_ptr`` 나
``std::make_shared`` 등을 타이핑할 필요가 없습니다. 우리의 ``Net`` 예시에서
모듈 holder API를 사용하면 아래와 같습니다.

.. code-block:: cpp

  struct NetImpl : torch::nn::Module {};
  TORCH_MODULE(Net);

  void a(Net net) { }

  int main() {
    Net net;
    a(net);
  }

여기서 언급할 만한 미묘한 문제가 하나 있습니다. 기본 생성자에 의해 만들어진
``std::shared_ptr`` 는 "비어" 있습니다. 즉, null 포인터입니다. 기본 생성자로
만들어진 ``Linear`` 이나 ``Net`` 은 무엇이어야 할까요? 음, 이건 어려운 결정입니다.
빈 (null) ``std::shared_ptr<LinearImpl>`` 로 정할 수 있습니다. 하지만
``Linear(3, 4)`` 가 ``std::make_shared<LinearImpl>(3, 4)`` 와 같다는 것을 기억합시다.
즉, ``Linear linear;`` 이 null 포인터여야 한다고 결정한다면
생성자에서 인자를 전혀 받지 않거나 모든 인자에 대해 기본값을 사용하는
모듈을 생성할 방법이 없어집니다. 이러한 이유로 현재
API에서 기본 생성자에 의해 만들어진 모듈 holder(``Linear()`` 등)는
기저 모듈(``LinearImpl()``)의 기본 생성자를 호출합니다. 만약
기저 모듈에 기본 생성자가 없으면 컴파일러 오류가 발생합니다.
반대로 빈 holder를 생성하려면 holder 생성자에 ``nullptr`` 를
전달하면 됩니다.

실제로는 앞에서와 같이 하위 모듈을 사용해 모듈을 *이니셜라이저 (initializer) 목록* 에
등록 및 생성하거나,

.. code-block:: cpp

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M)))
    { }
    torch::nn::Linear linear;
  };

파이썬 사용자들에게 더 친숙한 방법으로, 먼저 null 포인터로 홀더를 생성한 이후
생성자에서 값을 지정할 수 있습니다.

.. code-block:: cpp

  struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M) {
      linear = register_module("linear", torch::nn::Linear(N, M));
    }
    torch::nn::Linear linear{nullptr}; // construct an empty holder
  };

결론적으로 어떤 오너십 모델, 어떤 시맨틱을 사용하면 좋을까요? C++
프론트엔드 API는 모듈 holder가 제공하는 오너십 모델을 가장 잘 지원합니다.
이 메커니즘의 유일한 단점은 모듈 선언 아래에 boilerplate 한 줄이
추가된다는 것입니다. 즉, 가장 단순한 모델은 C++ 모듈의 기초를 배울 떄
나오는 밸류 시맨틱 모델입니다. 작고 간단한 스크립트의 경우,
이것만으로 충분할 수 있습니다. 그러나 언젠가는 기술적 이유로 인해
이 기능이 항상 지원되지는 않는다는 사실을 알게 될 것입니다. 예를 들어 직렬화
API(``torch::save`` 및 ``torch::load``)는 모듈 holder(혹은 일반
``shared_ptr``)만을 지원합니다. 따라서 C++ 프론트엔드로 모듈을
정의할 떄에는 모듈 holder API 방식이 권장되며, 앞으로 본 튜토리얼에서
이 API를 사용하겠습니다.

DCGAN 모듈 정의하기
^^^^^^^^^^^^^^^^

이제 이 글에서 해결하려는 머신러닝 태스크를 위한 모듈을 정의하는데
필요한 배경과 도입부 설명이 끝났습니다. 다시 상기하자면, 우리의 태스크는
`MNIST 데이터셋  <http://yann.lecun.com/exdb/mnist/>`_ 의 숫자 이미지를
생성하는 것입니다. 우리는 이 태스크를 풀기 위해
`적대적 생성 신경망(GAN) <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`_ 을
사용하고자 합니다. 그 중에서도 우리는 `DCGAN 아키텍처
<https://arxiv.org/abs/1511.06434>`_ 를 사용할 것입니다.
DCGAN은 가장 초기에 발표됐던 제일 간단한 GAN이지만 이 태스크를 위해서는
충분합니다.

.. tip::

  이 튜토리얼에 나온 소스 코드 전체는 `이 저장소
  <https://github.com/pytorch/examples/tree/master/cpp/dcgan>`_ 에서 확인할 수 있습니다.

GAN이 뭐였죠?
***********

GAN은 *생성기(generator)* 와 *판별기(discriminator)* 라는
두 가지 신경망 모델로 구성됩니다. 생성기는 노이즈 분포에서 샘플을 입력받고,
각 노이즈 샘플을 목표 분포(이 경우 MNIST 데이터셋)와 유사한 이미지로
변환하는 것이 목표입니다. 판별기는 MNIST 데이터셋의 *진짜*
이미지를 입력받거나 생성기로부터 *가짜* 이미지를 입력받습니다.
그리고 어떤 이미지가 얼마나 진짜같은지 (``1`` 에 가까운 출력)
혹은 가짜같은 지 (``0`` 에 가까운 출력) 판별합니다. 생성기가
만든 이미지가 얼마나 진짜같은 지 판별기가 피드백하고 이 피드백은 생성기
학습에 사용됩니다. 판별기가 진짜에 대한 안목이 얼마나 좋은 지에
대한 피드백은 판별기를 최적화하기 위해 사용됩니다. 이론적으로,
생성기와 판별기 사이의 섬세한 균형은 이 둘을 동시에 개선시킵니다.
이를 통해 생성기는 목표 분포와 구별할 수 없는 이미지를 생성하고,
(그때쯤이면) 잘 학습되어 있을 판별기의 안목을 속여 진짜와 가짜
이미지 모두에 대해 ``0.5`` 의 확률을 출력할 것입니다. 최종
결과물은 노이즈를 입력받아 실제 숫자의 이미지를 출력으로 생성하는
기계입니다.

생성기 (Generator) 모듈
********************

먼저 일련의 전치된 (transposed) 2D 합성곱, 배치 정규화 및
ReLU 활성화 유닛으로 구성된 생성기 모듈을 정의하겠습니다.
모듈의 ``forward()`` 메서드를 직접 정의하여 모듈 간 입력을
(함수형으로) 명시적으로 전달합니다.

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

이제 ``DCGANGenerator`` 의 ``forward()`` 를 호출해 노이즈 샘플을 이미지에 매핑할 수 있습니다.

여기서 사용한 ``nn::ConvTranspose2d`` 및 ``nn::BatchNorm2d`` 등의 모듈은
앞서 설명한 구조를 따릅니다. 상수 ``kNoiseSize`` 는 입력 노이즈 벡터의 크기를
결정하며 ``100`` 으로 설정됩니다. 하이퍼파라미터는 물론 대학원생들의 많은 노력을
통해 세팅됐습니다.

.. attention::

	하이퍼파라미터를 정하느라 다친 대학원생은 없었습니다. 그들은 서로서로 개사료를 먹이니까요.

.. note::

  C++ 프론트엔드의 ``Conv2d`` 와 같은 기본 제공 모듈에 옵션이 전달되는 방법에 대한
  간단히 설명하자면, 모든 모듈은 몇 가지 필수 옵션을 갖고 있습니다. (예: ``BatchNorm2d`` 의
  feature 개수) 만약 ``BatchNorm2d(128)``, ``Dropout(0.5)``, ``Conv2d(8, 4, 2)`` 와
  같이 필수 옵션만 설정하려 한다면 모듈 생성자에 직접 전달할 수 있습니다.
  (여기서는 각각 입력 채널 수, 출력 채널 수 및 커널 크기를 의미)
  그러나 만약 ``Conv2d`` 의 ``bias`` 와 같이 일반적으로 기본값을 사용하는
  다른 옵션을 수정해야 하는 경우, *options* 객체를 생성해 전달해야 합니다.
  C++ 프론트엔드의  모듈은 ``ModuleOptions`` 이라고 하는 연관된 옵션 struct를
  가지고 있습니다. 여기서 ``Module`` 은 해당 모듈의 이름으로, 예를 들어 ``Linear``
  의 경우 ``LinearOptions`` 와 같습니다. 우리는 위의 ``Conv2d`` 모듈에
  대해 이를 수행한 것입니다.


판별기(Discriminator) 모듈
************************

판별기는 마찬가지로 합성곱, 배치 정규화 및 활성화의
연속입니다. 하지만 이번에 합성곱은 전치되지 않은 기본
합성곱이며, 일반적 ReLU 대신에 알파 값이 0.2인 leaky ReLU를
사용합니다. 또한 최종 활성화는 값을 0과 1 사이의 범위로 압축하는
Sigmoid가 됩니다. 그런 다음 이렇게 압축된 값을 판별자가
이미지에 대해 출력하는 확률로 해석할 수 있습니다.

판별기를 만들기 위해 `Sequential` 모듈이라는 다른 것을 시도해 보겠습니다.
파이썬에서와 같이, PyTorch는 모델 정의를 위해 두 가지 API를 제공합니다.
(생성기 모듈 예시와 같이) 입력이 연속적인 함수를 통해 전달되는 함수형 API와
전체 모델을 하위 모듈로 포함하는 `Sequential` 모듈을 생성하는 객체 지향형
API입니다. `Sequential` 을 사용하면 판별기는 대략 다음과 같습니다.

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

  ``Sequential`` 모듈은 단순한 함수 합성만을 수행합니다. 첫 번째 하위 모듈의 출력은
  두 번째 하위 모듈의 입력이 되고 세 번째 하위 모듈의 출력은 네 번째 하위 모듈의 입력이
  되고 이후에도 마찬가지입니다.


데이터 불러오기
------------

이제 생성기와 판별기 모델을 정의했으므로 이러한 모델을 학습시킬
데이터가 필요합니다. 파이썬과 마찬가지로 C++ 프론트엔드는
강력한 병렬 데이터 로더(data loader)를 제공한다. 이 데이터 로더는
사용자가 직접 정의할 수 있는 데이터셋에서 데이터 배치를 읽을 수 있으며
설정을 위한 많은 옵션을 제공합니다.

.. note::

	파이썬 데이터 로더가 멀티 프로세싱을 사용하는 반면, C++ 데이터 로더는 실제로 멀티 스레딩을 사용해 어떠한 새로운 프로세스도 시작하지 않습니다.

데이터 로더는 ``torch::data::`` 네임스페이스에 포함된 C++ 프론트엔드의
``data`` API의 일부입니다. 이 API는 다음과 같은 몇 가지 컴포넌트로 구성됩니다.

- 데이터 로더 클래스
- 데이터셋을 정의하기 위한 API
- *변환* 을 정의하기 위한 API (데이터셋에 적용 가능)
- *샘플러* 를 정의하기 위한 API (데이터셋을 위한 인덱스를 생성)
- 기존 데이터셋, 변환, 샘플러들의 라이브러리

이 튜토리얼에서는 C++ 프론트엔드와 함께 제공되는 ``MNIST`` 데이터셋을
사용합니다. ``torch::data::datasets::MNIST`` 인스턴스를 만들어
다음 두 가지 변환을 적용해봅시다. 첫째, 이미지를 정규화하여 ``-1`` 과
``+1`` 사이에 있도록 합니다. (기존 범위는 ``0`` 과 ``1`` 사이)
둘째, 텐서 배치(batch)를 첫 번째 차원을 따라 단일 텐서로 쌓는 이른바
``Stack`` *collation* 을 적용합니다.

.. code-block:: cpp

  auto dataset = torch::data::datasets::MNIST("./mnist")
      .map(torch::data::transforms::Normalize<>(0.5, 0.5))
      .map(torch::data::transforms::Stack<>());

MNIST 데이터셋은 학습 바이너리 실행 위치를 기준으로 ``./mnist``
디렉토리에 위치해야 합니다. MNIST 데이터셋은 `이 스크립트
<https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03>`_ 를
사용해 다운로드할 수 있습니다.

다음으로, 데이터 로더를 만들고 이 데이터셋을 전달합니다. 새로운 데이터
로더를 만들기 위해 ``torch::data::make_data_loader`` 를 사용합니다.
이 로더는 올바른 타입(데이터셋 타입, 샘플러 타입 및 기타 구현 세부사항에
따라 결정됨)의 ``std::unique_ptr`` 를 반환합니다.

.. code-block:: cpp

  auto data_loader = torch::data::make_data_loader(std::move(dataset));

데이터 로더에는 많은 옵션이 제공됩니다. 전체 목록은 `여기
<https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/dataloader_options.h>`_
에서 확인할 수 있습니다.
예를 들어 데이터 로딩 속도를 높이기 위해 작업자 수를 늘릴 수
있습니다. 기본값은 0이며, 이는 주 쓰레드가 사용됨을 의미합니다.
``workers`` 를 ``2`` 로 설정하면 데이터를 동시에 로드하는 쓰레드가
두 개 생성됩니다. 또한 배치 크기를 기본값 ``1`` 에서 ``64`` (``kBatchSize`` 값)
와 같이 더 적당한 값으로 늘려야 합니다. 그러면
``DataLoaderOptions`` 객체를 만들어 적절한 속성을 설정해 보겠습니다.

.. code-block:: cpp

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));


이제 데이터 배치를 로드하는 루프를 작성할 수 있습니다. 지금은
콘솔에만 출력할 것입니다.

.. code-block:: cpp

  for (torch::data::Example<>& batch : *data_loader) {
    std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    for (int64_t i = 0; i < batch.data.size(0); ++i) {
      std::cout << batch.target[i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
  }

이 경우 데이터 로더가 반환하는 타입은 ``torch::data::Example`` 입니다.
이 타입은 데이터를 위한 ``data`` 필드와 레이블을 위한 ``target`` 필드가
있는 간단한 struct입니다. 앞서 ``Stack`` collation을 적용했기 때문에,
데이터 로더는 이 example을 하나만 반환합니다. 데이터 로더에 collation을
적용하지 않으면, ``std::vector<torch::data::Example<>>`` 를 yield하며,
각 배치의 example에는 하나의 element가 있을 것입니다.

이 코드를 다시 빌드하고 실행하면 대략 다음과 같은 내용을 얻을 것입니다.

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

즉, MNIST 데이터셋에서 데이터를 성공적으로 로드할 수 있습니다.

학습 루프 작성하기
-----------------

이제 예제의 알고리즘 부분을 마무리하고 생성기와 판별기 사이에서 일어나는 섬세한
작용을 구현해 보겠습니다. 먼저 생성기와 판별기 각각을 위해
총 두 개의 optimizer를 생성하겠습니다. 우리가 사용하는
optimizer는 `Adam <https://arxiv.org/pdf/1412.6980.pdf>`_ 알고리즘을 구현합니다.

.. code-block:: cpp

  torch::optim::Adam generator_optimizer(
      generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
  torch::optim::Adam discriminator_optimizer(
      discriminator->parameters(), torch::optim::AdamOptions(5e-4).beta1(0.5));

.. note::

	이 글 작성 당시, C++ 프론트엔드가 Adagrad, Adam, LBFGS, RMSprop 및 SGD를 구현하는 옵티마이저를 제공합니다. 최신 리스트는 `docs <https://pytorch.org/cppdocs/api/namespace_torch__optim.html>`_ 에 있습니다.

다음으로, 우리의 학습 루프를 수정해야 합니다. 매 에폭마다 데이터 로더를 반복 실행하는
바깥 루프를 추가해 다음의 GAN 학습 코드를 작성합니다.

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

위 코드는 먼저 진짜 (real) 이미지에 대해 판별기를 평가하는데, 이 때
판별기는 높은 확률을 출력해야 합니다. 이를 위해
``torch::empty(batch.data.size(0)).uniform_(0.8, 1.0)`` 를 목표 확률
값으로 사용합니다.

.. note::

  판별기를 보다 견고하게 학습하기 위해 모든 곳에서 1.0이 아닌 0.8과 1.0 사이의 균일 분포에서 임의의 값을 선택합니다. 이 트릭을 *label smoothing* 이라고 합니다.

판별기를 평가하기에 앞서 매개변수의 그래디언트를 0으로 만듭니다.
손실을 계산한 후 ``d_loss.backward()`` 를 호출해 이를
네트워크에 역전파합니다. 가짜 (fake) 이미지들에 대해서 이 과정을
반복합니다. 데이터셋의 이미지를 사용하는 대신, 생성자에
무작위 노이즈를 입력하여 여기서 사용할 가짜 이미지를 만듭니다.
그리고 그 가짜 이미지들을 판별기에 전달합니다. 이번에는
판별기가 낮은 확률, 이상적으로는 모두 0을 출력하기를 바랍니다.
진짜 이미지와 가짜 이미지 배치 모두에 대한 판별기 손실을 계산한
후에는, 판별기의 optimizer 매개변수 업데이트를 한 단계씩
진행할 수 있습니다.

생성기를 학습시키기 위해 우선 그래디언트를 다시 한번 0으로 설정하고
다시 가짜 이미지로 판별기를 평가합니다. 그러나 이번에는 판별기가
확률 1에 매우 근접하게 출력하게 하여, 생성기가 판별기를
속여 실제 (데이터셋에 있는) 진짜라고 생각하는 이미지를 생성할 수
있도록 하려 합니다. 이를 위해 ``fake_labels`` 텐서를 모두
1로 채우겠습니다. 마지막으로 매개변수를 업데이트하기 위해
생성기의 optimzier 매개변수 업데이트를 진행합니다.

이제 CPU로 모델을 학습시킬 준비가 되었습니다. 상태나 샘플 출력을
캡처할 수 있는 코드는 아직 없지만 잠시 후에 추가하겠습니다. 지금은
모델이 *무언가* 를 수행하고 있다는 것만을 관찰하고, 나중에는 생성된
이미지를 기반으로 이 무언가가 의미 있는지 여부를 확인할 것입니다.
다시 빌드하고 실행하면 다음과 같은 내용이 출력돼야 합니다.

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

GPU로 이동하기
--------------

이 스크립트는 CPU에서 잘 동작하지만, 합성곱 연산이 GPU에서 훨씬 빠르다는
것은 잘 알려진 사실입니다. 어떻게 학습을 GPU로 옮길 수 있을 지에 대해 빠르게 논의해
보겠습니다. 이를 위해 해야 할 일 두 가지로 GPU 장치(device) 사양을 우리가 직접 할당한
텐서에 전달하는 것과, C++ 프론트엔드의 모든 텐서와 모듈이 갖고 있는 ``to()``
메서드를 사용해 다른 모든 텐서를 GPU에 명시적으로 복사하는 것이 있습니다.
두 가지를 모두 달성하는 가장 간단한 방법으로 학습 스크립트 최상위에
``torch::Device`` 인스턴스를 만들어 ``torch::zeros`` 와 같은
텐서 팩토리 함수나 ``to()`` 메서드에 전달할 수 있습니다. 먼저 CPU device로
이를 구현해보겠습니다.

.. code-block:: cpp

  // 학습 스크립트 최상단에 이 코드를 넣으세요.
  torch::Device device(torch::kCPU);

아래와 같은 새로운 텐서 할당의 경우,

.. code-block:: cpp

  torch::Tensor fake_labels = torch::zeros(batch.data.size(0));

마지막 인자로 ``device`` 를 받도록 수정합니다.

.. code-block:: cpp

  torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);

MNIST 데이터셋의 텐서처럼 우리가 직접 생성하지 않는 텐서에서는
명시적으로 ``to()`` 호출을 삽입해야 합니다. 따라서 아래 코드의 경우,

.. code-block:: cpp

  torch::Tensor real_images = batch.data;

다음과 같이 변합니다.

.. code-block:: cpp

  torch::Tensor real_images = batch.data.to(device);

또한, 모델 매개변수를 올바른 장치로 옮겨야 합니다.

.. code-block:: cpp

  generator->to(device);
  discriminator->to(device);

.. note::

	만일 텐서가 이미 ``to()`` 에 전달된 장치 상에 있다면 그 호출은 아무 일도 하지 않습니다. 사본이 생성되지도 않습니다.

이제 CPU에서 실행되는 이전의 코드가 보다 명시적으로 바뀌었습니다.
하지만 이제는 장치를 CUDA 장치로 변경하는 것 또한 매우 쉽습니다.

.. code-block:: cpp

  torch::Device device(torch::kCUDA)

이제 모든 텐서가 GPU에 존재하며 어떠한 다운스트림 코드 변경 없이도
모든 연산을 위해 빠른 CUDA 커널을 호출합니다. 특정 인덱스의 장치를
지정하려면 ``Device`` 생성자의 두 번째 인자로 전달하면 됩니다.
서로 다른 장치에 서로 다른 텐서가 존재하기를 원하는 경우,
별도의 장치 인스턴스(예: CUDA 장치 0과 다른 CUDA 장치 1)를
전달할 수도 있습니다. 뿐만 아니라, 이러한 설정을 동적으로 수행할 수도
있어 다음과 같이 학습 스크립트의 휴대성을 높이는 데 종종 유용하게 사용됩니다.

.. code-block:: cpp

  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  }

나아가 아래와 같은 코드도 가능합니다.

.. code-block:: cpp

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

학습 상태 저장 및 복원하기
------------------------

마지막으로 학습 스크립트에 추가해야 할 내용은 모델 매개변수 및
옵티마이저의 상태, 그리고 생성된 몇 개의 이미지 샘플을
주기적으로 저장하는 것입니다. 학습 과정 도중에 컴퓨터가 다운되면
이렇게 저장된 상태로부터 학습 상태를 복원할 수 있습니다.
이는 장시간 지속되는 학습을 위해 필수로 요구됩니다. 다행히도
C++ 프론트엔드는 개별 텐서뿐만 아니라 모델 및 옵티마이저 상태를
직렬화하고 역직렬화할 수 있는 API를 제공합니다.

이를 위한 핵심 API는 ``torch::save(thing,filename)`` 와
``torch::load(thing,filename)`` 로, 여기서 ``thing`` 은
``torch::nn::Module`` 의 하위 클래스 혹은 우리의 학습 스크립트의 ``Adam``
객체와 같은 옵티마이저 인스턴스가 될 수 있습니다. 모델 및 옵티마이저 상태를
특정 주기마다 저장하도록 학습 루프를 수정해보겠습니다.

.. code-block:: cpp

  if (batch_index % kCheckpointEvery == 0) {
    // 모델 및 옵티마이저 상태를 저장합니다.
    torch::save(generator, "generator-checkpoint.pt");
    torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::save(discriminator, "discriminator-checkpoint.pt");
    torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
    // 생성기를 샘플링하고 이미지를 저장합니다.
    torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
    torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
    std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
  }

여기서 ``100`` 배치마다 상태를 저장하려면 ``kCheckpointEvery`` 를 ``100``
과 같은 정수로 설정할 수 있으며, ``checkpoint_counter`` 는 상태를 저장할 때마다
증가하는 카운터입니다.

학습 상태를 복원하기 위해 모델 및 옵티마이저를 모두 생성한 후 학습 루프 앞에
다음 코드를 추가할 수 있습니다.

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


생성한 이미지 검사하기
--------------------

학습 스크립트가 완성되어 CPU에서든 GPU에서든 GAN을 훈련시킬 준비가
됐습니다. 학습 과정의 중간 출력을 검사하기 위해
``"dcgan-sample-xxx.pt"`` 에 주기적으로 이미지 샘플을 저장하는 코드를
추가했으니, 텐서들을 불러와 matplotlib로 시각화하는 간단한 파이썬
스크립트를 작성해보겠습니다.

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

이제 모델을 약 30 에폭 정도 학습시킵시다.

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

숫자네요! 만세! 이제 여러분 차례입니다. 숫자가 보다 나아 보이도록
모델을 개선할 수 있나요?

결론
-----

이 튜토리얼을 통해 PyTorch C++ 프론트엔드에 대한 어느 정도 이해도가 생기셨기
바랍니다. 필연적으로 PyTorch 같은 머신러닝 라이브러리는 매우 다양하고
광범위한 API를 가지고 있습니다. 따라서, 여기서 논의하기에 시간과 공간이
부족했던 개념들이 많습니다. 그러나 직접 API를 사용해보고,
`문서 <https://pytorch.org/cppdocs/>`_, 그 중에서도 특히
`라이브러리 API <https://pytorch.org/cppdocs/api/library_root.html>`_
섹션을 참조해보는 것을 권장드립니다. 또한, C++ 프론트엔드가 파이썬
프론트엔드의 디자인과 시맨틱을 따른다는 사실을 잘 기억하면 보다 빠르게
학습할 수 있을 것입니다.

.. tip::

  본 튜토리얼에 대한 전체 소스코드는 `이 저장소
  <https://github.com/pytorch/examples/tree/master/cpp/dcgan>`_ 에 제공되어 있습니다.

언제나 그렇듯이 어떤 문제가 생기거나 질문이 있으면 저희
`포럼 <https://discuss.pytorch.org/>`_ 을 이용하거나 `Github 이슈
<https://github.com/pytorch/pytorch/issues>`_ 로 연락주세요.
