사용자 정의 C++ 연산자로 TORCHSCRIPT 확장하기
===============================================

PyTorch 1.0 릴리스는 PyTorch에 `TorchScript <https://pytorch.org/docs/master/jit.html>`_ 라는 새로운 
프로그래밍 모델을 도입하였습니다. TorchScript는 TorchScript 컴파일러에서 구문 분석, 컴파일 및 최적화할 수 있는 Python
프로그래밍 언어의 하위 집합입니다. 또한 컴파일된 TorchScript 모델에는 디스크에 있는 파일 형식으로 직렬화할 수 있는 옵션이 
있으며, 추론(inference)을 위해 순수 C++(Python뿐만 아니라)에서 로드하고 실행할 수 있습니다.

TorchScript는 ``torch`` 패키지에서 제공하는 작업의 큰 부분 집합을 지원 하므로 PyTorch의 "표준 라이브러리"에서 순수하게 
일련의 텐서 작업으로 많은 종류의 복잡한 모델을 표현할 수 있습니다. 그럼에도 불구하고 사용자 정의 C++ 또는 CUDA 기능으로 
TorchScript를 확장해야 하는 경우가 있습니다. 아이디어를 간단한 Python 함수로 표현할 수 없는 경우에만 이 옵션을 사용하는 
것이 좋지만 PyTorch의 고성능 C++ 텐서 라이브러리인 `ATen <https://pytorch.org/cppdocs/#aten>`_ 을 사용하여 
사용자 지정 C++ 및 CUDA 커널을 정의하기 위한 매우 친숙하고 간단한 인터페이스를 제공합니다. TorchScript에 바인딩되면 이러한 
사용자 지정 커널(또는 "ops")을 TorchScript 모델에 포함하고 Python 및 직렬화된 형식으로 C++에서 직접 실행할 수 있습니다.

다음 단락 에서는 C++로 작성된 컴퓨터 비전 라이브러리인 `OpenCV <https://www.opencv.org>`_ 를 호출하는 TorchScript 
사용자 지정 작업을 작성하는 예를 보여줍니다. C++에서 텐서를 사용하는 방법, 타사 텐서 형식(이 경우 OpenCV ``Mat`` ) 으로 
효율적으로 변환하는 방법, TorchScript 런타임에 연산자를 등록하는 방법, 마지막으로 Python과 C++에서 연산자를 컴파일하고 
사용하는 방법에 대해 설명합니다. 

C++에서 사용자 정의 연산자 구현
---------------------------------------

이 튜토리얼에서는 OpenCV에서 TorchScript로 이미지에 투영 변환을 적용하는 `warpPerspective <https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective>`_ 
함수를 사용자 정의 연산자로 노출합니다. 첫 번째 단계는 C++로 사용자 정의 연산자 구현을 작성하는 것입니다. 이 구현 ``op.cpp`` 
을 위한 파일을 호출하고 이 구현을 위한 파일을 호출하고 다음과 같이 보이도록 합시다 : 

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN warp_perspective
  :end-before: END warp_perspective

이 연산자의 코드는 매우 짧습니다. 파일 맨 위에 OpenCV 헤더 파일이 포함되어 있으며 ``opencv2/opencv.hpp`` 와 ``torch/
script.h`` 헤더와 함께 사용자 지정 TorchScript 연산자를 작성하는 데 필요한 PyTorch의 C++ API에서 필요한 모든 
항목을 노출합니다.

우리의 함수 ``warp_perspective`` 는 두 가지 인수를 취합니다: 입력 ``image`` 와 이미지에 적용하고자 하는 ``warp`` 
변환 행렬. 
이러한 입력 유형은 ``torch::Tensor`` , C++에서 PyTorch의 텐서 유형(파이썬에서 모든 텐서의 기본 유형이기도 함)입니다. 
 ``warp_perspective`` 함수의 반환 유형 또한 ``torch::Tensor`` 입니다. 

.. tip::

  `이 노트 <https://pytorch.org/cppdocs/notes/tensor_basics.html>`_ 에는 ``Tensor`` 클래스를 제공하는 
  라이브러리인 ATen에 대한 자세한 내용이 있습니다. 또, `이 튜토리얼 <https://pytorch.org/cppdocs/notes/tensor_creation.html>`_ 에서는 C++에서 새 텐서 개체를 할당하고 초기화하는 방법을 설명합니다(이 연산자에는 필요하지 않음).

.. attention::

  TorchScript 컴파일러는 고정된 수의 유형을 이해합니다. 이러한 유형만 사용자 지정 연산자에 대한 인수로 사용할 수 있습니다. 
  현재 이러한 유형은 다음과 같습니다: ``torch::Tensor`` , ``torch::Scalar`` , ``double`` , ``int64_t`` 및 ``std::vector`` 의 이러한 유형들. ``float`` 가 아니라 *오직* ``double`` 이며, ``int`` , ``short`` 이나 ``long`` 처럼 다른 정수타입이 아닌 *오직*  
  ``int64_t`` 을 지원합니다. 

함수 내부에서 가장 먼저 해야 할 일은 PyTorch 텐서를 OpenCV 행렬로 변환하는 것 입니다. OpenCV ``warpPerspective`` 는 ``cv::Mat`` 객체를 입력으로 기대하기 때문입니다. 다행히 데이터를 **복사하지 않고** 이 작업을 수행할 수 있는 방법이 있습니다. 
처음 몇 줄에는,

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: image_mat 시작
  :end-before: image_mat 완료

우리의 텐서를 ``Mat`` 객체로 변환하기 위해 OpenCV ``Mat`` 클래스의 `이 생성자 <https://docs.opencv.org/trunk/d3/d63/classcv_1_1Mat.html#a922de793eabcec705b3579c5f95a643e>`_ 를 호출합니다. 우리는 오리지널 ``이미지`` 
텐서의 행과 열의 수, 데이터 유형(이 예제에서는 ``float32`` 로 고칠 것), 그리고 마지막으로 기본 데이터에 대한 원시 포인터인 
-- a ``float*`` 를 전달합니다. 이 ``Mat``  클래스 생성자의 특별한 점은 입력 데이터를 복사하지 않는다는 것입니다. 대신 ``Mat`` 에서 수행된 모든 작업에 대해 이 메모리를 참조합니다. ``image_mat`` 에서 제자리 작업을 수행 하면 원본 ``이미지`` 
텐서에 반영됩니다. (반대의 경우도 마찬가지) 이것은 우리가 실제로 데이터를 PyTorch 텐서에 저장하고 있더라도 라이브러리의 기본 
매트릭스 유형으로 후속 OpenCV 루틴을 호출할 수 있도록 합니다. ``warp`` PyTorch 텐서를 ``warp_mat`` OpenCV 
매트릭스로 변환하기 위해 이 절차를 반복합니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: warp_mat 시작
  :end-before: warp_mat 끝

다음으로 TorchScript에서 사용하고 싶었던 OpenCV 함수를 호출할 준비가 되었습니다: ``warpPerspective`` . 이를 위해 
OpenCV 함수 ``image_mat`` 와 ``warp_mat`` 매트릭스, 빈 출력 매트릭스인 ``output_mat`` 를 전달합니다.  또한 출력 
매트릭스(이미지)의 원하는 크기 ``dsize`` 를 지정합니다 . 이 예제에서는 다음 ``8 x 8`` 과 같이 하드코딩됩니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: output_mat 시작
  :end-before: output_mat 끝

사용자 정의 연산자 구현의 마지막 단계는 ``output_mat`` 을 PyTorch에서 더 사용할 수 있도록 다시 PyTorch 텐서로 변환하는 
것입니다. 이것은 우리가 다른 방향으로 변환하기 위해 이전에 수행한 것과 놀랍도록 유사합니다. 이 경우 PyTorch에서 ``torch::from_blob`` 메소드를 제공합니다. 우리가 PyTorch 텐서로 해석하려는 *blob* 은 메모리에 약간 불투명한, 평면 포인터를 의미합니다. ``torch::from_blob`` 에 대한 호출은 다음과 같습니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: output_tensor 시작
  :end-before: output_tensor 끝

우리는 OpenCV ``Mat`` 클래스의 ``.ptr<float>()`` 메서드를 사용하여 기본 데이터에 대한 원시 포인터를 얻습니다.(이전의 PyTorch 텐서 ``.data_ptr<float>()`` 와 마찬가지로). 우리는 또한 ``8 x 8`` 처럼 하드코딩한 텐서의 출력 형태를 지정합니다. ``torch::from_blob`` 의 출력은 OpenCV 매트릭스가 소유한 메모리를 가리키는 ``torch::Tensor`` 입니다.

연산자 구현에서 이 텐서를 반환하기 전에, 기본 데이터의 메모리 복사를 수행하기 위해 ``.clone()`` 를 호출해야 합니다 . 그 
이유는 ``torch::from_blob`` 는 데이터를 소유하지 않는 텐서를 반환하기 때문입니다 . 그 시점에서 데이터는 여전히 OpenCV 
매트릭스에 의해 소유됩니다. 그러나 이 OpenCV 매트릭스는 범위를 벗어나 함수가 끝날 때 할당이 해제됩니다. ``output`` 텐서를 
있는 그대로 반환 하면 함수 외부에서 사용할 때까지 유효하지 않은 메모리를 가리킬 것입니다. ``.clone()`` 을 호출하면 새 텐서가 
자체적으로 소유한 원본 데이터의 복사본과 함께 새 텐서를 반환합니다. 따라서 바깥으로 돌아가는 것은(반환하는 것은) 안전합니다.

TorchScript에 사용자 정의 연산자 등록
------------------------------------------------

이제 C++에서 사용자 정의 연산자를 구현 했으므로 TorchScript 런타임 및 컴파일러에 *등록* 해야 합니다 . 이를 통해 
TorchScript 컴파일러는 TorchScript 코드에서 사용자 지정 연산자에 대한 참조를 확인할 수 있습니다. pybind11 라이브러리를 
사용한 적이 있다면 등록 구문이 pybind11 구문과 매우 유사합니다. 단일 함수를 등록하려면 다음과 같이 작성합니다:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: registry 시작
  :end-before: registry 끝

``op.cpp`` 파일의 최상위 레벨 어딘가에 있습니다. ``TORCH_LIBRARY`` 매크로는 프로그램이 시작될 때 호출되는 함수를 
작성합니다.  라이브러리 이름( ``my_ops`` )이 첫 번째 인수로 제공됩니다(따옴표로 묶지 않아야 함). 두 번째 인수(``m``) 는 
연산자를 등록하기 위한 기본 인터페이스 유형 ``torch::Library`` 의 변수를 정의합니다.
이 메서드 ``Library::def`` 는 실제로 ``warp_perspective`` 라는 연산자를 생성하여,Python과 TorchScript에 모두 
노출합니다. ``def`` 를 여러 번 호출하여 원하는 만큼 연산자를 정의할 수 있습니다.

뒤에서 def함수는 실제로 꽤 많은 작업을 수행하고 있습니다: 템플릿 메타프로그래밍을 사용하여 함수의 유형 특징을 검사하고 이를 
TorchScript의 유형 시스템 내에서 연산자 유형을 지정하는 연산자 스키마로 변환합니다.

사용자 정의 연산자 빌드
----------------------------

이제 C++로 사용자 정의 연산자를 구현하고 등록 코드를 작성했으므로 연구 및 실험을 위해 Python으로 로드하거나 Python이 아닌 
환경에서 추론을 위해 C++로 로드할 수 있는 (공유) 라이브러리로 연산자를 빌드할 때입니다. 순수 CMake 또는 ``setuptools`` 
과 같은 Python 대안을 사용하여 연산자를 빌드하는 여러 가지 방법이 있습니다. 간결함을 위해 아래 단락에서는 CMake 접근 방식에 
대해서만 설명합니다. 이 튜토리얼의 부록에서는 다른 대안을 다룹니다.

환경 설정
*****************

PyTorch와 OpenCV를 설치해야 합니다. 둘 다를 가장 쉽고 가장 플랫폼에 독립적으로 얻을 수 있는 방법 Conda:: 

  conda install -c pytorch pytorch
  conda install opencv

CMake로 빌드
*******************

`CMake <https://cmake.org>`_ 빌드 시스템을 사용하여 사용자 지정 연산자를 공유 라이브러리에 빌드하려면 짧은 
``CMakeLists.txt`` 파일 을 작성 하고 이전  ``op.cpp`` 파일과 함께 배치해야 합니다. 이를 위해 다음과 같은 디렉토리 
구조에 동의합니다 ::

  warp-perspective/
    op.cpp
    CMakeLists.txt

``CMakeLists.txt`` 파일의 내용은 다음과 같아야 합니다. : 

.. literalinclude:: ../advanced_source/torch_script_custom_ops/CMakeLists.txt
  :language: cpp


이제 연산자를 빌드하기 위해 ``warp_perspective`` 폴더에서 다음 명령을 실행할 수 있습니다:

.. code-block:: shell

  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
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
  -- Found torch: /libtorch/lib/libtorch.so
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /warp_perspective/build
  $ make -j
  Scanning dependencies of target warp_perspective
  [ 50%] Building CXX object CMakeFiles/warp_perspective.dir/op.cpp.o
  [100%] Linking CXX shared library libwarp_perspective.so
  [100%] Built target warp_perspective

``build`` 폴더에 ``libwarp_perspective.so`` 공유 라이브러리 파일을 저장합니다. 위의 ``cmake`` 명령에서는 helper 
변수 ``torch.utils.cmake_prefix_path`` 를 사용하여 PyTorch 설치를 위한 cmake 파일이 어디에 있는지 편리하게 
알려줍니다.

아래에서 연산자를 사용하고 호출하는 방법을 자세히 살펴보겠지만 더 일찍 성공을 느껴보기 위해 Python에서 다음 코드를 실행할 수 
있습니다. : 

.. literalinclude:: ../advanced_source/torch_script_custom_ops/smoke_test.py
  :language: python

모두 잘되었다면 다음과 같이 인쇄됩니다. ::

  <built-in method my_ops::warp_perspective of PyCapsule object at 0x7f618fc6fa50>

이것은 나중에 사용자 정의 연산자를 호출하는 데 사용할 Python 함수입니다.

Python에서 TorchScript 사용자 지정 연산자 사용
-----------------------------------------------

일단 사용자 지정 연산자를 공유 라이브러리에 만들어 내면 파이썬의 TorchScript 모델에서 연산자를
사용할 수 있습니다. 먼저 연산자를 파이썬에 로드하고, TorchScript 코드에서 연산자를 사용합니다.

연산자를 Python으로 가져오는 방법은 이미 보았듯이 ``torch.ops.load_library()`` 입니다.
이 함수는 사용자 지정 연산자가 포함된 공유 라이브러리의 경로를 통해 현재 프로세스에 사용자 지정
연산자를 로드합니다. 공유 라이브러리를 로드하면 ``TORCH_LIBRARY`` 블록도 실행됩니다. 이렇게
하면 TorchScript 컴파일러에 사용자 정의 연산자가 등록되고 TorchScript 코드에서 해당 연산자
를 사용할 수 있습니다.

로드된 연산자는 ``torch.ops.<namespace>.<function>`` 으로 참조 가능합니다. ``<namespace>``
는 연산자 이름의 네임스페이스 부분이며, ``<function>`` 은 연산자의 함수 이름입니다.위쪽의 예제
에서 볼 수 있는 연산자의 네임스페이스는 ``my_ops`` 이고 함수 이름은 ``warp_perspective``
였습니다. 따라서 연산자는 ``torch.ops.my_ops.warp_perspective``. 로 사용 가능합니다.
이 함수는 작성되거나 추적된 TorchScript 모듈에서 사용할 수 있지만, 기본 PyTorch에서도 사용
가능하고 일반 PyTorch 텐서로 전달할 수 있습니다..

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :prepend: import torch
  :start-after: BEGIN preamble
  :end-before: END preamble

출력:

.. code-block:: python

  tensor([[0.0000, 0.3218, 0.4611,  ..., 0.4636, 0.4636, 0.4636],
        [0.3746, 0.0978, 0.5005,  ..., 0.4636, 0.4636, 0.4636],
        [0.3245, 0.0169, 0.0000,  ..., 0.4458, 0.4458, 0.4458],
        ...,
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000]])


.. note::

    파이썬에서 처음 ``torch.ops.namespace.function`` 에 접근할 때 TorchScript 컴파일러(C++ 영역에서)
    는 함수의 ``namespace::function`` 이 등록되었는지, 등록되었다면 Python에서 C++ 연산자 구현을 호출하는
    데 사용할 수 있는 이 함수에 대한 Python 핸들을 반환합니다. 이것은 TorchScript 사용자 정의 연산자와 C++
    확장 사이의 주목할만한 차이점 중 하나입니다. (C++ 확장은 pybind11을 사용하여 수동으로 바인딩되지만
    TorchScript 사용자 지정 작업은 PyTorch 자체에 의해 즉석에서 바인딩됩니다. Pybind11은 파이썬에 바인딩할
    수 있는 유형과 클래스에 대해 더 많은 유연성을 제공하므로 순수하게 보이는 코드에 권장되지만 TorchScript
    작업에는 지원되지 않습니다.)

이제 ``torch`` 패키지의 다른 함수처럼 작성되거나 추적된 코드로 사용자 지정 연산자를 사용할수 있습니다.
실제로 ``torch.matmul`` 과 같은 '표준 라이브러리' 기능은 맞춤 연산자와 거의 동일한 등록 경로를 거칩니다.
이는 사용자 정의 연산자를 TorchScript에서 사용할 수 있는 방법과 위치에 관하여 최고 수준으로 만듭니다.
(그러나 한 가지 차이점은 표준 라이브러리 함수에는 ``torch.ops`` 인수 파싱과 다른 맞춤 작성 Python
인수 파싱 로직이 있다는 것입니다.)

트레이싱과 함께 사용자 지정 연산자 사용
**************************************

트레이싱된 함수에 연산자를 포함하는 것으로 시작하겠습니다. 트레이싱을 위해 바닐라 Pytorch
코드로 시작한다는 것을 기억해주시면 좋겠습니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN compute
  :end-before: END compute

위 코드에 이어 ``torch.jit.trace`` 를 호출합니다.  우리는 추가로 ``torch.jit.trace``
몇 가지 예제 입력을 전달합니다.그리고 이 입력을 통해 입력이 흐를 때 발생하는 작업 시퀀스를
기록하기 위해 구현에 전달할 것입니다. 그 결과TorchScript 컴파일러가 추가로 분석, 최적화
및 직렬화할 수 있는 Eager PyTorch 프로그램의 사실상 "고정" 버전이 생성됩니다.:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN trace
  :end-before: END trace

출력::

    graph(%x : Float(4:8, 8:1),
          %y : Float(8:5, 5:1),
          %z : Float(4:5, 5:1)):
      %3 : Float(4:5, 5:1) = aten::matmul(%x, %y) # test.py:10:0
      %4 : Float(4:5, 5:1) = aten::relu(%z) # test.py:10:0
      %5 : int = prim::Constant[value=1]() # test.py:10:0
      %6 : Float(4:5, 5:1) = aten::add(%3, %4, %5) # test.py:10:0
      return (%6)

이제 흥미로운 사실은 사용자 지정 연산자를 ``torch.relu`` 또는 다른 ``torch`` 함수처럼 PyTorch
트레이스에 간단히 드롭할 수 있다는 것입니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN compute2
  :end-before: END compute2

이전 코드와 같이 따라합니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN trace2
  :end-before: END trace2

출력::

    graph(%x.1 : Float(4:8, 8:1),
          %y : Float(8:5, 5:1),
          %z : Float(8:5, 5:1)):
      %3 : int = prim::Constant[value=3]() # test.py:25:0
      %4 : int = prim::Constant[value=6]() # test.py:25:0
      %5 : int = prim::Constant[value=0]() # test.py:25:0
      %6 : Device = prim::Constant[value="cpu"]() # test.py:25:0
      %7 : bool = prim::Constant[value=0]() # test.py:25:0
      %8 : Float(3:3, 3:1) = aten::eye(%3, %4, %5, %6, %7) # test.py:25:0
      %x : Float(8:8, 8:1) = my_ops::warp_perspective(%x.1, %8) # test.py:25:0
      %10 : Float(8:5, 5:1) = aten::matmul(%x, %y) # test.py:26:0
      %11 : Float(8:5, 5:1) = aten::relu(%z) # test.py:26:0
      %12 : int = prim::Constant[value=1]() # test.py:26:0
      %13 : Float(8:5, 5:1) = aten::add(%10, %11, %12) # test.py:26:0
      return (%13)

TorchScript 사용자 정의 작업을 추적된 PyTorch 코드에 통합하는 것은 이렇게 쉽습니다!

스크립트와 함께 사용자 정의 연산자 사용
*************************************

트레이싱 이외에도 PyTorch 프로그램의 TorchScript 표현에 도달하는 또 다른 방법은 TorchScript
에서 직접 코드를 작성하는 것입니다. TorchScript는 대부분 Python 언어의 하위 집합이며
TorchScript 컴파일러가 프로그램에 대해 더 쉽게 추론할 수 있도록 하는 몇 가지 제한 사항이
있습니다. 일반 PyTorch 코드를 함수의 경우 ``@torch.jit.script`` Annotation을, 클래스의
메서드의 경우 ``@torch.jit.script_method`` ( ``torch.jit.ScriptModule`` 에서
파생되어야 함) 어노테이션을 추가하여 TorchScript로 변환합니다. TorchScript Annotation에
대한 자세한 내용은 `여기 <https://pytorch.org/docs/master/jit.html>`_ 를 참조하십시오.

트레이싱 대신 TorchScript를 사용하는 한 가지 특별한 이유는 트레이싱이 PyTorch 코드에서
제어 흐름을 캡처할 수 없기 때문입니다. 따라서 제어 흐름을 사용하는 이 함수를 고려해봐야합니다.

.. code-block:: python

  def compute(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    return x.matmul(y) + z


이 함수를 바닐라 PyTorch에서 TorchScript로 변환하기 위해서는 ``@torch.jit.script`` Annotation을 달아야합니다.

.. code-block:: python

  @torch.jit.script
  def compute(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    return x.matmul(y) + z

이것은 ``compute`` 함수를 그래프로 적절한 때에 컴파일합니다.표현은 ``compute.graph``
속성에서 확인할 수 있습니다.

.. code-block:: python

  >>> compute.graph
  graph(%x : Dynamic
      %y : Dynamic) {
    %14 : int = prim::Constant[value=1]()
    %2 : int = prim::Constant[value=0]()
    %7 : int = prim::Constant[value=42]()
    %z.1 : int = prim::Constant[value=5]()
    %z.2 : int = prim::Constant[value=10]()
    %4 : Dynamic = aten::select(%x, %2, %2)
    %6 : Dynamic = aten::select(%4, %2, %2)
    %8 : Dynamic = aten::eq(%6, %7)
    %9 : bool = prim::TensorToBool(%8)
    %z : int = prim::If(%9)
      block0() {
        -> (%z.1)
      }
      block1() {
        -> (%z.2)
      }
    %13 : Dynamic = aten::matmul(%x, %y)
    %15 : Dynamic = aten::add(%13, %z, %14)
    return (%15);
  }

이제 이전과 같이 스크립트 코드 내에서 다른 함수처럼 사용자 지정 연산자를 사용할 수 있습니다.

.. code-block:: python

  torch.ops.load_library("libwarp_perspective.so")

  @torch.jit.script
  def compute(x, y):
    if bool(x[0] == 42):
        z = 5
    else:
        z = 10
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + z

TorchScript 컴파일러가 ``torch.ops.my_ops.warp_perspective`` 에 대한 참조를 볼 때
C++의  ``TORCH_LIBRARY`` 함수를 통해 등록한 구현을 찾아 그래프 표현으로 컴파일합니다.

.. code-block:: python

  >>> compute.graph
  graph(%x.1 : Dynamic
      %y : Dynamic) {
      %20 : int = prim::Constant[value=1]()
      %16 : int[] = prim::Constant[value=[0, -1]]()
      %14 : int = prim::Constant[value=6]()
      %2 : int = prim::Constant[value=0]()
      %7 : int = prim::Constant[value=42]()
      %z.1 : int = prim::Constant[value=5]()
      %z.2 : int = prim::Constant[value=10]()
      %13 : int = prim::Constant[value=3]()
      %4 : Dynamic = aten::select(%x.1, %2, %2)
      %6 : Dynamic = aten::select(%4, %2, %2)
      %8 : Dynamic = aten::eq(%6, %7)
      %9 : bool = prim::TensorToBool(%8)
      %z : int = prim::If(%9)
        block0() {
          -> (%z.1)
        }
        block1() {
          -> (%z.2)
        }
      %17 : Dynamic = aten::eye(%13, %14, %2, %16)
      %x : Dynamic = my_ops::warp_perspective(%x.1, %17)
      %19 : Dynamic = aten::matmul(%x, %y)
      %21 : Dynamic = aten::add(%19, %z, %20)
      return (%21);
    }

특히 그래프 끝 부분의 ``my_ops::warp_perspective``에 대한 참조를 주의하십시오.

.. 주의::

	TorchScript 그래프 형태는 변경될 수 있습니다. 위 예제의 형태에 의존하지 마십시오.
Python에서 사용자 지정 연산자를 사용할 때 그렇습니다. 간단히 말해서,
``torch.ops.load_library``, 를 사용하여 연산자가 포함된 라이브러리를 가져오고 트레이싱
되거나 스크립팅된 TorchScript 코드에서 다른 ``torch``연산자처럼 사용자 정의 연산자를
호출합니다.

C++에서 TorchScript 사용자 지정 연산자 사용
--------------------------------------------

TorchScript의 유용한 기능 중 하나는 모델을 디스크 상의 파일로 직렬화하는 기능입니다. 이
파일은 유선을 통해 전송되어 파일 시스템에 저장되거나 더 중요하게는 원본 소스 코드를 유지할
필요 없이 동적으로 역직렬화 및 실행될 수 있습니다. 이것은 Python뿐아니라 C++에서도 가능합
니다. 이를 위해 PyTorch는 TorchScript 모델 실행 및 역직렬화를 위한
`순수 C++ API <https://pytorch.org/cppdocs/>`_ 를 제공합니다. 아직 읽지 않았다면
다음에 나올 몇몇 부분에서 보여주고 있는
`C++에서 직렬화된 TorchScript 모델을 로드하고 실행하는 방법에 대한 튜토리얼 <https://tutorials.pytorch.kr/advanced/cpp_export.html>`_
을 읽어보세요.

간단히 말해서 사용자 정의 연산자는 파일에서 역직렬화되고 C++에서 실행되는 경우에도 일반
``torch`` 연산자처럼 실행할 수 있습니다. 이에 대한 유일한 요구 사항은 이전에 빌드한
사용자 지정 연산자 공유 라이브러리를 모델을 실행하는 C++ 응용 프로그램과 연결하는 것입
니다. Python에서는 단순히 ``torch.ops.load_library``를 호출하여 작동했습니다. C++
에서는 사용 중인 빌드 시스템에 관계없이 공유 라이브러리를 기본 애플리케이션과 연결해야
합니다. 다음 예제에서는 CMake를 사용하여 이를 보여줍니다.

.. note::

	기술적으로 Python에서 수행한 것과 거의 동일한 방식으로 런타임 시 C++ 애플리케이션에 공유
    라이브러리를 동적으로 로드할 수도 있습니다.Linux에서는 `dlopen을 사용하여
	<https://tldp.org/HOWTO/Program-Library-HOWTO/dl-libraries.html>`_. 이 작업을 수행할
    수 있습니다. 다른 플랫폼에도 같은 것이 있습니다.

위에 링크된 C++ 실행 자습서를 기반으로 직렬화된 TorchScript 모델을 로드하고 실행하는 사용자 지정 연산
자와 다른 폴더의 ``main.cpp`` 인 한 파일의 최소 C++ 응용 프로그램부터 시작해 보겠습니다.

결론
----------

이 튜토리얼에서는 C++에서 사용자 지정 TorchScript 연산자를 구현하는 방법, 
공유 라이브러리로 빌드하는 방법, Python에서 이를 사용하여 TorchScript 모델을 정의하는 방법, 
마지막으로 추론 워크로드(inference workloads)를 위해 C++ 애플리케이션에 로드하는 방법을 설명했습니다. 
이제 써드파티 C++ 라이브러리와 인터페이스하는 C++ 연산자로 TorchScript 모델을 확장하거나, 
맞춤형 고성능 CUDA 커널을 작성하거나, Python, TorchScript 및 C++ 간의 라인이 원활하게 
혼합되어야 하는 다른 사용 사례를 구현할 준비가 되었습니다.

항상 그렇듯이 문제가 발생하거나 질문이 있는 경우 
`포럼 <https://discuss.pytorch.org/>`_ 또는 `GitHub 이슈
<https://github.com/pytorch/pytorch/issues>`_ 를 통해 연락할 수 있습니다. 
`또한 자주 묻는 질문(FAQ) 페이지
<https://pytorch.org/cppdocs/notes/faq.html>`_ 에 유용한 정보가 있을 수 있습니다.

부록 A: 사용자 지정 연산자(Custom Operators)를 구축하는 더 많은 방법
--------------------------------------------------

"사용자 지정 연산자 빌드" 섹션에서는 CMake를 사용하여 사용자 지정 연산자를 
공유 라이브러리에 빌드하는 방법을 설명했습니다. 
이 부록에서는 추가적으로 컴파일을 위한 두 가지 접근 방식을 설명합니다. 
둘 다 Python을 컴파일 프로세스의 "드라이버" 또는 "인터페이스"로 사용합니다. 
또한 둘 다 `기존 인프라 <https://pytorch.org/docs/stable/cpp_extension.html>`_ 
PyTorch가 제공하는 `*C++ 확장*
<https://tutorials.pytorch.kr/advanced/cpp_extension.html>`_ 을 재사용합니다. 
이는 C++에서 Python으로 함수의 "명시적" 바인딩을 위해 pybind11에 의존하는 
TorchScript 사용자 정의 연산자와 동일한 바닐라(eager) PyTorch입니다.

첫 번째 접근 방식은 `C++ 확장의 편리한 JIT(Just-In-Time) 컴파일 인터페이스
<https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load>`_ 를 사용하여 
처음 실행할 때 PyTorch 스크립트의 백그라운드에서 코드를 컴파일합니다. 
두 번째 접근 방식은 ``setuptools`` 패키지에 의존하며 별도의 ``setup.py`` 파일 작성을 포함합니다. 
이를 통해 다른 ``setuptools`` 기반 프로젝트와의 통합뿐만 아니라 고급 구성이 가능합니다.
아래에서 두 가지 접근 방식을 자세히 살펴보겠습니다.

JIT 컴파일로 빌드
*****************************

PyTorch C++ 확장 툴킷에서 제공하는 JIT 컴파일 기능을 사용하면 사용자 지정 연산자의 컴파일을 
Python 코드에 직접 포함할 수 있습니다. 예시로 그것은 훈련 스크립트의 맨 위에 있습니다.

.. note::

  여기서 "JIT 컴파일"은 프로그램을 최적화하기 위해 TorchScript 컴파일러에서 
  발생하는 JIT 컴파일과 아무 관련이 없습니다. 
  이는 단순히 사용자 정의 연산자 C++ 코드가 미리 컴파일한 것처럼 처음 가져올 때 
  시스템의 /tmp 디렉토리 아래 폴더에 컴파일된다는 것을 의미합니다.

이 JIT 컴파일 기능은 두 가지 형태로 제공됩니다. 
첫 번째 단계에서는 동일하게 별도의 파일(``op.cpp``)에 연산자 구현을 유지한 다음 
``torch.utils.cpp_extension.load()`` 를 사용하여 확장을 컴파일합니다. 
일반적으로 이 함수는 C++ 확장을 노출하는 Python 모듈을 반환합니다. 
그러나 사용자 정의 연산자를 자체 Python 모듈로 컴파일하지 않기 때문에 일반 공유 라이브러리만 컴파일하려고 합니다. 
다행스럽게도 ``torch.utils.cpp_extension.load()`` 에는 Python 모듈이 아닌 공유 라이브러리 
구축에만 목적이 있음을 나타내기 위해 False로 설정할 수 있는 ``is_python_module`` 인수가 있습니다. 
그러면 ``torch.utils.cpp_extension.load()`` 는 이전 ``torch.ops.load_library`` 와 동일하게 
현재 프로세스에 공유 라이브러리를 컴파일하고 로드합니다.

.. code-block:: python

  import torch.utils.cpp_extension

  torch.utils.cpp_extension.load(
      name="warp_perspective",
      sources=["op.cpp"],
      extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
      is_python_module=False,
      verbose=True
  )

  print(torch.ops.my_ops.warp_perspective)

다음과 같이 출력됩니다:

.. code-block:: python

  <built-in method my_ops::warp_perspective of PyCapsule object at 0x7f3e0f840b10>

JIT 컴파일의 두 번째 flavor을 사용하면 사용자 지정 TorchScript 연산자에 대한 소스 코드를 문자열로 전달할 수 있습니다.
이를 위해 ``torch.utils.cpp_extension.load_inline`` 을 사용하십시오.

.. code-block:: python

  import torch
  import torch.utils.cpp_extension

  op_source = """
  #include <opencv2/opencv.hpp>
  #include <torch/script.h>

  torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
    cv::Mat image_mat(/*rows=*/image.size(0),
                      /*cols=*/image.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/image.data<float>());
    cv::Mat warp_mat(/*rows=*/warp.size(0),
                     /*cols=*/warp.size(1),
                     /*type=*/CV_32FC1,
                     /*data=*/warp.data<float>());

    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{64, 64});

    torch::Tensor output =
      torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{64, 64});
    return output.clone();
  }

  TORCH_LIBRARY(my_ops, m) {
    m.def("warp_perspective", &warp_perspective);
  }
  """

  torch.utils.cpp_extension.load_inline(
      name="warp_perspective",
      cpp_sources=op_source,
      extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
      is_python_module=False,
      verbose=True,
  )

  print(torch.ops.my_ops.warp_perspective)

당연히 소스 코드가 상당히 짧은 경우에만 
``torch.utils.cpp_extension.load_inline`` 을 사용하는 것이 가장 좋습니다.

Jupyter Notebook에서 이것을 사용하는 경우 각 실행이 새 라이브러리를 등록하고 
사용자 지정 연산자를 다시 등록하기 때문에 등록이 있는 셀을 여러 번 실행하면 안 됩니다. 
다시 실행해야 하는 경우 미리 노트북의 Python 커널을 다시 시작하십시오.

Setuptools를 사용하여 빌드
************************

Python에서만 사용자 정의 연산자를 구축하는 두 번째 접근 방식은 ``setuptools`` 를 사용하는 것입니다. 
이것은 ``setuptools`` 가 C++로 작성된 Python 모듈을 빌드하기 위한 매우 강력하고 
광범위한 인터페이스를 가지고 있다는 장점이 있습니다. 
그러나 ``setuptools`` 는 실제로 Python 모듈을 빌드하기 위한 것이며 
일반 공유 라이브러리(Python이 모듈에서 기대하는 필수 엔트리가 없음)가 아니라 이 경로가 약간 이상할 수 있습니다. 
즉, 다음과 같은 ``CMakeLists.txt`` 대신 ``setup.py`` 파일만 있으면 됩니다:

.. code-block:: python

  from setuptools import setup
  from torch.utils.cpp_extension import BuildExtension, CppExtension

  setup(
      name="warp_perspective",
      ext_modules=[
          CppExtension(
              "warp_perspective",
              ["example_app/warp_perspective/op.cpp"],
              libraries=["opencv_core", "opencv_imgproc"],
          )
      ],
      cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
  )

하단의 ``BuildExtension`` 에서 ``no_python_abi_suffix`` 옵션을 활성화했습니다. 
이는 생성된 공유 라이브러리의 이름에서 Python-3 특정 ABI 접미사를 생략하도록 ``setuptools`` 에 지시합니다.
그렇지 않으면 Python 3.7에서 라이브러리를 `` warp_perspective.cpython-37m-x86_64-linux-gnu`` 라고 할 수 있습니다.
따라서 ``cpython-37m-x86_64-linux-gnu`` 는 ABI 태그이지만 실제로는 ``warp_perspective.so`` 라고 부르기를 원합니다.

이제 ``setup.py`` 가 있는 폴더 내 터미널에서 
``python setup.py build development`` 를 실행하면 다음과 같이 표시되어야 합니다.

.. code-block:: shell

  $ python setup.py build develop
  running build
  running build_ext
  building 'warp_perspective' extension
  creating build
  creating build/temp.linux-x86_64-3.7
  gcc -pthread -B /root/local/miniconda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/torch/csrc/api/include -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/TH -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/THC -I/root/local/miniconda/include/python3.7m -c op.cpp -o build/temp.linux-x86_64-3.7/op.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=warp_perspective -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
  cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
  creating build/lib.linux-x86_64-3.7
  g++ -pthread -shared -B /root/local/miniconda/compiler_compat -L/root/local/miniconda/lib -Wl,-rpath=/root/local/miniconda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/op.o -lopencv_core -lopencv_imgproc -o build/lib.linux-x86_64-3.7/warp_perspective.so
  running develop
  running egg_info
  creating warp_perspective.egg-info
  writing warp_perspective.egg-info/PKG-INFO
  writing dependency_links to warp_perspective.egg-info/dependency_links.txt
  writing top-level names to warp_perspective.egg-info/top_level.txt
  writing manifest file 'warp_perspective.egg-info/SOURCES.txt'
  reading manifest file 'warp_perspective.egg-info/SOURCES.txt'
  writing manifest file 'warp_perspective.egg-info/SOURCES.txt'
  running build_ext
  copying build/lib.linux-x86_64-3.7/warp_perspective.so ->
  Creating /root/local/miniconda/lib/python3.7/site-packages/warp-perspective.egg-link (link to .)
  Adding warp-perspective 0.0.0 to easy-install.pth file

  Installed /warp_perspective
  Processing dependencies for warp-perspective==0.0.0
  Finished processing dependencies for warp-perspective==0.0.0

이것은 ``warp_perspective.so`` 라는 공유 라이브러리를 생성합니다. 
이 라이브러리는 이전에 TorchScript에 연산자를 표시했던 것처럼 ``torch.ops.load_library`` 에 전달할 수 있습니다.

.. code-block:: python

  >>> import torch
  >>> torch.ops.load_library("warp_perspective.so")
  >>> print(torch.ops.custom.warp_perspective)
  <built-in method custom::warp_perspective of PyCapsule object at 0x7ff51c5b7bd0>
