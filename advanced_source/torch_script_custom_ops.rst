사용자 정의 C++ 연산자로 TORCHSCRIPT 확장하기
===============================================

PyTorch 1.0 릴리스는 PyTorch에 `TorchScript <https://pytorch.org/docs/master/jit.html>`_ 라는 새로운 
프로그래밍 모델을 도입하였습니다 .TorchScript는 TorchScript 컴파일러에서 구문 분석, 컴파일 및 최적화할 수 있는 Python
프로그래밍 언어의 하위 집합입니다. 또한 컴파일된 TorchScript 모델에는 디스크에 있는 파일 형식으로 직렬화할 수 있는 옵션이 
있으며, 추론(inference)을 위해 순수 C++(Python뿐만 아니라)에서 로드하고 실행할 수 있습니다.

TorchScript는 ``torch`` 패키지에서 제공하는 작업의 큰 부분 집합을 지원 하므로 PyTorch의 "표준 라이브러리"에서 순수하게 
일련의 텐서 작업으로 많은 종류의 복잡한 모델을 표현할 수 있습니다. 그럼에도 불구하고 사용자 정의 C++ 또는 CUDA 기능으로 
TorchScript를 확장해야 하는 경우가 있습니다. 아이디어를 간단한 Python 함수로 표현할 수 없는 경우에만 이 옵션을 사용하는 
것이 좋지만 PyTorch의 고성능 C++ 텐서 라이브러리인 `ATen <https://pytorch.org/cppdocs/#aten>`_을 사용하여 
사용자 지정 C++ 및 CUDA 커널을 정의하기 위한 매우 친숙하고 간단한 인터페이스를 제공합니다. TorchScript에 바인딩되면 이러한 
사용자 지정 커널(또는 "ops")을 TorchScript 모델에 포함하고 Python 및 직렬화된 형식으로 C++에서 직접 실행할 수 있습니다.

다음 단락 에서는 C++로 작성된 컴퓨터 비전 라이브러리인 `OpenCV <https://www.opencv.org>`_를 호출하는 TorchScript 
사용자 지정 작업을 작성하는 예를 보여줍니다 . C++에서 텐서를 사용하는 방법, 타사 텐서 형식(이 경우 OpenCV ``Mat``) 으로 
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

이 연산자의 코드는 매우 짧습니다. 파일 맨 위에 OpenCV 헤더 파일이 포함되어 있으며 ``opencv2/opencv.hpp``와 ``torch/
script.h`` 헤더 와 함께 사용자 지정 TorchScript 연산자를 작성하는 데 필요한 PyTorch의 C++ API에서 필요한 모든 
항목을 노출합니다.

우리의 함수 ``warp_perspective`` 는 두 가지 인수를 취합니다: 입력 ``image`` 와 이미지에 적용하고자 하는 ``warp`` 
변환 행렬. 
이러한 입력 유형은 ``torch::Tensor``, C++에서 PyTorch의 텐서 유형(파이썬에서 모든 텐서의 기본 유형이기도 함)입니다. 
``warp_perspective`` 함수의 반환 유형 또한 ``torch::Tensor`` 입니다. 

.. tip::

  `이 노트 <https://pytorch.org/cppdocs/notes/tensor_basics.html>`_ 에는 ``Tensor`` 클래스를 제공하는 
  라이브러리인 ATen에 대한 자세한 내용이 있습니다. 또, `이 튜토리얼 <https://pytorch.org/cppdocs/notes/tensor_creation.html>`_ 에서는 C++에서 새 텐서 개체를 할당하고 초기화하는 방법을 설명합니다(이 연산자에는 필요하지 않
  음).

.. attention::

  TorchScript 컴파일러는 고정된 수의 유형을 이해합니다. 이러한 유형만 사용자 지정 연산자에 대한 인수로 사용할 수 있습니다. 
  현재 이러한 유형은 다음과 같습니다: ``torch::Tensor``, ``torch::Scalar``, ``double``, ``int64_t`` 및
  ``std::vector`` 의 이러한 유형들.``float``가 아니라 *오로지* ``double`` 이며, ``int``, ``short`` 이나 ``long`` 처럼 다른 정수타입이 아닌 *오로지* ``int64_t``을 지원합니다. 

함수 내부에서 가장 먼저 해야 할 일은 PyTorch 텐서를 OpenCV 행렬로 변환하는 것 입니다. OpenCV ``warpPerspective``는 
``cv::Mat``객체를 입력으로 기대하기 때문입니다. 다행히 데이터를 **복사하지 않고** 이 작업을 수행할 수 있는 방법이 있습니다. 
처음 몇 줄에는,

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN image_mat
  :end-before: END image_mat

우리의 텐서를 ``Mat`` 객체로 변환하기 위해 OpenCV ``Mat`` 클래스의 `이 생성자 <https://docs.opencv.org/trunk/d3/d63/classcv_1_1Mat.html#a922de793eabcec705b3579c5f95a643e>`_를 호출합니다. 우리는 오리지널 ``이미지`` 
텐서의 행과 열의 수, 데이터 유형(이 예제에서는 ``float32`` 로 고칠 것), 그리고 마지막으로 기본 데이터에 대한 원시 포인터인 
-- a ``float*`` 를 전달합니다. 이 ``Mat``  클래스 생성자의 특별한 점은 입력 데이터를 복사하지 않는다는 것입니다. 대신 ``
Mat``에서 수행된 모든 작업에 대해 이 메모리를 참조합니다. ``image_mat``에서 제자리 작업을 수행 하면 원본 ``이미지`` 
텐서에 반영됩니다.(반대의 경우도 마찬가지). 이것은 우리가 실제로 데이터를 PyTorch 텐서에 저장하고 있더라도 라이브러리의 기본 
매트릭스 유형으로 후속 OpenCV 루틴을 호출할 수 있도록 합니다. ``warp`` PyTorch 텐서를 ``warp_mat`` OpenCV 
매트릭스로 변환하기 위해 이 절차를 반복합니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN warp_mat
  :end-before: END warp_mat

다음으로 TorchScript에서 사용하고 싶었던 OpenCV 함수를 호출할 준비가 되었습니다: ``warpPerspective``. 이를 위해 
OpenCV 함수 ``image_mat``와 ``warp_mat``매트릭스, 빈 출력 매트릭스인 ``output_mat`` 를 전달합니다.  또한 출력 
매트릭스(이미지)의 원하는 크기 ``dsize``를 지정합니다 . 이 예제에서는 다음 ``8 x 8``과 같이 하드코딩됩니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN output_mat
  :end-before: END output_mat

사용자 정의 연산자 구현의 마지막 단계는 ``output_mat``을 PyTorch에서 더 사용할 수 있도록 다시 PyTorch 텐서로 변환하는 
것입니다. 이것은 우리가 다른 방향으로 변환하기 위해 이전에 수행한 것과 놀랍도록 유사합니다. 이 경우 PyTorch에서 ``torch::
from_blob``메소드를 제공합니다. 우리가 PyTorch 텐서로 해석하려는 *blob*은 메모리에 약간 불투명한, 평면 포인터를 의미합니다. ``torch::from_blob``에 대한 호출은 다음과 같습니다.

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN output_tensor
  :end-before: END output_tensor

우리는 OpenCV ``Mat`` 클래스의 ``.ptr<float>()``메서드를 사용하여 기본 데이터에 대한 원시 포인터를 얻습니다.(이전의 
PyTorch 텐서 ``.data_ptr<float>()``와 마찬가지로). 우리는 또한 ``8 x 8``처럼 하드코딩한 텐서의 출력 형태를 
지정합니다 . ``torch::from_blob``의 출력은 OpenCV 매트릭스가 소유한 메모리를 가리키는 ``torch::Tensor``입니다.

연산자 구현에서 이 텐서를 반환하기 전에, 기본 데이터의 메모리 복사를 수행하기 위해 ``.clone()``를 호출해야 합니다 . 그 
이유는 ``torch::from_blob``는 데이터를 소유하지 않는 텐서를 반환하기 때문입니다 . 그 시점에서 데이터는 여전히 OpenCV 
매트릭스에 의해 소유됩니다. 그러나 이 OpenCV 매트릭스는 범위를 벗어나 함수가 끝날 때 할당이 해제됩니다. ``output`` 텐서를 
있는 그대로 반환 하면 함수 외부에서 사용할 때까지 유효하지 않은 메모리를 가리킬 것입니다. ``.clone()``을 호출하면 새 텐서가 
자체적으로 소유한 원본 데이터의 복사본과 함께 새 텐서를 반환합니다. 따라서 바깥으로 돌아가는 것은(반환하는 것은) 안전합니다.

TorchScript에 사용자 정의 연산자 등록
------------------------------------------------

이제 C++에서 사용자 정의 연산자를 구현 했으므로 TorchScript 런타임 및 컴파일러에 *등록* 해야 합니다 . 이를 통해 
TorchScript 컴파일러는 TorchScript 코드에서 사용자 지정 연산자에 대한 참조를 확인할 수 있습니다. pybind11 라이브러리를 
사용한 적이 있다면 등록 구문이 pybind11 구문과 매우 유사합니다. 단일 함수를 등록하려면 다음과 같이 작성합니다:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN registry
  :end-before: END registry

``op.cpp`` 파일의 최상위 레벨 어딘가에 있습니다. ``TORCH_LIBRARY`` 매크로는 프로그램이 시작될 때 호출되는 함수를 
작성합니다.  라이브러리 이름(``my_ops``)이 첫 번째 인수로 제공됩니다(따옴표로 묶지 않아야 함). 두 번째 인수(``m``) 는 
연산자를 등록하기 위한 기본 인터페이스 유형 ``torch::Library``의 변수를 정의합니다.
이 메서드 ``Library::def``는 실제로 ``warp_perspective``라는 연산자를 생성하여,Python과 TorchScript에 모두 
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
``CMakeLists.txt`` 파일 을 작성 하고 이전  ``op.cpp``파일과 함께 배치해야 합니다. 이를 위해 다음과 같은 디렉토리 
구조에 동의합니다 ::

  warp-perspective/
    op.cpp
    CMakeLists.txt

The contents of our ``CMakeLists.txt`` file should then be the following:
``CMakeLists.txt`` 파일의 내용은 다음과 같아야 합니다.

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

which will place a ``libwarp_perspective.so`` shared library file in the
``build`` folder. In the ``cmake`` command above, we use the helper
variable ``torch.utils.cmake_prefix_path`` to conveniently tell us where
the cmake files for our PyTorch install are.

``build`` 폴더에 ``libwarp_perspective.so`` 공유 라이브러리 파일을 저장합니다. 위의 ``cmake`` 명령에서는 helper 
변수 ``torch.utils.cmake_prefix_path``를 사용하여 PyTorch 설치를 위한 cmake 파일이 어디에 있는지 편리하게 
알려줍니다.

아래에서 연산자를 사용하고 호출하는 방법을 자세히 살펴보겠지만 더 일찍 성공을 느껴보기 위해 Python에서 다음 코드를 실행할 수 
있습니다. : 

.. literalinclude:: ../advanced_source/torch_script_custom_ops/smoke_test.py
  :language: python

모두 잘되었다면 다음과 같이 인쇄됩니다.::

  <built-in method my_ops::warp_perspective of PyCapsule object at 0x7f618fc6fa50>

이것은 나중에 사용자 정의 연산자를 호출하는 데 사용할 Python 함수입니다.

Using the TorchScript Custom Operator in Python
-----------------------------------------------

Once our custom operator is built into a shared library  we are ready to use
this operator in our TorchScript models in Python. There are two parts to this:
first loading the operator into Python, and second using the operator in
TorchScript code.

You already saw how to import your operator into Python:
``torch.ops.load_library()``. This function takes the path to a shared library
containing custom operators, and loads it into the current process. Loading the
shared library will also execute the ``TORCH_LIBRARY`` block. This will register
our custom operator with the TorchScript compiler and allow us to use that
operator in TorchScript code.

You can refer to your loaded operator as ``torch.ops.<namespace>.<function>``,
where ``<namespace>`` is the namespace part of your operator name, and
``<function>`` the function name of your operator. For the operator we wrote
above, the namespace was ``my_ops`` and the function name ``warp_perspective``,
which means our operator is available as ``torch.ops.my_ops.warp_perspective``.
While this function can be used in scripted or traced TorchScript modules, we
can also just use it in vanilla eager PyTorch and pass it regular PyTorch
tensors:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :prepend: import torch
  :start-after: BEGIN preamble
  :end-before: END preamble

producing:

.. code-block:: python

  tensor([[0.0000, 0.3218, 0.4611,  ..., 0.4636, 0.4636, 0.4636],
        [0.3746, 0.0978, 0.5005,  ..., 0.4636, 0.4636, 0.4636],
        [0.3245, 0.0169, 0.0000,  ..., 0.4458, 0.4458, 0.4458],
        ...,
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000]])


.. note::

    What happens behind the scenes is that the first time you access
    ``torch.ops.namespace.function`` in Python, the TorchScript compiler (in C++
    land) will see if a function ``namespace::function`` has been registered, and
    if so, return a Python handle to this function that we can subsequently use to
    call into our C++ operator implementation from Python. This is one noteworthy
    difference between TorchScript custom operators and C++ extensions: C++
    extensions are bound manually using pybind11, while TorchScript custom ops are
    bound on the fly by PyTorch itself. Pybind11 gives you more flexibility with
    regards to what types and classes you can bind into Python and is thus
    recommended for purely eager code, but it is not supported for TorchScript
    ops.

From here on, you can use your custom operator in scripted or traced code just
as you would other functions from the ``torch`` package. In fact, "standard
library" functions like ``torch.matmul`` go through largely the same
registration path as custom operators, which makes custom operators really
first-class citizens when it comes to how and where they can be used in
TorchScript.  (One difference, however, is that standard library functions
have custom written Python argument parsing logic that differs from
``torch.ops`` argument parsing.)

Using the Custom Operator with Tracing
**************************************

Let's start by embedding our operator in a traced function. Recall that for
tracing, we start with some vanilla Pytorch code:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN compute
  :end-before: END compute

and then call ``torch.jit.trace`` on it. We further pass ``torch.jit.trace``
some example inputs, which it will forward to our implementation to record the
sequence of operations that occur as the inputs flow through it. The result of
this is effectively a "frozen" version of the eager PyTorch program, which the
TorchScript compiler can further analyze, optimize and serialize:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN trace
  :end-before: END trace

Producing::

    graph(%x : Float(4:8, 8:1),
          %y : Float(8:5, 5:1),
          %z : Float(4:5, 5:1)):
      %3 : Float(4:5, 5:1) = aten::matmul(%x, %y) # test.py:10:0
      %4 : Float(4:5, 5:1) = aten::relu(%z) # test.py:10:0
      %5 : int = prim::Constant[value=1]() # test.py:10:0
      %6 : Float(4:5, 5:1) = aten::add(%3, %4, %5) # test.py:10:0
      return (%6)

Now, the exciting revelation is that we can simply drop our custom operator into
our PyTorch trace as if it were ``torch.relu`` or any other ``torch`` function:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN compute2
  :end-before: END compute2

and then trace it as before:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN trace2
  :end-before: END trace2

Producing::

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

Integrating TorchScript custom ops into traced PyTorch code is as easy as this!

Using the Custom Operator with Script
*************************************

Besides tracing, another way to arrive at a TorchScript representation of a
PyTorch program is to directly write your code *in* TorchScript. TorchScript is
largely a subset of the Python language, with some restrictions that make it
easier for the TorchScript compiler to reason about programs. You turn your
regular PyTorch code into TorchScript by annotating it with
``@torch.jit.script`` for free functions and ``@torch.jit.script_method`` for
methods in a class (which must also derive from ``torch.jit.ScriptModule``). See
`here <https://pytorch.org/docs/master/jit.html>`_ for more details on
TorchScript annotations.

One particular reason to use TorchScript instead of tracing is that tracing is
unable to capture control flow in PyTorch code. As such, let us consider this
function which does use control flow:

.. code-block:: python

  def compute(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    return x.matmul(y) + z

To convert this function from vanilla PyTorch to TorchScript, we annotate it
with ``@torch.jit.script``:

.. code-block:: python

  @torch.jit.script
  def compute(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    return x.matmul(y) + z

This will just-in-time compile the ``compute`` function into a graph
representation, which we can inspect in the ``compute.graph`` property:

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

And now, just like before, we can use our custom operator like any other
function inside of our script code:

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

When the TorchScript compiler sees the reference to
``torch.ops.my_ops.warp_perspective``, it will find the implementation we
registered via the ``TORCH_LIBRARY`` function in C++, and compile it into its
graph representation:

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

Notice in particular the reference to ``my_ops::warp_perspective`` at the end of
the graph.

.. attention::

	The TorchScript graph representation is still subject to change. Do not rely
	on it looking like this.

And that's really it when it comes to using our custom operator in Python. In
short, you import the library containing your operator(s) using
``torch.ops.load_library``, and call your custom op like any other ``torch``
operator from your traced or scripted TorchScript code.

Using the TorchScript Custom Operator in C++
--------------------------------------------

One useful feature of TorchScript is the ability to serialize a model into an
on-disk file. This file can be sent over the wire, stored in a file system or,
more importantly, be dynamically deserialized and executed without needing to
keep the original source code around. This is possible in Python, but also in
C++. For this, PyTorch provides `a pure C++ API <https://pytorch.org/cppdocs/>`_
for deserializing as well as executing TorchScript models. If you haven't yet,
please read `the tutorial on loading and running serialized TorchScript models
in C++ <https://tutorials.pytorch.kr/advanced/cpp_export.html>`_, on which the
next few paragraphs will build.

In short, custom operators can be executed just like regular ``torch`` operators
even when deserialized from a file and run in C++. The only requirement for this
is to link the custom operator shared library we built earlier with the C++
application in which we execute the model. In Python, this worked simply calling
``torch.ops.load_library``. In C++, you need to link the shared library with
your main application in whatever build system you are using. The following
example will showcase this using CMake.

.. note::

	Technically, you can also dynamically load the shared library into your C++
	application at runtime in much the same way we did it in Python. On Linux,
	`you can do this with dlopen
	<https://tldp.org/HOWTO/Program-Library-HOWTO/dl-libraries.html>`_. There exist
	equivalents on other platforms.

Building on the C++ execution tutorial linked above, let's start with a minimal
C++ application in one file, ``main.cpp`` in a different folder from our
custom operator, that loads and executes a serialized TorchScript model:

.. code-block:: cpp

  #include <torch/script.h> // One-stop header.

  #include <iostream>
  #include <memory>


  int main(int argc, const char* argv[]) {
    if (argc != 2) {
      std::cerr << "usage: example-app <path-to-exported-script-module>\n";
      return -1;
    }

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({4, 8}));
    inputs.push_back(torch::randn({8, 5}));

    torch::Tensor output = module->forward(std::move(inputs)).toTensor();

    std::cout << output << std::endl;
  }

Along with a small ``CMakeLists.txt`` file:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
  project(example_app)

  find_package(Torch REQUIRED)

  add_executable(example_app main.cpp)
  target_link_libraries(example_app "${TORCH_LIBRARIES}")
  target_compile_features(example_app PRIVATE cxx_range_for)

At this point, we should be able to build the application:

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
  -- Build files have been written to: /example_app/build
  $ make -j
  Scanning dependencies of target example_app
  [ 50%] Building CXX object CMakeFiles/example_app.dir/main.cpp.o
  [100%] Linking CXX executable example_app
  [100%] Built target example_app

And run it without passing a model just yet:

.. code-block:: shell

  $ ./example_app
  usage: example_app <path-to-exported-script-module>

Next, let's serialize the script function we wrote earlier that uses our custom
operator:

.. code-block:: python

  torch.ops.load_library("libwarp_perspective.so")

  @torch.jit.script
  def compute(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + z

  compute.save("example.pt")

The last line will serialize the script function into a file called
"example.pt". If we then pass this serialized model to our C++ application, we
can run it straight away:

.. code-block:: shell

  $ ./example_app example.pt
  terminate called after throwing an instance of 'torch::jit::script::ErrorReport'
  what():
  Schema not found for node. File a bug report.
  Node: %16 : Dynamic = my_ops::warp_perspective(%0, %19)

Or maybe not. Maybe not just yet. Of course! We haven't linked the custom
operator library with our application yet. Let's do this right now, and to do it
properly let's update our file organization slightly, to look like this::

  example_app/
    CMakeLists.txt
    main.cpp
    warp_perspective/
      CMakeLists.txt
      op.cpp

This will allow us to add the ``warp_perspective`` library CMake target as a
subdirectory of our application target. The top level ``CMakeLists.txt`` in the
``example_app`` folder should look like this:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
  project(example_app)

  find_package(Torch REQUIRED)

  add_subdirectory(warp_perspective)

  add_executable(example_app main.cpp)
  target_link_libraries(example_app "${TORCH_LIBRARIES}")
  target_link_libraries(example_app -Wl,--no-as-needed warp_perspective)
  target_compile_features(example_app PRIVATE cxx_range_for)

This basic CMake configuration looks much like before, except that we add the
``warp_perspective`` CMake build as a subdirectory. Once its CMake code runs, we
link our ``example_app`` application with the ``warp_perspective`` shared
library.

.. attention::

  There is one crucial detail embedded in the above example: The
  ``-Wl,--no-as-needed`` prefix to the ``warp_perspective`` link line. This is
  required because we will not actually be calling any function from the
  ``warp_perspective`` shared library in our application code. We only need the
  ``TORCH_LIBRARY`` function to run. Inconveniently, this
  confuses the linker and makes it think it can just skip linking against the
  library altogether. On Linux, the ``-Wl,--no-as-needed`` flag forces the link
  to happen (NB: this flag is specific to Linux!). There are other workarounds
  for this. The simplest is to define *some function* in the operator library
  that you need to call from the main application. This could be as simple as a
  function ``void init();`` declared in some header, which is then defined as
  ``void init() { }`` in the operator library. Calling this ``init()`` function
  in the main application will give the linker the impression that this is a
  library worth linking against. Unfortunately, this is outside of our control,
  and we would rather let you know the reason and the simple workaround for this
  than handing you some opaque macro to plop in your code.

Now, since we find the ``Torch`` package at the top level now, the
``CMakeLists.txt`` file in the  ``warp_perspective`` subdirectory can be
shortened a bit. It should look like this:

.. code-block:: cmake

  find_package(OpenCV REQUIRED)
  add_library(warp_perspective SHARED op.cpp)
  target_compile_features(warp_perspective PRIVATE cxx_range_for)
  target_link_libraries(warp_perspective PRIVATE "${TORCH_LIBRARIES}")
  target_link_libraries(warp_perspective PRIVATE opencv_core opencv_photo)

Let's re-build our example app, which will also link with the custom operator
library. In the top level ``example_app`` directory:

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
  -- Build files have been written to: /warp_perspective/example_app/build
  $ make -j
  Scanning dependencies of target warp_perspective
  [ 25%] Building CXX object warp_perspective/CMakeFiles/warp_perspective.dir/op.cpp.o
  [ 50%] Linking CXX shared library libwarp_perspective.so
  [ 50%] Built target warp_perspective
  Scanning dependencies of target example_app
  [ 75%] Building CXX object CMakeFiles/example_app.dir/main.cpp.o
  [100%] Linking CXX executable example_app
  [100%] Built target example_app

If we now run the ``example_app`` binary and hand it our serialized model, we
should arrive at a happy ending:

.. code-block:: shell

  $ ./example_app example.pt
  11.4125   5.8262   9.5345   8.6111  12.3997
   7.4683  13.5969   9.0850  11.0698   9.4008
   7.4597  15.0926  12.5727   8.9319   9.0666
   9.4834  11.1747   9.0162  10.9521   8.6269
  10.0000  10.0000  10.0000  10.0000  10.0000
  10.0000  10.0000  10.0000  10.0000  10.0000
  10.0000  10.0000  10.0000  10.0000  10.0000
  10.0000  10.0000  10.0000  10.0000  10.0000
  [ Variable[CPUFloatType]{8,5} ]

Success! You are now ready to inference away.

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
<https://github.com/pytorch/pytorch/issues>`_를 통해 연락할 수 있습니다. 
`또한 자주 묻는 질문(FAQ) 페이지
<https://pytorch.org/cppdocs/notes/faq.html>`_에 유용한 정보가 있을 수 있습니다.

부록 A: 사용자 지정 연산자(Custom Operators)를 구축하는 더 많은 방법
--------------------------------------------------

"사용자 지정 연산자 빌드" 섹션에서는 CMake를 사용하여 사용자 지정 연산자를 
공유 라이브러리에 빌드하는 방법을 설명했습니다. 
이 부록에서는 추가적으로 컴파일을 위한 두 가지 접근 방식을 설명합니다. 
둘 다 Python을 컴파일 프로세스의 "드라이버" 또는 "인터페이스"로 사용합니다. 
또한 둘 다 `기존 인프라 <https://pytorch.org/docs/stable/cpp_extension.html>`_ 
PyTorch가 제공하는 `*C++ 확장*
<https://tutorials.pytorch.kr/advanced/cpp_extension.html>`_을 재사용합니다. 
이는 C++에서 Python으로 함수의 "명시적" 바인딩을 위해 pybind11에 의존하는 
TorchScript 사용자 정의 연산자와 동일한 바닐라(eager) PyTorch입니다.

첫 번째 접근 방식은 `C++ 확장의 편리한 JIT(Just-In-Time) 컴파일 인터페이스
<https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load>`_를 사용하여 
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
``torch.utils.cpp_extension.load()``를 사용하여 확장을 컴파일합니다. 
일반적으로 이 함수는 C++ 확장을 노출하는 Python 모듈을 반환합니다. 
그러나 사용자 정의 연산자를 자체 Python 모듈로 컴파일하지 않기 때문에 일반 공유 라이브러리만 컴파일하려고 합니다. 
다행스럽게도 ``torch.utils.cpp_extension.load()`` 에는 Python 모듈이 아닌 공유 라이브러리 
구축에만 목적이 있음을 나타내기 위해 False로 설정할 수 있는 ``is_python_module`` 인수가 있습니다. 
그러면 ``torch.utils.cpp_extension.load()``는 이전 ``torch.ops.load_library``와 동일하게 
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
이를 위해 ``torch.utils.cpp_extension.load_inline``을 사용하십시오.

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
``torch.utils.cpp_extension.load_inline``을 사용하는 것이 가장 좋습니다.

Jupyter Notebook에서 이것을 사용하는 경우 각 실행이 새 라이브러리를 등록하고 
사용자 지정 연산자를 다시 등록하기 때문에 등록이 있는 셀을 여러 번 실행하면 안 됩니다. 
다시 실행해야 하는 경우 미리 노트북의 Python 커널을 다시 시작하십시오.

Setuptools를 사용하여 빌드
************************

Python에서만 사용자 정의 연산자를 구축하는 두 번째 접근 방식은 ``setuptools``를 사용하는 것입니다. 
이것은 ``setuptools``가 C++로 작성된 Python 모듈을 빌드하기 위한 매우 강력하고 
광범위한 인터페이스를 가지고 있다는 장점이 있습니다. 
그러나 ``setuptools``는 실제로 Python 모듈을 빌드하기 위한 것이며 
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

하단의 ``BuildExtension``에서 ``no_python_abi_suffix`` 옵션을 활성화했습니다. 
이는 생성된 공유 라이브러리의 이름에서 Python-3 특정 ABI 접미사를 생략하도록 ``setuptools``에 지시합니다.
그렇지 않으면 Python 3.7에서 라이브러리를 ``warp_perspective.cpython-37m-x86_64-linux-gnu``라고 할 수 있습니다.
따라서 ``cpython-37m-x86_64-linux-gnu``는 ABI 태그이지만 실제로는 ``warp_perspective.so``라고 부르기를 원합니다.

이제 ``setup.py``가 있는 폴더 내 터미널에서 
``python setup.py build development``를 실행하면 다음과 같이 표시되어야 합니다.

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

이것은 ``warp_perspective.so``라는 공유 라이브러리를 생성합니다. 
이 라이브러리는 이전에 TorchScript에 연산자를 표시했던 것처럼 ``torch.ops.load_library``에 전달할 수 있습니다.

.. code-block:: python

  >>> import torch
  >>> torch.ops.load_library("warp_perspective.so")
  >>> print(torch.ops.custom.warp_perspective)
  <built-in method custom::warp_perspective of PyCapsule object at 0x7ff51c5b7bd0>
