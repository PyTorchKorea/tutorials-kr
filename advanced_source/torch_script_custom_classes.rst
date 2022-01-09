커스텀 C++ 클래스로 TorchScript 확장하기
======================================

이 튜토리얼은 :doc:`커스텀 오퍼레이터 <torch_script_custom_ops>` 튜토리얼의 후속이며
C++ 클래스를 TorchScript와 Python에 동시에 바인딩하기 위해 구축한 API를 소개합니다.
API는 `pybind11 <https://github.com/pybind/pybind11>`_ 과
매우 유사하며 해당 시스템에 익숙하다면 대부분의 개념이 이전됩니다.

C++에서 클래스 구현 및 바인딩
---------------------------

이 튜토리얼에서는 멤버 변수에서 지속 상태를 유지하는 간단한 C++ 클래스를 정의할 것입니다.

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN class
  :end-before: END class

몇 가지 주의할 사항이 있습니다:

- ``torch/custom_class.h`` 는 커스텀 클래스로 TorchScript를 확장하기 위해 포함해야하는 헤더입니다.
- 커스텀 클래스의 인스턴스로 작업할 때마다 ``c10::intrusive_ptr<>`` 의 인스턴스를 통해 작업을 수행합니다.
  ``intrusive_ptr`` 를 ``std::shared_ptr`` 과 같은 스마트 포인터로 생각하세요. 그러나 참조 계수는
  ``std::shared_ptr`` 같이 별도의 메타데이터 블록과 달리 객체에 직접 저장됩니다.
  ``torch::Tensor`` 는 내부적으로 동일한 포인터 유형을 사용합니다.
  커스텀 클래스도 ``torch::Tensor`` 포인터 유형을 사용해야 다양한 객체 유형을 일관되게 관리할 수 있습니다.
- 두 번째로 주목해야 할 점은 커스텀 클래스가 ``torch::CustomClassHolder`` 에서 상속되어야 한다는 것입니다.
  이렇게 하면 커스텀 클래스에 참조 계수를 저장할 공간이 있습니다.

이제 이 클래스를 어떻게 TorchScript에서 사용가능하게 하는지 살펴보겠습니다.
이런 과정은 클래스를 *바인딩* 한다고 합니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN binding
  :end-before: END binding
  :append:
      ;
    }



CMake를 사용하여 C++ 프로젝트로 예제 빌드
---------------------------------------

이제 `CMake <https://cmake.org>`_ 빌드 시스템을 사용하여 위의 C++ 코드를 빌드합니다.
먼저, 지금까지 다룬 모든 C++ code를 ``class.cpp`` 라는 파일에 넣습니다.
그런 다음 간단한 ``CMakeLists.txt`` 파일을 작성하여 동일한 디렉토리에 배치합니다.
``CMakeLists.txt`` 는 다음과 같아야 합니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/CMakeLists.txt
  :language: cmake

또한 ``build`` 디렉토리를 만듭니다. 파일 트리는 다음과 같아야 합니다::

  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/

:doc:`이전 튜토리얼 <torch_script_custom_ops>` 에서 설명한 것과 동일한 방식으로 환경을 설정했다고 가정합니다.
계속해서 cmake를 호출한 다음 make를 호출하여 프로젝트를 빌드합니다:

.. code-block:: shell

  $ cd build
  $ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
    -- The C compiler identification is GNU 7.3.1
    -- The CXX compiler identification is GNU 7.3.1
    -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc
    -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++
    -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++ -- works
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
    -- Found torch: /torchbind_tutorial/libtorch/lib/libtorch.so
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /torchbind_tutorial/build
  $ make -j
    Scanning dependencies of target custom_class
    [ 50%] Building CXX object CMakeFiles/custom_class.dir/class.cpp.o
    [100%] Linking CXX shared library libcustom_class.so
    [100%] Built target custom_class

이제 무엇보다도 빌드 디렉토리에 동적 라이브러리 파일이 있다는 것을 알게 될 것입니다.
리눅스에서는 아마도 ``libcustom_class.so`` 로 이름이 지정될 것입니다.
따라서 파일 트리는 다음과 같아야 합니다::

  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/
      libcustom_class.so

Python 및 TorchScript의 C++ 클래스 사용
--------------------------------------

이제 클래스와 등록이 ``.so`` 파일로 컴파일되었으므로 해당 `.so` 를 Python에 읽어들이고 사용해 볼 수 있습니다.
다음은 이를 보여주는 스크립트입니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/custom_test.py
  :language: python


커스텀 클래스를 사용하여 TorchScript 코드 저장, 읽기 및 실행
---------------------------------------------------------

libtorch를 사용하여 C++ 프로세스에서 커스텀 등록 C++ 클래스를 사용할 수도 있습니다.
예를 들어 MyStackClass 클래스에서 메소드를 인스턴스화하고 호출하는 간단한 ``nn.Module`` 을 정의해 보겠습니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/save.py
  :language: python

파일 시스템의 foo.pt는 방금 정의한 직렬화된 TorchScript 프로그램을 포함합니다.

이제 이 모델과 필요한 .so 파일을 읽어들이는 방법을 보여주기 위해 새 CMake 프로젝트를 정의하겠습니다.
이 작업을 수행하는 방법에 대한 자세한 내용은 `C++에서 TorchScript 모델 로딩하기 <https://tutorials.pytorch.kr/advanced/cpp_export.html>`_ 를
참조하세요.

이전과 유사하게 다음을 포함하는 파일 구조를 생성해 보겠습니다::

  cpp_inference_example/
    infer.cpp
    CMakeLists.txt
    foo.pt
    build/
    custom_class_project/
      class.cpp
      CMakeLists.txt
      build/

직렬화된 foo.pt 파일과 위의 ``custom_class_project`` 소스 트리를 복사했음을 주목하세요.
커스텀 클래스를 바이너리로 빌드할 수 있도록 ``custom_class_project`` 를 이 C++ 프로젝트에 의존성으로 추가할 것입니다.

``infer.cpp`` 를 다음으로 채우겠습니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/infer.cpp
  :language: cpp

마찬가지로 CMakeLists.txt 파일을 정의해 보겠습니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/CMakeLists.txt
  :language: cpp

``cd build``, ``cmake``, 및 ``make`` 에 대한 사용 방법을 알고 있습니다:

.. code-block:: shell

  $ cd build
  $ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
    -- The C compiler identification is GNU 7.3.1
    -- The CXX compiler identification is GNU 7.3.1
    -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc
    -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++
    -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++ -- works
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
    -- Found torch: /local/miniconda3/lib/python3.7/site-packages/torch/lib/libtorch.so
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /cpp_inference_example/build
  $ make -j
    Scanning dependencies of target custom_class
    [ 25%] Building CXX object custom_class_project/CMakeFiles/custom_class.dir/class.cpp.o
    [ 50%] Linking CXX shared library libcustom_class.so
    [ 50%] Built target custom_class
    Scanning dependencies of target infer
    [ 75%] Building CXX object CMakeFiles/infer.dir/infer.cpp.o
    [100%] Linking CXX executable infer
    [100%] Built target infer

이제 흥미로운 C++ 바이너리를 실행할 수 있습니다:

.. code-block:: shell

  $ ./infer
    momfoobarbaz

대단합니다!

커스텀 클래스를 IValues로/에서 이동
---------------------------------

TorchScript 메소드에서 ``IValue`` 를 가져오거나 반환하기, 또는 C++에서 커스텀 클래스 속성을 인스턴스화하려는
경우와 같이 커스텀 클래스를 ``IValue`` 안팎으로 이동해야 할 수도 있습니다.
커스텀 C++ 클래스 인스턴스에서 ``IValue`` 를 생성하려면:

- ``torch::make_custom_class<T>()`` 는 제공하는 인수 집합을 사용하고 해당 인수 집합과 일치하는
  T의 생성자를 호출하며 해당 인스턴스를 래핑하고 반환하는 c10::intrusive_ptr<T>와 유사한 API를 제공합니다.
  그러나 커스텀 클래스 객체에 대한 포인터만 반환하는 대신 객체를 래핑하는 ``IValue`` 를 반환합니다.
  그런 다음 이 ``IValue`` 를 TorchScript에 직접 전달할 수 있습니다.
- 이미 클래스를 가리키는 ``intrusive_ptr`` 이 있는 경우 생성자 ``IValue(intrusive_ptr<T>)`` 를 사용하여
  해당 클래스에서 IValue를 직접 생성할 수 있습니다.

``IValue`` 를 커스텀 클래스로 다시 변환하려면:

- ``IValue::toCustomClass<T>()`` 는 ``IValue`` 에 포함된 커스텀 클래스를 가리키는 ``intrusive_ptr<T>`` 를
  반환합니다. 내부적으로 이 함수는 ``T`` 가 커스텀 클래스로 등록되어 있고 ``IValue`` 에 실제로 커스텀 클래스가
  포함되어 있는지 확인합니다. ``isCustomClass()`` 를 호출하여 ``IValue`` 에 커스텀 클래스가 포함되어 있는지
  수동으로 확인할 수 있습니다.

커스텀 C++ 클래스에 대한 직렬화/역직렬화 방법 정의
-----------------------------------------------

커스텀 바인딩 된 C++ 클래스를 속성으로 사용하여 ``ScriptModule`` 을 저장하려고 하면
다음 오류가 발생합니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/export_attr.py
  :language: python

.. code-block:: shell

  $ python export_attr.py
  RuntimeError: Cannot serialize custom bound C++ class __torch__.torch.classes.my_classes.MyStackClass. Please define serialization methods via def_pickle for this class. (pushIValueImpl at ../torch/csrc/jit/pickler.cpp:128)

TorchScript가 C++ 클래스에서 저장한 정보를 자동으로 파악할 수 없기 때문입니다.
수동으로 지정해야 합니다. 그렇게 하는 방법은 ``class_`` 에서 특별한 ``def_pickle`` 메소드를
사용하여 클래스에서 ``__getstate__`` 및 ``__setstate__`` 메소드를 정의하는 것입니다.

.. note::
  TorchScript에서 ``__getstate__`` 및 ``__setstate__`` 의 의미는 Python pickle 모듈의 의미와 동일합니다.
  이러한 방법을 어떻게 사용하는지에 대하여 `자세한 내용 <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md#getstate-and-setstate>`_ 을
  참조하세요.

다음은 직렬화 메소드를 포함하기 위해 ``MyStackClass`` 등록에 추가할 수 있는 ``def_pickle`` 호출의 예시입니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN def_pickle
  :end-before: END def_pickle

.. note::
  pickle API에서 pybind11과 다른 접근 방식을 취합니다. pybind11이 ``class_::def()`` 로
  전달되는 특수 함수 ``pybind11::pickle()`` 인 반면, 이 목적을 위해 별도의 메소드
  ``def_pickle`` 를 가지고 있습니다. 이미 ``torch::jit::pickle`` 라는 이름이 사용되었고
  혼동을 일으키고 싶지 않았기 때문입니다.

이러한 방식으로 (역)직렬화 동작을 정의하면 이제 스크립트를 성공적으로 실행할 수 있습니다:

.. code-block:: shell

  $ python ../export_attr.py
  testing

바인딩된 C++ 클래스를 사용하거나 반환하는 커스텀 연산자 정의
--------------------------------------------------------

커스텀 C++ 클래스를 정의한 후에는 해당 클래스를 인수로 사용하거나 커스텀 연산자(예를 들어 free 함수)에서
반환할 수도 있습니다. 다음과 같은 free 함수가 있다고 가정합니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN free_function
  :end-before: END free_function

``TORCH_LIBRARY`` 블록 내에서 다음 코드를 실행하여 등록할 수 있습니다:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN def_free
  :end-before: END def_free

등록 API에 대한 자세한 내용은 `커스텀 C++ 연산자로 TorchScript 확장 <https://tutorials.pytorch.kr/advanced/torch_script_custom_ops.html>`_ 을
참조하세요.

이 작업이 완료되면 다음 예제와 같이 연산자를 사용할 수 있습니다:

.. code-block:: python

  class TryCustomOp(torch.nn.Module):
      def __init__(self):
          super(TryCustomOp, self).__init__()
          self.f = torch.classes.my_classes.MyStackClass(["foo", "bar"])

      def forward(self):
          return torch.ops.my_classes.manipulate_instance(self.f)

.. note::

  C++ 클래스를 인수로 사용하는 연산자를 등록하려면 커스텀 클래스가 이미 등록되어
  있어야 합니다. 커스텀 클래스 등록과 free 함수 정의가 동일한 ``TORCH_LIBRARY``
  블록에 있고 커스텀 클래스 등록이 먼저 오게 하여 이를 시행할 수 있습니다.
  향후 어떤 순서로든 등록할 수 있도록 이 요구사항을 완화할 수 있습니다.


결론
----

이 튜토리얼에서는 독립된 C++ 프로세스에서 C++ 클래스를 TorchScript 및
확장 Python에 나타내는 방법, 해당 메소드를 등록하는 방법, Python 및 TorchScript에서
해당 클래스를 사용하는 방법, 클래스를 사용하여 코드를 저장 및 읽어들이고 해당 코드를 실행하는 방법을 안내했습니다.
이제 타사 C++ 라이브러리와 인터페이스가 있는 C++ 클래스로 TorchScript 모델을 확장하거나,
Python, TorchScript 및 C++ 간의 라인이 원활하게 혼합되어야 하는 다른 사용 사례를 구현할 준비가 되었습니다.

언제나처럼 문제를 마주치거나 질문이 있으면 저희 `forum <https://discuss.pytorch.org/>`_ 또는
`GitHub issues <https://github.com/pytorch/pytorch/issues>`_ 에 올려주시면 되겠습니다.
또한 `자주 묻는 질문(FAQ) 페이지 <https://pytorch.org/cppdocs/notes/faq.html>`_ 에
유용한 정보가 있을 수 있습니다.
