사용자 정의 C++ 클래스로의 TorchScript 확장
===============================================

이 튜토리얼은 다음의 문서의 튜토리얼을 따릅니다.
:doc:`custom operator <torch_script_custom_ops>`
C++ 클래스를 TorchScript와 Python에 동시에 바인딩하기 위해 구축한 API를 소개합니다. API는 다음과 매우 유사합니다.
`pybind11 <https://github.com/pybind/pybind11>`_ 및 해당 시스템에 익숙하다면 대부분의 개념이 이전됩니다.


C++에서 클래스 구현 및 바인딩
-----------------------------------------

이 튜토리얼에서는 멤버 변수에서 지속 상태를 유지하는 간단한 C++ 클래스를 정의할 것입니다.

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN class
  :end-before: END class

몇가지 사항을 명시하겠습니다. :

- ``torch/custom_class.h`` 는 사용자 정의 클래스로 TorchScript를 확장하기 위해 포함해야 하는 헤더입니다.
- 커스텀 클래스의 인스턴스로 작업할 때마다 ``c10::intrusive_ptr<>``의 인스턴스를 통해 작업을 수행합니다. ``intrusive_ptr``을 ``std::shared_ptr``과 같은 스마트 포인터로 생각됩니다.
  그러나 참조 카운트는 별도의 메타데이터 블록과 달리 객체에 직접 저장됩니다.
  ``std::shared_ptr ``.``torch::Tensor``는 내부적으로 동일한 포인터 유형을 사용하며 사용자 정의 클래스도 이 포인터 유형을 사용해야 다양한 객체 유형을 일관되게 관리할 수 있습니다.
- 두 번째로 주목해야 할 점은 사용자 정의 클래스가 ``torch::CustomClassHolder``에서 상속되어야 한다는 것입니다. 이렇게 하면 사용자 지정 클래스에 참조 횟수를 저장할 공간이 있습니다.

이제 이 클래스를 *binding* 이라는 프로세스인 TorchScript에 표시하는 방법을 살펴보겠습니다.:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN binding
  :end-before: END binding
  :append:
      ;
    }



CMake를 사용하여 C++ 프로젝트로 예제 빌드
------------------------------------------------

이제 `CMake <https://cmake.org>`_ 빌드 시스템을 사용하여 위의 C++ 코드를 빌드합니다.
먼저, 지금까지 다룬 모든 C++ 코드를 ``class.cpp``라는 파일에 넣습니다.
그런 다음 간단한 ``CMakeLists.txt`` 파일을 작성하여 동일한 디렉토리에 배치합니다. 
``CMakeLists.txt``는 다음과 같아야 합니다. :

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/CMakeLists.txt
  :language: cmake


또한 ``build`` 디렉토리를 만듭니다. 파일 트리는 다음과 같아야 합니다::

  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/

:doc:`이전 튜토리얼 <torch_script_custom_ops>`에서 설명한 것과 동일한 방식으로 환경을 설정했다고 가정합니다.
계속해서 cmake를 호출한 다음 make를 호출하여 프로젝트를 빌드합니다.:

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

What you'll find is there is now (among other things) a dynamic library
file present in the build directory. On Linux, this is probably named
``libcustom_class.so``. So the file tree should look like::

  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/
      libcustom_class.so

Using the C++ Class from Python and TorchScript
-----------------------------------------------

Now that we have our class and its registration compiled into an ``.so`` file,
we can load that `.so` into Python and try it out. Here's a script that
demonstrates that:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/custom_test.py
  :language: python


Saving, Loading, and Running TorchScript Code Using Custom Classes
------------------------------------------------------------------

We can also use custom-registered C++ classes in a C++ process using
libtorch. As an example, let's define a simple ``nn.Module`` that
instantiates and calls a method on our MyStackClass class:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/save.py
  :language: python

``foo.pt`` in our filesystem now contains the serialized TorchScript
program we've just defined.

Now, we're going to define a new CMake project to show how you can load
this model and its required .so file. For a full treatment of how to do this,
please have a look at the `Loading a TorchScript Model in C++ Tutorial <https://tutorials.pytorch.kr/advanced/cpp_export.html>`_.

Similarly to before, let's create a file structure containing the following::

  cpp_inference_example/
    infer.cpp
    CMakeLists.txt
    foo.pt
    build/
    custom_class_project/
      class.cpp
      CMakeLists.txt
      build/

Notice we've copied over the serialized ``foo.pt`` file, as well as the source
tree from the ``custom_class_project`` above. We will be adding the
``custom_class_project`` as a dependency to this C++ project so that we can
build the custom class into the binary.

Let's populate ``infer.cpp`` with the following:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/infer.cpp
  :language: cpp

And similarly let's define our CMakeLists.txt file:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/CMakeLists.txt
  :language: cpp

You know the drill: ``cd build``, ``cmake``, and ``make``:

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

And now we can run our exciting C++ binary:

.. code-block:: shell

  $ ./infer
    momfoobarbaz

Incredible!

Moving Custom Classes To/From IValues
-------------------------------------

It's also possible that you may need to move custom classes into or out of
``IValue``s, such as when you take or return ``IValue``s from TorchScript methods
or you want to instantiate a custom class attribute in C++. For creating an
``IValue`` from a custom C++ class instance:

- ``torch::make_custom_class<T>()`` provides an API similar to c10::intrusive_ptr<T>
  in that it will take whatever set of arguments you provide to it, call the constructor
  of T that matches that set of arguments, and wrap that instance up and return it.
  However, instead of returning just a pointer to a custom class object, it returns
  an ``IValue`` wrapping the object. You can then pass this ``IValue`` directly to
  TorchScript.
- In the event that you already have an ``intrusive_ptr`` pointing to your class, you
  can directly construct an IValue from it using the constructor ``IValue(intrusive_ptr<T>)``.

For converting ``IValue`` back to custom classes:

- ``IValue::toCustomClass<T>()`` will return an ``intrusive_ptr<T>`` pointing to the
  custom class that the ``IValue`` contains. Internally, this function is checking
  that ``T`` is registered as a custom class and that the ``IValue`` does in fact contain
  a custom class. You can check whether the ``IValue`` contains a custom class manually by
  calling ``isCustomClass()``.

Defining Serialization/Deserialization Methods for Custom C++ Classes
---------------------------------------------------------------------

If you try to save a ``ScriptModule`` with a custom-bound C++ class as
an attribute, you'll get the following error:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/export_attr.py
  :language: python

.. code-block:: shell

  $ python export_attr.py
  RuntimeError: Cannot serialize custom bound C++ class __torch__.torch.classes.my_classes.MyStackClass. Please define serialization methods via def_pickle for this class. (pushIValueImpl at ../torch/csrc/jit/pickler.cpp:128)

This is because TorchScript cannot automatically figure out what information
save from your C++ class. You must specify that manually. The way to do that
is to define ``__getstate__`` and ``__setstate__`` methods on the class using
the special ``def_pickle`` method on ``class_``.

.. note::
  The semantics of ``__getstate__`` and ``__setstate__`` in TorchScript are
  equivalent to that of the Python pickle module. You can
  `read more <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md#getstate-and-setstate>`_
  about how we use these methods.

Here is an example of the ``def_pickle`` call we can add to the registration of
``MyStackClass`` to include serialization methods:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN def_pickle
  :end-before: END def_pickle

.. note::
  We take a different approach from pybind11 in the pickle API. Whereas pybind11
  as a special function ``pybind11::pickle()`` which you pass into ``class_::def()``,
  we have a separate method ``def_pickle`` for this purpose. This is because the
  name ``torch::jit::pickle`` was already taken, and we didn't want to cause confusion.

이러한 방식으로 (역)직렬화 동작을 정의하면 이제 스크립트를 성공적으로 실행할 수 있습니다.:

.. code-block:: shell

  $ python ../export_attr.py
  testing

바인딩된 C++ 클래스를 사용하거나 반환하는 사용자 지정 연산자 정의
---------------------------------------------------------------

사용자 정의 C++ 클래스를 정의한 후에는 해당 클래스를 인수로 사용하거나 사용자 정의 연산자(즉, 자유 함수)에서 반환할 수도 있습니다. 
다음과 같은 무료 기능이 있다고 가정합니다.:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN free_function
  :end-before: END free_function

 ``TORCH_LIBRARY`` 내부에서 다음 코드를 실행하여 등록할 수 있습니다.:

.. literalinclude:: ../advanced_source/torch_script_custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN def_free
  :end-before: END def_free

등록 API에 대한 자세한 내용은 `custom op tutorial <https://tutorials.pytorch.kr/advanced/torch_script_custom_ops.html>`_을 참고하세요.


Once this is done, you can use the op like the following example:

.. code-block:: python

  class TryCustomOp(torch.nn.Module):
      def __init__(self):
          super(TryCustomOp, self).__init__()
          self.f = torch.classes.my_classes.MyStackClass(["foo", "bar"])

      def forward(self):
          return torch.ops.foo.manipulate_instance(self.f)

.. note::

C++ 클래스를 인수로 사용하는 연산자를 등록하려면 사용자 정의 클래스가 이미 등록되어 있어야 합니다. 사용자 정의 클래스 등록과 자유 함수 정의가 동일한 ``TORCH_LIBRARY`` 블록에 있고 사용자 정의 클래스 등록이 먼저 오게 하여 이를 시행할 수 있습니다. 앞으로 어떤 순서로든 등록할 수 있도록 이 요구 사항을 완화할 수 있습니다.

결론
----------

이 튜토리얼에서는 C++ 클래스를 TorchScript(및 확장 Python)에 노출하는 방법, 해당 메서드를 등록하는 방법, Python 및 TorchScript에서 해당 클래스를 사용하는 방법, 클래스를 사용하여 코드를 저장 및 로드하고 해당 코드를 실행하는 방법을 안내했습니다. 독립 실행형 C++ 프로세스에서. 이제 타사 C++ 라이브러리와 인터페이스하는 C++ 클래스로 TorchScript 모델을 확장하거나 Python, TorchScript 및 C++ 간의 라인이 원활하게 혼합되어야 하는 다른 사용 사례를 구현할 준비가 되었습니다.

항상 그렇듯이 문제가 발생하거나 질문이 있는 경우 `포럼 <https://discuss.pytorch.org/>`_ 또는 `GitHub 문제 <https://github.com/pytorch/pytorch/ 문제>`_ 연락하십시오. 또한 '자주 묻는 질문(FAQ)' 페이지 <https://pytorch.org/cppdocs/notes/faq.html>`_에서 유용한 정보를 얻을 수 있습니다.
