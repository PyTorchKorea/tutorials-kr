.. _custom-ops-landing-page:

PyTorch Custom Operators
PyTorch 사용자 지정 연산자
===========================

PyTorch offers a large library of operators that work on Tensors (e.g. ``torch.add``,
``torch.sum``, etc). However, you may wish to bring a new custom operation to PyTorch
and get it to work with subsystems like ``torch.compile``, autograd, and ``torch.vmap``.
In order to do so, you must register the custom operation with PyTorch via the Python
`torch.library docs <https://pytorch.org/docs/stable/library.html>`_ or C++ ``TORCH_LIBRARY``
APIs.

PyTorch는 Tensor에서 작동하는 대규모 연산자 라이브러리(예: ``torch.add``, ``torch.sum`` 등)를 제공합니다.
그러나 PyTorch에 새로운 사용자 지정 연산을 도입하여 ``torch.compile``, autograd 및 ``torch.vmap`` 와 같은 
하위 시스템에서 작동하도록 할 수 있습니다. 
이렇게 하려면 Python `torch.library docs <https://pytorch.org/docs/stable/library.html>`_ 문서 또는 
C++ ``TORCH_LIBRARY`` API를 통해 PyTorch에 사용자 지정 연산을 등록해야 합니다.


Authoring a custom operator from Python
Python 에서 사용자 지정 연산자 작성
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see :ref:`python-custom-ops-tutorial`.
:ref:`python-custom-ops-tutorial`를 참조하십시오.

You may wish to author a custom operator from Python (as opposed to C++) if:
다음과 같은 경우 Python(C++와 반대)에서 사용자 지정 연산자를 작성할 수 있습니다.

- you have a Python function you want PyTorch to treat as an opaque callable, especially with
  respect to ``torch.compile`` and ``torch.export``.
- PyTorch가 불투명한 호출 가능한 것으로 취급하려는 Python 함수가 있는데, 특히 다음과 같은 경우에는 더욱 그렇습니다.
  ``torch.compile`` 및 ``torch.export``.

- you have some Python bindings to C++/CUDA kernels and want those to compose with PyTorch
  subsystems (like ``torch.compile`` or ``torch.autograd``)
- C++/CUDA 커널에 대한 Python 바인딩이 있으며 PyTorch 하위 시스템
  (예: ``torch.compile`` 또는 ``torch.autograd``)으로 구성되기를 원합니다.
  

Integrating custom C++ and/or CUDA code with PyTorch
사용자 지정 C++ 및/또는 CUDA 코드와 PyTorch 통합
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see :ref:`cpp-custom-ops-tutorial`.
:ref:`cpp-custom-ops-tutorial`를 참조하십시오.

You may wish to author a custom operator from C++ (as opposed to Python) if:
C++(Python과 반대)에서 사용자 지정 연산자를 작성할 수 있는 경우:
- you have custom C++ and/or CUDA code.
- 사용자 지정 C++ 및/또는 CUDA 코드가 있습니다.
- you plan to use this code with ``AOTInductor`` to do Python-less inference.
- 이 코드를 ``AOTInductor``와 함께 사용하여 Python 없는 추론을 수행할 계획입니다.

The Custom Operators Manual
사용자 지정 운영자 설명서
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For information not covered in the tutorials and this page, please see
`The Custom Operators Manual <https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU>`_
(we're working on moving the information to our docs site). We recommend that you
first read one of the tutorials above and then use the Custom Operators Manual as a reference;
it is not meant to be read head to toe.
튜토리얼과 이 페이지에서 다루지 않는 정보는 다음을 참조하시기 바랍니다.
`The Custom Operators Manual <https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU>`_
(정보를 문서 사이트로 옮기는 작업을 진행 중입니다.)
위의 튜토리얼 중 하나를 먼저 읽은 다음 사용자 지정 운영자 설명서를 참조로 사용하는 것이 좋습니다;
머리부터 발끝까지 읽어서는 안 됩니다.

When should I create a Custom Operator?
Custom Operator는 언제 생성해야 합니까?
---------------------------------------
If your operation is expressible as a composition of built-in PyTorch operators
then please write it as a Python function and call it instead of creating a
custom operator. Use the operator registration APIs to create a custom operator if you
are calling into some library that PyTorch doesn't understand (e.g. custom C/C++ code,
a custom CUDA kernel, or Python bindings to C/C++/CUDA extensions).
연산이 내장된 파이토치 연산자의 구성으로 표현할 수 있는 경우
사용자 지정 연산자를 만드는 대신 Python 함수로 작성하여 호출하세요.
다음과 같은 경우 운영자 등록 API를 사용하여 사용자 지정 연산자를 생성합니다.
PyTorch가 이해하지 못하는 라이브러리(예: 사용자 지정 C/C++ 코드를 호출하는 경우,
커스텀 CUDA 커널 또는 C/C++/CUDA 확장에 대한 Python 바인딩).

Why should I create a Custom Operator?
--------------------------------------

It is possible to use a C/C++/CUDA kernel by grabbing a Tensor's data pointer
and passing it to a pybind'ed kernel. However, this approach doesn't compose with
PyTorch subsystems like autograd, torch.compile, vmap, and more. In order
for an operation to compose with PyTorch subsystems, it must be registered
via the operator registration APIs.
