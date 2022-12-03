Custom C++ 과 CUDA 확장하기
==============================
**Author**: `Peter Goldsborough <https://www.goldsborough.me/>`_
**번역**: `하상호 <https://github.com/sangho0804>`_

PyTorch는 신경망(neural network), 임의의 텐서 대수(tensor algebra), 데이터 랭글링
(data wrangling) 및 기타 목적과 관련된 수많은 작업을 제공합니다. 그러나 여전히 
customized operation이 더 필요할 수 있습니다. 예를 들어 논문에서 찾은 새로운 활성화 
기능을 사용하거나 연구의 일부로 개발한 작업을 구현하고 싶을 수 있습니다.

PyTorch에서 이러한 custom operation을 통합하는 가장 쉬운 방법은  `여기
<https://pytorch.org/docs/master/notes/extending.html>`_ 에 설명된 대로
:class:`Function`과 :class:`Module`을 확장하여 Python으로 작성하는 것입니다. 이는 
파이썬의 일반적인 표현력뿐만 아니라 자동 미분(미분함수를 작성하지 않아도 됩니다.)의 
모든 기능을 제공합니다. 그러나 C++에서 작업이 더 잘 구현되는 경우가 있을 수 있습니다. 
예를 들어, 모델에서 자주 호출되거나 호출 횟수가 적더라도 매우 비싸기 때문에 코드가 매우 
빨라야 할 수 있습니다. 또 다른 그럴듯한 이유는 그것이 다른 C 나 C++ 라이브러리에 의존
하거나 상호작용하기 때문입니다. 이러한 경우를 해결하기 위해 PyTorch는 custom C++ extension을 
작성하는 매우 쉬운 방법을 제공합니다. 

C++ extension은 사용자가 소스 외부, 즉 PyTorch 백엔드와 별도로 정의된 PyTorch 연산자를 
생성할 수 있도록 개발한 메커니즘입니다. 이 접근 방식은 기본 PyTorch 작업이 구현되는 방식과 
다릅니다. C++ extension은 PyTorch 기반 프로젝트에 높은 수준의 유연성을 제공하는 동시에 
PyTorch의 백엔드와 작업을 통합하는 것과 관련된 많은 boilerplate를 절약하기 위한 것입니다. 
그럼에도 불구하고 작업을 C++ extension으로 정의한 후에 이를 기본 PyTorch 함수로 전환하는 
것은 주로 코드 구성의 문제이며 작업을 업스트림에 기여하기로 결정한 경우 사후에 처리할 수 
있습니다.

동기와 예
---------

이 문서의 마지막 부분에서는 C++(및 CUDA) 확장을 작성하고 사용하는 실제 예제를 볼 
수 있습니다. 당신이 시간에 쫓기고 있거나 하루가 끝날 때까지 작업을 완료하지 않으면 
누군가가 당신을 해고할 경우 이 섹션을 건너뛰고 바로 다음 섹션의 구현 세부 정보로 
이동할 수 있습니다.

최신 기술에 비해 우수한 속성을 가진 새로운 종류의 recurrent unit을 발견했다고 
가정해 보겠습니다. 이 recurrent unit은 LSTM과 유사하지만 * forget gate* 가 없고 
*Exponential Linear Unit* (ELU)를 내부 활성화 함수로 사용 한다는 점이 다릅니다. 
왜냐하면 이 단위는 절대 잊어버리지 않기 때문입니다. 따라서 우리는 이것을 LLTM 또는 
*Long-Long-Term-Memory* unit이라고 합니다.

LLTM이 vanilla LSTM과 다른 두 가지 방식은 우리가 목적에 맞게 PyTorch의 ``LSTMCell``을 
구성할 수 없을 정도로 꽤나 중요하므로 사용자 지정 셀을 만들어야 합니다. 이를 위한 
모든 경우에 대한 첫 번째이자 가장 쉬운 접근법은 Python을 사용하여 일반 PyTorch에서 
원하는 기능을 구현하는 것입니다. 이를 위해서는:class:`torch.nn.Module`을 하위 분류하고 
LLTM의 포워드 패스를 구현해야 합니다. 이는 다음과 같이 확인 할 수 있습니다 ::

  class LLTM(torch.nn.Module):
      def __init__(self, input_features, state_size):
          super(LLTM, self).__init__()
          self.input_features = input_features
          self.state_size = state_size
          # 3 * state_size (input gate, output gate 와 candidate cell gate)
          # input_features + state_size를 합니다. 왜냐하면 [input, h]로 곱해줄 것이기 때문입니다.
          self.weights = torch.nn.Parameter(
              torch.empty(3 * state_size, input_features + state_size))
          self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
          self.reset_parameters()

      def reset_parameters(self):
          stdv = 1.0 / math.sqrt(self.state_size)
          for weight in self.parameters():
              weight.data.uniform_(-stdv, +stdv)

      def forward(self, input, state):
          old_h, old_cell = state
          X = torch.cat([old_h, input], dim=1)

          # 하나의 MM으로 input, output 및 candidate cell gates를 계산합니다.
          gate_weights = F.linear(X, self.weights, self.bias)
          # 결합된 gate weight metrix를 components로 분할합니다.
          gates = gate_weights.chunk(3, dim=1)

          input_gate = torch.sigmoid(gates[0])
          output_gate = torch.sigmoid(gates[1])
          # 여기서는 일반적인 tanh 대신 ELU를 사용합니다.
          candidate_cell = F.elu(gates[2])

          # 새로운 cell state를 계산합니다.
          new_cell = old_cell + candidate_cell * input_gate
          # 새로운 hidden state 및 output을 계산합니다.
          new_h = torch.tanh(new_cell) * output_gate

          return new_h, new_cell

그러면 예상한 대로 사용할 수 있습니다 ::

  import torch

  X = torch.randn(batch_size, input_features)
  h = torch.randn(batch_size, state_size)
  C = torch.randn(batch_size, state_size)

  rnn = LLTM(input_features, state_size)

  new_h, new_C = rnn(X, (h, C))

당연히 모든게 가능하고 타당하다면 이 접근 방식을 사용하여 PyTorch를 확장해야 합니다. 
 pytorch는 `NVIDIA cuDNN
<https://developer.nvidia.com/cudnn>`_, `Intel MKL
<https://software.intel.com/en-us/mkl>`_ 또는 `NNPACK
<https://github.com/Maratyszcza/NNPACK>`_ 같은 라이브러리에 의해 구동되는 CPU 및 
GPU에 대해 매우 최적화된 구현을 가지고 있기 때문에, 위와 같은 PyTorch 코드는 종종 
충분히 빠릅니다. 그러나 특정 상황에서 추가 성능 개선의 여지가 있는 이유도 알 수 있습니다. 
가장 분명한 이유는 PyTorch가 당신이 구현하고 있는 *알고리즘*에 대한 지식이 없기 때문입니다. 
알고리즘을 구성하는 데 사용하는 개별 작업만 알고 있습니다. 따라서 PyTorch는 작업을 차례로 
개별적으로 실행해야 합니다. CUDA 커널의 시작을 포함할 수 있는 동작의 구현(또는 *커널*)에 
대한 각각의 개별 호출은 일정한 양의 오버헤드를 가지고 있기 때문에, 이 오버헤드는 많은 함수 
호출에서 중요해질 수 있습니다. 게다가, 코드를 실행하는 파이썬 인터프리터는 그 자체로 프로그램을 
느리게 할 수 있습니다.

따라서 작업 속도를 높이는 확실한 방법은 C++(또는 CUDA)로 어떤 부분을 다시 작성하고 특정 
작업 그룹을 *fuse* 하는 것입니다. Fusing은 많은 기능의 구현을 단일 기능으로 결합하는 것을 
의미하며, 이는 더 적은 커널 실행과 전반적인 데이터 흐름에 대한 향상된 가시성을 통해 수행
할 수 있는 기타 최적화의 이점을 얻습니다.

C++ extensions을 사용하여 LLTM의 *Fused* version을 구현하는 방법을 살펴보겠습니다. 
PyTorch 백엔드의 대부분을 지원하는 `ATen<https://github.com/zdevito/ATen>`_ 라이브러리를 
사용하여 일반 C++로 작성하는 것으로 시작하여 Python 코드를 얼마나 쉽게 변환할 수 있는지 
확인합니다. 그런 다음 모델의 일부를 CUDA 커널로 이동하여 GPU가 제공하는 대규모 병렬 처리의 
이점을 활용하여 작업 속도를 더욱 높일 것입니다.

C++ Extension 작성하기
----------------------

C++ extensions come in two flavors: They can be built "ahead of time" with
:mod:`setuptools`, or "just in time" via
:func:`torch.utils.cpp_extension.load`. We'll begin with the first approach and
discuss the latter later.
C++ 확장은 두 가지로 제공됩니다. :mod:`setuptools`를 사용하여 "ahead of time" 빌드하거나 
:func:`torch.utils.cpp_extension.load`를 통해 "just in time" 빌드할 수 있습니다. 
첫 번째 접근 방식부터 시작하여 나중에 후자에 대해 설명합니다.

:mod:`setuptools`를 사용하여 빌드하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"ahead of time" 기능을 위해 setuptools를 사용하여 C++ 코드를 컴파일하는 ``setup.py`` 스크립트를 
작성하여 C++ 확장을 구축합니다. LLTM의 경우 다음과 같이 간단합니다 ::

  from setuptools import setup, Extension
  from torch.utils import cpp_extension

  setup(name='lltm_cpp',
        ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension})

이 코드에서 :class:`CppExtension`은 올바른 포함 경로를 전달하고 확장 언어를 C++로 설정하는 
:class:`setuptools.Extension` 주변의 편의상의 wrapper 입니다. 동등한 vanilla :mod:`setuptools` 
코드는 간단히 다음과 같습니다 ::

  Extension(
     name='lltm_cpp',
     sources=['lltm.cpp'],
     include_dirs=cpp_extension.include_paths(),
     language='c++')

:class:`BuildExtension`은 여러 필수 구성 단계와 검사를 수행하고 C++/CUDA 확장이 혼합된 경우 
혼합된 컴파일을 관리합니다. 그리고 이것이 지금 당장 C++ 확장을 구축하는 데 알아야 할 전부입니다!
이제 'lltm cpp'로 들어가는 C++ 확장의 구현에 대해 살펴보겠습니다.

C++ Op 작성하기
^^^^^^^^^^^^^^^

C++에서 LLTM 구현을 시작해 봅시다! backward pass에 필요한 기능 중 하나는 시그모이드의 
도함수입니다. 이 코드는 C++ 확장을 작성할 때 사용할 수 있는 전체 환경에 대해 설명하기에 
충분합니다 :

.. code-block:: cpp

  #include <torch/extension.h>

  #include <iostream>

  torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
  }

``<torch/extension.h>``은 C++ 확장을 작성하는 데 필요한 모든 PyTorch bits를 포함하는 
one-stop header입니다. 여기에는 다음이 포함됩니다.

- 텐서 계산을 위한 주요 API인 ATen library,
- C++ 코드에 대한 Python bindings를 만드는 방법인 `pybind11 <https://github.com/pybind/pybind11>`_,
- ATEN과 pybind11 간의 상호 작용 세부 정보를 관리하는 headers

'd_sigmoid()'의 구현은 ATen API를 사용하는 방법을 보여준다. PyTorch의 텐서 및 variable 인터페이스는 
ATen 라이브러리에서 자동으로 생성되므로 Python 구현 1:1을 다소 C++로 변환할 수 있다. 
모든 계산에 대한 기본 데이터 유형은 :class:`torch::Tensor` 이다. 전체 API는 `여기
<https://pytorch.org/cppdocs/api/classat_1_1_tensor.html>`_에서 검사할 수 있습니다. 
또한 다른 *any other C or C++ header* 나 ``<iostream>`` 를 포함할 수 있고, C++11의 모든 기능을 
마음대로 사용할 수 있습니다. 

CUDA-11.5 nvcc는 Windows에서 torch/extension.h를 구문 분석하는 동안 내부 컴파일러 오류에 
부딪힙니다. 이 문제를 해결하려면 Python binding logic을 pure C++ 파일로 이동하십시오. 
사용 예:

.. code-block:: cpp

  #include <ATen/ATen.h>
  at::Tensor SigmoidAlphaBlendForwardCuda(....)

Instead of:

.. code-block:: cpp

  #include <torch/extension.h>
  torch::Tensor SigmoidAlphaBlendForwardCuda(...)

nvcc 버그에 대한 현재 공개된 문제는 `여기
<https://github.com/pytorch/pytorch/issues/69460>`_.
해결방법 코드 예제 `여기
<https://github.com/facebookresearch/pytorch3d/commit/cb170ac024a949f1f9614ffe6af1c38d972f7d48>`_. 

Forward Pass
************

다음으로 전체 forward pass를 C++로 port할 수 있습니다 :

.. code-block:: cpp

  #include <vector>

  std::vector<at::Tensor> lltm_forward(
      torch::Tensor input,
      torch::Tensor weights,
      torch::Tensor bias,
      torch::Tensor old_h,
      torch::Tensor old_cell) {
    auto X = torch::cat({old_h, input}, /*dim=*/1);

    auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
    auto gates = gate_weights.chunk(3, /*dim=*/1);

    auto input_gate = torch::sigmoid(gates[0]);
    auto output_gate = torch::sigmoid(gates[1]);
    auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

    auto new_cell = old_cell + candidate_cell * input_gate;
    auto new_h = torch::tanh(new_cell) * output_gate;

    return {new_h,
            new_cell,
            input_gate,
            output_gate,
            candidate_cell,
            X,
            gate_weights};
  }

Backward Pass
*************

C++ 확장 API는 현재 backwards function을 자동으로 생성하는 방법을 제공하지 않습니다. 
이와 같이 우리는 LLTM의 Backward Pass도 구현해야 합니다. 이것은 forward pass의 각 
입력에 대한 손실의 도함수를 계산합니다. 궁극적으로, 우리는 파이썬 파이딩을 만들기 위해 
forward와 backward function을 모두 :class:`torch.autograd.Function`로 만들 겁니다.
backward function은 약간 더 포함되어 있으므로 코드를 더 깊이 파고들지는 않을 것입니다.
(관심이 있으시다면, `Alex Graves' thesis
<https://www.cs.toronto.edu/~graves/phd.pdf>`_ 논문을 통해 이에 대한 더 많은 정보를 얻을 
수 있습니다.):

.. code-block:: cpp

  // tanh'(z) = 1 - tanh^2(z)
  torch::Tensor d_tanh(torch::Tensor z) {
    return 1 - z.tanh().pow(2);
  }

  // elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
  torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
    auto e = z.exp();
    auto mask = (alpha * (e - 1)) < 0;
    return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
  }

  std::vector<torch::Tensor> lltm_backward(
      torch::Tensor grad_h,
      torch::Tensor grad_cell,
      torch::Tensor new_cell,
      torch::Tensor input_gate,
      torch::Tensor output_gate,
      torch::Tensor candidate_cell,
      torch::Tensor X,
      torch::Tensor gate_weights,
      torch::Tensor weights) {
    auto d_output_gate = torch::tanh(new_cell) * grad_h;
    auto d_tanh_new_cell = output_gate * grad_h;
    auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

    auto d_old_cell = d_new_cell;
    auto d_candidate_cell = input_gate * d_new_cell;
    auto d_input_gate = candidate_cell * d_new_cell;

    auto gates = gate_weights.chunk(3, /*dim=*/1);
    d_input_gate *= d_sigmoid(gates[0]);
    d_output_gate *= d_sigmoid(gates[1]);
    d_candidate_cell *= d_elu(gates[2]);

    auto d_gates =
        torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

    auto d_weights = d_gates.t().mm(X);
    auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

    auto d_X = d_gates.mm(weights);
    const auto state_size = grad_h.size(1);
    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
  }

Python에 Binding 하기
^^^^^^^^^^^^^^^^^^^^^

C++ 및 ATen으로 operation을 작성하면 pybind11을 사용하여 매우 간단한 방식으로 
C++ 함수 또는 클래스를 Python에 바인딩할 수 있습니다. PyTorch C++ 확장에서의 이 부분에 
대한 질문이나 문제는 주로 `pybind11 documentation
<https://pybind11.readthedocs.io/en/stable/>`_ 설명서에서 해결됩니다.

extension을 위한 필수 binding 코드는 네 줄에 불과합니다.:

.. code-block:: cpp

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lltm_forward, "LLTM forward");
    m.def("backward", &lltm_backward, "LLTM backward");
  }

여기서 주목해야 할 한 가지는 매크로 ``TORCH_EXTENSION_NAME``입니다. 토치 확장 빌드는 
이를 ``setup.py`` 스크립트에서 확장에 지정한 이름으로 정의합니다. 이 경우
``TORCH_EXTENSION_NAME``의 값은 "lltm_cpp"입니다. 이는 두 위치(빌드 스크립트와 C++ 코드)에서 
확장 명을 유지해야 하는 것을 피하기 위한 것입니다. 둘 사이의 불일치가 심각하여 추적하기 어려운 
문제로 이어질 수 있기 때문입니다.

Extension 사용하기
^^^^^^^^^^^^^^^^^^

이제 PyTorch에서 확장을 가져오도록 설정되었습니다. 
이 시점에서 디렉터리 구조는 다음과 같을 수 있습니다.::

  pytorch/
    lltm-extension/
      lltm.cpp
      setup.py

이제 ``python setup.py install``를 실행 하여 확장 프로그램을 빌드하고 설치합니다.
이는 다음과 같이 실행됩니다.::

  running install
  running bdist_egg
  running egg_info
  creating lltm_cpp.egg-info
  writing lltm_cpp.egg-info/PKG-INFO
  writing dependency_links to lltm_cpp.egg-info/dependency_links.txt
  writing top-level names to lltm_cpp.egg-info/top_level.txt
  writing manifest file 'lltm_cpp.egg-info/SOURCES.txt'
  reading manifest file 'lltm_cpp.egg-info/SOURCES.txt'
  writing manifest file 'lltm_cpp.egg-info/SOURCES.txt'
  installing library code to build/bdist.linux-x86_64/egg
  running install_lib
  running build_ext
  building 'lltm_cpp' extension
  creating build
  creating build/temp.linux-x86_64-3.7
  gcc -pthread -B ~/local/miniconda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I~/local/miniconda/lib/python3.7/site-packages/torch/include -I~/local/miniconda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -I~/local/miniconda/lib/python3.7/site-packages/torch/include/TH -I~/local/miniconda/lib/python3.7/site-packages/torch/include/THC -I~/local/miniconda/include/python3.7m -c lltm.cpp -o build/temp.linux-x86_64-3.7/lltm.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=lltm_cpp -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11
  cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
  creating build/lib.linux-x86_64-3.7
  g++ -pthread -shared -B ~/local/miniconda/compiler_compat -L~/local/miniconda/lib -Wl,-rpath=~/local/miniconda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/lltm.o -o build/lib.linux-x86_64-3.7/lltm_cpp.cpython-37m-x86_64-linux-gnu.so
  creating build/bdist.linux-x86_64
  creating build/bdist.linux-x86_64/egg
  copying build/lib.linux-x86_64-3.7/lltm_cpp.cpython-37m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
  creating stub loader for lltm_cpp.cpython-37m-x86_64-linux-gnu.so
  byte-compiling build/bdist.linux-x86_64/egg/lltm_cpp.py to lltm_cpp.cpython-37.pyc
  creating build/bdist.linux-x86_64/egg/EGG-INFO
  copying lltm_cpp.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
  copying lltm_cpp.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
  copying lltm_cpp.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
  copying lltm_cpp.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
  writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
  zip_safe flag not set; analyzing archive contents...
  __pycache__.lltm_cpp.cpython-37: module references __file__
  creating 'dist/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
  removing 'build/bdist.linux-x86_64/egg' (and everything under it)
  Processing lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
  removing '~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg' (and everything under it)
  creating ~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
  Extracting lltm_cpp-0.0.0-py3.7-linux-x86_64.egg to ~/local/miniconda/lib/python3.7/site-packages
  lltm-cpp 0.0.0 is already the active version in easy-install.pth

  Installed ~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
  Processing dependencies for lltm-cpp==0.0.0
  Finished processing dependencies for lltm-cpp==0.0.0


컴파일러에 대한 작은 참고 사항: ABI 버전 관리 문제로 인해 C++ 확장을 빌드하는 데 사용 하는 
컴파일러는 PyTorch가 빌드된 컴파일러와 *ABI 호환 가능* 해야 합니다. 실제로 이는 Linux에서 
GCC 버전 4.9 이상을 사용해야 함을 의미합니다. Ubuntu 16.04 및 기타 최신 Linux 배포판의 경우 
이것이 이미 기본 컴파일러여야 합니다. MacOS에서는 clang(ABI 버전 관리 문제가 없음)을 사용해야 
합니다. 최악의 경우 컴파일러로 소스에서 PyTorch를 빌드한 다음 동일한 컴파일러로 확장을 빌드할 
수 있습니다.

확장이 구축되면 ``setup.py`` 스크립트에서 지정한 이름을 사용하여 Python에서 간단히 가져올 수 
있습니다. 먼저 ``import torch``를 수행하십시오. 그러면 dynamic linker에 표시되어야 하는 몇 
가지 symbols이 해결됩니다.::

  In [1]: import torch
  In [2]: import lltm_cpp
  In [3]: lltm_cpp.forward
  Out[3]: <function lltm.PyCapsule.forward>

 
``help()`` 함수나 모듈 을 호출하면 signature이 C++ 코드와 일치함을 알 수 
있습니다.::

  In[4] help(lltm_cpp.forward)
  forward(...) method of builtins.PyCapsule instance
      forward(arg0: torch::Tensor, arg1: torch::Tensor, arg2: torch::Tensor, arg3: torch::Tensor, arg4: torch::Tensor) -> List[torch::Tensor]

      LLTM forward

이제 파이썬에서 C++ 함수를 호출할 수 있으므로 :class:`torch.autograd.Function`와 
:class:`torch.nn.Module`로 감싸 파이토치의 first class citizens으로 
만들 수 있습니다.::


  import math
  import torch

  # Our module!
  import lltm_cpp

  class LLTMFunction(torch.autograd.Function):
      @staticmethod
      def forward(ctx, input, weights, bias, old_h, old_cell):
          outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
          new_h, new_cell = outputs[:2]
          variables = outputs[1:] + [weights]
          ctx.save_for_backward(*variables)

          return new_h, new_cell

      @staticmethod
      def backward(ctx, grad_h, grad_cell):
          outputs = lltm_cpp.backward(
              grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
          d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
          return d_input, d_weights, d_bias, d_old_h, d_old_cell


  class LLTM(torch.nn.Module):
      def __init__(self, input_features, state_size):
          super(LLTM, self).__init__()
          self.input_features = input_features
          self.state_size = state_size
          self.weights = torch.nn.Parameter(
              torch.empty(3 * state_size, input_features + state_size))
          self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
          self.reset_parameters()

      def reset_parameters(self):
          stdv = 1.0 / math.sqrt(self.state_size)
          for weight in self.parameters():
              weight.data.uniform_(-stdv, +stdv)

      def forward(self, input, state):
          return LLTMFunction.apply(input, self.weights, self.bias, *state)

성능 비교
*********

이제 PyTorch에서 C++ 코드를 사용하고 호출할 수 있으므로 small benchmark를 실행하여 
작업을 C++로 재작성하여 얼마나 많은 성능을 얻었는지 확인할 수 있습니다. LLTM을 forwards와
backwards를 통해 몇 번 실행하고 기간을 측정합니다.::

  import time

  import torch

  batch_size = 16
  input_features = 32
  state_size = 128

  X = torch.randn(batch_size, input_features)
  h = torch.randn(batch_size, state_size)
  C = torch.randn(batch_size, state_size)

  rnn = LLTM(input_features, state_size)

  forward = 0
  backward = 0
  for _ in range(100000):
      start = time.time()
      new_h, new_C = rnn(X, (h, C))
      forward += time.time() - start

      start = time.time()
      (new_h.sum() + new_C.sum()).backward()
      backward += time.time() - start

  print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))

이 post의 시작 부분에서 pure Python으로 작성한 원래 LLTM으로 이 코드를 실행하면 
다음과 같은 수치를 얻습니다(내 장치에서)::

  Forward: 506.480 us | Backward 444.694 us

and with our new C++ version::

  Forward: 349.335 us | Backward 443.523 us

이미 forward function의 상당한 속도 향상(30% 이상)을 볼 수 있습니다. backward function의 
경우 주요한 것은 아니지만 속도 향상이 눈에 띕니다. 위에서 쓴 backward pass는 특별히 최적화되지 
않았으며 확실히 개선될 수 있습니다. 또한 PyTorch의 자동 미분 엔진은 계산 그래프를 자동으로 
병렬화할 수 있고, 전반적으로 보다 효율적인 작업 흐름을 사용할 수 있으며, C++로도 구현되어 있어 
빠를 것으로 예상됩니다. 따라서 이것은 좋은 시작입니다.

GPU Devices의 성능
******************

PyTorch 의 ATEN 백엔드에 대한 놀라운 사실은 실행 중인 컴퓨팅 장치를 추상화한다는 것입니다. 
이는 우리가 CPU용으로 작성한 동일한 코드가 GPU 에서도 실행될 수 있으며 개별 작업이 그에 따라 
GPU 최적화 구현으로 발송됨을 의미합니다. 행렬 곱셈(``mm``또는 ``addmm`` 같은)과 같은 특정 
작업의 경우 이는 큰 이점입니다. CUDA 텐서로 C++ 코드를 실행하여 얼마나 많은 성능을 얻을 수 있는지 
살펴보겠습니다. 구현에 대한 변경이 필요하지 않으며 ``device=cuda_device``생성 시 인수를 
추가하거나 생성한 후 ``.to(cuda_device)``사용하여 Python에서 GPU 메모리에 텐서를 넣기만 하면 
됩니다.::

  import torch

  assert torch.cuda.is_available()
  cuda_device = torch.device("cuda")  # device object representing GPU

  batch_size = 16
  input_features = 32
  state_size = 128

  # Note the device=cuda_device arguments here
  X = torch.randn(batch_size, input_features, device=cuda_device)
  h = torch.randn(batch_size, state_size, device=cuda_device)
  C = torch.randn(batch_size, state_size, device=cuda_device)

  rnn = LLTM(input_features, state_size).to(cuda_device)

  forward = 0
  backward = 0
  for _ in range(100000):
      start = time.time()
      new_h, new_C = rnn(X, (h, C))
      torch.cuda.synchronize()
      forward += time.time() - start

      start = time.time()
      (new_h.sum() + new_C.sum()).backward()
      torch.cuda.synchronize()
      backward += time.time() - start

  print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

일반적인 PyTorch 코드를 이제 CUDA 장치에서 실행되는 C++ 버전과 다시 한 번 비교하면 성능이 
다시 향상되는 것을 볼 수 있습니다. Python/PyTorch의 경우::

  Forward: 187.719 us | Backward 410.815 us

And C++/ATen::

  Forward: 149.802 us | Backward 393.458 us

non-CUDA 코드에 비해 전반적으로 속도가 크게 향상되었습니다. 그러나 custom CUDA 커널을 
작성하여 C++ 코드에서 훨씬 더 많은 성능을 끌어낼 수 있습니다. 곧 자세히 살펴보겠습니다. 
그 전에 C++ 확장을 구축하는 또 다른 방법에 대해 논의해 봅시다.

JIT 컴파일 확장하기
^^^^^^^^^^^^^^^^^^

이전에, C++ 확장을 구축하는 두 가지 방법이 있다고 언급했습니다: 
:mod:`setuptools` 또는 JIT(Just in time) 사용. 전자를 다루었으니 후자에 대해 자세히 알아 봅시다. 
JIT 컴파일 메커니즘은 PyTorch의 API에서 :func:`torch.utils.cpp_extension.load`라는 간단한 
함수를 호출하여 확장을 즉시 컴파일하고 로드하는 방법을 제공합니다. LLTM의 경우 이는 다음과 같이 
간단해 보입니다.::

  from torch.utils.cpp_extension import load

  lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])

여기서는 :mod:`setuptools`'와 동일한 정보를 기능에 제공합니다. 
백그라운드에서 다음 작업을 수행합니다.::

1. 임시 디렉토리 생성 ``/tmp/torch_extensions/lltm``,
2. 해당 임시 디렉터리에 `Ninja <https://ninja-build.org/>`_ 빌드 파일을 내 보냅니다.
3. 소스 파일을 공유 라이브러리로 컴파일하고,
4. 이 공유 라이브러리를 Python 모듈로 가져옵니다.

실제로 ``verbose=True``를 :func:`cpp_extension.load`로 전달하면 프로세스에 대한 
정보가 표시됩니다.::

  Using /tmp/torch_extensions as PyTorch extensions root...
  Emitting ninja build file /tmp/torch_extensions/lltm_cpp/build.ninja...
  Building extension module lltm_cpp...
  Loading extension module lltm_cpp...

결과적으로 생성된 파이썬 모듈은 setuptool에 의해 생성된 것과 정확히 동일하지만 별도의 ``setup.py`` 
빌드 파일을 유지해야 하는 요구 사항을 제거합니다. 설정이 더 복잡하고 :mod:`setuptools`의 전체 
검정력이 필요한 경우에는 자신만의 ``setup.py``를 작성할 수 있지만, 대부분의 경우 이 JIT 기법으로 
충분합니다. 이 행을 처음 실행할 때는 확장자가 백그라운드에서 컴파일되기 때문에 시간이 좀 
걸립니다. Ninja 빌드 시스템을 사용하여 소스를 빌드하기 때문에, 재컴파일링은 증분되므로 파이썬 
모듈을 두 번째로 실행할 때 확장을 다시 로드하는 것이 빠르고 확장의 소스 파일을 변경하지 않았다면 
오버헤드가 낮습니다.

혼합된 C++/CUDA extension 작성하기
----------------------------------

구현을 한 단계 더 발전시키기 위해 custom CUDA 커널을 사용하여 forward 및 backward pass의 
일부를 손으로 작성할 수 있습니다. LLTM의 경우 단일 CUDA 커널에서 모두 융합되고 병렬화될 수 
있는 많은 수의 pointwise 연산이 순서대로 있기 때문에 이것은 특히 효과적일 전망이 있습니다. 이러한 
CUDA 커널을 작성하고 이 확장 메커니즘을 사용하여 PyTorch와 통합하는 방법을 살펴보겠습니다.

CUDA 확장자를 작성하기 위한 일반적인 전략은 먼저 파이썬에서 호출될 함수들을 정의하는 C++ 
파일을 작성하고 이러한 함수들을 파이썬에 pybind11로 바인딩하는 것입니다. 또한 이 파일은 
CUDA(.cu) 파일에 정의된 함수를 선언합니다. 그런 다음 C++ 함수는 몇 가지 검사를 수행하고 
궁극적으로 CUDA 함수로 호출을 전달합니다. CUDA 파일에는 실제 CUDA 커널을 작성합니다. 그런 다음 
:mod:`cpp_extension` 패키지는 gcc와 같은 C++ 컴파일러를 사용하여 C++ 소스를 컴파일하고 NVIDIA의 
nvcc 컴파일러를 사용하여 CUDA 소스를 컴파일합니다. 이것은 각 컴파일러가 컴파일하는 데 가장 
적합한 파일을 처리하도록 보장합니다. 궁극적으로, 이것들은 파이썬 코드에서 우리가 사용할 수 있는 
하나의 공유 라이브러리로 연결될 것입니다.

우리는 C++ 파일로 시작할 것이며, 이 파일을 ``lltm_cuda.cpp``라고 합니다. 예를 들어 다음과 같습니다:

.. code-block:: cpp

  #include <torch/extension.h>

  #include <vector>

  // CUDA forward declarations

  std::vector<torch::Tensor> lltm_cuda_forward(
      torch::Tensor input,
      torch::Tensor weights,
      torch::Tensor bias,
      torch::Tensor old_h,
      torch::Tensor old_cell);

  std::vector<torch::Tensor> lltm_cuda_backward(
      torch::Tensor grad_h,
      torch::Tensor grad_cell,
      torch::Tensor new_cell,
      torch::Tensor input_gate,
      torch::Tensor output_gate,
      torch::Tensor candidate_cell,
      torch::Tensor X,
      torch::Tensor gate_weights,
      torch::Tensor weights);

  // C++ interface

  #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
  #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
  #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

  std::vector<torch::Tensor> lltm_forward(
      torch::Tensor input,
      torch::Tensor weights,
      torch::Tensor bias,
      torch::Tensor old_h,
      torch::Tensor old_cell) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(old_h);
    CHECK_INPUT(old_cell);

    return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
  }

  std::vector<torch::Tensor> lltm_backward(
      torch::Tensor grad_h,
      torch::Tensor grad_cell,
      torch::Tensor new_cell,
      torch::Tensor input_gate,
      torch::Tensor output_gate,
      torch::Tensor candidate_cell,
      torch::Tensor X,
      torch::Tensor gate_weights,
      torch::Tensor weights) {
    CHECK_INPUT(grad_h);
    CHECK_INPUT(grad_cell);
    CHECK_INPUT(input_gate);
    CHECK_INPUT(output_gate);
    CHECK_INPUT(candidate_cell);
    CHECK_INPUT(X);
    CHECK_INPUT(gate_weights);
    CHECK_INPUT(weights);

    return lltm_cuda_backward(
        grad_h,
        grad_cell,
        new_cell,
        input_gate,
        output_gate,
        candidate_cell,
        X,
        gate_weights,
        weights);
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lltm_forward, "LLTM forward (CUDA)");
    m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
  }

보시는 바와 같이, 이것은 주로 CUDA 파일에서 정의할 기능에 대한 상용판, 검사 및 전달입니다. 
이 파일의 이름은 ``lltm_cuda_kernel.cu``입니다(``.cu`` 확장자 참고!). NVCC는 C++11을 합리적으로 
컴파일할 수 있으므로 ATen과 C++ 표준 라이브러리를 여전히 사용할 수 있습니다(``torch.h``는 제외). 
:mod:`setuptools`는 이름은 같지만 확장자가 다른 파일을 처리할 수 없으므로 JIT 방법 대신 ``setup.py`` 
방법을 사용하는 경우 CUDA 파일에 C++ 파일과 다른 이름을 지정해야 합니다(JIT 방법의 경우, 
``lltm.cpp`` 및 ``lltm.cu``이 제대로 작동합니다). 이 파일의 모양을 간단히 살펴보겠습니다.

.. code-block:: cpp

  #include <torch/extension.h>

  #include <cuda.h>
  #include <cuda_runtime.h>

  #include <vector>

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
    return 1.0 / (1.0 + exp(-z));
  }

여기서는 방금 설명한 헤더와 ``__device__`` 및 ``__forceinline__``과 같은 CUDA 고유 선언과 
``exp``와 같은 함수를 사용한다는 사실을 볼 수 있습니다. 계속해서 필요한 몇 가지 helper function을 
살펴보겠습니다:

.. code-block:: cpp

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
    const auto s = sigmoid(z);
    return (1.0 - s) * s;
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
    const auto t = tanh(z);
    return 1 - (t * t);
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
    return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
    const auto e = exp(z);
    const auto d_relu = z < 0.0 ? 0.0 : 1.0;
    return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
  }

이제 함수를 실제로 구현하려면 다시 두 가지가 필요합니다. 하나는 수동으로 명시적으로 작성하고 
싶지 않은 작업을 수행하고 CUDA 커널을 호출하는 함수이고 다른 하나는 속도를 높이고 싶은 부분에 
대한 실제 CUDA 커널입니다. forward pass의 경우 첫 번째 함수는 다음과 같아야 합니다:

.. code-block:: cpp

  std::vector<torch::Tensor> lltm_cuda_forward(
      torch::Tensor input,
      torch::Tensor weights,
      torch::Tensor bias,
      torch::Tensor old_h,
      torch::Tensor old_cell) {
    auto X = torch::cat({old_h, input}, /*dim=*/1);
    auto gates = torch::addmm(bias, X, weights.transpose(0, 1));

    const auto batch_size = old_cell.size(0);
    const auto state_size = old_cell.size(1);

    auto new_h = torch::zeros_like(old_cell);
    auto new_cell = torch::zeros_like(old_cell);
    auto input_gate = torch::zeros_like(old_cell);
    auto output_gate = torch::zeros_like(old_cell);
    auto candidate_cell = torch::zeros_like(old_cell);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
      lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
          gates.data<scalar_t>(),
          old_cell.data<scalar_t>(),
          new_h.data<scalar_t>(),
          new_cell.data<scalar_t>(),
          input_gate.data<scalar_t>(),
          output_gate.data<scalar_t>(),
          candidate_cell.data<scalar_t>(),
          state_size);
    }));

    return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
  }

여기서 주요 관심 사항은 ``AT_DISPATCH_FLOATING_TYPES``매크로 및 커널 실행(``<<<...>>>``로 표시됩니다.)
입니다. ATEN이 우리가 다루는 텐서의 장치와 데이터 유형을 추상화하는 동안, 텐서는 런타임 시 
구체적인 장치에 있는 구체적인 유형의 메모리로 여전히 뒷받침됩니다. 따라서 런타임에 텐서의 유형을 
결정한 다음 해당하는 올바른 유형 서명으로 함수를 선택적으로 호출하는 방법이 필요합니다. 수동으로 
수행하면 (개념적으로) 다음과 같이 표시됩니다:

.. code-block:: cpp

  switch (tensor.type().scalarType()) {
    case torch::ScalarType::Double:
      return function<double>(tensor.data<double>());
    case torch::ScalarType::Float:
      return function<float>(tensor.data<float>());
    ...
  }

``AT_DISPATCH_FLOATING_TYPES``의 목적은 이 디스패치를 처리하는 것입니다. 여기서는 
유형(우리의 경우 ``gates.type()``), 이름(오류 메시지의 경우) 및 람다 함수를 사용합니다. 이 
람다 함수 안에서 ``scalar_t``라는 형식 별칭을 사용할 수 있으며, 텐서가 실제로 해당 컨텍스트에서 
런타임에 있는 형식으로 정의됩니다. 이와 같이, 만약 우리가 템플릿 함수(CUDA 커널이 될 것)를 가지고 
있다면, 우리는 이 ``scalar_t`` 별칭으로 그것을 인스턴스화할 수 있고, 올바른 함수가 호출될 
것입니다. 이 경우 텐서의 데이터 포인터를 ``scalar_t`` 유형의 포인터로 검색하고자 합니다. 부동 
소수점 유형(``Float`` 및 ``Double``)뿐만 아니라 모든 유형에 대해 디스패치하려면 
``AT_DISPATCH_ALL_TYPES``를 사용할 수 있습니다.

우리는 일반 Aten으로 일부 작업을 수행합니다. 이러한 작업은 여전히 ​​GPU에서 실행되지만 ATEN의 
기본 구현을 사용합니다. 이것은 ATen이 행렬 곱셈(예: ``addmm``) 또는 우리가 구현하고 개선하기 훨씬 
더 어려운 컨볼루션과 같은 것에 대해 고도로 최적화된 루틴을 사용하기 때문에 의미가 있습니다.

커널 출시 자체에 대해서는 각 CUDA 블록에 1024개의 스레드가 있고, 전체 GPU 그리드가 구성 
요소당 하나의 스레드로 매트릭스를 채우는 데 필요한 만큼의 ``1 x 1024`` 스레드 블록으로 분할되도록 
명시하고 있습니다. 예를 들어 상태 크기가 2048이고 배치 크기가 4인 경우 1024개의 스레드를 사용하여 
총 ``4 x 2 = 8`` 블록을 실행합니다. 만약 여러분이 CUDA "블록"이나 "그리드"에 대해 들어본 적이 
없다면, CUDA에 대한 `introductory read
about CUDA <https://devblogs.nvidia.com/even-easier-introduction-cuda>`_가 도움이 될 것입니다.

실제 CUDA 커널은 매우 간단합니다. (이전에 GPU를 프로그래밍한 적이 있는 경우):

.. code-block:: cpp

  template <typename scalar_t>
  __global__ void lltm_cuda_forward_kernel(
      const scalar_t* __restrict__ gates,
      const scalar_t* __restrict__ old_cell,
      scalar_t* __restrict__ new_h,
      scalar_t* __restrict__ new_cell,
      scalar_t* __restrict__ input_gate,
      scalar_t* __restrict__ output_gate,
      scalar_t* __restrict__ candidate_cell,
      size_t state_size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * state_size + column;
    const int gates_row = blockIdx.y * (state_size * 3);
    if (column < state_size) {
      input_gate[index] = sigmoid(gates[gates_row + column]);
      output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
      candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
      new_cell[index] =
          old_cell[index] + candidate_cell[index] * input_gate[index];
      new_h[index] = tanh(new_cell[index]) * output_gate[index];
    }
  }

여기서 가장 흥미로운 점은 게이트 행렬의 각 개별 구성 요소에 대해 이러한 모든 pointwise 작업을 
완전히 병렬로 계산할 수 있다는 것입니다. 백만 개의 요소를 직렬로 연결 하는 거대한 ``for``루프로 
이 작업을 수행해야 한다고 상상하면 이것이 훨씬 더 빠른 이유를 알 수 있습니다.

accessors 사용하기
^^^^^^^^^^^^^^^^^

CUDA 커널에서 올바른 유형의 포인터에서 직접 작업하는 것을 볼 수 있습니다. 실제로, cuda 커널 
내부에서 높은 수준의 유형에 구애받지 않는 텐서로 직접 작업하는 것은 매우 비효율적입니다.

그러나 이것은 특히 고차원 데이터의 경우 사용 편의성과 가독성을 희생합니다. 
예를 들어 연속 ``gates``텐서가 3차원임을 알고 있습니다 :

1. batch, size of ``batch_size`` and stride of ``3*state_size``
2. row, size of ``3`` and stride of ``state_size``
3. index, size  of ``state_size`` and stride of ``1``

그렇다면 커널 내부의 ``gates[n][row][column]`` element에 어떻게 접근할 수 있을까? 
이것에 대해서 간단한 연산으로 element에 접근하기 위해서는 stride가 필요하다는 것이 밝혀졌다.

.. code-block:: cpp

  gates.data<scalar_t>()[n*3*state_size + row*state_size + column]

이 식은 장황할 뿐만 아니라 stride를 명시적으로 알려야 하므로 argument 내에서 커널 함수에 전달해야 
합니다. 크기가 다른 여러 텐서를 받아들이는 커널 함수의 경우 argument 목록이 매우 길어지는 것을 볼 수 
있습니다.

다행스럽게도 ATEN은 Tensor가 차원의 수와 type 이라는 단일 동적 검사로 생성된 접근자를 제공합니다. 
그런 다음 접근자는 단일 포인터로 변환할 필요 없이 Tensor 요소에 효율적으로 액세스하기 위한 
API를 노출합니다:

.. code-block:: cpp

  torch::Tensor foo = torch::rand({12, 12});

  // assert foo is 2-dimensional and holds floats.
  auto foo_a = foo.accessor<float,2>();
  float trace = 0;

  for(int i = 0; i < foo_a.size(0); i++) {
    // use the accessor foo_a to get tensor data.
    trace += foo_a[i][i];
  }

접근자 객체는 ``.size()`` 및 ``.stride()`` 메서드와 다차원 인덱싱을 사용하는 비교적 높은 
수준의 인터페이스를 가지고 있습니다. ``.accessor<>`` 인터페이스는 CPU 텐서에서 데이터에 
효율적으로 접근할 수 있도록 설계되었습니다. cuda 텐서에 해당하는 것은 ``packed_accessor64<>``와 
``packed_accessor32<>``로 64비트 또는 32비트 정수 인덱싱을 통해 패킹된 액세서를 생성합니다.

Accessor와의 근본적인 차이점은 Packed Accessor가 구조를 가리키는 대신 구조 내부에 크기 및 보폭 
데이터를 복사한다는 것입니다. 이를 통해 CUDA 커널 함수에 전달하고 내부 인터페이스를 사용할 수 
있습니다.

우리는 포인터 대신 압축된 접근자를 사용하는 함수를 설계할 수 있습니다.

.. code-block:: cpp

  __global__ void lltm_cuda_forward_kernel(
      const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell)

여기에 사용된 템플릿을 분해해 봅시다. 첫 번째 두 인수 ``scalar_t``와 ``2``는 일반적인 접근자와 
동일합니다. 인수 ``torch::RestrictPtrTraits``는 ``__restrict__`` 키워드를 사용해야 함을 
나타냅니다. 또한 크기와 보폭을 ``int32_t``로 저장하는 ``PackedAccessor32`` 변형을 사용했습니다. 
64비트 변형(``PackedAccessor64``)을 사용하면 커널 속도가 느려질 수 있기 때문에 이것은 중요하다.

함수 선언은 다음과 같습니다.

.. code-block:: cpp

  template <typename scalar_t>
  __global__ void lltm_cuda_forward_kernel(
      const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < gates.size(2)){
      input_gate[n][c] = sigmoid(gates[n][0][c]);
      output_gate[n][c] = sigmoid(gates[n][1][c]);
      candidate_cell[n][c] = elu(gates[n][2][c]);
      new_cell[n][c] =
          old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
      new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
    }
  }

구현이 훨씬 더 읽기 쉽습니다! 그런 다음 호스트 함수 내에서 ``.packed_accessor32<>``
메서드를 사용하여 패킹된 액세스 프로그램을 생성하여 이 함수를 호출합니다.

.. code-block:: cpp

  std::vector<torch::Tensor> lltm_cuda_forward(
      torch::Tensor input,
      torch::Tensor weights,
      torch::Tensor bias,
      torch::Tensor old_h,
      torch::Tensor old_cell) {
    auto X = torch::cat({old_h, input}, /*dim=*/1);
    auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

    const auto batch_size = old_cell.size(0);
    const auto state_size = old_cell.size(1);

    auto gates = gate_weights.reshape({batch_size, 3, state_size});
    auto new_h = torch::zeros_like(old_cell);
    auto new_cell = torch::zeros_like(old_cell);
    auto input_gate = torch::zeros_like(old_cell);
    auto output_gate = torch::zeros_like(old_cell);
    auto candidate_cell = torch::zeros_like(old_cell);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
      lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
          gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));

    return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
  }

backwards pass는 거의 동일한 패턴을 따르며 더 이상 자세히 설명하지 않겠습니다 :

.. code-block:: cpp

  template <typename scalar_t>
  __global__ void lltm_cuda_backward_kernel(
      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
      torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_gates,
      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell,
      const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gate_weights) {
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < d_gates.size(2)){
      const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
      const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
      const auto d_new_cell =
          d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


      d_old_cell[n][c] = d_new_cell;
      const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
      const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

      d_gates[n][0][c] =
          d_input_gate * d_sigmoid(gate_weights[n][0][c]);
      d_gates[n][1][c] =
          d_output_gate * d_sigmoid(gate_weights[n][1][c]);
      d_gates[n][2][c] =
          d_candidate_cell * d_elu(gate_weights[n][2][c]);
    }
  }

  std::vector<torch::Tensor> lltm_cuda_backward(
      torch::Tensor grad_h,
      torch::Tensor grad_cell,
      torch::Tensor new_cell,
      torch::Tensor input_gate,
      torch::Tensor output_gate,
      torch::Tensor candidate_cell,
      torch::Tensor X,
      torch::Tensor gates,
      torch::Tensor weights) {
    auto d_old_cell = torch::zeros_like(new_cell);
    auto d_gates = torch::zeros_like(gates);

    const auto batch_size = new_cell.size(0);
    const auto state_size = new_cell.size(1);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_cuda", ([&] {
      lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
          d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          d_gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));

    auto d_gate_weights = d_gates.reshape({batch_size, 3*state_size});
    auto d_weights = d_gate_weights.t().mm(X);
    auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

    auto d_X = d_gate_weights.mm(weights);
    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
  }


C++/CUDA Operation 와 PyTorch 통합하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA-enabled op와 PyTorch의 통합은 매우 간단합니다. ``setup.py`` 스크립트 를 작성하려는 경우 
다음과 같이 나타 낼 수 있습니다 ::

  from setuptools import setup
  from torch.utils.cpp_extension import BuildExtension, CUDAExtension

  setup(
      name='lltm',
      ext_modules=[
          CUDAExtension('lltm_cuda', [
              'lltm_cuda.cpp',
              'lltm_cuda_kernel.cu',
          ])
      ],
      cmdclass={
          'build_ext': BuildExtension
      })

이제  :func:`CppExtension` 대신 :func:`CUDAExtension`을 사용합니다. 
``.cpp`` 파일과 함께 ``.cu`` 파일을 지정하면 됩니다. 라이브러리에서 모든 번거로움을 처리합니다. 
JIT 메커니즘은 훨씬 더 간단합니다 ::

  from torch.utils.cpp_extension import load

  lltm = load(name='lltm', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])

성능 비교
**********************

우리의 목표는 코드의 pointwise 연산을 CUDA와 병렬화하고 융합하여 LLTM의 성능을 향상시키는 것이었습니다. 
그것이 사실인지 확인해 봅시다. 벤치마크를 실행하기 위해 앞서 나열한 코드를 실행할 수 있습니다. 
가장 빠른 버전은 CUDA 기반의 C++ 코드였습니다 ::

  Forward: 149.802 us | Backward 393.458 us


이제 custom CUDA 커널을 사용하여::

  Forward: 129.431 us | Backward 304.641 us

더 많은 성능이 향상됩니다!

결론
-----
이제 PyTorch의 C++ 확장 메커니즘에 대한 좋은 개요와 이를 사용하려는 동기를 갖추셨을 것입니다.
이 노트에 표시된 코드 예제는 `여기
<https://github.com/pytorch/extension-cpp>`_ 에서 찾을 수 있습니다.
질문이 있으면 저희 `포럼 
<https://discuss.pytorch.org/>`_ 을 이용해 주세요. 또한 문제가 발생할 경우 `FAQ
<https://pytorch.org/cppdocs/notes/faq.html>`_ 를 이용해 주세요.
