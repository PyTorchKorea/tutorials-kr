C++ 프론트엔드의 자동 미분 (autograd)
==================================

**번역**: `유용환 <https://github.com/yoosful>`_

``autograd`` 는 PyTorch로 유연하고 역동적인 신경망을 구축하기 위해
필수적인 패키지입니다. PyTorch 파이썬 프론트엔드의 자동 미분 API 대부분은 C++ 프론트엔드에서도
사용할 수 있으며, 파이썬에서 C++로 자동 미분 코드를 쉽게 변환할 수 있습니다.

이 튜토리얼에서는 PyTorch C++ 프론트엔드에서 자동 미분을 수행하는 몇 가지 예를 살펴보겠습니다.
이 튜토리얼은 여러분이 파이썬 프론트엔드의 자동 미분에 대해 기본적으로 이해하고 있다고
가정합니다. 그렇지 않은 경우 먼저 `Autograd: Automatic Differentiation
<https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html>`_ 을 읽어보세요.

기초 자동 미분 연산
---------------

(`이 튜토리얼 <https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html#autograd-automatic-differentiation>`_ 의 내용에 기반함)

텐서를 생성하고 그것의 계산을 추적하기 위해 ``torch::requires_grad()`` 를 실행해봅시다.

.. code-block:: cpp

  auto x = torch::ones({2, 2}, torch::requires_grad());
  std::cout << x << std::endl;

Out:

.. code-block:: shell

  1 1
  1 1
  [ CPUFloatType{2,2} ]


텐서 연산을 수행해보겠습니다.

.. code-block:: cpp

  auto y = x + 2;
  std::cout << y << std::endl;

Out:

.. code-block:: shell

   3  3
   3  3
  [ CPUFloatType{2,2} ]

``y`` 는 연산의 결과로 생성되었으므로 ``grad_fn`` 를 갖고 있습니다.

.. code-block:: cpp

  std::cout << y.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

  AddBackward1

``y`` 에 대해 더 많은 연산을 수행해봅시다.

.. code-block:: cpp

  auto z = y * y * 3;
  auto out = z.mean();

  std::cout << z << std::endl;
  std::cout << z.grad_fn()->name() << std::endl;
  std::cout << out << std::endl;
  std::cout << out.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

   27  27
   27  27
  [ CPUFloatType{2,2} ]
  MulBackward1
  27
  [ CPUFloatType{} ]
  MeanBackward0


``.requires_grad_( ... )`` 는 in-place로 텐서의 기존 ``requires_grad`` 플래그를 바꿉니다.

.. code-block:: cpp

  auto a = torch::randn({2, 2});
  a = ((a * 3) / (a - 1));
  std::cout << a.requires_grad() << std::endl;

  a.requires_grad_(true);
  std::cout << a.requires_grad() << std::endl;

  auto b = (a * a).sum();
  std::cout << b.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

  false
  true
  SumBackward0

이제 역전파를 수행해봅시다. ``out`` 이 단일 스칼라만을 포함하므로, ``out.backward()`` 는
``out.backward(torch::tensor(1.))`` 와 같습니다.


.. code-block:: cpp

  out.backward();

변화도 d(out)/dx를 출력해보겠습니다.

.. code-block:: cpp

  std::cout << x.grad() << std::endl;

Out:

.. code-block:: shell

   4.5000  4.5000
   4.5000  4.5000
  [ CPUFloatType{2,2} ]

``4.5`` 행렬이 출력돼야 합니다. 이 값을 얻는 과정에 대한 설명은 `이 튜토리얼의 해당 섹션
<https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html#gradients>`_ 에서 확인하세요.

이제 벡터-야코비안 곱의 예를 살펴보겠습니다.

.. code-block:: cpp

  x = torch::randn(3, torch::requires_grad());

  y = x * 2;
  while (y.norm().item<double>() < 1000) {
    y = y * 2;
  }

  std::cout << y << std::endl;
  std::cout << y.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

  -1021.4020
    314.6695
   -613.4944
  [ CPUFloatType{3} ]
  MulBackward1

벡터-야코비안 곱을 얻기 위해 벡터를 ``backward`` 의 인자로 넣어줍니다.

.. code-block:: cpp

  auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
  y.backward(v);

  std::cout << x.grad() << std::endl;

Out:

.. code-block:: shell

    102.4000
   1024.0000
      0.1024
  [ CPUFloatType{3} ]

또한 코드에 ``torch::NoGradGuard`` 를 넣어주면 자동 미분으로 하여금 그래디언트가
필요한 텐서를 추적하지 않도록 할 수 있습니다.

.. code-block:: cpp

  std::cout << x.requires_grad() << std::endl;
  std::cout << x.pow(2).requires_grad() << std::endl;

  {
    torch::NoGradGuard no_grad;
    std::cout << x.pow(2).requires_grad() << std::endl;
  }


Out:

.. code-block:: shell

  true
  true
  false

혹은 ``.detach()`` 를 사용하여 내용은 동일하지만 그래디언트가 필요 없는
새 텐서를 얻을 수도 있습니다.

.. code-block:: cpp

  std::cout << x.requires_grad() << std::endl;
  y = x.detach();
  std::cout << y.requires_grad() << std::endl;
  std::cout << x.eq(y).all().item<bool>() << std::endl;

Out:

.. code-block:: shell

  true
  false
  true

``grad`` / ``requires_grad`` / ``is_leaf`` / ``backward`` / ``detach`` / ``detach_`` /
``register_hook`` / ``retain_grad`` 등 C++ 텐서 자동 미분 API에 대한 자세한 내용은 `해당 C++ API 문서
<https://pytorch.org/cppdocs/api/classat_1_1_tensor.html>`_ 에서 확인하세요.

C++로 고차원 그래디언트 계산하기
----------------------------

고차원 그래디언트를 사용하는 사례로 그래디언트 패널티 계산이 있습니다.
``torch::autograd::grad`` 를 사용하는 예를 살펴봅시다.

.. code-block:: cpp

  #include <torch/torch.h>

  auto model = torch::nn::Linear(4, 3);

  auto input = torch::randn({3, 4}).requires_grad_(true);
  auto output = model(input);

  // Calculate loss
  auto target = torch::randn({3, 3});
  auto loss = torch::nn::MSELoss()(output, target);

  // Use norm of gradients as penalty
  auto grad_output = torch::ones_like(output);
  auto gradient = torch::autograd::grad({output}, {input}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true)[0];
  auto gradient_penalty = torch::pow((gradient.norm(2, /*dim=*/1) - 1), 2).mean();

  // Add gradient penalty to loss
  auto combined_loss = loss + gradient_penalty;
  combined_loss.backward();

  std::cout << input.grad() << std::endl;

Out:

.. code-block:: shell

  -0.1042 -0.0638  0.0103  0.0723
  -0.2543 -0.1222  0.0071  0.0814
  -0.1683 -0.1052  0.0355  0.1024
  [ CPUFloatType{3,4} ]

``torch::autograd::backward``
(`링크 <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1afa9b5d4329085df4b6b3d4b4be48914b.html>`_) 및
``torch::autograd::grad``
(`링크 <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1e03c42b14b40c306f9eb947ef842d9c.html>`_) 문서에서
이 함수들의 사용법에 대해 더 알아보세요.

C++에서 사용자 지정 자동 미분 함수 사용하기
-------------------------------------

(`이 튜토리얼 <https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd>`_ 의 내용에 기반함)

``torch::autograd`` 에 새로운 기본(elementary) 연산을 추가하려면 각 연산에 대해 새로운 ``torch::autograd::Function``
하위 클래스(subclass)를 구현해야 합니다. ``torch::autograd`` 는 결과와 그래디언트를 계산하고 연산 기록을 인코딩하기 위해 위해
이 ``torch::autograd::Function`` 들을 사용합니다. 모든 새로운 함수에는 두 가지 방법, 즉 ``forward`` 와 ``backward`` 를
구현해야 하며 자세한 요구사항은 `이 링크 <https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html>`__
에서 확인하세요.

아래 코드는 ``torch::nn`` 의 ``Linear`` 함수를 사용합니다.

.. code-block:: cpp

  #include <torch/torch.h>

  using namespace torch::autograd;

  // Inherit from Function
  class LinearFunction : public Function<LinearFunction> {
   public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static torch::Tensor forward(
        AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
      ctx->save_for_backward({input, weight, bias});
      auto output = input.mm(weight.t());
      if (bias.defined()) {
        output += bias.unsqueeze(0).expand_as(output);
      }
      return output;
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
      auto saved = ctx->get_saved_variables();
      auto input = saved[0];
      auto weight = saved[1];
      auto bias = saved[2];

      auto grad_output = grad_outputs[0];
      auto grad_input = grad_output.mm(weight);
      auto grad_weight = grad_output.t().mm(input);
      auto grad_bias = torch::Tensor();
      if (bias.defined()) {
        grad_bias = grad_output.sum(0);
      }

      return {grad_input, grad_weight, grad_bias};
    }
  };

이제 아래와 같이 ``LinearFunction`` 을 사용할 수 있습니다.

.. code-block:: cpp

  auto x = torch::randn({2, 3}).requires_grad_();
  auto weight = torch::randn({4, 3}).requires_grad_();
  auto y = LinearFunction::apply(x, weight);
  y.sum().backward();

  std::cout << x.grad() << std::endl;
  std::cout << weight.grad() << std::endl;

Out:

.. code-block:: shell

   0.5314  1.2807  1.4864
   0.5314  1.2807  1.4864
  [ CPUFloatType{2,3} ]
   3.7608  0.9101  0.0073
   3.7608  0.9101  0.0073
   3.7608  0.9101  0.0073
   3.7608  0.9101  0.0073
  [ CPUFloatType{4,3} ]

여기서, 텐서가 아닌 인자를 매개변수로 갖는 또 다른 함수를 예로 들어 보겠습니다.

.. code-block:: cpp

  #include <torch/torch.h>

  using namespace torch::autograd;

  class MulConstant : public Function<MulConstant> {
   public:
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor tensor, double constant) {
      // ctx is a context object that can be used to stash information
      // for backward computation
      ctx->saved_data["constant"] = constant;
      return tensor * constant;
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
      // We return as many input gradients as there were arguments.
      // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
      return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
    }
  };

이제 아래와 같이 ``MulConstant`` 를 사용할 수 있습니다.

.. code-block:: cpp

  auto x = torch::randn({2}).requires_grad_();
  auto y = MulConstant::apply(x, 5.5);
  y.sum().backward();

  std::cout << x.grad() << std::endl;

Out:

.. code-block:: shell

   5.5000
   5.5000
  [ CPUFloatType{2} ]

``torch::autograd::Function`` 에 대한 더 많은 내용은
`이 문서 <https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html>`_ 에서 확인할 수 있습니다.

파이썬 자동 미분 코드를 C++로 변환하기
------------------------------

개략적으로 말하면, C++에서 자동 미분을 사용하는 가장 쉬운 방법은 먼저
파이썬에서 동작하는 자동 미분 코드를 작성한 후, 아래 표를 참고해 C++ 코드로
변환하는 것입니다.

+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Python                         | C++                                                                                                                                                                    |
+================================+========================================================================================================================================================================+
| ``torch.autograd.backward``    | ``torch::autograd::backward`` (`링크 <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1afa9b5d4329085df4b6b3d4b4be48914b.html>`_)                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.autograd.grad``        | ``torch::autograd::grad`` (`링크 <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1e03c42b14b40c306f9eb947ef842d9c.html>`_)                      |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.detach``        | ``torch::Tensor::detach`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor6detachEv>`_)                                              |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.detach_``       | ``torch::Tensor::detach_`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7detach_Ev>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.backward``      | ``torch::Tensor::backward`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8backwardERK6Tensorbb>`_)                                |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.register_hook`` | ``torch::Tensor::register_hook`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4I0ENK2at6Tensor13register_hookE18hook_return_void_tI1TERR1T>`_) |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.requires_grad`` | ``torch::Tensor::requires_grad_`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor14requires_grad_Eb>`_)                             |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.retain_grad``   | ``torch::Tensor::retain_grad`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor11retain_gradEv>`_)                                   |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.grad``          | ``torch::Tensor::grad`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4gradEv>`_)                                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.grad_fn``       | ``torch::Tensor::grad_fn`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7grad_fnEv>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.set_data``      | ``torch::Tensor::set_data`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8set_dataERK6Tensor>`_)                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.data``          | ``torch::Tensor::data`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4dataEv>`_)                                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.output_nr``     | ``torch::Tensor::output_nr`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor9output_nrEv>`_)                                        |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.is_leaf``       | ``torch::Tensor::is_leaf`` (`링크 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7is_leafEv>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

대부분의 변환된 파이썬 자동 미분 코드가 C++에서도 잘 동작할 것입니다.
동작하지 않을 경우, `GitHub issues <https://github.com/pytorch/pytorch/issues>`_ 에 버그 리포트를 제출해 주시면
최대한 빨리 고쳐드리겠습니다.

결론
-----

이제 PyTorch의 C++ 자동 미분 API에 대한 개괄적인 이해가 생겼을 것입니다.
여기서 사용된 코드 예제들은 `여기 <https://github.com/pytorch/examples/tree/master/cpp/autograd>`_ 에서
확인할 수 있습니다. 언제나 그렇듯이 어떤 문제가 생기거나 질문이 있으면 저희
`포럼 <https://discuss.pytorch.org/>`_ 을 이용하거나 `Github 이슈
<https://github.com/pytorch/pytorch/issues>`_ 로 연락주세요.
