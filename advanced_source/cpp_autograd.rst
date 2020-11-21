C ++ 프론트 엔드의 AUTOGRAD
==================================

이 ``autograd`` 패키지는 pytorch에서 매우 유연하고 동적인 신경망을 구축하는 데 중요합니다.
pytorch 파이썬(python) 프론트 엔드의 대부분의 autograd API는 C ++ 프론트 엔드에서도
사용할 수 있으므로  autograd 코드를 파이썬(python)에서 C ++로 쉽게 변환 할 수 있습니다.

이 튜토리얼에서는 pytorch C++ 프론트엔드에서 autograd를 수행하는 몇 가지 예를 살펴보겠습니다.
이 튜토리얼에서는 파이썬(python) 프론트 엔드의 autograd에 대한 기본적인 이해가 이미 있다고 가정합니다.
그렇지 않은 경우 먼저
`Autograd: Automatic Differentiation <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_ 을 읽어보십시오.

기본 autograd 작업
-------------------------

(`이 튜토리얼 <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#autograd-automatic-differentiation>`_ 에서 수정됨)
tensor를 생성하고 ``torch::requires_grad()`` 계산을 추적하도록 설정

.. code-block:: cpp

  auto x = torch::ones({2, 2}, torch::requires_grad());
  std::cout << x << std::endl;

Out:

.. code-block:: shell

  1 1
  1 1
  [ CPUFloatType{2,2} ]


tensor operation 수행:

.. code-block:: cpp

  auto y = x + 2;
  std::cout << y << std::endl;

Out:

.. code-block:: shell

   3  3
   3  3
  [ CPUFloatType{2,2} ]

``y`` 가 operation의 결과로 생성되었으므로, 이것은 ``grad_fn`` 을 갖습니다.

.. code-block:: cpp

  std::cout << y.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

  AddBackward1

``y`` 에서 더 많은 operation 수행

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


``.requires_grad_( ... )`` 은 기존 tensor의 ``requires_grad`` flag를 제자리에 변경합니다.

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

지금 backprop 합시다. ``out`` 에는 하나의 scalar가 포함되어 있기 때문에, ``out.backward()``
는 ``out.backward(torch::tensor(1.))`` 와 같습니다.

.. code-block:: cpp

  out.backward();

증감률 d(out)/dx 출력

.. code-block:: cpp

  std::cout << x.grad() << std::endl;

Out:

.. code-block:: shell

   4.5000  4.5000
   4.5000  4.5000
  [ CPUFloatType{2,2} ]

``4.5`` 의 매트릭스를 얻었어야 했습니다. 이 값에 도달하는 방법에 대한 설명은,
`이 튜토리얼의 해당 섹션을 참조하십시오<https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients>`_.

이제 vector -Jacobian product의 예를 살펴 보겠습니다:

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

vector-Jacobian product를 원하면 벡터 ``backward`` 인수로 전달하십시오:

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

또한 ``torch::NoGradGuard`` 를 코드 블록에 삽입 하여 gradients가 필요한 tensor의 추적 기록에서 autograd를 중지 할 수도 있습니다.

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

또는 ``.detach()`` 를 사용하여 동일한 콘텐츠를 포함하지만 gradients가 필요하지 않은 새 tensor를 가져옵니다:

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

``grad`` / ``requires_grad`` / ``is_leaf`` / ``backward`` / ``detach`` / ``detach_`` / ``register_hook`` / ``retain_grad``
와 같은 C++ tensor autograd API에 대한 자세한 내용은 `해당 C++ API 문서 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html>`_ 를 참조하십시오.

C++에서 고차 gradient 연산
---------------------------------------

고차 gradient의 응용 프로그램 중 하나는 gradient 패널티를 계산하는 것입니다.
 ``torch::autograd::grad`` 를 사용하여 예제를 보겠습니다:

.. code-block:: cpp

  #include <torch/torch.h>
  
  auto model = torch::nn::Linear(4, 3);
  
  auto input = torch::randn({3, 4}).requires_grad_(true);
  auto output = model(input);
  
  // 손실 계산
  auto target = torch::randn({3, 3});
  auto loss = torch::nn::MSELoss()(output, target);
  
  // gradient의 규범을 penalty로 사용
  auto grad_output = torch::ones_like(output);
  auto gradient = torch::autograd::grad({output}, {input}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true)[0];
  auto gradient_penalty = torch::pow((gradient.norm(2, /*dim=*/1) - 1), 2).mean();
  
  // 손실에 gradient penalty 추가
  auto combined_loss = loss + gradient_penalty;
  combined_loss.backward();
  
  std::cout << input.grad() << std::endl;

Out:

.. code-block:: shell

  -0.1042 -0.0638  0.0103  0.0723
  -0.2543 -0.1222  0.0071  0.0814
  -0.1683 -0.1052  0.0355  0.1024
  [ CPUFloatType{3,4} ]

사용 방법에 대한 자세한 내용은 ``torch::autograd::backward``
(`link <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1afa9b5d4329085df4b6b3d4b4be48914b.html>`_)
및 ``torch::autograd::grad``
(`link <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1e03c42b14b40c306f9eb947ef842d9c.html>`_)
설명서를 참조하십시오.

C ++에서 사용자 지정 autograd 함수 사용
-------------------------------------

(`이 튜토리얼 <https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd>`_ 에서 수정됨)

``torch::autograd`` 에 새로운 기본적인 운영을 추가하려면 각 운영마다 새로운 ``torch::autograd::Function`` 하위 클래스를 구현해야 합니다.
``torch::autograd::Function`` 은 ``torch::autograd`` 가 결과와 gradient를 계산하고,
operation history를 인코딩하는데 사용됩니다. 모든 새 기능을 사용하려면 ``forward`` 와 ``backward``, 두 가지 메소드를 구현해야합니다.
자세한 사항은 `이 링크 <https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html>`_ 를 참조하십시오.

아래는 ``torch::nn`` 에서 ``Linear`` 함수에 대한 코드를 찾을 수 있습니다:

.. code-block:: cpp

  #include <torch/torch.h>
  
  using namespace torch::autograd;
  
  // 함수에서 상속
  class LinearFunction : public Function<LinearFunction> {
   public:
    // 전방과 후방 모두 정적 함수라는 점을 유의하십시오
  
    // bias는 선택적인 주장이다
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

그런 다음 ``LinearFunction`` 을 다음과 같이 사용할 수 있습니다:

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

여기에서는 tensor가 아닌 non-tensor argument로 매개 변수화된 함수의 추가 예제를 제공합니다:

.. code-block:: cpp

  #include <torch/torch.h>
  
  using namespace torch::autograd;
  
  class MulConstant : public Function<MulConstant> {
   public:
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor tensor, double constant) {
      // ctx는 정보를 넣어 두는데 사용할 수 있는 context 개체이다
      // for backward computation
      ctx->saved_data["constant"] = constant;
      return tensor * constant;
    }
  
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
      // 논쟁이 있었던 만큼 많은 입력 gradient를 돌려준다.
      // 전달할 non-tensor argument의 gradient는 `torch::Tensor()` 여야 한다.
      return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
    }
  };

그 다음에 ``MulConstant`` 를 다음과 같이 사용할 수 있습니다:

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

``torch::autograd::Function`` 에 대한 자세한 내용은
`해당 설명서 <https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html>`_ 를 참조하십시오.

Python에서 C++로 autograd 코드 변역
--------------------------------------------

높은 수준에서 C ++에서 autograd를 사용하는 가장 쉬운 방법은 먼저 Python에서 작동하는 autograd 코드를 만든 다음
다음 표를 사용하여 Python에서 C ++로 autograd 코드를 변환하는 것입니다:


+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Python                         | C++                                                                                                                                                                    |
+================================+========================================================================================================================================================================+
| ``torch.autograd.backward``    | ``torch::autograd::backward`` (`link <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1afa9b5d4329085df4b6b3d4b4be48914b.html>`_)                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.autograd.grad``        | ``torch::autograd::grad`` (`link <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1e03c42b14b40c306f9eb947ef842d9c.html>`_)                      |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.detach``        | ``torch::Tensor::detach`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor6detachEv>`_)                                              |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.detach_``       | ``torch::Tensor::detach_`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7detach_Ev>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.backward``      | ``torch::Tensor::backward`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8backwardERK6Tensorbb>`_)                                |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.register_hook`` | ``torch::Tensor::register_hook`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4I0ENK2at6Tensor13register_hookE18hook_return_void_tI1TERR1T>`_) |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.requires_grad`` | ``torch::Tensor::requires_grad_`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor14requires_grad_Eb>`_)                             |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.retain_grad``   | ``torch::Tensor::retain_grad`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor11retain_gradEv>`_)                                   |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.grad``          | ``torch::Tensor::grad`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4gradEv>`_)                                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.grad_fn``       | ``torch::Tensor::grad_fn`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7grad_fnEv>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.set_data``      | ``torch::Tensor::set_data`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8set_dataERK6Tensor>`_)                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.data``          | ``torch::Tensor::data`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4dataEv>`_)                                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.output_nr``     | ``torch::Tensor::output_nr`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor9output_nrEv>`_)                                        |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.is_leaf``       | ``torch::Tensor::is_leaf`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7is_leafEv>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
번역후에는 대부분의 Python autograd 코드가 C++에서만 작동합니다.
그렇지 않은 경우`GitHub issue에 <https://github.com/pytorch/pytorch/issues>`_ 버그 보고서를 제출해주시면 최대한 빨리 수정하겠습니다.


결론
----------

이제 PyTorch의 C ++ autograd API에 대한 좋은 개요가 있어야합니다.
 `여기 <https://github.com/pytorch/examples/tree/master/cpp/autograd>`_ 에서 이 노트에 표시된 코드 예제를 찾을 수 있습니다.
항상 그렇듯이, 문제가 발생하거나 질문이 있는 경우
 `포럼 <https://discuss.pytorch.org/>`_
또는 `GitHub issues <https://github.com/pytorch/pytorch/issues>`_ 에 올려주십시오.
