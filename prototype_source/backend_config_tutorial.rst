(prototype) PyTorch BackendConfig Tutorial
==========================================
**저자**: `Andrew Or <https://github.com/andrewor14>`_
**번역**: `장승호 <https://github.com/jason9865>`_

BackendConfig API를 통해 백엔드 환경에서 PyTorch 양자화를 사용할 수 있습니다.
기존 환경에서는 FX 그래프 모드 양자화 만을 사용할 수 있지만 
추후에는 다른 모드 또한 지원 할 예정입니다.
본 튜토리얼에서는 특정 백엔드 환경에서 양자화 기능을 커스터마이징하기 위해 
BackendConfig API를 사용하는 방법에 대해 다룹니다.
BackendConfig가 만들어진 동기와 구현 방법에 대한 세부정보를 알고 싶으시다면
아래 사이트를 참고하세요.
`README <https://github.com/pytorch/pytorch/tree/master/torch/ao/quantization/backend_config>`__.

여러분이 PyTorch의 양자화 API를 백엔드 환경에서 사용하고 싶어하는 백엔드 개발자라고 가정해봅시다.
백엔드 환경에서 사용할 수 있는 선택지는 양자화된 선형(Linear) 연산자와 합성곱(Convolution) ReLU 연산자가 있습니다.
이번 장에서는 `prepare_fx`와 `convert_fx`를 통해 커스텀 BackendConfig를 만들고,
이를 활용하여 예시 모델을 양자화하여 백엔드 환경을 구축하는 방법에 대해 살펴보겠습니다.

.. code:: ipython3

    import torch
    from torch.ao.quantization import (
        default_weight_observer,
        get_default_qconfig_mapping,
        MinMaxObserver,
        QConfig,
        QConfigMapping,
    )
    from torch.ao.quantization.backend_config import (
        BackendConfig,
        BackendPatternConfig,
        DTypeConfig,
        DTypeWithConstraints,
        ObservationType,
    )
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

1. 양자화된 연산자를 위한 참조 패턴 유도하기
--------------------------------------------------------

양자화된 선형연산자를 위해 백엔드 환경에서는 `[dequant - fp32_linear - quant]` 참조 패턴을
양자화된 단일 선형 연산자로 축소하여 사용한다고 가정합시다.
이를 위해 우선 quant-dequant연산자를 부동소수점 선형 연산자 앞 뒤로 삽입하여
아래와 같은 추론 모델을 만들 수 있습니다.::

  quant1 - [dequant1 - fp32_linear - quant2] - dequant2

이와 유사하게 양자화된 합성곱 ReLU 연산자를 만들기 위해서는
대괄호 안에 있는 참조패턴을 하나의 양자화된 합성곱 ReLU 연산자로 변환하여 사용합니다.::

  quant1 - [dequant1 - fp32_conv_relu - quant2] - dequant2

2. 백엔드 환경 제약조건을 DTypeConfig로 설정하기
---------------------------------------------

앞서 언급한 추론 패턴에서 DTypeConfig에 명시된 입력값의 데이터 타입은 
quant1 변수의 데이터 타입 인자로, 출력값의 데이터 타입은 quant2 변수의 
데이터 타입 인자로 전달됩니다. 동적 양자화(dynamic quantization)의 경우, 
출력값의 데이터 타입이 fp32일 경우 출력값의 quant-dequant 쌍은 삽입되지 않습니다.
아래 예제 코드에서 양자화 시 필요한 제약조건을 나타내고
특정 데이터 타입의 범위를 지정하는 방법을 확인할 수 있습니다.

.. code:: ipython3

    quint8_with_constraints = DTypeWithConstraints(
        dtype=torch.quint8,
        quant_min_lower_bound=0,
        quant_max_upper_bound=255,
        scale_min_lower_bound=2 ** -12,
    )
    
    # Specify the dtypes passed to the quantized ops in the reference model spec
    weighted_int8_dtype_config = DTypeConfig(
        input_dtype=quint8_with_constraints,
        output_dtype=quint8_with_constraints,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float)

3. 합성곱 ReLU 결합(fusion)하기
-------------------------------

초기 사용자 모델에서는 합성곱 연산자와 ReLU 연산자가 분리되어 있습니다.
따라서 먼저 합성곱 연산자와 ReLU 연산자를 결합하여 하나의 합성곱-ReLU연산자를 만든 후
선형 연산자를 양자화한 것과 유사하게 합성곱-ReLU 연산자를 양자화를 진행합니다.
이 때 3개의 인자를 갖는 함수를 정의합니다. 첫번째 인자는 QAT이 적용되는지 여부를 나타내며
나머지 2개의 인자는 결합된 패턴의 개별 요소(여기서는 합성곱 연산자와 ReLU)를 가리킵니다.

.. code:: ipython3

   def fuse_conv2d_relu(is_qat, conv, relu):
       """Return a fused ConvReLU2d from individual conv and relu modules."""
       return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)

4. BackendConfig 정의하기
----------------------------

이제 필요한 것은 모두 준비가 되었으니 BackendConfig를 정의해봅시다.
선형 연산자의 입력값과 출력값에 대해 서로 다른 observer(명칭은 추후 변경 예정)를 사용합니다.
이를 통해 양자화 매개변수가 서로 다른 양자화 연산자(quant1과 quant2)를 거치며
이와 같은 방식은 선형 연산이나 합성곱 연산과 같이 가중치를 사용하는 연산에서 
일반적으로 사용합니다.

합성곱-ReLU 연산자의 경우 observation의 타입은 동일합니다.
하지만 BackendPatternConfig의 경우 결합과 양자화에 사용하기 위해 2개가 필요합니다.
합성곱-ReLU와 선형 연산자에는 앞서 정의한 DTypeConfig를 활용합니다.

.. code:: ipython3

    linear_config = BackendPatternConfig() \
        .set_pattern(torch.nn.Linear) \
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
        .add_dtype_config(weighted_int8_dtype_config) \
        .set_root_module(torch.nn.Linear) \
        .set_qat_module(torch.nn.qat.Linear) \
        .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

    # For fusing Conv2d + ReLU into ConvReLU2d
    # No need to set observation type and dtype config here, since we are not
    # inserting quant-dequant ops in this step yet
    conv_relu_config = BackendPatternConfig() \
        .set_pattern((torch.nn.Conv2d, torch.nn.ReLU)) \
        .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
        .set_fuser_method(fuse_conv2d_relu)
    
    # For quantizing ConvReLU2d
    fused_conv_relu_config = BackendPatternConfig() \
        .set_pattern(torch.ao.nn.intrinsic.ConvReLU2d) \
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
        .add_dtype_config(weighted_int8_dtype_config) \
        .set_root_module(torch.nn.Conv2d) \
        .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d) \
        .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)

    backend_config = BackendConfig("my_backend") \
        .set_backend_pattern_config(linear_config) \
        .set_backend_pattern_config(conv_relu_config) \
        .set_backend_pattern_config(fused_conv_relu_config)

5. 백엔드 환경 제약조건을 만족시키는 QConfigMapping 설정하기
----------------------------------------------------------------

앞서 정의한 연산자를 사용하기 위해서는 DTypeConfig의 제약조건을 만족하는 
QConfig를 정의해야합니다. 자세한 내용은 `DTypeConfig <https://pytorch.org/docs/stable/generated/torch.ao.quantization.backend_config.DTypeConfig.html>`__을 참고하세요.
그리고 양자화하려는 패턴들에 사용되는 모든 모듈에 QConfig를 사용합니다.

.. code:: ipython3

    # 주의 : quant_max 값은 127이지만 추후 255까지 늘어날 수 있습니다.(`quint8_with_constraints`를 참고하세요)
    activation_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127, eps=2 ** -12)
    qconfig = QConfig(activation=activation_observer, weight=default_weight_observer)

    # 주의 : (Conv2d, ReLU) 내부 Conv2d와 ReLU와 같은 결합된 패턴의 모든 개별 요소들은
    # 반드시 같은 QConfig여야합니다.
    qconfig_mapping = QConfigMapping() \
        .set_object_type(torch.nn.Linear, qconfig) \
        .set_object_type(torch.nn.Conv2d, qconfig) \
        .set_object_type(torch.nn.BatchNorm2d, qconfig) \
        .set_object_type(torch.nn.ReLU, qconfig)

6. 사전 처리(prepare)와 변환(convert)을 통한 모델 양자화
--------------------------------------------------

마지막으로 앞서 정의한 BackendConfig를 prepare과 convert를 거쳐 양자화합니다.
이를 통해 양자화된 선형 모듈과 결합된 합성곱-ReLU 모델을 만들 수 있습니다.

.. code:: ipython3

    class MyModel(torch.nn.Module):
        def __init__(self, use_bn: bool):
            super().__init__()
            self.linear = torch.nn.Linear(10, 3)
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.bn = torch.nn.BatchNorm2d(3)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.use_bn = use_bn

        def forward(self, x):
            x = self.linear(x)
            x = self.conv(x)
            if self.use_bn:
                x = self.bn(x)
            x = self.relu(x)
            x = self.sigmoid(x)
            return x

    example_inputs = (torch.rand(1, 3, 10, 10, dtype=torch.float),)
    model = MyModel(use_bn=False)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
    prepared(*example_inputs)  # calibrate
    converted = convert_fx(prepared, backend_config=backend_config)

.. parsed-literal::

    >>> print(converted)

    GraphModule(
      (linear): QuantizedLinear(in_features=10, out_features=3, scale=0.012136868201196194, zero_point=67, qscheme=torch.per_tensor_affine)
      (conv): QuantizedConvReLU2d(3, 3, kernel_size=(3, 3), stride=(1, 1), scale=0.0029353597201406956, zero_point=0)
      (sigmoid): Sigmoid()
    )
    
    def forward(self, x):
        linear_input_scale_0 = self.linear_input_scale_0
        linear_input_zero_point_0 = self.linear_input_zero_point_0
        quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
        linear = self.linear(quantize_per_tensor);  quantize_per_tensor = None
        conv = self.conv(linear);  linear = None
        dequantize_2 = conv.dequantize();  conv = None
        sigmoid = self.sigmoid(dequantize_2);  dequantize_2 = None
        return sigmoid

(7. 오류가 있는 BackendConfig 설정 실험하기)
-------------------------------------------------

실험의 일환으로 합성곱-ReLU 연산자 대신 합성곱-배치정규화-ReLU(conv-bn-relu) 모델을 이용합니다.
이 때 BackendConfig는 이전과 동일한 것을 사용하며 합성곱-배치정규화-ReLU 양자화 관련된 정보는 없습니다.
실험 결과, 선형 모델의 경우 양자화가 성공적으로 진행되었지만 합성곱-배치정규화-ReLU의 경우
결합과 양자화 모두 이루어지지 않았습니다.

.. code:: ipython3
    # 합성곱-배치정규화-ReLU와 관련된 정보가 없기 때문에 선형 모델 만 양자화되었습니다.
    example_inputs = (torch.rand(1, 3, 10, 10, dtype=torch.float),)
    model = MyModel(use_bn=True)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
    prepared(*example_inputs)  # calibrate
    converted = convert_fx(prepared, backend_config=backend_config)

.. parsed-literal::

    >>> print(converted)

    GraphModule(
      (linear): QuantizedLinear(in_features=10, out_features=3, scale=0.015307803638279438, zero_point=95, qscheme=torch.per_tensor_affine)
      (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (sigmoid): Sigmoid()
    )
    
    def forward(self, x):
        linear_input_scale_0 = self.linear_input_scale_0
        linear_input_zero_point_0 = self.linear_input_zero_point_0
        quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
        linear = self.linear(quantize_per_tensor);  quantize_per_tensor = None
        dequantize_1 = linear.dequantize();  linear = None
        conv = self.conv(dequantize_1);  dequantize_1 = None
        bn = self.bn(conv);  conv = None
        relu = self.relu(bn);  bn = None
        sigmoid = self.sigmoid(relu);  relu = None
        return sigmoid

백엔드 환경에 데이터 타입 제약조건을 만족하지 않는 기본 QConfigMapping을 이용하여 또 다른 실험을 진행했습니다.
실혐 결과 QConfig가 무시되어 어떤 모델도 양자화 되지 않았습니다.

.. code:: ipython3
    # Nothing is quantized or fused, since backend constraints are not satisfied
    example_inputs = (torch.rand(1, 3, 10, 10, dtype=torch.float),)
    model = MyModel(use_bn=True)
    prepared = prepare_fx(model, get_default_qconfig_mapping(), example_inputs, backend_config=backend_config)
    prepared(*example_inputs)  # calibrate
    converted = convert_fx(prepared, backend_config=backend_config)

.. parsed-literal::

    >>> print(converted)

    GraphModule(
      (linear): Linear(in_features=10, out_features=3, bias=True)
      (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (sigmoid): Sigmoid()
    )
    
    def forward(self, x):
        linear = self.linear(x);  x = None
        conv = self.conv(linear);  linear = None
        bn = self.bn(conv);  conv = None
        relu = self.relu(bn);  bn = None
        sigmoid = self.sigmoid(relu);  relu = None
        return sigmoid


기본 BackendConfig
-----------------------

PyTorch 양자화는 ``torch.ao.quantization.backend_config`` 네임스페이스 하위
여러 기본 BackendConfig를 지원합니다.

- `get_fbgemm_backend_config <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/fbgemm.py>`__:
  서버 세팅용 BackendConfig
- `get_qnnpack_backend_config <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/qnnpack.py>`__:
  모바일 및 엣지 장비, XNNPack 양자화 연산자 지원 BackendConfig
- `get_native_backend_config <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/native.py>`__
  (기본값): FBGEMM과 QNNPACK BackendConfig 내에서 제공되는 연산자 패턴을
  지원하는 BackendConfig

그 밖에 다른 BackendConfig(TensorRT, x86 등)가 개발 중이지만
아직 실험 단계에 머물러 있습니다. 새로운 커스텀 백엔드 환경에서
PyTorch 양자화 API를 사용하기 원한다면 예제 코드에 정의된 
API 코드를 바탕으로 자체적인 BackendConfig를 정의할 수 있습니다.

참고자료
---------------

FX 그래프 모드 양자화에서 BackendConfig를 사용하는 법:
https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/README.md

BackendConfig가 만들어진 동기와 구현 방법
https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md

BackendConfig의 초기 설계:
https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md
