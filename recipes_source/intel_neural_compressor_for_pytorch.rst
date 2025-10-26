PyTorch에서 Intel® Neural Compressor를 활용한 손쉬운 양자화(Quantization)
==================================================================

**번역**: `정휘수 <https://github.com/Brdy8294>`_

개요(Overview)
--------------

대부분의 딥러닝 애플리케이션은 추론(inference)을 위해 32비트 부동소수점(floating-point) 정밀도를 사용합니다.
하지만 FP8과 같은 저정밀(low-precision) 데이터 타입은 성능 향상이 크기 때문에 점점 더 많은 주목을 받고 있습니다.
저정밀 방식을 채택할 때의 핵심 과제는 정확도를 최대한 유지하면서도 사전 정의된 요구 사항을 충족하는 것입니다.

Intel® Neural Compressor는 PyTorch에 정확도 기반(accuracy-driven) 자동 튜닝(auto-tuning) 기법을 확장하여 이 문제를 해결하고, 사용자가 Intel 하드웨어에서 가장 적합한 양자화된 모델을 쉽게 찾을 수 있도록 돕습니다.

Intel® Neural Compressor는 오픈소스 프로젝트이며, `Github <https://github.com/intel/neural-compressor>`_ 에서 확인할 수 있습니다.

특징(Features)
---------------

- 사용이 간편한 API(Ease-of-use API): PyTorch의 ``prepare`` 및 ``convert`` API를 재사용해 쉽게 적용할 수 있습니다.
- 정확도 기반 튜닝(Accuracy-driven Tuning): 정확도 기반 자동 튜닝 프로세스를 지원하며 ``autotune`` API를 제공합니다.
- 다양한 양자화 방식(Kinds of Quantization): 고전적인 INT8 양자화, 가중치-전용(weight-only) 양자화, FP8 양자화를 지원합니다.
  또한 시뮬레이션 기반의 최신 연구로, MX 데이터 타입 에뮬레이션(emulation) 양자화도 포함됩니다.
  자세한 내용은 `Supported Matrix <https://github.com/intel/neural-compressor/blob/master/docs/source/3x/PyTorch.md#supported-matrix>`_ 를 참조하세요.

시작하기(Getting Started)
---------------------------

설치(Installation)
~~~~~~~~~~~~~~~~~~

.. code:: bash

    # pip을 이용한 안정(stable) 버전 설치
    pip install neural-compressor-pt
..

참고: Neural Compressor는 HPU, Intel GPU, CUDA, CPU 등 가속기를 자동 감지합니다.
특정 디바이스를 지정하려면 환경변수 ``INC_TARGET_DEVICE`` 를 사용하세요(예: ``export INC_TARGET_DEVICE=cpu``).

예제(Examples)
~~~~~~~~~~~~~~

이 섹션에서는 Intel® Neural Compressor로 여러 종류의 양자화를 수행하는 예제를 보여줍니다.

FP8 양자화(FP8 Quantization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FP8 양자화는 Intel® Gaudi® 2 및 3 AI Accelerator(HPU)에서 지원됩니다.
환경 설정 방법은 `Intel® Gaudi® Documentation <https://docs.habana.ai/en/latest/index.html>`_ 를 참조하세요.

.. code-block:: python

    # FP8 Quantization Example
    from neural_compressor.torch.quantization import (
        FP8Config,
        prepare,
        convert,
    )

    import torch
    import torchvision.models as models

    # 사전 학습된 ResNet18 모델 로드
    model = models.resnet18()

    # FP8 양자화 설정
    qconfig = FP8Config(fp8_config="E4M3")
    model = prepare(model, qconfig)

    # 보정(calibration) 수행 (실제 보정 데이터로 교체)
    calibration_data = torch.randn(1, 3, 224, 224).to("hpu")
    model(calibration_data)

    # FP8 모델로 변환
    model = convert(model)

    # 추론 수행
    input_data = torch.randn(1, 3, 224, 224).to("hpu")
    output = model(input_data).to("cpu")
    print(output)

..

가중치-전용 양자화(Weight-only Quantization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

가중치-전용 양자화 역시 Intel® Gaudi® 2 및 3 AI Accelerator에서 지원됩니다.
양자화된 모델은 다음과 같이 로드할 수 있습니다.

.. code-block:: python

    from neural_compressor.torch.quantization import load

    # 모델 이름은 HuggingFace Model Hub에서 가져옵니다.
    model_name = "TheBloke/Llama-2-7B-GPTQ"
    model = load(
        model_name_or_path=model_name,
        format="huggingface",
        device="hpu",
        torch_dtype=torch.bfloat16,
    )
..

참고: Intel Neural Compressor는 처음 로드할 때 auto-gptq 형식을 HPU 형식으로 변환하고,
다음 로드를 위해 로컬 캐시에 ``hpu_model.safetensors`` 파일을 저장합니다.
따라서 첫 로드에는 시간이 다소 걸릴 수 있습니다.

PT2E 백엔드 기반 정적 양자화(Static Quantization with PT2E Backend)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PT2E 경로는 ``torch.dynamo`` 로 Eager 모델을 FX 그래프 모델로 캡처하고,
그 위에 관찰자(observers)와 Q/DQ 쌍을 삽입합니다.
마지막으로 ``torch.compile`` 로 패턴 매칭을 수행하여 Q/DQ 쌍을 최적화된 양자화 연산자로 대체합니다.

W8A8 정적 양자화 절차는 ``export`` → ``prepare`` → ``convert`` → ``compile`` 의 네 단계입니다.

.. code-block:: python

   import torch
   from neural_compressor.torch.export import export
   from neural_compressor.torch.quantization import StaticQuantConfig, prepare, convert

   # float 모델과 예시 입력 준비
   model = UserFloatModel()
   example_inputs = ...

   # Eager 모델을 FX 그래프 모델로 내보내기
   exported_model = export(model=model, example_inputs=example_inputs)

   # 모델 양자화
   quant_config = StaticQuantConfig()
   prepared_model = prepare(exported_model, quant_config=quant_config)

   # 보정(calibration)
   run_fn(prepared_model)

   q_model = convert(prepared_model)

   # Q/DQ 패턴을 Q-operator로 교체하며 컴파일
   from torch._inductor import config

   config.freezing = True
   opt_model = torch.compile(q_model)
..

정확도 기반 자동 튜닝(Accuracy-driven Tuning)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

정확도 기반 자동 튜닝을 활용하려면 튜닝 공간(tuning space)을 명시해야 합니다.
``autotune`` 은 튜닝 공간을 순회하며 지정된 고정밀(high-precision) 모델에 설정을 적용하고,
기준선(baseline)과 비교해 평가 결과를 기록합니다.
튜닝은 종료 정책(exit policy)에 도달하면 중단됩니다.

.. code-block:: python

   from neural_compressor.torch.quantization import RTNConfig, TuningConfig, autotune


   def eval_fn(model) -> float:
       return ...


   tune_config = TuningConfig(
       config_set=RTNConfig(use_sym=[False, True], group_size=[32, 128]),
       tolerable_loss=0.2,
       max_trials=10,
   )
   q_model = autotune(model, tune_config=tune_config, eval_fn=eval_fn)
..

튜토리얼(Tutorials)
-------------------

자세한 튜토리얼은 Intel® Neural Compressor 공식 문서 `사이트 <https://intel.github.io/neural-compressor/latest/docs/source/Welcome.html>`_ 에서 확인할 수 있습니다.
