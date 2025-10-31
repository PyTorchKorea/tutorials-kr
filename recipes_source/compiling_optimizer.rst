(beta) torch.compile로 옵티마이저 컴파일하기
==========================================================================================

**저자:** `Michael Lazos <https://github.com/mlazos>`_
**번역:** `김승환 <https://github.com/7SH7>`_

옵티마이저는 딥러닝 모델을 훈련하는 핵심 알고리즘입니다.
모든 모델 파라미터를 업데이트하는 역할을 하기 때문에, 대규모 모델에서는 종종 훈련 성능의 병목이 될 수 있습니다.
이 레시피에서는 옵티마이저에 ``torch.compile``을 적용하여 GPU 성능 향상을 관찰해보겠습니다.

.. note::

    이 튜토리얼은 PyTorch 2.2.0 이상이 필요합니다.

모델 설정
~~~~~~~~~~~~~~~~~~~~~
이 예제에서는 간단한 선형 계층의 시퀀스를 사용할 것입니다.
우리는 옵티마이저의 성능만 벤치마킹할 것이기 때문에, 모델의 선택은 중요하지 않습니다.
옵티마이저의 성능은 파라미터의 수에 따라 달라지기 때문입니다.

사용하는 머신에 따라 정확한 결과는 다를 수 있습니다.

.. code-block:: python

   import torch
   
   model = torch.nn.Sequential(
       *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
   )
   input = torch.rand(1024, device="cuda")
   output = model(input)
   output.sum().backward()

옵티마이저 벤치마크 설정 및 실행
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
이 예제에서는 Adam 옵티마이저를 사용하고, ``torch.compile()``에서 step()을 감싸는 도우미 함수를 생성합니다.

.. note::
   
   ``torch.compile`` is only supported on cuda devices with compute capability >= 7.0

.. code-block:: python

   # torch.compile이 지원되지 않는 디바이스에서는 깔끔하게 종료합니다.
   if torch.cuda.get_device_capability() < (7, 0):
       print("Exiting because torch.compile is not supported on this device.")
       import sys
       sys.exit(0)


   opt = torch.optim.Adam(model.parameters(), lr=0.01)


   @torch.compile(fullgraph=False)
   def fn():
       opt.step()
   
   
   # 유용한 벤치마킹 함수를 정의해봅시다.
   import torch.utils.benchmark as benchmark
   
   
   def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
       t0 = benchmark.Timer(
           stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
       )
       return t0.blocked_autorange().mean * 1e6


   # 함수를 컴파일하기 위한 웜업 실행
   for _ in range(5):
       fn()
   
   eager_runtime = benchmark_torch_function_in_microseconds(opt.step)
   compiled_runtime = benchmark_torch_function_in_microseconds(fn)
   
   assert eager_runtime > compiled_runtime
   
   print(f"eager runtime: {eager_runtime}us")
   print(f"compiled runtime: {compiled_runtime}us")

샘플 결과:

* Eager runtime: 747.2437149845064us
* Compiled runtime: 392.07384741178us

See Also
~~~~~~~~~

* 심층적인 기술 개요를 위해서, `PT2로 옵티마이저 컴파일하기 <https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669>`__ 를 참조하세요.