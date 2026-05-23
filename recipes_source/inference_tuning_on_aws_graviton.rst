(Beta) AWS Graviton 프로세서에서의 PyTorch 추론 성능 튜닝
======================================================================

**저자**: `Sunita Nadampalli <https://github.com/snadampal>`_

`AWS Graviton <https://aws.amazon.com/ec2/graviton/>`_ 은 AWS에서 설계된 ARM 기반 프로세서 시리즈입니다. AWS Graviton3 프로세서는 머신러닝(ML) 워크로드에 최적화 되어있으며 ``bfloat16``, Scalable Vector Extension (SVE) 그리고 Graviton2 대비 두 배의 Single Instruction Multiple Data(SIMD) 대역폭을 지원합니다.

PyTorch는 합성곱(convolution), matmul, relu 등 머신러닝 연산자를 위한 기본 참조 ATen 커널을 제공합니다. 이러한 연산자는 기초 선형대수학(BLAS, Basic Linear Algebra) 라이브러리에서 제공하는 플랫폼별 커널 구현을 통해 가속할 수 있습니다. AWS Graviton CPU에서는 MKLDNN과 Arm Compute 라이브러리(`ACL <https://github.com/ARM-software/ComputeLibrary>`_) 그리고 `OpenBLAS <https://github.com/OpenMathLib/OpenBLAS>`_ 라이브러리가 일부 연산자에 대해 최적화된 구현을 제공합니다. 두 라이브러리는 모두 PyTorch 2.0 버전부터 PyTorch에 통합되었습니다.

이 튜토리얼에서는 ``bfloat16`` 커널과 적절한 백엔드 선택을 통해 AWS Graviton3 CPU (`AWS c7g 인스턴스 <https://aws.amazon.com/ec2/instance-types/c7g/>`_)에서 선형 계층 신경망의 최적 추론 성능을 얻는 방법을 다룹니다.

목차
--------
1. 기본 사용법
2. Bfloat16 고속 연산 커널을 사용한 추론 속도 향상
3. 작은 배치 차원에서 OpenBLAS를 사용한 추론 성능 개선
4. Linux Transparent Huge Pages를 사용한 메모리 할당 오버헤드 최적화
5. 결론

.. note::
   이 튜토리얼을 성공적으로 실행하고 아래에 제시된 속도 향상 수치를 재현하려면 Graviton3 계열 (``c7g/r7g/m7g``) 하드웨어 인스턴스가 필요합니다. 이 튜토리얼에서 `c7g.xl (4vcpu) 인스턴스 <https://aws.amazon.com/ec2/instance-types/c7g/>`_\ 를 사용했습니다.

기본 사용법
---------------

PyTorch는 PyTorch 2.0 버전부터 AWS Graviton3 최적화를 기본적으로 지원합니다.
최적화에 대한 자세한 내용은 이 `블로그 <https://pytorch.org/blog/optimized-pytorch-w-graviton/>`_ 를 참고하세요.

1. 다음 명령어를 실행하여 PyTorch를 설치합니다.

   .. code-block::

      python3 -m pip install torch

2. 먼저 필요한 라이브러리를 import하고 실행할 장치를 정의합니다.
    
.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.profiler import profile, record_function, ProfilerActivity

    # AWS Graviton3 cpu
    device = ("cpu")
    print(f"Using {device} device")


3. 선형 계층은 트랜스포머를 포함한 여러 신경망의 핵심 요소이므로 이 데모에서는 선형 계층을 사용합니다. ``nn.Module`` 을 서브클래싱하고 ``__init__`` 에서 계층을 초기화하여 신경망을 정의합니다. 실제 환경에 가까운 시나리오를 만들기 위해 일반적인 대형 언어 모델 매개변수를 사용하여 네트워크를 구성합니다.

.. code-block:: python

  class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 11008),
            nn.ReLU(),
            nn.Linear(11008, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

4. ``MyNeuralNetwork`` 의 인스턴스를 생성하고 장치로 이동합니다.

.. code-block:: python

    model = MyNeuralNetwork().to(device)
    print(model)

다음으로 ``nn.Softmax`` 모듈의 인스턴스를 통과시켜 예측 확률을 얻습니다.

.. code-block:: python

    X = torch.rand(1, 64, 64, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

출력:

.. code-block::

    Predicted class: tensor([2])

네트워크의 기능이 검증되었습니다. 다음으로 성능을 프로파일링합니다. 작은 배치 차원과 큰 배치 차원, 두 가지 시나리오를 확인해 보겠습니다.

**시나리오 1:** 예를 들어 256과 같은 큰 배치 차원

.. code-block:: python

    # 충분한 실행 시간을 확보하기 위해 먼저 워밍업하고 여러 번 반복합니다.

    X = torch.rand(256, 64, 64, device=device)

    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


다음은 기본 PyTorch 설정에서의 프로파일러 출력입니다.

.. table::
   :widths: auto

   ======================  ============   ===========  =============  ===========  ============  ============
                  Name      Self CPU %      Self CPU    CPU total %    CPU total   CPU time avg    # of Calls
   ======================  ============   ===========  =============  ===========  ============  ============
           aten::addmm        97.61%         15.813s        98.61%       15.977s      53.255ms           300
       aten::clamp_min         1.09%       177.032ms         1.09%     177.032ms     885.160us           200
            aten::copy         1.00%       162.054ms         1.00%     162.054ms     540.180us           300
     mymodel_inference         0.22%        35.738ms       100.00%       16.201s       16.201s             1
          aten::linear         0.02%         2.955ms        98.66%       15.985s      53.282ms           300
               aten::t         0.01%         2.421ms         0.03%       5.043ms      16.810us           300
            aten::relu         0.01%         2.356ms         1.11%     179.388ms     896.940us           200
   ======================  ============   ===========  =============  ===========  ============  ============

**Self CPU time total:** 16.201s


``bfloat16`` 고속 연산 커널을 사용한 추론 속도 향상
----------------------------------------------------------

AWS Graviton3 프로세서는 `bfloat16 MMLA 명령어 <https://developer.arm.com/documentation/ddi0596/2020-12/SVE-Instructions/BFMMLA--BFloat16-floating-point-matrix-multiply-accumulate->`_ 를 지원합니다. Arm Compute 라이브러리 (`ACL <https://github.com/ARM-software/ComputeLibrary>`_)는 AWS Graviton 프로세서를 위한 최적화된 ``bfloat16`` General Matrix Multiplication(GEMM) 커널을 제공하며, PyTorch 2.0부터 MKLDNN 백엔드를 통해 PyTorch에 통합되었습니다. 고속 연산 GEMM 커널을 사용하면 추론 성능을 최적화할 수 있습니다. 고속 연산 모드는 기본적으로 활성화되어 있지 않습니다. 이 커널들은 GEMM을 ``float`` 대신 ``bfloat16`` 정밀도로 수행하므로 모델 추론 정확도가 약간 낮아질 수 있기 때문입니다. 하지만 정확도 하락은 ``torchbench`` 테스트 스위트에서 ``bfloat16`` 백엔드에 대해 정의한 ``코사인 유사도`` 임계값 안에 있으므로, 대부분의 애플리케이션에서 허용 가능한 수준입니다. 고속 연산 GEMM 커널을 활성화하려면 다음 환경 변수를 설정합니다.

.. code-block:: bash

    $ export DNNL_DEFAULT_FPMATH_MODE=BF16


위의 추론 스크립트를 실행하면 MKLDNN 고속 연산 모드가 활성화된 상태에서 다음과 같은 프로파일러 출력을 볼 수 있습니다.

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ============  ============
                  Name      Self CPU %     Self CPU    CPU total %     CPU total   CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  ============  ============
           aten::addmm        95.61%        6.943s        97.10%        7.052s      23.507ms           300
       aten::clamp_min         2.31%     167.653ms         2.31%     167.653ms     838.265us           200
            aten::copy         1.48%     107.593ms         1.48%     107.593ms     358.643us           300
     mymodel_inference         0.43%      31.167ms       100.00%        7.262s        7.262s             1
          aten::linear         0.04%       2.911ms        97.21%        7.060s      23.533ms           300
               aten::t         0.03%       2.414ms         0.07%       4.892ms      16.307us           300
            aten::relu         0.03%       2.281ms         2.34%     169.934ms     849.670us           200
   ======================  ============  ============  ============  ============  ============  ============

**Self CPU time total:** 7.262s


이는 bfloat16 고속 연산 커널을 사용했을 때 약 ``2배 (7.262s vs 16.201s)`` 의 성능 향상입니다. 다음으로 더 작은 배치 차원의 시나리오를 살펴보겠습니다.

**시나리오 2:** 예를 들어 32와 같은 작은 배치 차원

.. code-block:: python

    X = torch.rand(32, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


위 스크립트를 PyTorch 기본 설정으로 실행하면 다음과 같은 프로파일러 출력을 볼 수 있습니다.

.. table::
   :widths: auto

   ======================  =============  ============  ============  ============  ============  ============
                     Name    Self CPU %      Self CPU   CPU total %     CPU total   CPU time avg    # of Calls
   ======================  =============  ============  ============  ============  ============  ============
           aten::addmm        95.51%         5.821s        97.04%        5.914s      19.713ms           300
       aten::clamp_min         2.33%      142.244ms         2.33%     142.244ms     711.220us           200
            aten::copy         1.51%       92.322ms         1.51%      92.322ms     307.740us           300
     mymodel_inference         0.45%       27.713ms       100.00%        6.094s        6.094s             1
          aten::linear         0.04%        2.495ms        97.16%        5.921s      19.736ms           300
               aten::t         0.03%        2.131ms         0.07%       4.441ms      14.803us           300
            aten::relu         0.03%        1.942ms         2.37%     144.186ms     720.930us           200
   ======================  =============  ============  ============  ============  ============  ============

**Self CPU time total:** 6.094s


다음 출력은 MKLDNN 고속 연산 모드를 활성화하여 실행했을 때의 프로파일러 출력입니다.

.. code-block:: bash

   $ export DNNL_DEFAULT_FPMATH_MODE=BF16

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ============   =============
                   Name     Self CPU %      Self CPU    CPU total %   CPU total    CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  ============   =============
           aten::addmm        93.31%        3.848s        95.66%        3.944s      13.148ms           300
       aten::clamp_min         3.43%     141.309ms         3.43%     141.309ms     706.545us           200
            aten::copy         2.33%      95.916ms         2.33%      95.916ms     319.720us           300
     mymodel_inference         0.67%      27.431ms       100.00%        4.123s        4.123s             1
          aten::linear         0.06%       2.471ms        95.83%        3.951s      13.170ms           300
               aten::t         0.05%       2.027ms         0.10%       4.243ms      14.143us           300
            aten::relu         0.05%       1.928ms         3.47%     143.237ms     716.185us           200
   ======================  ============  ============  ============  ============  ============   =============

**Self CPU time total:** 4.123s

MKLDNN 고속 연산 모드는 작은 배치 차원에서 약 **1.47x  (4.123s vs 6.094s)** 의 성능 향상을 제공합니다. 이 개선은 의미가 있지만, 전체 성능에는 여전히 개선의 여지가 있습니다. 작은 배치 연산에서는 oneDNN과 ACL 백엔드에서 발생하는 런타임 오버헤드 (가중치 재정렬과 커널 실행 시간)가 ACL GEMM 커널의 연산 이점보다 더 크기 때문입니다.


작은 배치 차원에서 OpenBLAS를 사용한 추론 성능 개선
------------------------------------------------------------------------

작은 배치 차원에서의 추론 성능은 작은 shape를 MKLDNN에서 OpenBLAS 백엔드로 넘김으로써 개선할 수 있습니다. 향후 릴리스에서는 안정적인 휴리스틱을 사용해 백엔드 선택을 자동화하는 작업을 진행하고 있습니다. 이러한 휴리스틱이 구현되기 전까지는 MKLDNN 백엔드 선택 임계값을 높여 작은 shape을 OpenBLAS로 넘길 수 있습니다. 다음 예제에서는 ``64`` 를 임계값으로 사용하여 ``배치 차원이 32`` 인 입력이 MKLDNN으로 디스패치되지 않도록 합니다. 대신 OpenBLAS로 디스패치됩니다.

.. code-block:: bash

   $ export TORCH_MKLDNN_MATMUL_MIN_DIM=64

다음은 OpenBLAS 백엔드를 사용했을 때의 프로파일러 출력입니다.

.. table::
   :widths: auto

   ======================  ============  ============  ============  =============  ============  =============
                     Name    Self CPU %      Self CPU   CPU total %     CPU total   CPU time avg    # of Calls
   ======================  ============  ============  ============  =============  ============  =============
           aten::addmm        96.25%        1.958s        97.51%        1.984s        6.612ms           300
       aten::clamp_min         1.28%      26.124ms         1.28%      26.124ms      130.620us           200
            aten::copy         1.23%      24.951ms         1.23%      24.951ms       83.170us           300
     mymodel_inference         0.86%      17.423ms       100.00%        2.034s         2.034s             1
          aten::linear         0.08%       1.691ms        97.74%        1.988s        6.628ms           300
               aten::t         0.07%       1.520ms         0.14%       2.945ms        9.817us           300
            aten::relu         0.06%       1.258ms         1.35%      27.382ms      136.910us           200
   ======================  ============  ============  ============  =============  ============  =============

**Self CPU time total:** 2.034s


위에서 볼 수 있듯이 OpenBLAS로 전환하면 기본 MKLDNN 백엔드 설정과 비교해 성능이 두 배 향상되었습니다 **(2.034s vs 4.123s)**. 이는 배치 차원이 더 작을수록 더욱 중요해집니다. 예를 들어 배치 차원이 10인 경우를 살펴보겠습니다.


.. code-block:: python

    X = torch.rand(10, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


다음은 고속 연산 모드를 사용했을 때의 프로파일러 출력입니다.

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  =============  =============
                     Name    Self CPU %      Self CPU   CPU total %     CPU total   CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  =============  =============
           aten::addmm        87.81%        3.613s        91.90%        3.781s      12.604ms           300
       aten::clamp_min         7.18%     295.437ms         7.18%     295.437ms       1.477ms           200
            aten::copy         4.07%     167.516ms         4.07%     167.516ms     558.387us           300
     mymodel_inference         0.67%      27.708ms       100.00%        4.115s        4.115s             1
          aten::linear         0.06%       2.499ms        92.06%        3.788s      12.627ms           300
               aten::t         0.05%       1.982ms         0.11%       4.385ms      14.617us           300
            aten::relu         0.05%       1.932ms         7.23%     297.369ms       1.487ms           200
   ======================  ============  ============  ============  ============  =============  =============

**Self CPU time total:** 4.115s


그리고 다음은 OpenBLAS 백엔드를 사용했을 때의 출력입니다.

.. code-block:: bash

   $ export TORCH_MKLDNN_MATMUL_MIN_DIM=64

.. table::
   :widths: auto

   ======================  =============  ============  ============  ============  =============  ============
                   Name     Self CPU %      Self CPU     CPU total %   CPU total    CPU time avg    # of Calls
   ======================  =============  ============  ============  ============  =============  ============
           aten::addmm        92.66%        1.179s        95.23%        1.211s         4.038ms           300
       aten::clamp_min         2.83%      36.060ms         2.83%      36.060ms       180.300us           200
            aten::copy         2.52%      32.013ms         2.52%      32.013ms       106.710us           300
     mymodel_inference         1.38%      17.521ms       100.00%        1.272s          1.272s             1
          aten::linear         0.14%       1.750ms        95.60%        1.216s         4.054ms           300
               aten::t         0.12%       1.475ms         0.24%       3.033ms        10.110us           300
            aten::relu         0.10%       1.285ms         2.94%      37.345ms       186.725us           200
   ======================  =============  ============  ============  ============  =============  ============

**Self CPU time total:** 1.272s


여기서는 백엔드 임계값을 적절히 조정함으로써 **3.2배(1.272s vs 4.115s)** 의 성능 향상을 확인했습니다.


Linux Transparent Huge Pages (THP)를 사용한 메모리 할당 오버헤드 최적화
---------------------------------------------------------------------------

또한 이러한 큰 네트워크에서는 Tensor 메모리 할당이 추론 지연 시간의 상당 부분을 차지한다는 점을 확인했습니다. 이는 PyTorch C10 메모리 할당자에서 Linux transparent huge page 할당을 활성화하여 최적화할 수 있습니다. 현재 이 특징은 메모리 전체 사용량을 약간 증가시키기 때문에 기본적으로 활성화되어 있지 않습니다. 활성화하려면 다음 환경 변수를 설정합니다.

.. code-block:: bash

    $ export THP_MEM_ALLOC_ENABLE=1

배치 차원이 256이고 MKLDNN 고속 연산 모드를 사용하는 경우:

.. code-block:: python

    X = torch.rand(256, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


다음은 THP 메모리 할당을 활성화했을 때의 프로파일러 출력입니다.

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ==============  ============
                     Name   Self CPU %    Self CPU     CPU total %    CPU total     CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  ==============  ============
           aten::addmm        91.31%        6.115s        94.39%        6.321s      21.069ms           300
       aten::clamp_min         4.82%     322.568ms         4.82%     322.568ms       1.613ms           200
            aten::copy         3.06%     204.602ms         3.06%     204.602ms     682.007us           300
     mymodel_inference         0.61%      40.777ms       100.00%        6.697s        6.697s             1
          aten::linear         0.05%       3.082ms        94.51%        6.329s      21.097ms           300
            aten::relu         0.04%       2.547ms         4.85%     325.115ms       1.626ms           200
   ======================  ============  ============  ============  ============  ==============  ============

**Self CPU time total:** 6.697s

이는 앞서 측정한 이미 최적화된 MKLDNN 고속 연산 모드에 더해 추가로 **1.08배 또는 8%(6.697s vs 7.262s)** 의 성능 향상입니다.


결론
------------

이 튜토리얼에서는 AWS Graviton3 인스턴스에서의 PyTorch 추론을 다루었습니다. 기본 사용법을 살펴보고 고속 연산 커널을 사용한 속도 향상을 보여주었으며 배치 차원에 따른 서로 다른 백엔드를 비교하고, Linux transparent huge pages를 사용하여 Tensor 메모리 할당 지연 시간을 최적화하는 방법을 설명했습니다. 권장 사항은 큰 Tensor shape에는 Bfloat16 고속 연산 모드와 THP 메모리 할당을 함께 사용한 MKLDNN 백엔드를 사용하고, 작은 Tensor shape에는 OpenBLAS 백엔드를 사용하는 것입니다. 직접 시도해 보길 바랍니다!
