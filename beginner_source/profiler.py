"""
PyTorch 모듈 프로파일링 하기
---------------------------
**Author:** `Suraj Subramanian <https://github.com/suraj813>`_

**번역:** `이재복 <http://github.com/zzaebok>`_

PyTorch는 코드 내의 다양한 Pytorch 연산에 대한 시간과 메모리 비용을 파악하는 데 유용한 프로파일러(profiler) API를 포함하고 있습니다.
프로파일러는 코드에 쉽게 통합될 수 있으며, 프로파일링 결과는 표로 출력되거나 JSON 형식의 추적(trace) 파일로 반환될 수 있습니다.

.. note::
    프로파일러는 멀티스레드화된 모델들을 지원합니다.
    프로파일러는 연산이 이루어지는 스레드와 같은 스레드에서 실행되지만 다른 스레드에서 실행되는 자식 연산
    또한 프로파일링할 수 있습니다.
    동시에 실행되는 프로파일러들은 결과가 섞이지 않도록 각자의 스레드 범위에 한정됩니다.

.. note::
    Pytorch 1.8은 미래의 릴리즈에서 기존의 프로파일러 API를 대체할 새로운 API를 소개하고 있습니다.
    새로운 API를 `이 페이지 <https://pytorch.org/docs/master/profiler.html>`__ 에서 확인하세요.

프로파일러 API 사용법에 대해 빠르게 살펴보고 싶다면 `이 레시피 문서 <https://tutorials.pytorch.kr/recipes/recipes/profiler_recipe.html>`__ 를 확인하세요.


--------------
"""

import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler


######################################################################
# 프로파일러를 이용하여 성능 디버깅하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 프로파일러는 모델에서 성능의 병목을 파악할 때 유용할 수 있습니다.
# 이번 예제에서, 두 가지 하위 작업을 수행하는 사용자 정의 모듈을 만들겠습니다:
#
# - 입력에 대한 선형 변환
# - 변환 결과를 이용한 마스크 텐서(mask Tensor)에서 인덱스 추출
#
# 각 하위 작업들에 대한 코드는 ``profiler.record_function("label")`` 을 이용하여
# 레이블된 컨텍스트 매니저(context manager) 들에 의해 감쌉니다.
# 프로파일러의 출력에서, 하위 작업들의 모든 연산에 대한 집계(aggregate) 성능 지표들이 해당 레이블 아래 나타나게 됩니다.
#
#
# 프로파일러를 사용하는 것은 약간의 오버헤드가 발생하며, 코드를 분석할 때에만 사용하는 것이 가장 좋습니다.
# 만일 실행시간을 벤치마킹하는 경우에는 이를 제거하는 것을 잊지 마십시오.
#

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx


######################################################################
# 순전파 단계(forward pass) 프로파일링하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 입력과 마스크 텐서, 그리고 모델을 임의로 초기화합니다.
#
# 프로파일러를 실행하기 전, 정확한 성능 벤치마킹을 보장하기 위해 CUDA를 워밍업(warm-up) 시킵니다.
# 모델의 순전파 단계를 ``profiler.profile`` 컨텍스트 매니저를 통해 감쌉니다.
# ``with_stack=True`` 인자는 연산의 추적(trace) 파일 내부에 파일과 줄번호를 덧붙입니다.
#
# .. WARNING::
#     ``with_stack=True`` 는 추가적인 오버헤드를 발생시키기 때문에 코드를 분석할 때에 사용하는 것이 바람직합니다.
#     성능을 벤치마킹한다면 이를 제거하는 것을 잊지 마십시오.
#

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

# 워밍업(warm-up)
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)


######################################################################
# 프로파일러의 결과 출력하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 최종적으로 프로파일러의 결과를 출력합니다.
# ``profiler.key_averages`` 는 연산자의 이름에 따라 결과를 집계하는데,
# 선택적으로 입력의 shape과/또는 스택 추적(stack trace) 이벤트에 따라서도 결과를 집계할 수 있습니다.
# 입력의 shape에 따라서 그룹화 하는 것은 어떠한 shape의 텐서들이 모델에 의해 사용되는지 파악하는 데 유용합니다.
#
# 여기서, ``group_by_stack_n=5`` 를 사용하는데 이는 연산(operation)과 traceback(가장 최근 5개의 이벤트에 대한)을
# 기준으로 실행시간을 집계하는 것이고, 이벤트들이 등록된 순서로 정렬되어 표시됩니다.
# 결과 표는 ``sort_by`` 인자 (유효한 정렬 키는 `docs <https://pytorch.org/docs/stable/autograd.html#profiler>`__ 에서
# 확인하세요) 를 넘겨줌으로써 정렬될 수 있습니다.
#
# .. Note::
#   notebook에서 프로파일러를 실행할 때 스택 추적(stacktrace)에서 파일명 대신
#   ``<ipython-input-18-193a910735e8>(13): forward`` 와 같은 항목을 볼 수 있습니다.
#   이는 ``<notebook-cell>(line number): calling-function`` 의 형식에 대응됩니다.

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

"""
(Some columns are omitted)

-------------  ------------  ------------  ------------  ---------------------------------
         Name    Self CPU %      Self CPU  Self CPU Mem   Source Location
-------------  ------------  ------------  ------------  ---------------------------------
 MASK INDICES        87.88%        5.212s    -953.67 Mb  /mnt/xarfuse/.../torch/au
                                                         <ipython-input-...>(10): forward
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(9): <module>
                                                         /mnt/xarfuse/.../IPython/

  aten::copy_        12.07%     715.848ms           0 b  <ipython-input-...>(12): forward
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(9): <module>
                                                         /mnt/xarfuse/.../IPython/
                                                         /mnt/xarfuse/.../IPython/

  LINEAR PASS         0.01%     350.151us         -20 b  /mnt/xarfuse/.../torch/au
                                                         <ipython-input-...>(7): forward
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(9): <module>
                                                         /mnt/xarfuse/.../IPython/

  aten::addmm         0.00%     293.342us           0 b  /mnt/xarfuse/.../torch/nn
                                                         /mnt/xarfuse/.../torch/nn
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(8): forward
                                                         /mnt/xarfuse/.../torch/nn

   aten::mean         0.00%     235.095us           0 b  <ipython-input-...>(11): forward
                                                         /mnt/xarfuse/.../torch/nn
                                                         <ipython-input-...>(9): <module>
                                                         /mnt/xarfuse/.../IPython/
                                                         /mnt/xarfuse/.../IPython/

-----------------------------  ------------  ---------- ----------------------------------
Self CPU time total: 5.931s

"""

######################################################################
# 메모리 성능 향상시키기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 메모리와 시간 측면에서 가장 비용이 큰 연산은 MASK INDICES 내 ``forward(10)`` 연산입니다.
# 먼저 메모리 소모 문제를 해결해봅시다.
# 12번째 줄의 ``.to()`` 연산은 953.67 Mb를 소모하는 것을 확인할 수 있습니다.
# 이 연산은 ``mask`` 를 CPU에 복사합니다.
# ``mask`` 는 ``torch.double`` 데이터 타입으로 초기화됩니다.
# 이를 ``torch.float`` 으로 변환하여 메모리 사용량을 줄일 수 있을까요?
#

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

# 워밍업(warm-up)
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

"""
(Some columns are omitted)

-----------------  ------------  ------------  ------------  --------------------------------
             Name    Self CPU %      Self CPU  Self CPU Mem   Source Location
-----------------  ------------  ------------  ------------  --------------------------------
     MASK INDICES        93.61%        5.006s    -476.84 Mb  /mnt/xarfuse/.../torch/au
                                                             <ipython-input-...>(10): forward
                                                             /mnt/xarfuse/  /torch/nn
                                                             <ipython-input-...>(9): <module>
                                                             /mnt/xarfuse/.../IPython/

      aten::copy_         6.34%     338.759ms           0 b  <ipython-input-...>(12): forward
                                                             /mnt/xarfuse/.../torch/nn
                                                             <ipython-input-...>(9): <module>
                                                             /mnt/xarfuse/.../IPython/
                                                             /mnt/xarfuse/.../IPython/

 aten::as_strided         0.01%     281.808us           0 b  <ipython-input-...>(11): forward
                                                             /mnt/xarfuse/.../torch/nn
                                                             <ipython-input-...>(9): <module>
                                                             /mnt/xarfuse/.../IPython/
                                                             /mnt/xarfuse/.../IPython/

      aten::addmm         0.01%     275.721us           0 b  /mnt/xarfuse/.../torch/nn
                                                             /mnt/xarfuse/.../torch/nn
                                                             /mnt/xarfuse/.../torch/nn
                                                             <ipython-input-...>(8): forward
                                                             /mnt/xarfuse/.../torch/nn

      aten::_local        0.01%     268.650us           0 b  <ipython-input-...>(11): forward
      _scalar_dense                                          /mnt/xarfuse/.../torch/nn
                                                             <ipython-input-...>(9): <module>
                                                             /mnt/xarfuse/.../IPython/
                                                             /mnt/xarfuse/.../IPython/

-----------------  ------------  ------------  ------------  --------------------------------
Self CPU time total: 5.347s

"""

######################################################################
#
# 이 연산을 위한 CPU 메모리 사용량이 절반으로 줄었습니다.
#
# 시간 성능 향상시키기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 소모된 시간이 조금 줄긴 했지만, 이는 아직도 너무 높은 수치입니다.
# CUDA 에서 CPU 로 행렬을 복사하는 것이 꽤 비용이 큰 연산인 것이 밝혀졌습니다.
# ``forward(12)`` 의 ``aten::copy_`` 연산은 ``mask`` 를 CPU에 복사하여 NumPy 의 ``argwhere`` 함수를 사용할 수 있게 합니다.
# ``forward(13)`` 의 ``aten::copy_`` 는 배열을 다시 텐서로 CUDA에 복사합니다.
# 이곳에서 ``torch`` 함수 ``nonzero()`` 를 대신 사용한다면 두 연산을 모두 제거할 수 있습니다.
#

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return out, hi_idx


model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

# 워밍업(warm-up)
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

"""
(Some columns are omitted)

--------------  ------------  ------------  ------------  ---------------------------------
          Name    Self CPU %      Self CPU  Self CPU Mem   Source Location
--------------  ------------  ------------  ------------  ---------------------------------
      aten::gt        57.17%     129.089ms           0 b  <ipython-input-...>(12): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/
                                                          /mnt/xarfuse/.../IPython/

 aten::nonzero        37.38%      84.402ms           0 b  <ipython-input-...>(12): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/
                                                          /mnt/xarfuse/.../IPython/

   INDEX SCORE         3.32%       7.491ms    -119.21 Mb  /mnt/xarfuse/.../torch/au
                                                          <ipython-input-...>(10): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/

aten::as_strided         0.20%    441.587us          0 b  <ipython-input-...>(12): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/
                                                          /mnt/xarfuse/.../IPython/

 aten::nonzero
     _numpy             0.18%     395.602us           0 b  <ipython-input-...>(12): forward
                                                          /mnt/xarfuse/.../torch/nn
                                                          <ipython-input-...>(25): <module>
                                                          /mnt/xarfuse/.../IPython/
                                                          /mnt/xarfuse/.../IPython/
--------------  ------------  ------------  ------------  ---------------------------------
Self CPU time total: 225.801ms

"""


######################################################################
# 더 읽을거리
# ~~~~~~~~~~~~~~~~~
# PyTorch 모델에서 시간과 메모리 병목을 분석하기 위해 프로파일러가 어떻게 사용될 수 있는지를 살펴보았습니다.
# 아래에 프로파일러에 대한 읽을거리가 더 있습니다:
#
# - `프로파일러 사용 레시피 <https://tutorials.pytorch.kr/recipes/recipes/profiler_recipe.html>`__
# - `Profiling RPC-Based Workloads <https://tutorials.pytorch.kr/recipes/distributed_rpc_profiling.html>`__
# - `Profiler API Docs <https://pytorch.org/docs/stable/autograd.html?highlight=profiler#profiler>`__
