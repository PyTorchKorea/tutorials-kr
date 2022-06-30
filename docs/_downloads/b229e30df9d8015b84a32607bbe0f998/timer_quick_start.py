"""
Timer 빠르게 시작하기
========================

이 튜토리얼에서는 `torch.utils.benchmark.Timer`\ 의 주요 API들을 다뤄보도록
하겠습니다. PyTorch Timer는
`timeit.Timer <https://docs.python.org/3/library/timeit.html#timeit.Timer>`__
API 기반으로, 몇몇 PyTorch 특화된 기능(modification)을 제공합니다.
내장 `Timer` 클래스에 익숙하실 필요는 없지만, 성능 측정(work)의 기본적인
내용들에는 익숙하다고 가정하겠습니다.

보다 종합적인 성능 튜닝 튜토리얼은 다음 링크를 참고해주세요:

    https://tutorials.pytorch.kr/recipes/recipes/benchmark.html


**목차:**
    1. `Timer 정의하기 <#id1>`__
    2. `실제 시간(wall time): \`Timer.blocked_autorange(...)\` <#wall-time-timer-blocked-autorange>`__
    3. `C++ 코드조각(snippet) <#c-snippet>`__
    4. `명령어 실행 횟수(instruction counts): \`Timer.collect_callgrind(...)\` <#instruction-counts-timer-collect-callgrind>`__
    5. `명령어 실행 횟수: 더 깊이 파보기 <#id2>`__
    6. `Callgrind를 사용한 A/B 테스트 <#callgrind-a-b>`__
    7. `마무리 <#id3>`__
    8. `각주 <#id4>`__
"""


###############################################################################
# 1. Timer 정의하기
# ~~~~~~~~~~~~~~~~~~~
#
# `Timer` 는 작업을 정의하기 위해 사용합니다.
#

from torch.utils.benchmark import Timer

timer = Timer(
    # 반복문(loop)에서 실행하고 시간을 측정할 연산을 정의합니다
    stmt="x * y",

    # `setup` 은 반복 측정을 시작하기 전에 실행되므로,
    # `stmt` 에서 필요한 모든 상태를 준비(populate)하는데 사용됩니다
    setup="""
        x = torch.ones((128,))
        y = torch.ones((128,))
    """,

    # 또는, `globals` 를 사용하여 외부 범위(outer scope)에서 사용하는 변수들을
    # 전달할 수 있습니다
    # -------------------------------------------------------------------------
    # globals={
    #     "x": torch.ones((128,)),
    #     "y": torch.ones((128,)),
    # },

    # PyTorch에서 사용하는 쓰레드(thread)의 수를 조절합니다 (기본값: 1)
    num_threads=1,
)

###############################################################################
# 2. 실제 실행 시간(wall time): `Timer.blocked_autorange(...)`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 메서드(method)는 몇 번이나 반복할지 적절한 횟수를 고르거나, 쓰레드의 수를
# 변경(fix)하거나,결과를 편하게 표현하는 방법을 제공하는 등, 세부적인 사항들을
# 처리(handle)합니다.
#

# Measurement 객체는 여러번 반복하여 측정한 결과를 저장하고, 다양한 편의 기능
# (utility feature)을 제공합니다.
from torch.utils.benchmark import Measurement

m: Measurement = timer.blocked_autorange(min_run_time=1)
print(m)

###############################################################################
# .. code-block:: none
#    :caption: **Snippet wall time.**
#
#         <torch.utils.benchmark.utils.common.Measurement object at 0x7f1929a38ed0>
#         x * y
#         setup:
#           x = torch.ones((128,))
#           y = torch.ones((128,))
#
#           Median: 2.34 us
#           IQR:    0.07 us (2.31 to 2.38)
#           424 measurements, 1000 runs per measurement, 1 thread
#

###############################################################################
# 3. C++ 코드 조각(snippet)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from torch.utils.benchmark import Language

cpp_timer = Timer(
    "x * y;",
    """
        auto x = torch::ones({128});
        auto y = torch::ones({128});
    """,
    language=Language.CPP,
)

print(cpp_timer.blocked_autorange(min_run_time=1))

###############################################################################
# .. code-block:: none
#    :caption: **C++ snippet wall time.**
#
#         <torch.utils.benchmark.utils.common.Measurement object at 0x7f192b019ed0>
#         x * y;
#         setup:
#           auto x = torch::ones({128});
#           auto y = torch::ones({128});
#
#           Median: 1.21 us
#           IQR:    0.03 us (1.20 to 1.23)
#           83 measurements, 10000 runs per measurement, 1 thread
#

###############################################################################
# 당연히 C++ 코드 조각(snippet)이 더 빠르고 편차(variation)가 적습니다.
#

###############################################################################
# 4. 명령어 실행 횟수(instruction counts): `Timer.collect_callgrind(...)`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 더 자세한 정보를 제공하기 위해, `Timer.collect_callgrind` 는
# 명령어 실행 횟수(instruction count)를 수집하는
# `Callgrind <https://valgrind.org/docs/manual/cl-manual.html>` 를 감싸고(wrap) 있습니다.
# 이는 코드 조각(snippet)이 어떻게 실행되는지에 대해 세분화되고 결정적인(deterministic)
# 통찰을 제공하므로 유용합니다.
#

from torch.utils.benchmark import CallgrindStats, FunctionCounts

stats: CallgrindStats = cpp_timer.collect_callgrind()
print(stats)

###############################################################################
# .. code-block:: none
#    :caption: **C++ Callgrind stats (summary)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0x7f1929a35850>
#         x * y;
#         setup:
#           auto x = torch::ones({128});
#           auto y = torch::ones({128});
#
#                                 All          Noisy symbols removed
#             Instructions:       563600                     563600
#             Baseline:                0                          0
#         100 runs per measurement, 1 thread
#

###############################################################################
# 5. 명령어 실행 횟수: 더 깊이 파보기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# CallgrindStats의 문자열 표현은 Measurement의 그것과 유사합니다.
# `Noisy symbol` 은 Python의 개념입니다. (CPython 인터프리터(interpreter)에서는
# 불필요하다(noisy)고 알려진 호출들을 제외합니다)
#
# 일단 더 자세한 분석을 위해, 특정 호출(call)을 살펴보겠습니다.
# `CallgrindStats.stats()` 은 이를 더 쉽게해주는 FunctionCounts 객체를 반환합니다.
# 개념적으로, FunctionCounts는 각 쌍(pair)이 `(명령어 호출 횟수, 파일 경로 및 함수 이름)`
# 인 형태로 구성된, 유용한 메서드(utility method)가 있는 쌍(pair)의 튜플(tuple)로
# 생각할 수 있습니다.
#
# 경로(path)에 대한 참고 사항:
#   일반적으로 절대경로(absolute path)는 신경쓰지 않습니다. 예를 들어, 곱하기 호출의
#   전체 경로와 함수 이름은 이런 식일 것입니다:
#
#       /the/prefix/to/your/pytorch/install/dir/pytorch/build/aten/src/ATen/core/TensorMethods.cpp:at::Tensor::mul(at::Tensor const&) const [/the/path/to/your/conda/install/miniconda3/envs/ab_ref/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so]
#
#   실제로 우리가 관심을 갖는 정보들은 이런 식으로 표현이 가능합니다:
#
#       build/aten/src/ATen/core/TensorMethods.cpp:at::Tensor::mul(at::Tensor const&) const
#
#   CallgrindStats.as_standardized()는 파일 경로의 의미없는 부분(low signal portion)뿐만
#   아니라, 공유 객체(shared object)들도 제거(strip)하는데 최선을 다하므로, 대부분의 경우
#   사용하는 것을 권합니다.
#

inclusive_stats = stats.as_standardized().stats(inclusive=False)
print(inclusive_stats[:10])

###############################################################################
# .. code-block:: none
#    :caption: **C++ Callgrind stats (detailed)**
#
#         torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192a6dfd90>
#           47264  ???:_int_free
#           25963  ???:_int_malloc
#           19900  build/../aten/src/ATen/TensorIter ... (at::TensorIteratorConfig const&)
#           18000  ???:__tls_get_addr
#           13500  ???:malloc
#           11300  build/../c10/util/SmallVector.h:a ... (at::TensorIteratorConfig const&)
#           10345  ???:_int_memalign
#           10000  build/../aten/src/ATen/TensorIter ... (at::TensorIteratorConfig const&)
#            9200  ???:free
#            8000  build/../c10/util/SmallVector.h:a ... IteratorBase::get_strides() const
#
#         Total: 173472
#

###############################################################################
# 이 외에도 요약해야 할 내용들이 많습니다. `FunctionCounts.transform` 메소드를
# 사용하여 함수 경로의 일부를 자르고, 호출된 함수를 제거(discard)합니다.
# 그렇게 하면 중복(collision, 예. `foo.h` 에 같이 매핑된 `foo.h:a()` 와 `foo.h:b()` )된
# 횟수는 더해집니다.
#

import os
import re

def group_by_file(fn_name: str):
    if fn_name.startswith("???"):
        fn_dir, fn_file = fn_name.split(":")[:2]
    else:
        fn_dir, fn_file = os.path.split(fn_name.split(":")[0])
        fn_dir = re.sub("^.*build/../", "", fn_dir)
        fn_dir = re.sub("^.*torch/", "torch/", fn_dir)

    return f"{fn_dir:<15} {fn_file}"

print(inclusive_stats.transform(group_by_file)[:10])

###############################################################################
# .. code-block:: none
#    :caption: **Callgrind stats (condensed)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192995d750>
#           118200  aten/src/ATen   TensorIterator.cpp
#            65000  c10/util        SmallVector.h
#            47264  ???             _int_free
#            25963  ???             _int_malloc
#            20900  c10/util        intrusive_ptr.h
#            18000  ???             __tls_get_addr
#            15900  c10/core        TensorImpl.h
#            15100  c10/core        CPUAllocator.cpp
#            13500  ???             malloc
#            12500  c10/core        TensorImpl.cpp
#
#         Total: 352327
#

###############################################################################
# 6. Callgrind를 사용한 A/B 테스트
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 명령어 실행 횟수 측정의 가장 유용한 기능 중 하나는 성능을 분석할 때
# 중요한 것으로, 연산을 세밀하게 비교할 수 있다는 것입니다.
#
# 이를 실제로 확인해보기 위해, 텐서(Tensor)를 브로드캐스트(broadcast)하여
# 128 크기의 텐서(Tensor)와 곱하는 {128} x {1} 곱셈과 비교해보겠습니다:
#   result = {a0 * b0, a1 * b0, ..., a127 * b0}
#

broadcasting_stats = Timer(
    "x * y;",
    """
        auto x = torch::ones({128});
        auto y = torch::ones({1});
    """,
    language=Language.CPP,
).collect_callgrind().as_standardized().stats(inclusive=False)

###############################################################################
# 종종 서로 다른 두 환경에서 A/B 테스트를 진행하고 싶을 때가 있습니다. (예.
# PR을 테스트하거나, 컴파일 플래그(flag) 실험 등)
# 이는 CallgrindStats와 FunctionCounts, Measurement는 모두 pickle화(picklalbe)가
# 가능하기 때문에 매우 간단합니다. 각 환경에서 측정한 결과들을 저장하고, 단일
# 프로세스에서 불러와서 분석하기만 하면 됩니다.
#

import pickle

# 가능하다는 것을 보여주기 위해 `broadcasting_stats` 을 저장하고 불러옵니다.
broadcasting_stats = pickle.loads(pickle.dumps(broadcasting_stats))


# 두 작업을 비교합니다:
delta = broadcasting_stats - inclusive_stats

def extract_fn_name(fn: str):
    """Trim everything except the function name."""
    fn = ":".join(fn.split(":")[1:])
    return re.sub(r"\(.+\)", "(...)", fn)

# `.transform` 을 사용하여 diff를 읽을 수 있게 만듭니다:
print(delta.transform(extract_fn_name))


###############################################################################
# .. code-block:: none
#    :caption: **Instruction count delta**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192995d750>
#             17600  at::TensorIteratorBase::compute_strides(...)
#             12700  at::TensorIteratorBase::allocate_or_resize_outputs()
#             10200  c10::SmallVectorImpl<long>::operator=(...)
#              7400  at::infer_size(...)
#              6200  at::TensorIteratorBase::invert_perm(...) const
#              6064  _int_free
#              5100  at::TensorIteratorBase::reorder_dimensions()
#              4300  malloc
#              4300  at::TensorIteratorBase::compatible_stride(...) const
#               ...
#               -28  _int_memalign
#              -100  c10::impl::check_tensor_options_and_extract_memory_format(...)
#              -300  __memcmp_avx2_movbe
#              -400  at::detail::empty_cpu(...)
#             -1100  at::TensorIteratorBase::numel() const
#             -1300  void at::native::(...)
#             -2400  c10::TensorImpl::is_contiguous(...) const
#             -6100  at::TensorIteratorBase::compute_fast_setup_type(...)
#            -22600  at::TensorIteratorBase::fast_set_up(...)
#
#         Total: 58091
#

###############################################################################
# 브로드캐스팅했던 버전은 호출당(샘플 당 100번의 실행을 수집하였음을 기억하세요)
# 580번, 대략 10%만큼 명령어가 더 실행되었습니다. TensorIterator 호출이 제법 많으므로
# 조금 더 깊이 살펴보겠습니다. FunctionCounts.filter를 사용하여 이를 쉽게 수행할 수
# 있습니다.
#

print(delta.transform(extract_fn_name).filter(lambda fn: "TensorIterator" in fn))

###############################################################################
# .. code-block:: none
#    :caption: **Instruction count delta (filter)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f19299544d0>
#             17600  at::TensorIteratorBase::compute_strides(...)
#             12700  at::TensorIteratorBase::allocate_or_resize_outputs()
#              6200  at::TensorIteratorBase::invert_perm(...) const
#              5100  at::TensorIteratorBase::reorder_dimensions()
#              4300  at::TensorIteratorBase::compatible_stride(...) const
#              4000  at::TensorIteratorBase::compute_shape(...)
#              2300  at::TensorIteratorBase::coalesce_dimensions()
#              1600  at::TensorIteratorBase::build(...)
#             -1100  at::TensorIteratorBase::numel() const
#             -6100  at::TensorIteratorBase::compute_fast_setup_type(...)
#            -22600  at::TensorIteratorBase::fast_set_up(...)
#
#         Total: 24000
#

###############################################################################
# 이렇게 보면 진행 내역이 명확합니다:TensorIterator 구성(setup) 시 더 빠른 경로가
# 있지만, {128} x {1} 경우에는 이것이 아닌 더 비용이 많이 드는 일반적인 분석을
# 수행해야 합니다. 필터에서 생략(omit)된 가장 주요한 호출은
# `c10::SmallVectorImpl<long>::operator=(...)` 으로, 일반적인 구성(setup)의 일부
# 이기도 합니다.
#

###############################################################################
# 7. 마무리
# ~~~~~~~~~~~~~~
#
# 요약하면 `Timer.blocked_autorange` 를 사용하여 실제 실행 시간(wall time)을 수집합니다.
# 시간 편차가 너무 크면, `min_run_time` 을 늘리거나, 만약 C++이 더 편하면 C++ 코드
# 조각을 사용하도록 합니다.
#
# 세분화된 분석을 위해, `Timer.collect_callgrind` 를 사용하여 명령어 실행 횟수를
# 측정하고 `FunctionCounts.(__add__ / __sub__ / transform / filter)` 를 사용하여
# 결과를 쪼개어 분석(slice-and-dice)합니다.
#

###############################################################################
# 8. 각주
# ~~~~~~~~~~~~
#
#   - 묵시적(implied) `import torch`
#       `globals` 가 "torch"를 포함하지 않으면, Timer가 자동으로 불러옵니다.
#       즉, `Timer("torch.empty(())")` 가 동작합니다. (다른 불러오기(import)
#       는 반드시 `setup` 에 포함되어 있어야 합니다 -
#       예. `Timer("np.zeros(())", "import numpy as np")` )
#
#   - REL_WITH_DEB_INFO
#       실행되는 PyTorch 내부에 대한 전체 정보를 제공하기 위해, Callgrind는
#       C++ 디버그 심볼(debug symbol)에 접근해야 합니다. 이를 위해 PyTorch를
#       빌드할 때 REL_WITH_DEB_INFO=1 을 설정해야 합니다. 그렇지 않으면
#       함수 호출이 불투명(opaque)해집니다. (이런 경우 CallgrindStats가
#       디버그 심볼 누락을 경고합니다.)
