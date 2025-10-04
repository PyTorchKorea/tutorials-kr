# -*- coding: utf-8 -*-

"""
사용자 정의 Triton 커널을 ``torch.compile``과 함께 사용하기
=========================================================
**저자:** `Oguz Ulgen <https://github.com/oulgen>`_
**번역:** `구경선 <https://github.com/kookyungseon>`_, `이채운 <https://github.com/dlcodns>`_
"""

######################################################################
# 사용자 정의 Triton 커널을 사용하면 모델의 특정 부분의 계산을 최적화할 수 있습니다.
# 이 커널들은 Triton의 언어로 작성된 것으로 설계되었습니다.
# 사용자 정의 Triton을 사용하여 하드웨어 성능을 최고로 향상시킵니다.
# ``torch.compile``를 사용하는 커널은 이러한 최적화된 계산을 통합할 수 있습니다.
# PyTorch 모델을 통해 상당한 성능 향상을 실현할 수 있습니다.
#
# 이 레시피는 사용자 정의 Triton 커널을  ``torch.compile``과 함께 사용할 수 있는 방법을 보여줍니다.
#
# 전제조건
# -------------------
#
#  이 레시피를 시작하기 전에 다음이 있는지 확인합니다:
# * ``torch.compile`` 및 Triton에 대한 기본적인 이해. 참조:
#
#   * `torch.compiler API 설명서 <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler>`__
#   * `torch.compile 소개 <https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html>`__
#   * `Triton 언어 문서 <https://triton-lang.org/main/index.html>`__
#
# * PyTorch 2.3 이상
# * Triton을 지원하는 GPU
#

import torch
from torch.utils._triton import has_triton

######################################################################
# 기본 사용법
# --------------------
#
# 이 예에서는 Triton 문서의 간단한 벡터 덧셈 커널을 사용합니다.
# ``torch.compile``과 함께.
# 참고, `Triton 문서를 참고하세요 <https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html>`__.
#

if not has_triton():
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    from triton import language as tl

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @torch.compile(fullgraph=True)
    def add_fn(x, y):
        output = torch.zeros_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
        return output

    x = torch.randn(4, device="cuda")
    y = torch.randn(4, device="cuda")
    out = add_fn(x, y)
    print(f"Vector addition of\nX:\t{x}\nY:\t{y}\nis equal to\n{out}")

######################################################################
# 고급 사용법
# -------------------------------------------------------------------
#
# Triton의 자동 튜닝 기능은 Triton 커널의 구성 매개변수를 자동으로 최적화해주는 강력한 도구입니다.
# 다양한 설정을 검토하여 특정 사용 사례에 최적의 성능을 제공하는 구성을 선택합니다.
#
# ``torch.compile``과 함께 사용할 경우 ``triton.autotune``을 사용하면 PyTorch 모델을 최대한 효율적으로
# 실행할 수 있습니다. 아래는 ``torch.compile``과 ``triton.autotune``을 사용하는 예제입니다.
#
# .. note::
#   ``torch.compile``은 ``triton.autotune``에 대한 configs와 key 인수만 지원합니다.
#

if not has_triton():
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    from triton import language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 4}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=4, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @torch.compile(fullgraph=True)
    def add_fn(x, y):
        output = torch.zeros_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel_autotuned[grid](x, y, output, n_elements)
        return output

    x = torch.randn(4, device="cuda")
    y = torch.randn(4, device="cuda")
    out = add_fn(x, y)
    print(f"Vector addition of\nX:\t{x}\nY:\t{y}\nis equal to\n{out}")

######################################################################
# Composability
# -------------------------------------------------------------------
#
# User-defined Triton kernels do not automatically support all PyTorch
# subsystems. This can be seen in the following use cases:
#
# - Adding a CPU fallback
# - Adding a ``FlopCounter`` formula
# - Composing with Tensor Subclasses
#
# To compose with additional PyTorch subsystems, use ``torch.library.triton_op``.
#
# ``triton_op is`` a structured way of defining a custom operator that is backed by one
# or more Triton kernels: like regular custom operators (``torch.library.custom_op``),
# you are able to specify the interactions with PyTorch subsystems via ``torch.library``.
# However, unlike ``torch.library.custom_op``, which creates opaque callables with respect to
# ``torch.compile``, ``torch.compile`` traces into ``triton_op`` to apply optimizations.
#
# Here’s a chart of which API to use when integrating Triton kernels with PyTorch.
#
# .. list-table::
#    :header-rows: 1
#
#    * -
#      - Triton kernel (no explicit ``torch.library`` wrapper)
#      - ``torch.library.triton_op``
#      - ``torch.library.custom_op``
#    * - Supports inference
#      - Yes
#      - Yes
#      - Yes
#    * - Supports training
#      - In the majority of cases
#      - Yes
#      - Yes
#    * - Supports ``torch.compile``
#      - Yes
#      - Yes
#      - Yes
#    * - Supports ``torch.compile(fullgraph=True)``
#      - In the majority of cases
#      - In the majority of cases
#      - In all cases
#    * - Does torch.compile trace into the implementation?
#      - Yes
#      - Yes
#      - No
#    * - Supports AOTInductor
#      - Yes
#      - Yes
#      - No
#    * - Supports PyTorch Subsystems like FlopCounterMode, CPU Fallback, Tensor Subclasses
#      - No
#      - Yes
#      - Yes

######################################################################
# Wrapping Triton kernels with ``triton_op``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use ``torch.library.triton_op`` to wrap a function that may invoke one or more Triton kernels.
# Use ``torch.library.wrap_triton`` to wrap the calls to the Triton kernel.

from torch.library import triton_op, wrap_triton

@triton_op("mylib::mysin", mutates_args={})
def mysin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    wrap_triton(sin_kernel)[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)
    return out

@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)

######################################################################
# You can invoke the ``triton_op`` in one of the following two ways.

x = torch.randn(3, device="cuda")
y = mysin(x)
z = torch.ops.mylib.mysin.default(x)

assert torch.allclose(y, x.sin())
assert torch.allclose(z, x.sin())

######################################################################
# The resulting ``triton_op`` works with ``torch.compile`` and ``AOTInductor``.

y = torch.compile(mysin)(x)
assert torch.allclose(y, x.sin())

######################################################################
# Adding training support
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Use ``register_autograd`` to add an autograd formula for the ``triton_op``.
# Prefer this to using ``torch.autograd.Function`` (which has various composability footguns
# with ``torch.compile``).

def backward(ctx, grad):
    x, = ctx.saved_tensors
    return grad * x.cos()

def setup_context(ctx, inputs, output):
    x, = inputs
    ctx.save_for_backward(x)

mysin.register_autograd(backward, setup_context=setup_context)

######################################################################
# Note that the backward must be a composition of PyTorch-understood operators.
# If you want the backward to call Triton kernels, then those must be wrapped in ``triton_op`` as well:

@triton.jit
def cos_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.cos(x)
    tl.store(out_ptr + offsets, output, mask=mask)

@triton_op("mylib::mycos", mutates_args={})
def mycos(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    wrap_triton(cos_kernel)[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)
    return out

def backward(ctx, grad):
    x, = ctx.saved_tensors
    return grad * mycos(x)

def setup_context(ctx, inputs, output):
    x, = inputs
    ctx.save_for_backward(x)

mysin.register_autograd(backward, setup_context=setup_context)

######################################################################
# Adding a CPU Fallback
# ^^^^^^^^^^^^^^^^^^^^^
# Triton kernels don’t run on CPU. Use  ``register_kernel`` to add a CPU (or any other device) fallback for the ``triton_op``:

@mysin.register_kernel("cpu")
def _(x):
    return torch.sin(x)

x = torch.randn(3)
y = mysin(x)
assert torch.allclose(y, x.sin())

######################################################################
# The fallback must be composed of PyTorch operators.

######################################################################
# Adding a FlopCounter Formula
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To specify how many flops the triton kernel reports under PyTorch's flop counter,
# use ``register_flop_formula``.

from torch.utils.flop_counter import FlopCounterMode, register_flop_formula

@register_flop_formula(torch.ops.mylib.mysin)
def _(x_shape):
    numel = 1
    for s in x_shape:
        numel *= s
    return numel

x = torch.randn(3, device="cuda")

#########################################################
# ``FlopCounterMode`` requires `tabulate <https://pypi.org/project/tabulate/>`__.
# Before running the code below, make sure you have ``tabulate`` installed or install by
# running ``pip install tabulate``.
#
# >>> with FlopCounterMode() as flop_counter:
# >>>     y = mysin(x)

######################################################################
# Limitations
# --------------------------------------------------------------------
#
# PyTorch 2.3 버전 기준으로, ``torch.compile``의 사용자 정의 Triton 커널에는 동적 모양
# ``torch.autograd.Function``, JIT inductor, AOT inductor가 지원됩니다. 이 기능들을
# 조합하여 복잡하고 고성능인 모델을 구축할 수 있습니다.
#
# PyTorch 2.6 버전에서 ``torch.library.triton_op``이 추가되어, Tensor 하위 클래스 및
# 기타 고급 기능에 대한 사용자 정의 Triton 커널 지원이 추가되었습니다.
#
# 그러나 알아두어야 할 몇 가지 제한 사항이 있습니다.
#
# * **Triton Features:** ``triton.heuristics`` 는 단독으로 사용하거나 ``triton.autotune`` 앞에서
# 사용할 수 있지만, ``triton.autotune`` 뒤에서는 사용할 수 없습니다. 따라서 ``triton.heuristics`` 와
# ``triton.autotune``을 함께 사용하려면 ``triton.heuristics`` 를 먼저 사용해야 합니다.
#
# 결론
# -----------
#
# 이 레시피에서는 사용자 정의 Triton 커널을 ``torch.compile`` 로 활용하는 방법을 알아보았습니다. 간단한
# 벡터 덧셈 커널의 기본 사용법과 Triton의 자동 튜닝 기능을 포함한 고급 사용법에 대해 다뤘습니다. 또한 사용자
# 정의 Triton 커널과 다른 Pytorch 기능의 조합 가능성에 대해 논의하고 현재의 몇 가지 제한 사항을 강조했습니다.
#
# 관련 항목
# ---------
#
# * `Optimizer 컴파일하기 <https://tutorials.pytorch.kr/recipes/compiling_optimizer.html>`__
# * `Scaled Dot Product Attention을 활용한 고성능 Transformer 구현하기 <https://tutorials.pytorch.kr/intermediate/scaled_dot_product_attention_tutorial.html>`__
