# -*- coding: utf-8 -*-

"""
Using User-Defined Triton Kernels with ``torch.compile``
=========================================================
**Author:** `Oguz Ulgen <https://github.com/oulgen>`_
"""

######################################################################
# User-defined Triton kernels can be used to optimize specific parts of your
# model's computation. These kernels are written in Triton's language, which is designed
# to make it easier to achieve peak hardware performance. By using user-defined Triton
# kernels with ``torch.compile``, you can integrate these optimized computations into
# your PyTorch model, potentially achieving significant performance improvements.
#
# This recipes demonstrates how you can use user-defined Triton kernels with ``torch.compile``.
#
# Prerequisites
# -------------------
#
# Before starting this recipe, make sure that you have the following:
#
# * Basic understanding of ``torch.compile`` and Triton. See:
#
#   * `torch.compiler API documentation <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler>`__
#   * `Introduction to torch.compile <https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html>`__
#   * `Triton language documentation <https://triton-lang.org/main/index.html>`__
#
# * PyTorch 2.3 or later
# * A GPU that supports Triton
#

import torch
from torch.utils._triton import has_triton

######################################################################
# Basic Usage
# --------------------
#
# In this example, we will use a simple vector addition kernel from the Triton documentation
# with ``torch.compile``.
# For reference, see `Triton documentation <https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html>`__.
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
# Triton의 자동 튜닝 기능은 Triton 커널의 구성 파라미터를 자동으로 최적화해주는 강력한 도구입니다.
# 다양한 설정을 검토하여 특정 사용 사례에 최적의 성능을 제공하는 구성을 선택합니다.
#
# ``torch.compile``과 함께 사용할 경우 ``triton.autotune``을 사용하면 PyTorch 모델을 최대한 효율적으로 
# 실행할 수 있습니다. 아래는 ``torch.compile``과 ``triton.autotune``을 사용하는 예제입니다.
# 
# .. 참고::
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
# 호환성과 제한사항
# --------------------------------------------------------------------
#
# PyTorch 2.3 버전 기준으로, ``torch.compile``의 사용자 정의 Triton 커널에는 동적 모양 
# ``torch.autograd.Function``, JIT inductor, AOT inductor가 지원됩니다. 이 기능들을 
# 조합하여 복잡하고 고성능인 모델을 구축할 수 있습니다.
# 
# 그러나 알아두어야 할 몇 가지 제한 사항이 있습니다:
# 
# * **Tensor Subclasses:** 현재로서는 Tensor 하위 클래스 및 기타 고급 기능은 지원되지 않습니다.
#
# * **Triton Features:** ``triton.heuristics``는 단독으로 사용하거나 ``triton.autotune`` 앞에서 
# 사용할 수 있지만, ``triton.autotune`` 뒤에서는 사용할 수 없습니다. 따라서 ``triton.heuristics``와 
# ``triton.autotune``을 함께 사용하려면 ``triton.heuristics``를 먼저 사용해야 합니다.
# 
# 결론
# -----------
# 
# 이 레시피에서는 사용자 정의 Triton 커널을 ``torch.compile``로 활용하는 방법을 알아보았습니다. 간단한 
# 벡터 덧셈 커널의 기본 사용법과 Triton의 자동 튜닝 기능을 포함한 고급 사용법에 대해 다뤘습니다. 또한 사용자 
# 정의 Triton 커널과 다른 Pytorch 기능의 조합 가능성에 대해 논의하고 현재의 몇 가지 제한 사항을 강조했습니다.
#
# 관련 항목
# ---------
#
# * `Optimizer 컴파일하기 <https://tutorials.pytorch.kr/recipes/compiling_optimizer.html>`__
# * `Scaled Dot Product Attention을 활용한 고성능 Transformer 구현하기 <https://tutorials.pytorch.kr/intermediate/scaled_dot_product_attention_tutorial.html>`__
