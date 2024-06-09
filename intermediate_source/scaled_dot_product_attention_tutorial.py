"""
(Beta) Scaled Dot Product Attention (SDPA)로 고성능 트랜스포머(Transformers) 구현하기
=================================================================================

**Author:** `Driss Guessous <https://github.com/drisspg>`_
  **번역:** `이강희 <https://github.com/khleexv>`_

"""

######################################################################
# 요약
# ~~~~
#
# 이 튜토리얼에서, 트랜스포머(Transformer) 아키텍처 구현에 도움이 되는 새로운
# ``torch.nn.functional`` 모듈의 함수를 소개합니다. 이 함수의 이름은 ``torch.nn.functional.scaled_dot_product_attention``
# 입니다. 함수에 대한 자세한 설명은 `PyTorch 문서 <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention>`__
# 를 참고하세요. 이 함수는 이미 ``torch.nn.MultiheadAttention`` 과 ``torch.nn.TransformerEncoderLayer``
# 에서 사용되고 있습니다.
#
# 개요
# ~~~~
# 고수준에서, 이 PyTorch 함수는 쿼리(query), 키(key), 값(value) 사이의
# scaled dot product attention (SDPA)을 계산합니다.
# 이 함수의 정의는 `Attention is all you need <https://arxiv.org/abs/1706.03762>`__
# 논문에서 찾을 수 있습니다. 이 함수는 기존 함수를 사용하여 PyTorch로 작성할 수 있지만,
# 퓨즈드(fused) 구현은 단순한 구현보다 큰 성능 이점을 제공할 수 있습니다.
#
# 퓨즈드 구현
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 이 함수는 CUDA tensor 입력을 다음 중 하나의 구현을 사용합니다.
#
# 구현:
#
# * `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`__
# * `Memory-Efficient Attention <https://github.com/facebookresearch/xformers>`__
# * A PyTorch implementation defined in C++
#
# .. note::
#
#   이 튜토리얼은 PyTorch 버전 2.0.0 이상이 필요합니다.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# 사용 예시:
query, key, value = torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device)
F.scaled_dot_product_attention(query, key, value)


######################################################################
# 명시적 Dispatcher 제어
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# 이 함수는 암시적으로 세 가지 구현 중 하나를 사용합니다. 하지만 컨텍스트 매니저를
# 사용하면 명시적으로 어떤 구현을 사용할 지 제어할 수 있습니다. 컨텍스트 매니저를 통해
# 특정 구현을 명시적으로 비활성화 할 수 있습니다. 특정 입력에 대한 가장 빠른 구현을 찾고자
# 한다면, 컨텍스트 매니저로 모든 구현의 성능을 측정해볼 수 있습니다.

# 벤치마크 함수를 정의합니다
import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# 입력의 하이퍼파라미터를 정의합니다
batch_size = 32
max_sequence_len = 1024
num_heads = 32
embed_dimension = 32

dtype = torch.float16

query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# 세 가지 구현의 속도를 측정합니다
from torch.nn.attention import SDPBackend, sdpa_kernel


with sdpa_kernel(SDPBackend.MATH):
    math_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
    print(f"The math implementation runs in {math_time:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")


######################################################################
# 하드웨어 의존성
# ~~~~~~~~~~~~~~~~~~
#
# 위 셀을 어떤 머신에서 실행했는지와 사용 가능한 하드웨어에 따라 결과가 다를 수 있습니다.
# - GPU가 없고 CPU에서 실행 중이라면 컨텍스트 매니저는 효과가 없고 세 가지 실행 모두
# 유사한 시간을 반환할 것입니다.
# - 그래픽 카드가 지원하는 컴퓨팅 능력에 따라 flash attention 또는
# memory efficient 구현이 동작하지 않을 수 있습니다.


######################################################################
# Causal Self Attention
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# 아래는 multi-head causal self attention 블록의 구현 예시입니다.
# `Andrej Karpathy NanoGPT <https://github.com/karpathy/nanoGPT>`__ 저장소를 참고했습니다.
#

class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y


num_heads = 8
heads_per_dim = 64
embed_dimension = num_heads * heads_per_dim
dtype = torch.float16
model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to("cuda").to(dtype).eval()
print(model)


#####################################################################
# ``NestedTensor`` 및 Dense tensor 지원
# ------------------------------------
#
# SDPA는 ``NestedTensor`` 와 Dense tensor 입력을 모두 지원합니다.
# ``NestedTensors`` 는 입력이 가변 길이 시퀀스로 구성된 배치인 경우에
# 배치 내 시퀀스의 최대 길이에 맞춰 각 시퀀스를 패딩할 필요가 없습니다. ``NestedTensors`` 에 대한 자세한 내용은
# `torch.nested <https://pytorch.org/docs/stable/nested.html>`__ 와 `NestedTensors 튜토리얼 <https://tutorials.pytorch.kr/prototype/nestedtensor.html>`__ 을 참고하세요.
#

import random
def generate_rand_batch(
    batch_size,
    max_sequence_len,
    embed_dimension,
    pad_percentage=None,
    dtype=torch.float16,
    device="cuda",
):
    if not pad_percentage:
        return (
            torch.randn(
                batch_size,
                max_sequence_len,
                embed_dimension,
                dtype=dtype,
                device=device,
            ),
            None,
        )
    # Random sequence lengths
    seq_len_list = [
        int(max_sequence_len * (1 - random.gauss(pad_percentage, 0.01)))
        for _ in range(batch_size)
    ]
    # Make random entry in the batch have max sequence length
    seq_len_list[random.randint(0, batch_size - 1)] = max_sequence_len
    return (
        torch.nested.nested_tensor(
            [
                torch.randn(seq_len, embed_dimension,
                            dtype=dtype, device=device)
                for seq_len in seq_len_list
            ]
        ),
        seq_len_list,
    )

random_nt, _ = generate_rand_batch(32, 512, embed_dimension, pad_percentage=0.5, dtype=dtype, device=device)
random_dense, _ = generate_rand_batch(32, 512, embed_dimension, pad_percentage=None, dtype=dtype, device=device)

# 현재 퓨즈드(fused) 구현은 ``NestedTensor`` 로 학습하는 것을 지원하지 않습니다.
model.eval()

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        print(f"Random NT runs in {benchmark_torch_function_in_microseconds(model, random_nt):.3f} microseconds")
        print(f"Random Dense runs in {benchmark_torch_function_in_microseconds(model, random_dense):.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")


######################################################################
# ``torch.compile`` 과 함께 SDPA 사용하기
# ============================================
#
# PyTorch 2.0 릴리즈와 함께 ``torch.compile()`` 라는 새로운 기능이 추가되었는데,
# 이는 eager mode보다 상당한 성능 향상을 제공할 수 있습니다.
# Scaled dot product attention은 ``torch.compile()`` 로 완전히 구성할 수 있습니다.
# 이를 확인하기 위해 ``torch.compile()`` 을 통해 ``CausalSelfAttention`` 모듈을 컴파일하고
# 결과적으로 얻어지는 성능 향상을 알아봅시다.
#

batch_size = 32
max_sequence_len = 256
x = torch.rand(batch_size, max_sequence_len,
               embed_dimension, device=device, dtype=dtype)
print(
    f"The non compiled module runs in  {benchmark_torch_function_in_microseconds(model, x):.3f} microseconds")


compiled_model = torch.compile(model)
# Let's compile it
compiled_model(x)
print(
    f"The compiled module runs in  {benchmark_torch_function_in_microseconds(compiled_model, x):.3f} microseconds")


######################################################################
#
# 정확한 실행 시간은 환경에 따라 다르지만, 다음은 저자의 결과입니다.
# 컴파일 되지 않은 모듈은 실행에 166.616ms 가 소요되었습니다.
# 컴파일 된 모듈은 실행에 166.726ms 가 소요되었습니다.
# 이는 우리의 예상과는 다릅니다. 좀 더 자세히 알아봅시다.
# PyTorch는 코드의 성능 특성을 점검할 수 있는 놀라운 내장(built-in) 프로파일러를 제공합니다.
#

from torch.profiler import profile, record_function, ProfilerActivity
activities = [ProfilerActivity.CPU]
if device == 'cuda':
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=False) as prof:
    with record_function(" Non-Compilied Causal Attention"):
        for _ in range(25):
            model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


with profile(activities=activities, record_shapes=False) as prof:
    with record_function("Compiled Causal Attention"):
        for _ in range(25):
            compiled_model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

######################################################################
#
# 더 많은 정보를 얻기 위해 추적(trace)를 내보내고 ``chrome://tracing``을 사용하여 결과를 확인해보세요.
#
# .. code-block:: python
#
#    prof.export_chrome_trace("compiled_causal_attention_trace.json").




######################################################################
# 이전 코드 조각(snippet)은 컴파일 된 모듈과 컴파일되지 않은 모듈 모두에 대해
# 가장 많은 GPU 실행 시간을 차지한 상위 10개의 PyTorch 함수에 대한 보고서를 생성합니다.
# 분석 결과, 두 모듈 모두 GPU에서 소요된 시간의 대부분이
# 동일한 함수들에 집중되어 있음을 보여줍니다.
# PyTorch가 프레임워크 오버헤드를 제거하는 데 매우 탁월한 ``torch.compile`` 를
# 제공하기 때문입니다. ``CausalSelfAttention`` 같은 경우처럼 크고, 효율적인 CUDA 커널을
# 사용하는 모델에서 PyTorch 오버헤드는 작아질 것입니다.
#
# 사실, 모듈은 보통 ``CausalSelfAttention`` 블럭 하나만으로 구성되지 않습니다.
# `Andrej Karpathy NanoGPT <https://github.com/karpathy/nanoGPT>`__ 저장소에서 실험한 경우,
# 모듈을 컴파일 하는 것은 학습의 각 단계별 소요 시간을 ``6090.49ms`` 에서 ``3273.17ms`` 로
# 줄일 수 있었습니다. 이 실험은 NanoGPT 저장소의 ``ae3a8d5`` 커밋에서 Shakespeare
# 데이터셋을 사용하여 진행되었습니다.
#

######################################################################
# SDPA를 ``atteition.bias`` 하위 클래스와 사용하기
# ====================================================
#
# PyTorch 2.3부터 텐서 하위 클래스를 포함하는 새로운 서브모듈을 추가했습니다.
# 추가된 모듈의 이름은 ``torch.nn.attention.bias`` 이며, ``torch.nn.functional.scaled_dot_product_attention``
# 와 함께 사용할 수 있도록 설계되었습니다. 또한, 인과적 어텐션 변형(Causal Attention Variants)을 생성하기
# 위한 다음 2가지 기능(utilities)을 포함하고 있습니다:
#
# - ``torch.nn.attention.bias.causal_upper_left``
# - ``torch.nn.attention.bias.causal_lower_right``
#
# .. note::
#    현재 ``torch.nn.functional.scaled_dot_product_attention`` 의 ``is_causal`` 인자(argument)는
#    ``torch.nn.attention.bias.causal_upper_left`` 를 사용하는 것과 동일합니다.
#

from torch.nn.attention.bias import causal_lower_right, causal_upper_left

batch_size = 32
sequence_length_q = 2
sequence_length_kv = 10
num_heads = 16
embed_dimension = 32

dtype = torch.float16

query = torch.rand(batch_size, num_heads, sequence_length_q, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, sequence_length_kv, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, sequence_length_kv, embed_dimension, device=device, dtype=dtype)

upper_left_bias = causal_upper_left(sequence_length_q, sequence_length_kv)
lower_right_bias = causal_lower_right(sequence_length_q, sequence_length_kv)

print(type(upper_left_bias))
print(type(lower_right_bias))

assert type(upper_left_bias) == type(lower_right_bias)
assert issubclass(type(upper_left_bias), torch.Tensor)

# 위의 출력에서 볼 수 있듯, 두 객체는 같은 타입인 ``torch.nn.attention.bias.CausalBias`` 이며,
# ``torch.Tensor`` 의 하위 클래스(subclass)입니다.

# 각 텐서들이 어떻게 생겼는지 살펴보겠습니다.
print(upper_left_bias)
print(lower_right_bias)

# Upper Left Bias는 인과적 어텐션 마스크(causal attention mask)를 어텐션 점수 행렬(attention scores matrix)의 왼쪽 상단에 정렬합니다.
# 이는 어텐션 점수 행렬이 정사각형이 아닌 경우에만 영향을 미치며, 이는 디코딩 상황에서 일반적인 경우입니다.
# 이 개념을 다른 방식으로 생각하는 방법은, upper left bias를 사용할 때는 쿼리(query)의 0번째 토큰이 키(key)의 0번째 토큰과 정렬된다고
# 생각하는 것입니다. 즉, 어텐션 점수 행렬(attention score matrix)이 2차원이라고 가정할 때, ``attn_score[0][0]`` 이 쿼리의 0번째 토큰과
# 키의 0번째 토큰 사이의 어텐션 점수인 것입니다.
# Lower Right Bias의 경우에는 쿼리(query)의 마지막 토큰이 키(key)의 마지막 토큰과 정렬되도록 쿼리(query)의 시퀀스를 정렬합니다.
# 예를 들어, ``attn_score[-1][-1]`` 은 쿼리와 키의 길이가 서로 다르더라도 쿼리의 마지막 토큰과 키의 마지막 토큰이 같은 위치에 있기 때문에
# 모두 True입니다.

# SDPA와 함께 사용하기 위한 객체들입니다.
out_upper_left = F.scaled_dot_product_attention(query, key, value, upper_left_bias)
out_lower_right = F.scaled_dot_product_attention(query, key, value, lower_right_bias)
out_is_causal = F.scaled_dot_product_attention(query, key, value, is_causal=True)

assert torch.allclose(out_upper_left, out_is_causal)
assert not torch.allclose(out_upper_left, out_lower_right)

# 아래 어텐션 편향(attention bias)들은 torch.compile과 호환됩니다.
compiled_sdpa = torch.compile(F.scaled_dot_product_attention, fullgraph=True)
out_upper_left = compiled_sdpa(query, key, value, upper_left_bias)

######################################################################
#
# 결론
# =======
#
# 이 튜토리얼에서, ``torch.nn.functional.scaled_dot_product_attention`` 의 기본적인
# 사용법을 살펴봤습니다. ``sdpa_kernel`` 컨텍스트 매니저로 GPU가 특정 구현을
# 사용하도록 할 수 있다는 것을 보았습니다. 또한, 간단한 ``NestedTensor`` 에서 작동하고
# 컴파일 가능한 ``CausalSelfAttention`` 모듈을 만들었습니다.
# 이 과정에서 프로파일링 도구를 사용하여 유저가 정의한 모듈의 성능 특성을 어떻게
# 확인할 수 있는지도 살펴봤습니다.
#
