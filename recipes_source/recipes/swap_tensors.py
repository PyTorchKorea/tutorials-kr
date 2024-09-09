"""
nn.Module에서 ``load_state_dict`` 및 텐서 서브클래스의 확장 포인트
===============================================================================
**저자:** `Mikayla Gawarecki <https://github.com/mikaylagawarecki>`_

이 레시피는 새로운 유틸리티 함수 ``torch.utils.swap_tensors``
뿐만 아니라 이를 통합한 두 가지 새로운 확장 지점을 소개합니다
``nn.Module``:

* ``nn.Module.to()`` 및 관련 메서드
* ``nn.Module.load_state_dict()``

.. 주의::
    이 레시피는 PyTorch 2.3.0 이상이 필요합니다.
"""

###############################################################################
# ``torch.utils.swap_tensors``
# ----------------------------
# ``torch.utils.swap_tensors`` (이하 ``swap_tensors``로 언급됨)은
# 두 개의 파이썬 텐서를 입력받아 서로 교환하는 유틸리티 함수입니다.

import torch
import torch.nn as nn
t1 = torch.arange(2)
t2 = torch.arange(3)
print(f"Before swapping, t1: {t1}, t2: {t2}")
torch.utils.swap_tensors(t1, t2)
print(f"After swapping, t1: {t1}, t2: {t2}")

################################################################################
# 더 구체적으로, ``swap_tensors``는 두 텐서의 파이썬 ``__class__``, ``__dict__``와
# ``__slots__``뿐만 아니라 관련된 ``at::Tensor``도 교환합니다.
#
#
# ``nn.Module``에의 적용
# ----------------------------
# 이 유틸리티는 모듈 외부의 파이썬 객체가 모듈의 파라미터에 대한
# 참조를 보유하고 있을 때 ``nn.Module``에 관련이 있습니다. 만약 ``nn.Module``
# 이 파라미터를 제자리에 수정하면, 파라미터에 대한 참조를 보유한 객체는
# 변경 사항을 볼 수 없습니다. 고전적인 예로는 ``nn.Module``의 파라미터에 대한
# 참조를 보유하는 옵티마이저가 있습니다. 이로 인해 ``optimizer.step()``이
# 오류 없이 실행되지만, ``nn.Module``의 가중치는 업데이트되지 않는
# 무성의 정확성 문제를 초래할 수 있습니다.

mod = torch.nn.Linear(1, 2, bias=False)
optimizer = torch.optim.SGD(mod.parameters())
print(f"weight in mod: {mod.weight}")
print(f"weight in optimizer: {optimizer.param_groups[0]['params']}")
mod.weight = torch.nn.Parameter(2 * mod.weight)
print(f"weight in mod: {mod.weight}")
print(f"weight in optimizer: {optimizer.param_groups[0]['params']}")

################################################################################
# ``nn.Module.to()`` 및 관련 메서드
# --------------------------------------
# 여기에는 모듈의 디바이스를 변경하는 메서드(예: ``nn.Module.cpu()``),
# 모듈의 ``dtype``을 변경하는 메서드(예: ``nn.Module.float()``)
# 뿐만 아니라 모듈을 구체화할 수 있게 해주는 메서드
# (예: ``nn.Module.to_empty()``)가 포함됩니다.
#
# 처음에는 이러한 메서드가 모듈의 파라미터를 제자리에서 수정할 수 있다는 것이
# 직관적이지 않을 수 있습니다. 기존의 접근 방식은 PyTorch 초기부터 사용된
# 복잡한 해킹 방법을 사용했습니다.
#
# 특히, 기존 접근 방식은 다음과 같은 경우에 작동하지 않습니다:
#
# * ``__torch_dispatch__`` 서브클래스를 사용할 때
# * ``param``과 ``new_param``의 파이썬 ``type()``이 동일하지 않을 때
# * 특수 C++ 표현을 가진 텐서(예: 희소 텐서 및 ``XLA`` 텐서)
#
# 이 레시피의 다음 부분에서는 양자화된 선형 가중치를 나타내는
# 장난감 ``__torch_dispatch__`` 서브클래스 ``MyQuantizedLinearWeight``를 정의할 것입니다.
# 이 서브클래스는 튜토리얼의 나머지 부분에서 설명을 위해 사용됩니다.
# 간결함을 위해 대부분의 ``__torch_dispatch__``
# 구현은 생략합니다.
aten = torch.ops.aten

class MyQuantizedLinearWeight(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, scale):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.shape,
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            strides=elem.stride(),
            storage_offset=elem.storage_offset())

    def __init__(self, elem: torch.Tensor, scale: float):
        self.elem = elem
        self.scale = scale

    def __repr__(self):
        return f"MyQuantizedLinearWeight({self.elem}, scale={self.scale})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in (aten.detach.default, aten._to_copy.default):
            new_elem = func(args[0].elem, *args[1:], **kwargs)
            return cls(new_elem, args[0].scale)
        # Implementations for certain ops would be added to ``OP_TABLE``.
        # We omit this for brevity.
        OP_TABLE = dict()
        if func in OP_TABLE:
          return OP_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"Unsupported function {func}")

#################################################################################
# ``dtype``가 ``torch.float32``인 ``nn.Linear`` 레이어를 생성하고, 가중치를
# ``MyQuantizedLinearWeight``로 설정한 후, 이를 ``torch.bfloat16``으로 변환해 봅니다.
# 가중치의 ``dtype``이 예상대로 변경되는 것을 관찰할 수 있습니다. 그러나
# 서브클래스의 페이로드(``elem``)의 ``dtype``은 변경되지 않습니다.

m = nn.Linear(3, 5, dtype=torch.float32)
m.weight = torch.nn.Parameter(MyQuantizedLinearWeight(m.weight, 0.5))
print(f"Before: id(m.weight)={id(m.weight)}, id(m.bias)={id(m.bias)}")
m.bfloat16()
print(f"After: id(m.weight)={id(m.weight)}, id(m.bias)={id(m.bias)}")
print(f"m.weight.dtype: {m.weight.dtype}")
print(f"m.weight.elem.dtype: {m.weight.elem.dtype}")
print(f"m.bias.dtype: {m.bias.dtype}")

################################################################################
# 이를 위해 글로벌 구성을 도입합니다
# ``torch.__future__.set_swap_module_params_on_conversion``을 사용할 것입니다.
# 이 구성은 ``swap_tensors``를 사용하여 모듈의 매개변수를 교환하며,
# ``.data`` 설정 대신 참조를 보존합니다. 이 구성이 설정되면,
# 변환 과정에서 ``swap_tensors``가 사용되며, 이를 통해
# 페이로드의 ``dtype``이 올바르게 변환되도록 보장합니다.

torch.__future__.set_swap_module_params_on_conversion(True)
m = nn.Linear(3, 5, dtype=torch.float32)
m.weight = torch.nn.Parameter(MyQuantizedLinearWeight(m.weight, 0.5))
print(f"Before: id(m.weight)={id(m.weight)}, id(m.bias)={id(m.bias)}")
m.bfloat16()
print(f"After: id(m.weight)={id(m.weight)}, id(m.bias)={id(m.bias)}")
print(f"m.weight.dtype: {m.weight.dtype}")
print(f"m.weight.elem.dtype: {m.weight.elem.dtype}")
print(f"m.bias.dtype: {m.bias.dtype}")
torch.__future__.set_swap_module_params_on_conversion(False)

################################################################################
# ``nn.Module.load_state_dict()``
# --------------------------------
# ``load_state_dict()``에 전달된 ``assign`` 키워드 인수의 값에 따라,
# ``state_dict``를 로드하는 두 가지 방법이 있습니다:
#
# * ``assign=False``: ``module.param``의 속성을 보존하고, ``state_dict['param_name']``의
#   값만 가져옵니다.
# * ``assign=True``: ``state_dict['param_name']``의 속성과 값을 모두 보존합니다.
#
#
# 이전에는 각각 제자리에서 ``copy_``와 ``__setattr__``로 구현되었습니다.
# 기존 구현에서는 각각의 접근 방식에 고유한 제한 사항이 있었습니다 -- ``assign=False``는
# ``state_dict``의 매개변수 타입이
# 모듈의 매개변수 타입과 동일해야 한다는 제약을 부과하는 반면, ``assign=True``는
# 모듈의 매개변수에 대한 참조를 보유하는 모든 것이
# ``nn.Module.load_state_dict()`` 이후에 초기화되어야 한다는 제약을 부과합니다.
#
# 이제 우리는 ``load_state_dict()``에 ``swap_tensors`` 경로를 추가하여 두 가지 제약을 해결합니다.
# 그리고 새로운 확장 포인트 ``torch.Tensor.module_load(self, other, assign=False)``를 도입합니다.
# 위에서 언급한 ``__future__``를 통해 ``swap_tensors`` 경로가 활성화되면,
# ``module_load``에 대한 ``__torch_function__`` 핸들러를 사용하여
# ``state_dict``의 값에 사용자 정의 변환을 적용할 수 있습니다. 이 변환의 결과는
# 모듈의 매개변수와 교체됩니다.
#
# 다음 예제에서는 ``MyQuantizedLinearWeight`` 서브클래스를 사용하여
# 위에서 정의된 기능을 사용하여
# ``state_dict``를 로드할 때.
# 선형 레이어의 가중치에 사용자 정의 양자화 방식을 적용하는 방법을 보여줍니다.
#
# ``module_load``에 대한 ``__torch_function__`` 
# ``self`` 또는 ``other`` (이 경우 ``param`` 또는
# ``state_dict[param_key]``)가 ``MyQuantizedLinearWeight`` 서브클래스인 경우  핸들러는 호출됩니다.
#
# ``state_dict``가 일반 텐서를 포함하고 있다고 가정하고,
# 모듈이 ``MyQuantizedLinearWeight`` 파라미터를 포함하고 있으며,
# ``state_dict``의 텐서가 서브클래스로 변환되기를 원합니다. 그럼,
# 우리는 ``torch.Tensor.module_load``에 대한 ``__torch_function__`` 핸들러를 다음과 같이 정의할 수 있습니다:
# 다음과 같이:

@classmethod
def custom_torch_function(cls, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs

    if func is torch.Tensor.module_load:
        dest, src = args[0], args[1]
        assert type(dest) == cls and type(src) == torch.Tensor
        return MyQuantizedLinearWeight(src, dest.scale)
    else:
        with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

MyQuantizedLinearWeight.__torch_function__ = custom_torch_function

#################################################################################
# 먼저, 메타 디바이스에서 모델의 스켈레톤을 생성하여 저장소를 실체화하는 것을 피합시다.
# 저장소를 실체화하지 않습니다. 우리는 모듈의 모든 가중치를
# `MyQuantizedLinearWeight` 서브클래스로 변환하면서 바이어스는 그대로 유지합니다.

def fn(m):
    if isinstance(m, nn.Linear):
        requires_grad = m.weight.requires_grad
        m.weight = torch.nn.Parameter(
                    MyQuantizedLinearWeight(m.weight, 0.5), requires_grad=requires_grad
                   )

with torch.device("meta"):
    m = nn.Linear(3, 5)
    m.apply(fn)

#################################################################################
# 그러면 ``state_dict``를 로드할 수 있습니다. 바이어스의 경우 ``assign=True``를 사용하는데,
# 바이어스의 경우, ``state_dict``에 있는 텐서의 속성을 유지하고자 합니다.
# ``state_dict``에 있는 텐서의 속성을 유지하기 위해서입니다 (예를 들어, 로드 후 바이어스가 ``meta`` 디바이스에 있지 않도록).

torch.__future__.set_swap_module_params_on_conversion(True)
print(f"Before: id(weight)={id(m.weight)}, id(bias)={id(m.bias)}")
print(f"m.state_dict() before load_state_dict():\n {m.state_dict()}")
state_dict = nn.Linear(3, 5).state_dict()
print(f"state_dict:\n {state_dict}")
m.load_state_dict(state_dict, assign=True)
print(f"After: id(weight)={id(m.weight)}, id(bias)={id(m.bias)}")
print(f"m.state_dict() after load_state_dict():\n {m.state_dict()}")

#################################################################################
# 위의 예제는 ``nn.Module.load_state_dict()``에서 새로운 확장 지점을 사용하는 방법을 보여주는 장난감 예제입니다.
# ``nn.Module.load_state_dict()``에서 새로운 확장 지점을 사용하는 방법을 보여줍니다. 또한 다른 시나리오를 상상할 수도 있습니다.
# 예를 들어, ``state_dict``에 텐서 서브클래스가 있고 모듈에 일반 ``nn.Parameters``/ 
# 모듈에 텐서가 있거나 둘 다 텐서 서브클래스일 때 등 다양한 시나리오를 상상할 수 있습니다. 사용에 따라
# 시나리오에 따라 ``module_load``에 대한 ``__torch_function__`` 핸들러를 정의할 수 있습니다.
# 필요에 따라 변환을 적용합니다.
#
# 결론
# ----------
# 이번 레시피에서는 ``swap_tensors``와 ``nn.Module``에서 파라미터의 참조를 보존하는 것의 중요성에 대해 배웠습니다.
# ``nn.Module``에서 파라미터의 참조를 보존하는 것과 
# ``torch.__future__.set_swap_module_params_on_conversion``에 의해
# 제어되는 두 가지 새로운 확장 지점을 사용하는 방법에 대해서도 배웠습니다.
