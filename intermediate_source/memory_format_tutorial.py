# -*- coding: utf-8 -*-
"""
(실험용) PyTorch를 사용한 Channels Last Memory Format
*******************************************************
**Author**: `Vitaly Fedyunin <https://github.com/VitalyFedyunin>`_  

**번역**: `Choi Yoonjeong <https://github.com/potatochips178>`_

Channels last에 대하여
------------------------
Channels Last memory format 는 차원을 보존하는 메모리에서 NCHW 텐서를 정렬하는 대안책입니다. Channels Last tensors는 채널(Channel)이 가장 밀도가 높은 방식으로 정렬됩니다.

예를 들어, 전형적인(연속적인) NCHW tensor의 저장소(이 경우 2개의 2x2 이미지에 3개의 채널이 존재함) 는 다음과 같이 보입니다:

.. figure:: /_static/img/classic_memory_format.png
   :alt: classic_memory_format

Channels Last memory 형식은 데이터를 다르게 정렬합니다: 

.. figure:: /_static/img/channels_last_memory_format.png
   :alt: channels_last_memory_format

Pytorch는 존재하는 스트라이드(strides) 구조를 사용함으로써 메모리 형식을 지원합니다(그리고  eager, JIT, 그리고 TorchScrip를 포함한 기존의 모델들과 하위 호환성을 제공합니다). 
예를들어, Channels Last 형식에서 10x3x16x16 배치(batch)는 (768, 1, 48, 3)와 같은 폭(strides)을 가지고 있을 것입니다. 

"""

######################################################################
# Channels Last memory 형식은 오직 4D NCWH Tensors에서만 실행 가능 합니다. 
#

import torch
N, C, H, W = 10, 3, 32, 32

######################################################################
# 메모리 형식 API
# -----------------------
#
# 연속 메모리 형식과 channel last메모리 형식 간에 tensor를 변형시키는 방법은 다음과 같습니다. 

######################################################################
# 전형적인 PyTorch의 연속적인 tensor
x = torch.empty(N, C, H, W)
print(x.stride()) # Ouputs: (3072, 1024, 32, 1)

######################################################################
# 변환 연산자
x = x.contiguous(memory_format=torch.channels_last)
print(x.shape) # Outputs: (10, 3, 32, 32) 차원 순서는 보존함
print(x.stride()) # Outputs: (3072, 1, 96, 3)

######################################################################
# 연속적인 형식으로 되돌아가기
x = x.contiguous(memory_format=torch.contiguous_format)
print(x.stride()) # Outputs: (3072, 1024, 32, 1)

######################################################################
# 대안
x = x.to(memory_format=torch.channels_last)
print(x.stride()) # Ouputs: (3072, 1, 96, 3)

######################################################################
# 형식 확인
print(x.is_contiguous(memory_format=torch.channels_last)) # Ouputs: True

######################################################################
# Channels Last로 생성하기
x = torch.empty(N, C, H, W, memory_format=torch.channels_last)
print(x.stride()) # Ouputs: (3072, 1, 96, 3)

######################################################################
# ``clone`` 은 메모리 형식을 보존합니다. 
y = x.clone()
print(y.stride()) # Ouputs: (3072, 1, 96, 3)

######################################################################
# ``to``, ``cuda``, ``float`` 등은 메모리를 보존합니다. 
if torch.cuda.is_available():
    y = x.cuda()
    print(y.stride()) # Ouputs: (3072, 1, 96, 3)

######################################################################
# ``empty_like``, ``*_like`` 연산자는 메모리 형식을 보존합니다. 
y = torch.empty_like(x)
print(y.stride()) # Ouputs: (3072, 1, 96, 3)

######################################################################
# 점 별(pointwise) 연산자는 메모리 형식을 보존합니다. 
z = x + y
print(z.stride()) # Ouputs: (3072, 1, 96, 3)

######################################################################
#  Conv, Batchnorm 모듈은 Channels Last를 지원합니다. 
#  (오직 CndNN >=7.6 에서만 작동)
if torch.backends.cudnn.version() >= 7603:
    input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, device="cuda", requires_grad=True)
    model = torch.nn.Conv2d(8, 4, 3).cuda().float()

    input = input.contiguous(memory_format=torch.channels_last)
    model = model.to(memory_format=torch.channels_last) # 모듈 인자들은 Channels Last로 변환이 필요합니다 

    out = model(input)
    print(out.is_contiguous(memory_format=torch.channels_last)) # Ouputs: True

######################################################################
# 성능 향상
# -------------------------------------------------------------------------------------------
# 가장 중요한 성능 향상은 Tensor Cores 지원과 함께하는 NVidia의 하드웨어에서 관찰됩니다. 
# NVidia https://github.com/NVIDIA/apex 가 지원하는 스크립트를 훈련하는 AMP(Automated Mixed Precision)를 실행시키는 동안 22% 넘는 성능향상을 이룰 수 있습니다. 
#
# ``python main_amp.py -a resnet50 --b 200 --workers 16 --opt-level O2  ./data``

# opt_level = O2
# keep_batchnorm_fp32 = None <class 'NoneType'>
# loss_scale = None <class 'NoneType'>
# CUDNN VERSION: 7603
# => creating model 'resnet50'
# Selected optimization level O2:  FP16 training with FP32 batchnorm and FP32 master weights.
# Defaults for this optimization level are:
# enabled                : True
# opt_level              : O2
# cast_model_type        : torch.float16
# patch_torch_functions  : False
# keep_batchnorm_fp32    : True
# master_weights         : True
# loss_scale             : dynamic
# Processing user overrides (additional kwargs that are not None)...
# After processing overrides, optimization options are:
# enabled                : True
# opt_level              : O2
# cast_model_type        : torch.float16
# patch_torch_functions  : False
# keep_batchnorm_fp32    : True
# master_weights         : True
# loss_scale             : dynamic
# Epoch: [0][10/125] Time 0.866 (0.866) Speed 230.949 (230.949) Loss 0.6735125184 (0.6735) Prec@1 61.000 (61.000) Prec@5 100.000 (100.000)
# Epoch: [0][20/125] Time 0.259 (0.562) Speed 773.481 (355.693) Loss 0.6968704462 (0.6852) Prec@1 55.000 (58.000) Prec@5 100.000 (100.000)
# Epoch: [0][30/125] Time 0.258 (0.461) Speed 775.089 (433.965) Loss 0.7877287269 (0.7194) Prec@1 51.500 (55.833) Prec@5 100.000 (100.000)
# Epoch: [0][40/125] Time 0.259 (0.410) Speed 771.710 (487.281) Loss 0.8285319805 (0.7467) Prec@1 48.500 (54.000) Prec@5 100.000 (100.000)
# Epoch: [0][50/125] Time 0.260 (0.380) Speed 770.090 (525.908) Loss 0.7370464802 (0.7447) Prec@1 56.500 (54.500) Prec@5 100.000 (100.000)
# Epoch: [0][60/125] Time 0.258 (0.360) Speed 775.623 (555.728) Loss 0.7592862844 (0.7472) Prec@1 51.000 (53.917) Prec@5 100.000 (100.000)
# Epoch: [0][70/125] Time 0.258 (0.345) Speed 774.746 (579.115) Loss 1.9698858261 (0.9218) Prec@1 49.500 (53.286) Prec@5 100.000 (100.000)
# Epoch: [0][80/125] Time 0.260 (0.335) Speed 770.324 (597.659) Loss 2.2505953312 (1.0879) Prec@1 50.500 (52.938) Prec@5 100.000 (100.000)

######################################################################
# ``--channels-last true`` 를 지나가는 것은 22% 성능향상을 보이며 Channels Lasst 형식을 가진 모델을 실행시키는 것을 허용합니다. 
# 
# ``python main_amp.py -a resnet50 --b 200 --workers 16 --opt-level O2 --channels-last true ./data``

# opt_level = O2
# keep_batchnorm_fp32 = None <class 'NoneType'>
# loss_scale = None <class 'NoneType'>
#
# CUDNN VERSION: 7603
#
# => creating model 'resnet50'
# Selected optimization level O2:  FP16 training with FP32 batchnorm and FP32 master weights.
#
# Defaults for this optimization level are:
# enabled                : True
# opt_level              : O2
# cast_model_type        : torch.float16
# patch_torch_functions  : False
# keep_batchnorm_fp32    : True
# master_weights         : True
# loss_scale             : dynamic
# Processing user overrides (additional kwargs that are not None)...
# After processing overrides, optimization options are:
# enabled                : True
# opt_level              : O2
# cast_model_type        : torch.float16
# patch_torch_functions  : False
# keep_batchnorm_fp32    : True
# master_weights         : True
# loss_scale             : dynamic
#
# Epoch: [0][10/125] Time 0.767 (0.767) Speed 260.785 (260.785) Loss 0.7579724789 (0.7580) Prec@1 53.500 (53.500) Prec@5 100.000 (100.000)
# Epoch: [0][20/125] Time 0.198 (0.482) Speed 1012.135 (414.716) Loss 0.7007197738 (0.7293) Prec@1 49.000 (51.250) Prec@5 100.000 (100.000)
# Epoch: [0][30/125] Time 0.198 (0.387) Speed 1010.977 (516.198) Loss 0.7113101482 (0.7233) Prec@1 55.500 (52.667) Prec@5 100.000 (100.000)
# Epoch: [0][40/125] Time 0.197 (0.340) Speed 1013.023 (588.333) Loss 0.8943189979 (0.7661) Prec@1 54.000 (53.000) Prec@5 100.000 (100.000)
# Epoch: [0][50/125] Time 0.198 (0.312) Speed 1010.541 (641.977) Loss 1.7113249302 (0.9551) Prec@1 51.000 (52.600) Prec@5 100.000 (100.000)
# Epoch: [0][60/125] Time 0.198 (0.293) Speed 1011.163 (683.574) Loss 5.8537774086 (1.7716) Prec@1 50.500 (52.250) Prec@5 100.000 (100.000)
# Epoch: [0][70/125] Time 0.198 (0.279) Speed 1011.453 (716.767) Loss 5.7595844269 (2.3413) Prec@1 46.500 (51.429) Prec@5 100.000 (100.000)
# Epoch: [0][80/125] Time 0.198 (0.269) Speed 1011.827 (743.883) Loss 2.8196096420 (2.4011) Prec@1 47.500 (50.938) Prec@5 100.000 (100.000)

######################################################################
# 다음 모델의 리스트는 Channels Last를 전적으로 지원하고 Volta 기기에서 8%-35%의 성능 향상을 보입니다:
# ``alexnet``, ``mnasnet0_5``, ``mnasnet0_75``, ``mnasnet1_0``, ``mnasnet1_3``, ``mobilenet_v2``, ``resnet101``, ``resnet152``, ``resnet18``, ``resnet34``, ``resnet50``, ``resnext50_32x4d``, ``shufflenet_v2_x0_5``, ``shufflenet_v2_x1_0``, ``shufflenet_v2_x1_5``, ``shufflenet_v2_x2_0``, ``squeezenet1_0``, ``squeezenet1_1``, ``vgg11``, ``vgg11_bn``, ``vgg13``, ``vgg13_bn``, ``vgg16``, ``vgg16_bn``, ``vgg19``, ``vgg19_bn``, ``wide_resnet101_2``, ``wide_resnet50_2``
#

######################################################################
# 기존 모델들 변환하기
# --------------------------
#
# Channels Last 지원은 기존 모델에 의해 제한되지 않고 어떠한 모델도 Channels Last로 바뀔 수 있으며 입력이 정확히 형식에 맞춰지자마자 그래프를 통해 형식을 전할 수 있습니다. 
#

# 모델 초기화(혹은 가져오기) 이후, 한 번 실행이 필요합니다. 
model = model.to(memory_format=torch.channels_last) # Replace with your model # 원하는 모델로 교체하기 

# 모든 입력에 대해서 실행이 필요합니다. 
input = input.to(memory_format=torch.channels_last) # Replace with your input # 원하는 입력으로 교체하기
output = model(input)

#######################################################################
# 그러나, 모든 연산자들이 Channels Last를 지원하도록 완전히 바뀐 것은 아닙니다(대신 보통 연속적인 출력을 반환합니다).
# 이것은 사용된 연산자들의 리스트를 지원되는 연산자 리스트에 https://github.com/pytorch/pytorch/wiki/Operators-with-Channels-Last-support 대해 확인하거나 혹은 메모리 형식 검사를 빠른 실행 모드로 도입해 모델을 실행시켜야 합니다.  
# 
# 아래 코드를 실행시킨 후, 만약 연산자들의 출력이 입력의 메모리 형식과 일치하지 않는다면 연산자들은 예외를 일으킬 것입니다. 
#
#
def contains_cl(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.is_contiguous(memory_format=torch.channels_last) and not t.is_contiguous():
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_cl(list(t)):
                return True
    return False


def print_inputs(args, indent=''):
    for t in args:
        if isinstance(t, torch.Tensor):
            print(indent, t.stride(), t.shape, t.device, t.dtype)
        elif isinstance(t, list) or isinstance(t, tuple):
            print(indent, type(t))
            print_inputs(list(t), indent=indent + '    ')
        else:
            print(indent, t)


def check_wrapper(fn):
    name = fn.__name__

    def check_cl(*args, **kwargs):
        was_cl = contains_cl(args)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            print("`{}` inputs are:".format(name))
            print_inputs(args)
            print('-------------------')
            raise e
        failed = False
        if was_cl:
            if isinstance(result, torch.Tensor):
                if result.dim() == 4 and not result.is_contiguous(memory_format=torch.channels_last):
                    print("`{}` got channels_last input, but output is not channels_last:".format(name),
                          result.shape, result.stride(), result.device, result.dtype)
                    failed = True
        if failed and True:
            print("`{}` inputs are:".format(name))
            print_inputs(args)
            raise Exception(
                'Operator `{}` lost channels_last property'.format(name))
        return result
    return check_cl


def attribute(m):
    for i in dir(m):
        e = getattr(m, i)
        exclude_functions = ['is_cuda', 'has_names', 'numel',
                             'stride', 'Tensor', 'is_contiguous', '__class__']
        if i not in exclude_functions and not i.startswith('_') and '__call__' in dir(e):
            try:
                setattr(m, i, check_wrapper(e))
            except Exception as e:
                print(i)
                print(e)


attribute(torch.Tensor)
attribute(torch.nn.functional)
attribute(torch)


######################################################################
# 만약 Channels Last tensors를 지원하지 않는 연산자를 찾는다면, 그리고 기여하기를 원한다면 
# 다음 개발자 안내서를 따르면 됩니다. 
# https://github.com/pytorch/pytorch/wiki/Writing-memory-format-aware-operators
#

######################################################################
# 해야할 일
# ----------
# 다음과 같이 많은 할 것들이 아직 남아있습니다:
#
# - N1HW와 NC11 Tensors의 모호성 해결하기;
# - 분산 훈련을 지원하는 지 확인해보기;
# - 연산자 범위를 향상시키기
#
# 만약 향상을 위한 피드백이나 제안이 있다면 issue를 만들어 알려주십시오.
#
# `issue 만들기 <https://github.com/pytorch/pytorch/issues>`_
