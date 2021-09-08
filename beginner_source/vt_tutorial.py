"""
배포를 위한 비전 트랜스포머(Vision Transformer) 모델 최적화하기
==============================================================
Authors : `Jeff Tang <https://github.com/jeffxtang>`_,
`Geeta Chauhan <https://github.com/gchauhan/>`_
번역 : `김태영 <https://github.com/Taeyoung96/>`_

비전 트랜스포머(Vision Transformer)은 자연어 처리 분야에서 소개된
비교할 수 없는 최첨단의 결과를 달성한 최신의 어텐션 기반(attention-based) 트랜스포머 모델을
컴퓨터 비전 분야에 적용을 한 모델입니다.
FaceBook에서 발표한 Data-efficient Image Transformers는 `DeiT <https://ai.facebook.com/blog/data-efficient-image-transformers-a-promising-new-technique-for-image-classification>`_ 
이미지 분류를 위해 ImageNet 데이터셋을 통해 훈련된
비전 트랜스포머 모델입니다.

이번 튜토리얼에서는, DeiT가 무엇인지 그리고 어떻게 사용하는지 다룰 것입니다.
그 다음 스크립팅(scripting), 양자화, 최적화, 그리고 iOS와 안드로이드 앱 안에서 
모델을 사용하는 전체적인 단계를 수행해볼 것입니다.
또한, 양자화와 최적화가 된 모델과 양자화와 최적화가 되지않은 모델을 비교해볼 것이며,
단계를 수행해 가면서 양자화와 최적화를 적용한 모델이 얼마나 이점을 가지는지 볼 것입니다

"""  

######################################################################
# .
#



######################################################################
# DeiT란 무엇인가
# --------------------
#

######################################################################
# 컨벌루션 신경망(CNNs)은 2012년 딥러닝이 시작된 이후
# 이미지 분류를 수행할 때 주요한 모델이였습니다. 그러나 컨벌루션 신경망은 일반적으로
# 최첨단의 결과를 달성하기 위해 훈련에 수억 개의 이미지가 필요했습니다.
# DeiT는 이미지 분류를 수행하는데 있어서 최신 CNN 모델과 경쟁을 하는데
# 훈련에 더 적은 데이터와 컴퓨팅 자원이 필요로 하는 비전 트랜스포머 모델입니다.
# 이는 DeiT의 두 가지 주요 구성 요소에 의해 가능하게 되었습니다:
#
# -  훨씬 더 큰 데이터 세트에 대한 훈련을 시뮬레이션하는 데이터 증강(augmentation)
# -  트랜스포머 네트워크에 CNN의 출력값을 그대로 증류(distillation)하여 학습할 수 있도록 하는 기법
#
# DeiT는 제한된 데이터와 자원을 활용하여 컴퓨터 비전 태스크(Task)에 트랜스포머 모델을  
# 성공적으로 적용할 수 있음을 보여줍니다. 
# DeiT의 좀 더 자세한 내용을 원한다면, `저장소 <https://github.com/facebookresearch/deit>`_
# 와 `논문 <https://arxiv.org/abs/2012.12877>`_ 을 참고하시길 바랍니다.
#


######################################################################
# DeiT를 이용하여 이미지 분류하기
# -------------------------------
#
# DeiT를 사용하여 어떻게 이미지들을 분류하는지 자세한 정보는 DeiT 저장소에 README를 참고하시길 바랍니다.
# 빠른 테스트를 위해선, 첫번째로 요구되는 패키지들을
# 설치해 봅시다:
#
# ::
#
#    pip install torch torchvision
#    pip install timm
#    pip install pandas
#    pip install requests 
#
# 그런 다음 아래 스크립트를 실행해 봅시다:
#


from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

print(torch.__version__)
# 1.8.0 이여야 합니다.


model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
img = transform(img)[None,]
out = model(img)
clsidx = torch.argmax(out)
print(clsidx.item())


######################################################################
# ImageNet 목록에 따른 `라벨(labels) 파일 <https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`_
# 클래스 인덱스의 따라 출력은 269여야 하며, 이는 '목공 늑대, 회색 늑대, 큰개자리 루푸스'에 매핑됩니다.  
#
# 이제 DeiT 모델을 사용하여 이미지들을 분류할 수 있음을 확인했습니다.
# iOS 및 Android 앱에서 실행할 수 있도록 모델을 수정하는 방법을 살펴보겠습니다. 
#

######################################################################
# DeiT 스크립팅(Scripting) 하기
# ----------------------
# To use the model on mobile, we first need to script the
# model. See the `Script and Optimize recipe <https://tutorials.pytorch.kr/recipes/script_optimized.html>`_ for a
# quick overview. Run the code below to convert the DeiT model used in the
# previous step to the TorchScript format that can run on mobile.
#


model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("fbdeit_scripted.pt")


######################################################################
# The scripted model file fbdeit_scripted.pt of size about 346MB is
# generated.
#


######################################################################
# DeiT 양자화하기
# ---------------------
# To reduce the trained model size significantly while
# keeping the inference accuracy about the same, quantization can be
# applied to the model. Thanks to the transformer model used in DeiT, we
# can easily apply dynamic-quantization to the model, because dynamic
# quantization works best for LSTM and transformer models (see `here <https://pytorch.org/docs/stable/quantization.html?highlight=quantization#dynamic-quantization>`_
# for more details).
#
# Now run the code below:
#

# Use 'fbgemm' for server inference and 'qnnpack' for mobile inference
backend = "fbgemm" # replaced with qnnpack causing much worse inference speed for quantized model on this notebook
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
scripted_quantized_model = torch.jit.script(quantized_model)
scripted_quantized_model.save("fbdeit_scripted_quantized.pt")


######################################################################
# This generates the scripted and quantized version of the model
# fbdeit_quantized_scripted.pt, with size about 89MB, a 74% reduction of
# the non-quantized model size of 346MB!
#

######################################################################
# You can use the ``scripted_quantized_model`` to generate the same
# inference result:
#

out = scripted_quantized_model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
# The same output 269 should be printed

######################################################################
# DeiT 최적화하기
# ---------------------
# The final step before using the quantized and scripted
# model on mobile is to optimize it:
#

from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
optimized_scripted_quantized_model.save("fbdeit_optimized_scripted_quantized.pt")


######################################################################
# The generated fbdeit_optimized_scripted_quantized.pt file has about the
# same size as the quantized, scripted, but non-optimized model. The
# inference result remains the same.
#



out = optimized_scripted_quantized_model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
# Again, the same output 269 should be printed


######################################################################
# Using Lite Interpreter
# ------------------------
#
# To see how much model size reduction and inference speed up the Lite
# Interpreter can result in, let’s create the lite version of the model.
#

optimized_scripted_quantized_model._save_for_lite_interpreter("fbdeit_optimized_scripted_quantized_lite.ptl")
ptl = torch.jit.load("fbdeit_optimized_scripted_quantized_lite.ptl")


######################################################################
# Although the lite model size is comparable to the non-lite version, when
# running the lite version on mobile, the inference speed up is expected.
#


######################################################################
# Comparing Inference Speed
# ---------------------------
#
# To see how the inference speed differs for the four models - the
# original model, the scripted model, the quantized-and-scripted model,
# the optimized-quantized-and-scripted model - run the code below:
#

with torch.autograd.profiler.profile(use_cuda=False) as prof1:
    out = model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof2:
    out = scripted_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof3:
    out = scripted_quantized_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof4:
    out = optimized_scripted_quantized_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof5:
    out = ptl(img)

print("original model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
print("scripted model: {:.2f}ms".format(prof2.self_cpu_time_total/1000))
print("scripted & quantized model: {:.2f}ms".format(prof3.self_cpu_time_total/1000))
print("scripted & quantized & optimized model: {:.2f}ms".format(prof4.self_cpu_time_total/1000))
print("lite model: {:.2f}ms".format(prof5.self_cpu_time_total/1000))

######################################################################
# The results running on a Google Colab are:
#
# ::
#
#    original model: 1236.69ms
#    scripted model: 1226.72ms
#    scripted & quantized model: 593.19ms
#    scripted & quantized & optimized model: 598.01ms
#    lite model: 600.72ms
#


######################################################################
# The following results summarize the inference time taken by each model
# and the percentage reduction of each model relative to the original
# model.
#

import pandas as pd
import numpy as np

df = pd.DataFrame({'Model': ['original model','scripted model', 'scripted & quantized model', 'scripted & quantized & optimized model', 'lite model']})
df = pd.concat([df, pd.DataFrame([
    ["{:.2f}ms".format(prof1.self_cpu_time_total/1000), "0%"],
    ["{:.2f}ms".format(prof2.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof2.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    ["{:.2f}ms".format(prof3.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof3.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    ["{:.2f}ms".format(prof4.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof4.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    ["{:.2f}ms".format(prof5.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof5.self_cpu_time_total)/prof1.self_cpu_time_total*100)]],
    columns=['Inference Time', 'Reduction'])], axis=1)

print(df)

"""
        Model                             Inference Time    Reduction
0	original model                             1236.69ms           0%
1	scripted model                             1226.72ms        0.81%
2	scripted & quantized model                  593.19ms       52.03%
3	scripted & quantized & optimized model      598.01ms       51.64%
4	lite model                                  600.72ms       51.43%
"""

######################################################################
# Learn More
# ~~~~~~~~~~~~~~~~~
#
# - `Facebook Data-efficient Image Transformers <https://ai.facebook.com/blog/data-efficient-image-transformers-a-promising-new-technique-for-image-classification>`__
# - `Vision Transformer with ImageNet and MNIST on iOS <https://github.com/pytorch/ios-demo-app/tree/master/ViT4MNIST>`__
# - `Vision Transformer with ImageNet and MNIST on Android <https://github.com/pytorch/android-demo-app/tree/master/ViT4MNIST>`__
