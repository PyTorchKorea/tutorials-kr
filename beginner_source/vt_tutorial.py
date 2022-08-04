"""
배포를 위한 비전 트랜스포머(Vision Transformer) 모델 최적화하기
=================================================================
Authors : `Jeff Tang <https://github.com/jeffxtang>`_, `Geeta Chauhan <https://github.com/gchauhan/>`_
번역 : `김태영 <https://github.com/Taeyoung96/>`_

비전 트랜스포머(Vision Transformer)는 자연어 처리 분야에서 소개된
최고 수준의 결과를 달성한 최신의 어텐션 기반(attention-based) 트랜스포머 모델을
컴퓨터 비전 분야에 적용을 한 모델입니다.
FaceBook에서 발표한 Data-efficient Image Transformers는 `DeiT <https://ai.facebook.com/blog/data-efficient-image-transformers-a-promising-new-technique-for-image-classification>`_
이미지 분류를 위해 ImageNet 데이터셋을 통해 훈련된
비전 트랜스포머 모델입니다.

이번 튜토리얼에서는, DeiT가 무엇인지 그리고 어떻게 사용하는지 다룰 것입니다.
그 다음 스크립팅, 양자화, 최적화, 그리고 iOS와 안드로이드 앱 안에서
모델을 사용하는 전체적인 단계를 수행해 볼 것입니다.
또한, 양자화와 최적화가 된 모델과 양자화와 최적화가 되지 않은 모델을 비교해 볼 것이며,
단계를 수행해 가면서 양자화와 최적화를 적용한 모델이 얼마나 이점을 가지는지 볼 것입니다.

"""

######################################################################
# DeiT란 무엇인가
# --------------------
#
# 합성곱 신경망(CNNs)은 2012년 딥러닝이 시작된 이후
# 이미지 분류를 수행할 때 주요한 모델이였습니다. 그러나 합성곱 신경망은 일반적으로
# 최첨단의 결과를 달성하기 위해 훈련에 수억 개의 이미지가 필요했습니다.
# DeiT는 훈련에 더 적은 데이터와 컴퓨팅 자원을 필요로 하는 비전 트랜스포머 모델이며,
# 최신 CNN 모델과 이미지 분류를 수행하는데 경쟁을 합니다.
# 이는 DeiT의 두 가지 주요 구성 요소에 의해 가능하게 되었습니다.
#
# -  훨씬 더 큰 데이터 세트에 대한 훈련을 시뮬레이션하는 데이터 증강(augmentation)
# -  트랜스포머 네트워크에 CNN의 출력값을 그대로 증류(distillation)하여 학습할 수 있도록 하는 기법
#
# DeiT는 제한된 데이터와 자원을 활용하여 컴퓨터 비전 태스크(task)에 트랜스포머 모델을
# 성공적으로 적용할 수 있음을 보여줍니다.
# DeiT의 좀 더 자세한 내용을 원한다면, `저장소 <https://github.com/facebookresearch/deit>`_
# 와 `논문 <https://arxiv.org/abs/2012.12877>`_ 을 참고하시길 바랍니다.
#


######################################################################
# DeiT를 활용한 이미지 분류
# -------------------------------
#
# DeiT를 사용하여 이미지를 분류하는 방법에 대한 자세한 정보는 DeiT 저장소에 README를 참고하시길 바랍니다.
# 빠른 테스트를 위해서, 먼저 필요한 패키지들을
# 설치합니다:
#
# pip install torch torchvision timm pandas requests

#######################################################
# Google Colab에서는 아래와 같이 실행합니다:

# !pip install timm pandas requests

#############################
# 그런 다음 아래 스크립트를 실행합니다:
#

from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

print(torch.__version__)
# Pytorch 버전은 1.8.0 이어야 합니다.


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
# ImageNet 목록에 따라 `라벨(labels) 파일 <https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`_
# 클래스 인덱스의 출력은 269여야 하며, 이는 ‘timber wolf, grey wolf, gray wolf, Canis lupus’에 매핑됩니다.
#
# 이제 DeiT 모델을 사용하여 이미지들을 분류할 수 있음을 확인했습니다.
# iOS 및 Android 앱에서 실행할 수 있도록 모델을 수정하는 방법을 살펴보겠습니다.
#

######################################################################
# DeiT 스크립팅
# ----------------------
# 모바일에서 이 모델을 사용하려면, 우리는 첫번째로 모델 스크립팅이 필요합니다.
# 전체적인 개요는 `스크립트 그리고 최적화 레시피 <https://tutorials.pytorch.kr/recipes/script_optimized.html>`_
# 에서 확인할 수 있습니다. 아래 코드를 실행하여 이전 단계에서 사용한 DeiT 모델을
# 모바일에서 실행할 수 있는 TorchScript 형식으로 변환합니다.
#


model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("fbdeit_scripted.pt")


######################################################################
# 약 346MB 크기의 스크립팅된 모델 파일 fbdeit_scripted.pt가 생성됩니다.
#
#


######################################################################
# DeiT 양자화
# ---------------------
# 추론 정확도를 거의 동일하게 유지하면서 훈련된 모델 크기를 크게 줄이기 위해
# 모델에 양자화를 적용할 수 있습니다.
# DeiT에서 사용된 트랜스포머 모델 덕분에,
# 모델에 동적 양자화를 쉽게 적용할 수 있습니다.
# 왜나하면 동적 양자화는 LSTM 모델과 트랜스포머 모델에서 가장 잘 적용되기 때문입니다.
# (자세한 내용은 `여기 <https://pytorch.org/docs/stable/quantization.html?highlight=quantization#dynamic-quantization>`_
# 를 참고하세요.)
#
# 아래의 코드를 실행시켜 봅시다.
#

# 서버 추론을 위해 'fbgemm'을, 모바일 추론을 위해 'qnnpack'을 사용해 봅시다.
backend = "fbgemm" # 이 주피터 노트북에서는 양자화된 모델의 더 느린 추론 속도를 일으키는 qnnpack으로 대체되었습니다.
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
scripted_quantized_model = torch.jit.script(quantized_model)
scripted_quantized_model.save("fbdeit_scripted_quantized.pt")


######################################################################
# fbdeit_quantized_scripted.pt 모델의 스크립팅과 양자화가 적용된 버전이 만들어졌습니다.
# 모델의 크기는 단지 89MB 입니다.
# 양자화가 적용되지 않은 모델의 크기인 346MB보다 74%나 감소했습니다!
#

######################################################################
# 동일한 추론 결과를 만들기 위해 ``scripted_quantized_model`` 을
# 사용해 봅시다.
#

out = scripted_quantized_model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
# 동일한 출력 결과인 269가 출력 되어야 합니다.

######################################################################
# DeiT 최적화
# ---------------------
# 모바일에 스크립트 되고 양자화된 모델을 사용하기 위한
# 마지막 단계는 최적화입니다.
#

from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
optimized_scripted_quantized_model.save("fbdeit_optimized_scripted_quantized.pt")


######################################################################
# 생성된 fbdeit_optimized_scripted_quantized.pt 파일은
# 양자화되고 스크립트되지만 최적화되지 않은 모델과 크기가 거의 같습니다.
# 추론 결과는 동일하게 유지됩니다.
#



out = optimized_scripted_quantized_model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
# 다시 한번, 동일한 출력 결과인 269가 출력 되어야 합니다.


######################################################################
# 라이트 인터프리터(Lite interpreter) 사용
# -----------------------------------------
#
# 라이트 인터프리터를 사용하면 얼마나 모델의 사이즈가 작아지고, 추론 시간이 짧아지는지
# 결과를 확인해 봅시다. 이제 좀 더 가벼운 버전의 모델을 만들어 봅시다.
#

optimized_scripted_quantized_model._save_for_lite_interpreter("fbdeit_optimized_scripted_quantized_lite.ptl")
ptl = torch.jit.load("fbdeit_optimized_scripted_quantized_lite.ptl")


######################################################################
# 가벼운 모델의 크기는 그렇지 않은 버전의 모델 크기와 비슷하지만,
# 모바일에서 가벼운 버전을 실행하면 추론 속도가 빨라질 것으로 예상됩니다.
#


######################################################################
# 추론 속도 비교
# ---------------------------
#
# 네 가지 모델(원본 모델, 스크립트된 모델, 스크립트와 양자화를 적용한 모델,
# 스크립트와 양자화를 적용한 후 최적화한 모델)의 추론 속도가 어떻게 다른지 확인해 봅시다.
#
# 아래의 코드를 실행해 봅시다.
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
# Google Colab에서 실행 시킨 결과는 다음과 같습니다.
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
# 다음 결과는 각 모델이 소요한 추론 시간과
# 원본 모델에 대한 각 모델의 감소율을 요약한 것입니다.
#
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
# 더 읽을거리
# ~~~~~~~~~~~~~~~~~
#
# - `Facebook Data-efficient Image Transformers <https://ai.facebook.com/blog/data-efficient-image-transformers-a-promising-new-technique-for-image-classification>`__
# - `Vision Transformer with ImageNet and MNIST on iOS <https://github.com/pytorch/ios-demo-app/tree/master/ViT4MNIST>`__
# - `Vision Transformer with ImageNet and MNIST on Android <https://github.com/pytorch/android-demo-app/tree/master/ViT4MNIST>`__
