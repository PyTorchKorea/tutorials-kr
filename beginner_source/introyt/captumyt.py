"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
`Building Models <modelsyt_tutorial.html>`_ ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
**Model Understanding**

Captum으로 모델 이해(Understanding)하기 
===============================
**번역**: '송진영 <https://github.com/diligejy>'

아래 `유튜브 <https://www.youtube.com/watch?v=Am2EF9CLu-g>`__ 영상을 참조하세요. 노트북과 관련 파일 다운로드는 
`여기 <https://pytorch-tutorial-assets.s3.amazonaws.com/youtube-series/video7.zip>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/Am2EF9CLu-g" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

`Captum <https://captum.ai/>`__ (라틴어로 “이해”라는 뜻의) 은 는 PyTorch 기반의 모델 해석력을 위한 오픈 소스 확장형 라이브러리입니다.

모델의 복잡성이 증가하고 그로 인해 모델을 설명하는 능력(Transparency)이 부족해져 모델 해석력을 갖춘 방법이 점점 더 중요해지고 있습니다.
모델 이해는 머신 러닝을 사용하는 산업 전반에 걸쳐 실무적 활용을 위한 핵심 영역이자 활발한 연구 영역입니다.

Captum은 통합된 그레디언트(Integrated Gradients)를 포함한 최신 알고리즘을 제공하여 연구자와 개발자에게 어떤 기능이 모델의 출력에 속성하고 있는지 쉽게 파악할 수 있는 방법을 제공합니다.

전체 문서나 API 자료 그리고 튜토리얼 같은 주제는 `captum.ai <https://captum.ai/>`__ 사이트를 참고하세요.


도입부
------------

모델 해석력에 대한 Captum의 접근 방식은 *속성(attribution)* 중심으로 이루어집니다

Captum에는 세 가지 속성(attribution)가 있습니다:

- **특징 속성(Feature Attribution)**는 특정 출력물을 생성한 입력물의 특징적인 측면에서 설명하고자 하며
  , 영화 리뷰가 긍정적이었는지 부정적이었는지를 리뷰에서 특정 단어로 설명하는 것은 특징 속성의 예입니다.
  
- **계층 속성(Layer Attribution)**는 특정 입력에 따른 모델의 은닉층의 활동을 조사합니다. 
    계층 속성의 예시로, 입력 이미지에 대응하여 컨벌루션 계층의 공간 매핑(spatially-mapped)된 출력을 검사하는 방법이 있습니다.
-  **뉴런 속성(Neuron Attribution)**는 계층 속성과 유사하지만 단일 뉴런의 활성에 초점을 맞춥니다.

이 페이지(interactive notebook)에서는, 특징 속성와 계층 속성에 대해 알아보겠습니다.


세 가지 속성 유형은 각각 여러 **속성 알고리즘(attribution algorithm)**이 연관되어 있습니다.
속성 알고리즘은 크게 두 가지로 나눌 수 있습니다:

-  **그래디언트 기반 알고리즘(Gradient-based algorithms)**은 
    입력에 대한 모델 출력, 계층 출력 또는 뉴런 활성화의 역전파(backward gradient)를 계산합니다.
   
   **통합 그래디언트(Integrated Gradients)** (특징에 대한), **계층 그래디언트(Layer Gradient) \*
   활성화**, and **뉴런 전도도(Neuron Conductance)** 모두 그래디언트 기반 알고리즘입니다.
   
-  **퍼터베이션 기반 알고리즘(Perturbation-based algorithms)**은 입력의 변화에 대응하여 모델, 계층 또는 뉴런 출력의 변화를 확인합니다.
    입력 퍼터베이션(input perturbations)은 일정한 방향을 가지거나 랜덤할 수 있습니다.
    **폐색(Occlusion),** **특징 절제(Feature Ablation),** and **기능 순환(Feature Permutation)**은 모두 퍼터베이션 기반 알고리즘입니다.

아래에서 두 가지 유형의 알고리즘을 모두 검토하겠습니다.

특히 대형 모델이 포함된 경우 평가 중인 입력 피쳐와 쉽게 연관시킬 수 있는 방식으로 속성 데이터를 시각화하는 것이 중요할 수 있습니다. 
Matplotlib, Plotly 또는 이와 유사한 도구를 사용하여 자신만의 시각화를 생성하는 것은 확실히 가능하지만, 
Captum은 해당 속성에 특화된 향상된 도구를 제공합니다:


- "captum.attr. visualization" 모듈(아래에서 "viz"로 import됨)은 이미지와 관련된 속성을 시각화하는 데 유용한 기능을 제공합니다.
- **Captum Insights**는 이미지, 텍스트 및 임의 모델 유형에 대한 이미 만들어진 시각화 위젯을 제공하는 API이며 Captum 위에서 사용하기 쉽습니다.

이 두 가지 시각화 도구 세트는 모두 이 노트북에서 설명할 것입니다. 
처음 몇 가지 예에서는 컴퓨터 비전 사용 사례에 초점을 두겠지만, 
마지막 Captum 인사이트 섹션에서는 다중 모델, 시각적 질의응답 모델에서 속성의 시각화를 설명할 것입니다.

설치
------------

시작하기 전에 Python 환경을 다음과 같이 구성해야 합니다:

- 파이썬 버전 3.6 이상
- Captum Insights 예제의 경우 Flask 1.1 이상
- PyTorch 버전 1.2 이상 (최신 버전 권장)
- TorchVision 버전 0.6 이상 (최신 버전 권장)
- Captum (최신 버전 권장)

Anaconda 또는 pip 가상 환경에 Captum을 설치하려면 환경에 맞게 적절한 명령을 사용합니다:

만약 ``conda``일 경우 ::

    conda install pytorch torchvision captum -c pytorch

만약 ``pip`` 일 경우 ::

    pip install torch torchvision captum

설정한 환경에서 이 노트북을 다시 시작하면 준비가 완료됩니다!


첫 번째 예시
---------------
먼저, 단순하고 시각적인 예시로 ImageNet 데이터 세트에 대해 사전학습된 ResNet 모델부터 시작하겠습니다. 
테스트 입력을 받고, 다양한 **특징 속성(Feature Attribution)** 알고리즘을 사용하여 
입력 이미지가 출력에 어떤 영향을 미치는지 조사하고, 일부 테스트 이미지에 대해 입력 속성을 시각화하여 확인해보겠습니다. 

먼저, 라이브러리들을 불러옵니다.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


#########################################################################
# 이제 TorchVision 모델 라이브러리를 사용하여 사전학습된 ResNet을 다운로드해 보겠습니다.
# 우리는 학습을 시키지 않았기 때문에 일단 평가모드를 사용하겠습니다.
#

model = models.resnet101(weights='IMAGENET1K_V1')
model = model.eval()


#######################################################################
# 대화형 노트북이 위치한 곳에는 "cat.jpg" 파일이 포함된 "img" 폴더도 있어야 합니다.
#
# 

test_img = Image.open('img/cat.jpg')
test_img_data = np.asarray(test_img)
plt.imshow(test_img_data)
plt.show()


##########################################################################
# ResNet 모델은 ImageNet 데이터셋으로 학습되었고, 특정 크기의 이미지와 채널 데이터가 특정 범위의 값으로 정규화될 것으로 예상합니다. 
# 또한 모델이 인식하는 범주에 대해 사람이 읽을 수 있는 레이블 목록을 수집할 것이며, 
# 이 레이블은 "img" 폴더에도 있어야 합니다.


#
# 모델은 224x183 3색 이미지를 필요로 합니다
#
transform = transforms.Compose([
 transforms.Resize(224),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

# 표준 ImageNet 정규화
transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0) # 모형에 더미 배치 차원(dimension)이 필요함

labels_path = 'img/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


######################################################################
# 이제 우리는 질문을 던질 수 있습니다. 우리 모델은 이 이미지가 무엇을 나타낸다고 생각하나요?
# 
#

output = model(input_img)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')


######################################################################
# ResNet이 고양이 이미지를 고양이라고 생각하는 것을 확인했습니다. 
# 하지만 *왜* 모델은 이것이 고양이 이미지라고 생각합니까?
#
# 그에 대한 답은 Captum에서 찾아보겠습니다. 

##########################################################################
# 통합 그라디언트(Gradient)를 사용한 특징 속성
# ---------------------------------------------
# 
# **특징 속성 (Feature attribution)**는 특정한 출력 데이터에서 입력 데이터의 특징을 찾아내는 일을 합니다. 
# 
# 다시 말해, 특징 속성은 특정한 입력 데이터(여기서는 테스트 이미지)을 사용하여 
# 특정한 출력 특징 데이터에 대한 각 입력 특징 데이터의 상대적 중요도에 대한 맵을 생성합니다.
#
# 
# '통합 그래디언트(Integrated Gradients) <https://captum.ai/api/integrated_gradients.html >'__ 는 
# Captum에서 사용할 수 있는 특징 속성 알고리즘 중 하나입니다. 
# 통합 그래디언트는 입력에 대한 모델 출력 데이터의 그래디언트의 적분을 근사화하여 
# 각 입력 특징 데이터에 중요도 점수를 할당합니다
# 
# 우리의 경우에는 출력 벡터의 특정 요소, 
# 즉 선택한 범주에 대한 모델의 신뢰도를 나타내는 요소를 취하고 
# 입력 이미지의 어떤 부분이 이 출력에 기여했는지를 이해하기 위해 
# 통합 그라디언트를 사용할 것입니다.
# 
# 통합 그라디언트에서 중요도 맵(the importance map)을 가져오면 
# Captum의 시각화 도구를 사용하여 중요도 맵을 유용하게 표현할 수 있습니다.
# Captum의 ``visualize_image_attr()`` 함수는 속성 데이터의 표시를 원하는대로 만들 수 있는 다양한 옵션을 제공합니다.
# 여기서는 Matplotlib 색상 맵을 사용하겠습니다.
# 
# "integrated_gradients.attribute()" 호출을 사용하여 셀을 실행하는 데는 보통 1-2분 정도 걸립니다.
# 

# 모델을 사용하여 속성 알고리즘 초기화
integrated_gradients = IntegratedGradients(model)

# 알고리즘에 출력 대상을 다음과 같이 지정하도록 요청합니다 
attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)

# 비교를 위해 원본 이미지 표시
_ = viz.visualize_image_attr(None, np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)), 
                      method="original_image", title="Original Image")

default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=256)

_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             title='Integrated Gradients')


#######################################################################
# 위의 이미지에서 통합 그래디언트(Integrated Gradients)는 이미지에서 고양이 위치 주변에서 가장 강력한 신호를 제공합니다.
#


##########################################################################
# 폐색(Occlusion)에 따른 특징 속성
#
# ----------------------------------
# 
# 그래디언트 기반 속성 방법은 입력에 대한 출력 변화를 직접 계산한다는 측면에서 모델을 이해하는 데 도움이 됩니다.
#
# *퍼터베이션(Perturbation) 기반 속성* 방법은 출력에 미치는 영향을 측정하기 위해 
# 입력에 변화를 도입함으로써 이를 보다 직접적으로 접근합니다.
# 'Occlusion <https://captum.ai/api/occlusion.html >'_' 도 그러한 방법 중 하나입니다.
# 입력 영상의 단면을 교체하고 출력 신호에 미치는 영향을 조사하는 작업이 포함됩니다.
# 
# 아래에서는 Occlusion 속성을 설정합니다. 
# 컨볼루션 신경망을 구성하는 것과 유사하게 대상 영역의 크기와 개별 측정의 간격을 결정하는 스트라이드 길이를 지정할 수 있습니다.
# Occlusion 속성의 출력을 ``visualize_image_attr_multiple()`` 로 시각화하고, 
# 영역별 양과 음의 속성 모두의 히트맵을 보여주며, 원래 이미지를 양의 속성 영역으로 마스킹하여 시각화합니다.
#
# 
# 마스킹은 모델이 가장 "고양이와 유사한" 것으로 판단한 고양이 사진의 영역을 잘 보여줍니다.
#

occlusion = Occlusion(model)

attributions_occ = occlusion.attribute(input_img,
                                       target=pred_label_idx,
                                       strides=(3, 8, 8),
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)


_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map", "heat_map", "masked_image"],
                                      ["all", "positive", "negative", "positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                      fig_size=(18, 6)
                                     )


######################################################################
# 고양이를 포함하는 이미지의 영역에 더 큰 의미를 부여하는 것을 다시 한 번 확인할 수 있습니다.
#
# 


#########################################################################
# Layer GradCAM을 이용한 계층 속성(Layer Attribution)
#
# ------------------------------------
# 
# **계층 속성**을 사용하면 모델 내 은닉층의 활동을 입력 데이터의 특징으로 파악할 수 있습니다. 
# 아래에서는 계층 속성 알고리즘을 사용하여 모델 내 컨볼루션 계층 중 하나의 활동을 검토해보겠습니다.
# 
# GradCAM은 지정된 레이어에 대한 타겟 출력의 그래디언트를 계산하고 
# 각 출력 채널에 대한 평균(출력의 차원 2)을 계산하고 
# 각 채널에 대한 평균 그래디언트를 레이어 활성화에 곱합니다.
# 
# 결과는 모든 채널에 걸쳐 합산됩니다. 
# GradCAM은 convnets을 위해 설계되었습니다. 
# 컨볼루션 계층의 활동은 종종 입력 데이터에 공간적으로 매핑되기 때문에 
# GradCAM 속성은 종종 업샘플링되어 입력 데이터를 마스킹하는 데 사용됩니다.
#
# 계측 속성은 입력 속성과 유사하게 설정됩니다. 
# 다만, 모델에 검토하고자 하는 은닉층을 특정(specify)해야 합니다. 
# 위와 같이, ``attribute()`` 를 호출 할 때 우리는 관심있는 목표 클래스를 특정합니다.
# 
# 

layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)

_ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                             sign="all",
                             title="Layer 3 Block 1 Conv 2")


##########################################################################
# 입력 이미지와 비교하려면 속성 데이터를 업샘플링해야 하기 때문에
# 편리한 메소드인 ``interpolate()``를  사용합니다.
# 이 메소드는 `LayerAttribution <https://captum.ai/api/base_classes.html?highlight=layerattribution#captum.attr.LayerAttribution>`__ base 클래스에 있습니다.
# 
# 

upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])

print(attributions_lgc.shape)
print(upsamp_attr_lgc.shape)
print(input_img.shape)

_ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                      transformed_img.permute(1,2,0).numpy(),
                                      ["original_image","blended_heat_map","masked_image"],
                                      ["all","positive","positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Masked"],
                                      fig_size=(18, 6))


#######################################################################
# 이와 같은 시각화를 통해 숨겨진 계층이 입력에 어떻게 반응하는지에 대한 새로운 통찰력을 얻을 수 있습니다.
#
# 


##########################################################################
# Captum Insights를 이용한 시각화
# ----------------------------------
# 
# Captum Insights는 모델 이해를 돕기 위해 Captum 기반으로 제작된 모델 해석력 시각화 위젯입니다.
# Captum Insights는 이미지, 텍스트 등 전반에 걸쳐 작동하여 사용자가 특징 속성(feature attribution)을 
# 이해하는 데 도움을 줍니다.
# 여러 입력/출력 쌍에 대한 속성을 시각화할 수 있으며 이미지, 텍스트 및 임의의 데이터에 대한 시각화 도구를 제공합니다.
# 
# 이 섹션에서는 Captum Insights를 사용하여 여러가지 이미지 분류 추론을 시각화해 보겠습니다.
# 
# 먼저 이미지를 모아 모델이 어떻게 생각하는지 알아보겠습니다.
# 다양성을 위해 고양이, 찻주전자, 삼엽충 화석을 사용하겠습니다:
#
#  

imgs = ['img/cat.jpg', 'img/teapot.jpg', 'img/trilobite.jpg']

for img in imgs:
    img = Image.open(img)
    transformed_img = transform(img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '/', pred_label_idx.item(), ' (', prediction_score.squeeze().item(), ')')


##########################################################################
# …그리고 모델이 이 모든 것들을 정확하게 식별하고 있는 것처럼 보입니다(물론 더 깊이 확인해보고 싶긴 합니다)
# 이를 위해 우리는 Captum Insights 위젯을 사용할 것입니다.  
# ``AttributionVisualizer`` 개체를 만들고 아래와 같이 불러올 것입니다.
# ``AttributionVisualizer`` 개체는 배치 데이터(batches of data)를 필요로 하기 때문에 Captum의 ``Batch`` 헬퍼 클래스를 사용하겠습니다.  
# 그리고 우리는 이미지를 구체적으로 살펴보기 위해 ``ImageFeature`` 도 불러오도록 하겠습니다.
# 
#
# ``AttributionVisualizer``은 아래의 요소들로 구성합니다:
# 
# -  검사할 모델의 배열 (튜토리얼 예제의 경우 한 개의 모델)
# -  Captum Insights가 모형에서 상위 k 예측을 끌어낼 수 있는 스코어링 함수
# -  모델로 학습된 클래스 목록(사람이 읽을 수 있으며 순서대로 정렬되어 있어야 함)
# -  찾고 싶은 특징 목록 (튜토리얼 예제의 경우 ``ImageFeature``)
# -  데이터셋은 입력 및 레이블의 배치(batch)를 반환하는 반복 가능한(iterable) 개체로, 
#    학습에 사용하는 것과 같은 것을 사용
#
# 

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

# 기준은 모두 0으로 입력합니다. (데이터에 따라 다를 수 있습니다)
def baseline_func(input):
    return input * 0

# 위에서 변환한 이미지와 병합(merging)하기
def full_img_transform(input):
    i = Image.open(input)
    i = transform(i)
    i = transform_normalize(i)
    i = i.unsqueeze(0)
    return i


input_imgs = torch.cat(list(map(lambda i: full_img_transform(i), imgs)), 0)

visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=list(map(lambda k: idx_to_labels[k][1], idx_to_labels.keys())),
    features=[
        ImageFeature(
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[],
        )
    ],
    dataset=[Batch(input_imgs, labels=[282,849,69])]
)


#########################################################################
# 위의 셀을 실행하는 데는 위의 속성과 달리 시간이 많이 걸리지 않았습니다. 
# Captum Insight를 사용하면 시각적 위젯에서 서로 다른 속성 알고리즘을 구성할 수 있기 때문에 속성을 계산하고 표시할 수 있습니다. 
# *그 과정은 몇 분 정도 소요됩니다.

# 
# 아래 셀을 실행하면 Captum Insight 위젯이 렌더링됩니다. 
# 그런 다음 속성 방법과 그 인수(arguments)를 선택하고, 예측 클래스 또는 예측 정확도에 따라 모델 응답을 필터링하고, 
# 연관된 확률로 모델의 예측을 보고, 원래 이미지와 비교하여 속성의 히트맵을 볼 수 있습니다.

# 

visualizer.render()
