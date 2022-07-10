"""
PyTorch에서 데이터 불러오기
=======================
PyTorch는 인공신경망을 만드는데 필요한 다양한 기본 요소를 간단하고 직관적이며
안정적인 API로 제공합니다. PyTorch는 공용 데이터셋을 쉽게 사용할 수 있도록
도와주는 패키지를 포함하고 있습니다.

개요
------------
PyTorch 데이터 불러오기 기능의 핵심은
`torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__
클래스입니다. 데이터를 파이썬 iterable로써 접근할 수 있게 해주는 클래스입니다.
또한, `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__
클래스를 통해 PyTorch에 내장된 다양한 고품질 데이터셋을 이용하실 수 있습니다.


개별 데이터셋은 아래 패키지에서 확인하실 수 있으며, 데이터셋은 계속해서 추가될 예정입니다.

* `torchvision <https://pytorch.org/vision/stable/datasets.html>`__
* `torchaudio <https://pytorch.org/audio/stable/datasets.html>`__
* `torchtext <https://pytorch.org/text/stable/datasets.html>`__


이번 레시피에서는 ``torchaudio.datasets.YESNO`` 데이터셋을 살펴보면서,
PyTorch ``Dataset`` 에서 PyTorch ``DataLoader`` 로 데이터를 효과적이고 효율적으로
불러오는 방법을 살펴보겠습니다.
"""



######################################################################
# 초기 설정(Setup)
# -----------------------------------------------------------------
# 시작하기 전에, 데이터셋이 포함된 ``torchaudio`` 패키지를 설치합니다.

# pip install torchaudio

######################################################
# Google Colab에서는 아래와 같이 실행합니다:

# !pip install torchaudio

#############################
# 단계(Steps)
# -----------------------------------------------------------------
#
# 1. 데이터를 불러오는데 필요한 라이브러리 import하기
# 2. 데이터 접근하기
# 3. 데이터 불러오기
# 4. 데이터 순회하기
# 5. [선택 사항] 데이터 시각화하기
#
#
# 1. 데이터를 불러오는데 필요한 라이브러리 import하기
# -----------------------------------------------------------------
#
# 이번 레시피는 ``torch`` 와 ``torchaudio`` 를 사용합니다. 다른 내장 데이터셋이
# 필요하다면 ``torchvision`` 혹은 ``torchtext`` 를 설치해서 사용해도 됩니다.
#

import torch
import torchaudio


###########################################################################
# 2. 데이터에 접근하기
# -----------------------------------------------------------------
#
# ``torchaudio`` 의 YesNo 데이터셋은 한 사람이 히브리어로 yes 혹은
# no를 녹음한 오디오 클립 60개로 구성되어 있습니다. 오디오 클립 각각의 길이는 단어 8개입니다.
# ( `더 알아보기 <https://www.openslr.org/1/>`__ ).
#
# ``torchaudio.datasets.YESNO`` 클래스를 사용하여 YesNo 데이터셋을 생성합니다.
torchaudio.datasets.YESNO(
     root='./',
     url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
     folder_in_archive='waves_yesno',
     download=True)

###########################################################################
# 각각의 데이터 항목 (item)은 튜플 형태 (waveform: 파형, sample_rate: 샘플 속도, labels: 라벨)를 갖습니다.
#
# YesNo 데이터셋을 불러올 때 ``root`` 매개변수는 꼭 지정해주셔야 합니다. ``root`` 는
# 학습(training) 및 테스트(testing) 데이터셋이 존재하는 위치를 가르켜야 합니다.
# 그 외의 매개변수는 선택 사항이며, 위 예시에서 기본값을 확인하실 있습니다. 아래와
# 같은 매개변수도 사용 가능합니다.
#
# * ``download``: 참(True)인 경우, 데이터셋 파일을 인터넷에서 다운받고 root 폴더에 저장합니다. 파일이 이미 존재하면 다시 다운받지 않습니다.
#
# 이제 YesNo 데이터를 확인해봅시다:

# YesNo 안에 각각의 데이터 항목은 튜플 형태 (파형, 샘플 속도, 라벨)를 가지며,
# 이때 labels는 0(no)과 1(yes)을 담은 리스트 형태로 되어 있습니다.
yesno_data = torchaudio.datasets.YESNO('./', download=True)

# 실제 데이터에 접근해서 yesno_data의 형태를 확인합니다. 세 번째 항목을 예시로 살펴봅니다.
n = 3
waveform, sample_rate, labels = yesno_data[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))


######################################################################
# 실제 상황에서는 데이터를 "학습(training)" 데이터셋과 "테스트(testing)" 데이터셋으로 나누는 것이
# 권장됩니다. 모델의 성능을 제대로 평가하려면 학습에 쓰이지 않은 out-of-sample
# 데이터를 이용해야 하기 때문입니다.
#
# 3. 데이터 불러오기
# -----------------------------------------------------------------
#
# 데이터셋에 성공적으로 접근했으니, 이제 데이터셋을 ``torch.utils.data.DataLoader`` 로 넘겨줍니다.
# ``DataLoader`` 는 데이터셋을 sampler와 조합시켜 데이터셋을 순회할 수 있는 iterable을 만들어줍니다.
#

data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True)


######################################################################
# 4. 데이터 순회하기
# -----------------------------------------------------------------
#
# 이제 ``data_loader`` 를 이용해서 데이터를 순회할 수 있습니다. 모델을 학습하려면 이처럼
# 데이터를 순회할 수 있어야 합니다. 아래 예시를 보시면 ``data_loader`` 안에 있는 각각의
# 데이터 항목이 파형, 샘플 속도, 라벨을 담은 텐서로 바뀌었음을 확인할 수 있습니다.
#

for data in data_loader:
  print("Data: ", data)
  print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1], data[2]))
  break


######################################################################
# 5. [선택 사항] 데이터 시각화하기
# -----------------------------------------------------------------
#
# ``DataLoader`` 의 데이터를 시각화해서 더 자세히 확인해보실 수 있습니다.
#

import matplotlib.pyplot as plt

print(data[0][0].numpy())

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# 축하드립니다! PyTorch에서 데이터를 불러오는데 성공하셨습니다.
#
# 더 알아보기
# -----------------------------------------------------------------
#
# 다른 레시피를 둘러보고 계속 배워보세요:
#
# - :doc:`/recipes/recipes/defining_a_neural_network`
# - :doc:`/recipes/recipes/what_is_state_dict`
