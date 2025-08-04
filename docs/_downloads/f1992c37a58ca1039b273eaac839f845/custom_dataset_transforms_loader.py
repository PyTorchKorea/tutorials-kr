"""
사용자 정의 PyTorch Dataloader 작성하기
===========================================================

머신러닝 알고리즘을 개발하기 위해서는 데이터 전처리에 많은 노력이 필요합니다. PyTorch는 데이터를 로드하는데 쉽고 가능하다면
더 좋은 가독성을 가진 코드를 만들기위해 많은 도구들을 제공합니다. 이 레시피에서는 다음 세 가지를 배울 수 있습니다.

 1. PyTorch 데이터셋 API들을 이용하여 사용자 정의 데이터셋 만들기.
 2. 구성가능하며 호출 될 수 있는 사용자 정의 transform 만들기.
 3. 이러한 컴포넌트들을 합쳐서 사용자 정의 dataloader 만들기.

이 튜토리얼을 실행하기 위해서는 다음의 패키지들이 설치 되었는지 확인해 주세요.
 -  ``scikit-image``: 이미지 I/O와 이미지 변형에 필요합니다.
 -  ``pandas``: CSV를 더 쉽게 파싱하기 위해 필요합니다.

작성되고 있는 이 시점에서, 이 레시피는 `Sasank Chilamkurthy <https://chsasank.github.io>`__ 의 오리지널 튜토리얼을 바탕으로 하며
나중에는 `Joe Spisak <https://github.com/jspisak>`__ 에 의해 수정되었습니다.
한국어로 `Jae Joong Lee <https://https://github.com/JaeLee18>`__ 에 의해 번역되었습니다.
"""


######################################################################
# 설정
# ----------------------
#
# 먼저 이 레시피에 필요한 모든 라이브러리들을 불러오도록 하겠습니다.
#
#

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # 반응형 모드 설정


######################################################################
# 첫 번째: 데이터셋
# ----------------------
#

######################################################################
#
# 우리가 다룰 데이터셋은 얼굴 포즈입니다.
# 전반적으로, 한 얼굴에는 68개의 랜드마크들이 표시되어 있습니다.
#
# 다음 단계로는, `여기 <https://download.pytorch.org/tutorial/faces.zip>`_ 에서 
# 데이터셋을 다운 받아 이미지들이 ‘data/faces/’ 의 경로에 위치하게 해주세요.
#
# **알림:** 사실 이 데이터셋은 imagenet 데이터셋에서 ‘face’ 태그를 포함하고 있는 이미지에
# `dlib` 의 포즈 예측 `<https://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`__ 을 적용하여 생성하였습니다.
# 
# ::
#
#    !wget https://download.pytorch.org/tutorial/faces.zip
#    !mkdir data/faces/
#    import zipfile
#    with zipfile.ZipFile("faces.zip","r") as zip_ref:
#    zip_ref.extractall("/data/faces/")
#    %cd /data/faces/



######################################################################
# 이 데이터셋은 다음과 같은 설명이 달려있는 CSV파일이 포함되어 있습니다.
#
# ::
#
#      image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
#      0805personali01.jpg,27,83,27,98, ... 84,134
#      1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
#
# 이제 CSV파일을 빠르게 읽고 파일 안에 있는 설명들은 (N, 2) 배열로 읽어봅시다.
# 여기서 N은 랜드마크의 갯수입니다.
#


landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

######################################################################
# 1.1 이미지를 표시하기 위해 간단한 도움 함수 작성하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 다음으로는 이미지를 보여주기 위해 간단한 도움 함수를 작성하여 이미지가 가지고 있는 랜드마크들과
# 이미지 샘플을 보여주도록 하겠습니다.
#

def show_landmarks(image, landmarks):
    """ 랜드마크와 함께 이미지 보여주기 """
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  #  잠시 멈추어 도표가 업데이트 되게 합니다

plt.figure()
show_landmarks(io.imread(os.path.join('faces/', img_name)),
               landmarks)
plt.show()


######################################################################
# 1.2 데이터셋 클래스 만들기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 이제 PyTorch 데이터셋 클래스에 대해 알아봅시다.
#
#


######################################################################
# ``torch.utils.data.Dataset`` 은 추상 클래스로서 데이터셋을 맡고 있습니다
# ``Dataset`` 을 상속받아야 하며 다음의 메소드들을 오버라이드 해야합니다.
#
# -  ``__len__`` 에는 ``len(dataset)`` 데이터셋의 사이즈를 반환합니다.
# -  ``__getitem__`` 는 이러한 인덱싱을 지원하고 ``dataset[i]`` 
#     :math:``i`` 번째 샘플을 얻기 위해 사용됩니다.
#
# 우리의 얼굴 랜드마크 데이터셋을 위한 데이터셋 클래스를 만들어 봅시다.
# 우리는 csv파일은 ``__init__`` 에서 읽고 이미지들은 ``__getitem__`` 에서 읽도록 남겨두겠습니다.
# 이러한 방법은 메모리를 효율적으로 사용하도록 하는데 그 이유는 모든 이미지를 한 번에 메모리에 저장하지 않고
# 필요할 때마다 불러오기 때문입니다.
#
# 우리 데이터셋의 샘플은 dict 형태로 이렇게 ``{'image': image, 'landmarks': landmarks}`` 되어있습니다.
# 데이터셋은 선택적 매개변수인 ``transform`` 을 가지고 있어서
# 필요한 프로세싱 어느것이나 샘플에 적용 될 수 있습니다.
# ``transform`` 이 얼마나 유용한지는 다른 레시피에서 확인 해 볼 수 있습니다.
#

class FaceLandmarksDataset(Dataset):
    """ 얼굴 랜드마크 데이터셋. """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        매개변수 :
            csv_file (문자열): 설명이 포함된 csv 파일 경로. 
            root_dir (문자역): 모든 이미지가 있는 폴더 경로.
            transform (호출가능한 함수, 선택적 매개변수): 샘플에 적용 될 수 있는 선택적 변환.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


######################################################################
# 1.3 반복문을 통한 데이터 샘플 사용
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# 다음으로는 이 클래스를 인스턴스화하고 데이터 샘플을 반복문을 이용하여 사용해봅시다. 
# 우리는 첫 4개의 샘플들만 출력하고 그 4개 샘플들의 랜드마크를 보여주겠습니다.
#

face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                    root_dir='faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


######################################################################
# 두 번째: 데이터 변형
# ---------------------------
#


######################################################################
# 우리는 지금까지 어느정도 사용자 정의 데이터셋을 만들어 보았는데 이제는 사용자 정의 변형을 만들 차례 입니다.
# 컴퓨터 비전에서는 사용자 정의 변형은 알고리즘을 일반화시키고 정확도를 올리는데 도움을 줍니다.
# 변형들은 훈련시에 사용이 되며 주로 데이터 증강으로 참조되며 최근의 모델 개발에선 흔히 사용됩니다.
#
# 데이터셋을 다룰때 자주 일어나는 문제중 하나는 모든 샘플들이 같은 크기를 가지고 있지 않을 경우입니다.
# 대부분의 신경망들은 미리 정해진 크기의 이미지들을 받아들입니다.
# 그렇기 때문에 우리는 전처리 코드를 작성해야할 필요가 있습니다.
# 이제 세개의 변형을 만들어 봅시다.
#
# -  ``Rescale``: 이미지 크기를 변경할때 사용됩니다.
# -  ``RandomCrop``: 무작위로 이미지를 잘라내며 데이터 증강에 쓰입니다.
# -  ``ToTensor``: Numpy 이미지들을 파이토치 이미지로 변환할때 사용됩니다. (그러기 위해서는 이미지 차원의 순서를 바꿔야합니다.)
#
# 우리는 위의 세개의 변형들을 단순한 함수 대신에 호출가능한 클래스로 만들어서 매번 변형이 호출될때 항상 매개변수가 넘겨지지 않도록 할겁니다.
# 그러기 위해서는 우리는 단순히 ``__call__`` 메소드를 만들고 필요하다면 ``__init__`` 를 만들면 됩니다.
# 그러면 우리는 변형을 이런식으로 사용할 수 있습니다.
#
# ::
#
#    tsfm = Transform(params)
#    transformed_sample = tsfm(sample)
#
# 어떻게 이런 변형들이 이미지와 랜드마크에 적용이 되었는지 아래를 봐주시길 바랍니다.
#


######################################################################
# 2.1 호출 가능한 클래스들 작성하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 각각의 변형에 맞는 호출 가능한 클래스 작성을 시작해 봅시다.
#
#

class Rescale(object):
    """ 주어진 크기로 샘플안에 있는 이미지를 재변환 합니다.

    Args:
        output_size (tuple 또는 int): 원하는 결과값의 크기입니다.
        tuple로 주어진다면 결과값은 output_size 와 동일해야하며
        int일때는 설정된 값보다 작은 이미지들의 가로와 세로는 output_size 에 적절한 비율로 변환됩니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h 와 w  는 이미지의 랜드마크들 때문에 서로 바뀝니다.
        # x 와 y 축들은 각각 1과 0 값을 가집니다.
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """ 샘플에 있는 이미지를 무작위로 자르기.

    Args:
        output_size (tuple 또는 int): 원하는 결과값의 크기입니다.
        int로 설정하시면 정사각형 형태로 자르게 됩니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """ 샘플 안에 있는 n차원 배열을 Tensor로 변홥힙니다. """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # 색깔 축들을 바꿔치기해야하는데 그 이유는 numpy와 torch의 이미지 표현방식이 다르기 때문입니다.
        # numpy 이미지: H x W x C
        # torch 이미지: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


######################################################################
# 2.2 변환들을 구성하고 샘플에 적용해보기.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 다음에는 작성해왔던 변환들을 구성하고 샘플에 적용해봅시다.
#
#
# 우리가 한 이미지의 가로나 세로중에서 더작은 쪽을 256으로 크기를 바꾸고싶고 
# 바뀐 이미지에서 무작위하게 가로 세로 전부 224로 자르고 싶다고 상황을 가정해봅시다.
# 예를들면, 우리는 ``Rescale`` 과 ``RandomCrop`` 변환을 구성해야 합니다.
# ``torchvision.transforms.Compose`` 는 간단한 호출가능한 클래스로 이러한것들을 우리에게 가능하게 해줍니다.
#

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# 위에 있는 변환들을 각각 샘플에 적용 시킵니다.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()


######################################################################
# 2.3 데이터셋을 반복문을 통해 사용하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 다음으로 우리는 데이터셋을 반복문을 통해 사용해보도록 하겠습니다.
#
#
# 이제 이 모든것을 다 꺼내어서 변환을 구성하고 데이터셋을 만들어봅시다.
# 요약하자면 항상 이 데이터셋을 다음과 같이 불러와집니다.
#
# -  이미지는 읽으려고 할때마다 불러옵니다.
# -  변형들은 읽은 이미지에 적용이 됩니다.
# -  변형들중 하나는 무작위를 이용하기 때문에, 데이터는 샘플링에 따라 증강됩니다.
#
# 저번에 해본것처럼 생성된 데이터셋을 ``for i in range`` 이라는 반복문을 통해 사용할 수 있습니다.
#

transformed_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                           root_dir='faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break


######################################################################
# 세번째: Dataloader
# ----------------------
#


######################################################################
# 직접적으로 데이터셋을 ``for``  반복문으로 데이터를 이용하는건 많은 특성들을 놓칠 수 밖에 없습니다.
# 특히, 우리는 다음과 같은 특성들을 놓친다고 할 수 있습니다.
#
# -  데이터 배치
# -  데이터 섞기
# -  ``multiprocessing`` 를 이용하여 병렬적으로 데이터 불러오기
#
# ``torch.utils.data.DataLoader`` 는 반복자로서 위에 나와있는 모든 특성들을 제공합니다.
# 아래에 제시된 사용되는 매개변수들은 쉽게 이해가 될겁니다. 흥미로운 배개변수는 ``collate_fn`` 인데
# 이것은 정확하게 ``collate_fn`` 을 통해 몇개의 샘플들이 배치가 되어야하는지 지정할 수 있습니다.
# 하지만 굳이 수정하지 않아도 대부분의 경우에는 잘 작동할겁니다.
#

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# 배치를 보여주기위한 도움 함수
def show_landmarks_batch(sample_batched):
    """ 샘플들의 배치에서 이미지와 함께 랜드마크를 보여줍니다. """
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # 4번째 배치를 보여주고 반복문을 멈춥니다.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


######################################################################
# 이제 PyTorch를 이용해서 어떻게 사용자 정의 dataloader를 만드는지 배웠습니다.
# 저희는 좀 더 관련된 문서들을 깊게 읽으셔서 더욱 맞춤화된 작업 흐림을 가지길 추천 드립니다.
# 더 배워보시려면 ``torch.utils.data`` 문서를 `여기 <https://pytorch.org/docs/stable/data.html>`__ 에서 읽어 보실 수 있습니다.
