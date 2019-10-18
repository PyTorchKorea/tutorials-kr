# -*- coding: utf-8 -*-

"""
데이터 로딩 프로세싱 튜토리얼
저자 : Sasank Chilamkurthy <https://chsasank.github.io>
번역 : 정윤성

데이터를 준비하는것은 어떤 머신 러닝 문제를 푸는 과정속에 많은 노력이 필요합니다.
Pytorch는 데이터 로딩을 쉽게 ,  코드를 보다 읽기 편하게 만들어줍니다.
이번 튜토리얼에서는, 일반적이지 않은 데이터에서 로드하는 방법과 
데이터를 전처리/증가시키는 방법에 대해서 알아볼 것입니다.

이번 튜토리얼을 진행하기 위해서, 아래에 있는 패키지를 설치해주시기 바랍니다.

-  ``scikit-image``: 이미지 I/O 와 변형을 위해 필요합니다.
-  ``pandas``: CSV파일 파싱을 보다 쉽게 해줍니다.


"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings 
# 경고 무시해주세요
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode #반응형 모드

######################################################################

######################################################################
# 다룰 데이터셋은 아래 조건과 같은 랜드마크(landmark)가 있는 얼굴 사진입니다.
#
# .. figure:: /_static/img/landmarked_face2.png
#    :width: 400
#
# 각각의 얼굴에 68개의 서로다른 중요 포인트들이 존재합니다.
#
# .. note::
# 주의사항::
#     링크를 통해서 데이터셋을 다운로드 해주세요.<https://download.pytorch.org/tutorial/faces.zip>`_
#     다운로드한 데이터셋은 'data/faces/'에 위치해야 합니다.
#     다운로드 하신 데이터셋은 'facedlib의 pose estimation이 적용된 데이터 셋입니다.
#     <https://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>
#
# 데이터셋은 아래와 같은 특징을 가진 CSV파일이 포함되어 있습니다.
#
# ::
#
#     image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
#     0805personali01.jpg,27,83,27,98, ... 84,134
#     1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
#
# 이제 CSV 파일을 불러와서 (N,2)배열안에 있는 특징들을 잡아보겠습니다.
# N은 랜드마크(landmarks)의 개수입니다.

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))


######################################################################
# 이미지와 랜드마크(landmark)를 보여주는 간단한 함수를 작성해보고, 실제로 적용해보겠습니다.
#
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    """ 랜드마크(landmark)와 이미지를 보여줍니다. """
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # 업데이트가 될 수 있도록 잠시 멈춥니다.

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
plt.show()


######################################################################
# 데이터셋의 클래스입니다.
# -------------
#
# ``torch.utils.data.Dataset`` 은 데이터셋을 나타내는 추상클래스입니다.
# Your custom dataset should inherit ``Dataset`` and override the following
# 여러분의 데이터셋은 ``Dataset``에 상속하고 아래와같이 오버라이드 해야합니다.
# methods:
#
# -  ``len(dataset)``같은 기능을 하는 ``__len__`` 은 데이터셋의 크기를 리턴해야합니다.
# -  ``dataset[i]`` 처럼 인덱스를 찾는것을 지원하는 ``__getitem__``은
#    'i'번째 샘플을 찾는데 사용됩니다.
#
# 이제 데이터셋 클래스를 만들어보도록 하겠습니다.
# ``__init__``을 사용해서 CSV파일 안에 있는 데이터를 읽지만,
# ``__getitem__``을 이용해서 이미지의 판독을 합니다.
# 이방법은 모든 이미지를 메모리에 저장하지 않고 필요할때마다 읽기 때문에 
# 메모리를 효율적으로 사용합니다.
#
# Sample of our dataset will be a dict
# ``{'image': image, 'landmarks': landmarks}``. Our dataset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. We will see the usefulness of ``transform`` in the
# next section.
# 
# 데이터셋의 샘플은  ``{'image': image, 'landmarks': landmarks}`` 형태를 갖습니다.
# 필요한 처리가 샘플에 적용될 수 있도록  Optional argument인 ``transform`` 을 갖습니다
# 다음장에서 전이``transform``를 활용하는 방법에 대해서 알아보겠습니다.
#

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
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
# 클래스를 인스턴스화 하고 데이터 샘플을 통해서 반복해봅시다.
# 첫번째 4개의 샘플의 크기를 출력 하고, 샘플들의 랜드마크(landmarks)를 보여줄 것 입니다.
#

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

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
# Transforms
# ----------
#
# 위에서 볼 수 있었던 한가지 문제점은 샘플들이 다 같은 사이즈가 아니라는 것입니다.
# 대부분의 neural networks 는 고정된 크기의 이미지라고 가정합니다.
# 그러므로 우리는 neural networks에게 주기전에 처리할 과정을 작성해야합니다.
#
# 3가지의 transforms 을 만들어 봅시다:
# -  ``Rescale``: 이미지의 크기를 조절합니다.
# -  ``RandomCrop``: to 이미지를 무작위로 자릅니다. 
#                    이것을 data augmentation이라 합니다.
# -  ``ToTensor``: numpy images에서 torch images로 변경합니다. 
#                  (축변환이 필요합니다)
#
# parameter가 호출될 때 마다 항상 통과될 필요가없도록 
# Transform 함수대신에 부를 수 있는 클래스로 작성 합니다.
# 이것 때문에, ``__call__`` method를 구현할 필요가 있습니다.
# 필요하다면, ``__init__ method 에도 구현해야합니다.
# 그런 후에, transform을 다음과 같이 사용할 수 있습니다.
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)

# 아래에서는 이미지와 특징들을 어떻게 적용하는지 살펴보도록 하겠습니다.

class Rescale(object):
    """주어진 사이즈로 샘플크기를 조정합니다.

    Args:
        output_size(tuple or int) : 얻고자하는 사이즈 값이 출력됩니다.
            만약 tuple 이라면 결과값이 output_size(기대값)과 일치하지만,
            int라면 이미지의 가장자리중 최소값은 종횡비를 유지하기 위해서
            output_size 와 일치합니다.
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

        # 이미지가 x축과 y축이 1과 0이기 때문에
        # h 와 w 의 랜드마크(landmarks)의 위치를 바꿔줍니다.
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """샘플데이터를 무작위로 자릅니다.

    Args:
        output_size (tuple or int): 줄이고자 하는 크기입니다.
                        int라면, 정사각형으로 나올 것 입니다.
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
    """numpy를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


######################################################################
# Compose transforms
# ~~~~~~~~~~~~~~~~~~
#
# 이제, 샘플에 전이(transform)를 적용해 봅시다.
#
# 이미지의 가장 짧은 측면을 256개로 rescale하고, 
# 그후에 무작위로 224개를 자른다고 가정합시다.
# 다시말해, ``Rescale`` 과 ``RandomCrop``을 사용해봅시다.
#
# ``torchvision.transforms.Compose``는 위의 두작업을 하는 간단한 호출할 수 있는 클래스입니다.
#

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
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
# 데이터셋을 이용한 반복작업
# -----------------------------
#
# transforms을 적용한 dataset을 만들기위해서 만들었던것을 다 집어 넣어 봅시다.
#
# 요약하자면, 데이터셋은 다음과 같이 샘플링됩니다.
#
# -  이미지는 파일 전체를 메모리에 올리지않고 필요할때마다 불러와서 읽습니다.
# -  그 후에 읽은 이미지에 Transform을 적용합니다.
# -  transfroms 중 하나가 랜덤이기 때문에, 데이터는 샘플링때 증가합니다.
#
#
# 우리는 이제 이전에 사용하던 것 처럼 for문을 사용해서 
# 생성된 데이터셋을 반복작업에 사용할 수 있습니다.
# 

transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
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
# 그러나, 데이터에``for``문을 이용한 반복을 사용해서 많은 특징들을 놓칠수 있습니다.
# 특히, 아래와 같은것을 놓칠수 있습니다.
#
# -  데이터를 처리하는 과정
# -  데이터를 섞는 과정
# -  병렬처리 과정에서 ``multiprocessing`` 을 사용할때 데이터를 불러오는 것
#
# ``torch.utils.data.DataLoader``는 위의 기능들을 제공하는 반복자(iterartor)입니다.
#  아래에 사용되는 Parameters 들은 명확해야합니다.
#  여러개 중에 ``collate_fn``를 주의깊게 봐야합니다.
#  이것을 이용해서 샘플들이 정확하게 배치하는 방법을 구체화 할 수 있습니다.
#  그러나, 일반적인 결합하는 경우에 대해서는 작동을 정확하게 수행해야합니다.

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# 배치하는 과정을 보여주는 함수입니다.
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

######################################################################
# Afterword: torchvision
# ----------------------
#
# 이번 튜토리얼에서는, 데이터셋과, 트랜스폼(transforms), 데이터로더(dataloader)를
# 작성방법과 사용방법에 대해 알아보았습니다.
#  ``torchvision``패키지는 몇개의 dataset과 transform을 제공합니다.
# 따라서 여러분은 클래스들을 작성해야할 필요는 없습니다.
# ``ImageFolder``는 torchvision 에서 사용할수 있는 일반적인 데이터셋 입니다.
# 다음과 같이 파일이 있다고 가정해 봅시다.: ::
#
#     root/ants/xxx.png
#     root/ants/xxy.jpeg
#     root/ants/xxz.png
#     .
#     .
#     .
#     root/bees/123.jpg
#     root/bees/nsdf3.png
#     root/bees/asd932_.png
#
# 여기서 'ants', 'bees' 는 class labels 입니다.
# Similarly generic transforms 
# which operate on ``PIL.Image`` like  ``RandomHorizontalFlip``, ``Scale``,
# are also available. 
# ``RandomHorizontalFlip``, ``Scale``와 같은``PIL.Image``에서 
# 동작하는 일반적인 전이(transform)는 다 사용가능합니다.
# 아래와 같은 데이터로더(dataloader)를 작성하여 사용할수도 있습니다. ::
#
#   import torch
#   from torchvision import transforms, datasets
#
#   data_transform = transforms.Compose([
#           transforms.RandomSizedCrop(224),
#           transforms.RandomHorizontalFlip(),
#           transforms.ToTensor(),
#           transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#       ])
#   hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                              transform=data_transform)
#   dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                                batch_size=4, shuffle=True,
#                                                num_workers=4)
#
#  training code에 대한 예시를 알고 싶다면, 
#  `transfer_learning_tutorial` 문서를 참고해주세요
