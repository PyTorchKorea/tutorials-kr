# -*- coding: utf-8 -*-

"""
사용자 정의 Dataset, Dataloader, Transforms 작성하기
==========================================================

**저자** : Sasank Chilamkurthy <https://chsasank.github.io>
**번역** : 정윤성 <https://github.com/Yunseong-Jeong>

머신러닝 문제를 푸는 과정에서 데이터를 준비하는데 많은 노력이 필요합니다.
PyTorch는 데이터를 불러오는 과정을 쉽게해주고, 또 잘 사용한다면 코드의 가독성도 보다 높여줄 수 있는 도구들을
제공합니다. 이 튜토리얼에서 일반적이지 않은 데이터셋으로부터 데이터를 읽어오고
전처리하고 증가하는 방법을 알아보겠습니다.

이번 튜토리얼을 진행하기 위해 아래 패키지들을 설치해주세요.

-  ``scikit-image``: 이미지 I/O 와 변형을 위해 필요합니다.
-  ``pandas``: CSV 파일 파싱을 보다 쉽게 해줍니다.


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

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # 반응형 모드

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
#     이 `링크 <https://download.pytorch.org/tutorial/faces.zip>`_ 를 통해 데이터셋을 다운로드 해주세요.
#     다운로드한 데이터셋은 'data/faces/'에 위치해야 합니다.
#     이 데이터셋은 ImageNet에서 '얼굴'이라는 태그를 가진 몇몇 이미지들에 대해
#     `dlib의 pose estimation <https://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ 을
#     적용한 데이터셋입니다.
#
#
# 데이터셋은 아래와 같은 특징을 가진 CSV 파일이 포함되어 있습니다.
#
# ::
#
#     image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
#     0805personali01.jpg,27,83,27,98, ... 84,134
#     1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
#
# 이제 CSV에서 이미지 이름과 그에 해당하는 데이터(annotation)을 가져와 보겠습니다. 예시로 person-7.jpg가 있는
# 65번째 줄(row index number)을 가져오겠습니다.이미지 이름을 읽어 ``img_name`` 에 저장하고, 데이터는 (L, 2)
# 배열인 ``landmarks`` 에 저장합니다. 이 때 L은 해당 행의 랜드마크의 개수입니다.

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
# 이미지와 랜드마크(landmark)를 보여주는 간단한 함수를 작성해보고,
# 실제로 적용해보겠습니다.
#
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    """ 랜드마크(landmark)와 이미지를 보여줍니다. """
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # 갱신이 되도록 잠시 멈춥니다.

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
plt.show()


######################################################################
# Dataset 클래스
# ----------------
#
# ``torch.utils.data.Dataset`` 은 데이터셋을 나타내는 추상클래스입니다.
# 여러분의 데이터셋은 ``Dataset`` 에 상속하고 아래와 같이 오버라이드 해야합니다.
#
# -  ``len(dataset)`` 에서 호출되는 ``__len__`` 은 데이터셋의 크기를 리턴해야합니다.
# -  ``dataset[i]`` 에서 호출되는 ``__getitem__`` 은
#    :math:`i`\ 번째 샘플을 찾는데 사용됩니다.
#
# 이제 데이터셋 클래스를 만들어보도록 하겠습니다.
# ``__init__`` 을 사용해서 CSV 파일 안에 있는 데이터를 읽지만,
# ``__getitem__`` 을 이용해서 이미지의 판독을 합니다.
# 이 방법은 모든 이미지를 메모리에 저장하지 않고 필요할때마다 읽기 때문에
# 메모리를 효율적으로 사용합니다.
#
# 데이터셋의 샘플은  ``{'image': image, 'landmarks': landmarks}`` 의 사전 형태를 갖습니다.
# 선택적 인자인 ``transform`` 을 통해 필요한 전처리 과정을 샘플에 적용할 수 있습니다.
# 다음 장에서 전이 ``transform`` 의 유용성에 대해 알아보겠습니다.
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
# 위에서 볼 수 있었던 한가지 문제점은 샘플들이 다 같은 크기가 아니라는 것입니다.
# 대부분의 신경망(neural networks)은 고정된 크기의 이미지라고 가정합니다.
# 그러므로 우리는 신경망에 주기 전에 처리할 과정을 작성해야 합니다.
#
# 3가지의 transforms 을 만들어 봅시다:
#
# - ``Rescale``: 이미지의 크기를 조절합니다.
# - ``RandomCrop``: 이미지를 무작위로 자릅니다.
#   이것을 data augmentation이라 합니다.
# - ``ToTensor``: numpy 이미지에서 torch 이미지로 변경합니다.
#   (축변환이 필요합니다)
#
# 간단한 함수대신에 호출 할 수 있는 클래스로 작성 합니다.
# 이렇게 한다면, 클래스가 호출 될 때마다 전이(Transform)의 매개변수가 전달 되지 않아도 됩니다.
# 이와 같이, ``__call__`` 함수를 구현해야 합니다.
# 필요하다면, ``__init__`` 함수도 구현해야 합니다. 다음과 같이 전이(transform)를 사용할 수 있습니다.
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# 아래에서는 이미지와 랜드마크(landmark)들을 어떻게 적용하는지 살펴보도록 하겠습니다.

class Rescale(object):
    """주어진 크기로 샘플크기를 조정합니다.

    Args:
        output_size(tuple or int) : 원하는 출력 크기가
            tuple인 경우 해당 tuple(output_size)이 결과물(output)의 크기가 되고,
            int라면 비율을 유지하면서, 길이가 작은 쪽이 output_size가 됩니다.
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
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

######################################################################
# .. note::
#     위 예시에서, `RandomCrop` 은 외부 라이브러리의 난수 생성기(random number generator; 이 경우, Numpy의 `np.random.int` )를
#     사용하고 있습니다. 이는 `DataLoader` 가 예상치 못한 동작을 하도록 할 수 있습니다.
#     (https://pytorch.org/docs/stable/notes/faq.html#my-data-loader-workers-return-identical-random-numbers 를 참고하세요)
#     실제 상황에서는 `torch.randint` 와 같은 PyTorch가 제공하는 난수 생성기를 사용하는 것이 안전합니다.

######################################################################
# Compose transforms
# ~~~~~~~~~~~~~~~~~~
#
# 이제, 샘플에 전이(transform)를 적용해 봅시다.
#
# 이미지의 가장 짧은 측면을 256개로 rescale하고,
# 그후에 무작위로 224개를 자른다고 가정합시다.
# 다시말해, ``Rescale`` 과 ``RandomCrop`` 을 사용해봅시다.
#
# ``torchvision.transforms.Compose`` 는 위의 두작업을 하는 간단한 호출할 수 있는 클래스입니다.
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
# 전이(transform)를 적용한 dataset을 만들기위해서 만들었던것을 다 집어 넣어 봅시다.
#
# 요약하자면, 데이터셋은 다음과 같이 샘플링 됩니다.
#
# -  이미지는 파일 전체를 메모리에 올리지않고 필요할때마다 불러와서 읽습니다.
# -  그 후에 읽은 이미지에 Transform을 적용합니다.
# -  transfroms 중 하나가 랜덤이기 때문에, 데이터는 샘플링때 증가합니다.
#
#
# 우리는 이제 이전에 사용하던 것 처럼 ``for i in range`` 를 사용해서
# 생성된 데이터셋을 반복 작업에 사용할 수 있습니다.
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
# 그러나, 데이터 상에서 반복하는 ``for`` 문은 많은 특징(features)를 놓칠 수 있습니다.
# 특히, 아래와 같은 것을 놓칠 수 있습니다:
#
# -  데이터를 묶는 과정
# -  데이터를 섞는 과정
# -  병렬처리 과정에서 ``multiprocessing`` 을 사용할때 데이터를 불러오는 것
#
# ``torch.utils.data.DataLoder`` 는 위와 같은 기능을 모두 제공해주는 반복자(iterator)입니다.
# 사용되는 매개변수(Parameters)는 명확해야 합니다.
# ``collate_fn`` 는 흥미로운 매개변수(Parameters) 중 하나입니다.
# ``collate_fn`` 을 이용하여 샘플들을 정확하게 배치하는 방법을 명시할 수 있습니다.
# 그러나, 대부분의 경우에 대해서 정확하게 작동해야 합니다.

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)


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

# Windows를 사용 중이라면, 다음 줄의 주석을 제거하고 for 반복문을 들여쓰기합니다.
# "num_workers"를 0으로 변경해야 할 수도 있습니다.

# if __name__ == '__main__':
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
# 이번 튜토리얼에서는, 데이터셋 작성과 사용, 전이(transforms), 데이터를 불러오는 방법에 대해서 알아봤습니다.
# ``torchvision`` 패키지는 몇몇의 일반적인 데이터셋과 전이(transforms)들을 제공합니다.
# 클래스들을 따로 작성하지 않아도 될 것입니다.
# torchvision에서의 사용가능한 일반적인 데이터셋 중 하나는 ``ImageFolder`` 입니다.
# 이것은 다음과 같은 방식으로 구성되어 있다고 가정합니다: ::
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
# 여기서'ants', 'bees'는 class labels입니다.
# 비슷하게, ``RandomHorizontalFlip`` , ``Scale`` 과 같이  ``PIL.Image`` 에서 작동하는
# 일반적인 전이(transforms)도 사용가능합니다. 이와 같이 데이터로더(dataloader)를 사용할 수 있습니다: ::
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
#  :doc:`transfer_learning_tutorial` 문서를 참고해주세요
