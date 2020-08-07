"""
사용자 정의 PyTorch dataloader 작성하기
=====================================

머신러닝 알고리즘을 개발하기 위해서는 데이터 전처리에 많은 노력이 필요합니다. PyTorch는 데이터를 로드하는데 쉽고 가능하다면
더 좋은 가독성을 가진 코드를 만들기위해 많은 도구들을 제공합니다. 이 Recipe에서는 다음 세 가지를 배울 수 있습니다.

 1. PyTorch 데이터셋 API들을 이용하여 사용자 정의 데이터셋 만들기.
 2. 구성가능하며 호출 될 수 있는 사용자 정의 transform 만들기.
 3. 이러한 컴포넌트들 합쳐서 사용자 정의 dataloader 만들기.

이 튜토리얼을 실행하기 위해서는 다음의 패키지들이 설치 되었는지 확인해 주세요.
 -  ``scikit-image``: 이미지 I/O와 이미지 변형에 필요합니다.
 -  ``pandas``: CSV를 더 쉽게 파싱하기 위해 필요합니다.

작성되고 있는 이 시점에서, 이 레시피는 `Sasank Chilamkurthy <https://chsasank.github.io>`__ 의 오리지널 튜토리얼을 기반으로
나중에는 `Joe Spisak <https://github.com/jspisak>`__ 에 의해 수정되었습니다.
"""


######################################################################
# 설정
# ----------------------
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
# -------------------
#

######################################################################
# 우리가 다룰 데이터셋은 얼굴 포즈입니다.
# 전반적으로, 한 얼굴에는 68개의 랜드마크들이 표시되어 있습니다.
#
# 다음 단계로는, `여기 <https://download.pytorch.org/tutorial/faces.zip>`_ 에서 
# 데이터셋을 다운 받아 이미지들이 ‘data/faces/’ 의 경로에 위치하게 해주세요.
#
# **알림:** 사실 이 데이터셋은 imagenet 데이터셋에서 ‘face’ 태그를 포함하고 있는 이미지에
# `dlib` 의 포즈 예측 <https://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ 을 적용하여 생성하였습니다.
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
# 이 데이터셋은 다음과 같은 주석이 달려있는 CSV파일이 포함되어 있습니다.
#
# ::
#
#      image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
#      0805personali01.jpg,27,83,27,98, ... 84,134
#      1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
#
# 이제 CSV파일을 빠르게 읽고 파일 안에 있는 주석들은 (N, 2) 배열로 읽어봅시다.
# 여기서 N은 랜드마크의 갯수입니다.
#


landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

######################################################################
# 1.1 이미지를 표시하기 위해 간단한 헬퍼 함수 작성하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 다음으로는 이미지를 보여주기위해 간단한 헬퍼 함수를 작성하여 이미지가 가지고 있는 랜드마크들과
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
# 이제 파이토치 데이터셋 클래스에 대해 알아봅시다.
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
# 우리는 csv파일을 ``__init__`` 에서 읽고 이미지들은 ``__getitem__`` 에서 읽도록 남겨주세요.
# 이러한 방법은 메모리를 효율적으로 사용 하도록 하는데 그 이유는 모든 이미지들을 한번에 메모리에 저장하지않고
# 필요할때마다 불러오게 됩니다.
#
# 우리 데이터셋의 샘플은 dict 형태로 이렇게``{'image': image, 'landmarks': landmarks}`` 되어있습니다.
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
# 1.3 Iterate through data samples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Next let’s instantiate this class and iterate through the data samples.
# We will print the sizes of first 4 samples and show their landmarks.
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
# Part 2: Data Tranformations
# ---------------------------
#


######################################################################
# Now that we have a dataset to work with and have done some level of
# customization, we can move to creating custom transformations. In
# computer vision, these come in handy to help generalize algorithms and
# improve accuracy. A suite of transformations used at training time is
# typically referred to as data augmentation and is a common practice for
# modern model development.
#
# One issue common in handling datasets is that the samples may not all be
# the same size. Most neural networks expect the images of a fixed size.
# Therefore, we will need to write some prepocessing code. Let’s create
# three transforms:
#
# -  ``Rescale``: to scale the image
# -  ``RandomCrop``: to crop from image randomly. This is data
#    augmentation.
# -  ``ToTensor``: to convert the numpy images to torch images (we need to
#    swap axes).
#
# We will write them as callable classes instead of simple functions so
# that parameters of the transform need not be passed everytime it’s
# called. For this, we just need to implement ``__call__`` method and if
# required, ``__init__`` method. We can then use a transform like this:
#
# ::
#
#    tsfm = Transform(params)
#    transformed_sample = tsfm(sample)
#
# Observe below how these transforms had to be applied both on the image
# and landmarks.
#


######################################################################
# 2.1 Create callable classes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let’s start with creating callable classes for each transform
#
#

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
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

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
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
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


######################################################################
# 2.2 Compose transforms and apply to a sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next let’s compose these transforms and apply to a sample
#
#
# Let’s say we want to rescale the shorter side of the image to 256 and
# then randomly crop a square of size 224 from it. i.e, we want to compose
# ``Rescale`` and ``RandomCrop`` transforms.
# ``torchvision.transforms.Compose`` is a simple callable class which
# allows us to do this.
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
# 2.3 Iterate through the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next we will iterate through the dataset
#
#
# Let’s put this all together to create a dataset with composed
# transforms. To summarize, every time this dataset is sampled:
#
# -  An image is read from the file on the fly
# -  Transforms are applied on the read image
# -  Since one of the transforms is random, data is augmentated on
#    sampling
#
# We can iterate over the created dataset with a ``for i in range`` loop
# as before.
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
# Part 3: The Dataloader
# ----------------------
#


######################################################################
# By operating on the dataset directly, we are losing out on a lot of
# features by using a simple ``for`` loop to iterate over the data. In
# particular, we are missing out on:
#
# -  Batching the data
# -  Shuffling the data
# -  Load the data in parallel using ``multiprocessing`` workers.
#
# ``torch.utils.data.DataLoader`` is an iterator which provides all these
# features. Parameters used below should be clear. One parameter of
# interest is ``collate_fn``. You can specify how exactly the samples need
# to be batched using ``collate_fn``. However, default collate should work
# fine for most use cases.
#

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
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

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


######################################################################
# Now that you’ve learned how to create a custom dataloader with PyTorch,
# we recommend diving deeper into the docs and customizing your workflow
# even further. You can learn more in the ``torch.utils.data`` docs
# `here <https://pytorch.org/docs/stable/data.html>`__.
#
