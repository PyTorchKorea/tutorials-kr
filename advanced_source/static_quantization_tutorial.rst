(베타) PyTorch에서 Eager Mode를 이용한 정적 양자화
=========================================================
**저자**: `Raghuraman Krishnamoorthi <https://github.com/raghuramank100>`_
**편집**: `Seth Weidman <https://github.com/SethHWeidman/>`_, `Jerry Zhang <https:github.com/jerryzh168>`_
**번역**: `김현길 <https://github.com/des00>`_, `Choi Yoonjeong <https://github.com/potatochips178/>`_

이 튜토리얼에서는 어떻게 학습 후 정적 양자화(post-training static quantization)를 하는지 보여주며,
모델의 정확도(accuracy)을 더욱 높이기 위한 두 가지 고급 기술인 채널별 양자화(per-channel quantization)와
양자화 자각 학습(quantization-aware training)도 살펴봅니다. 현재 양자화는 CPU만 지원하기에,
이 튜토리얼에서는 GPU/ CUDA를 이용하지 않습니다.
이 튜토리얼을 끝내면 PyTorch에서 양자화가 어떻게 속도는 향상시키면서 모델 사이즈를 큰 폭으로 줄이는지
확인할 수 있습니다. 게다가 `여기 <https://arxiv.org/abs/1806.08342>`_ 에 소개된 몇몇 고급 양자화 기술을
얼마나 쉽게 적용하는지도 볼 수 있고, 이런 기술들이 다른 양자화 기술들보다 모델의 정확도에 부정적인 영향을
덜 끼치는 것도 볼 수 있습니다.

주의: 다른 PyTorch 저장소의 상용구 코드(boilerplate code)를 많이 사용합니다.
예를 들어 ``MobileNetV2`` 모델 아키텍처 정의, DataLoader 정의 같은 것들입니다.
물론 이런 코드들을 읽는 것을 추천하지만, 양자화 특징만 알고 싶다면
"4. 학습 후 정적 양자화" 부분으로 넘어가도 됩니다.
필요한 것들을 import 하는 것부터 시작해 봅시다:

.. code:: python

    import os
    import sys
    import time
    import numpy as np

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    import torchvision
    from torchvision import datasets
    import torchvision.transforms as transforms

    # # warnings 설정
    import warnings
    warnings.filterwarnings(
        action='ignore',
        category=DeprecationWarning,
        module=r'.*'
    )
    warnings.filterwarnings(
        action='default',
        module=r'torch.ao.quantization'
    )

    # 반복 가능한 결과를 위한 랜덤 시드 지정하기
    torch.manual_seed(191009)

1. 모델 아키텍처
---------------------

처음으로 MobileNetV2 모델 아키텍처를 정의합니다.
이 모델은 양자화를 위한 몇 가지 중요한 변경사항들이 있습니다:

- 덧셈을 ``nn.quantized.FloatFunctional`` 으로 교체
- 신경망의 처음과 끝에 ``QuantStub`` 및 ``DeQuantStub`` 삽입
- ReLU를 ReLU6로 교체

알림: `여기 <https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py>`_ 에서
이 코드를 가져왔습니다.

.. code:: python

    from torch.ao.quantization import QuantStub, DeQuantStub

    def _make_divisible(v, divisor, min_value=None):
        """
        이 함수는 원본 TensorFlow 저장소에서 가져왔습니다.
        모든 계층이 8로 나누어지는 채널 숫자를 가지고 있습니다.
        이곳에서 확인 가능합니다:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # 내림은 10% 넘게 내려가지 않는 것을 보장합니다.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


    class ConvBNReLU(nn.Sequential):
        def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
            padding = (kernel_size - 1) // 2
            super(ConvBNReLU, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.1),
                # ReLU로 교체
                nn.ReLU(inplace=False)
            )


    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio):
            super(InvertedResidual, self).__init__()
            self.stride = stride
            assert stride in [1, 2]

            hidden_dim = int(round(inp * expand_ratio))
            self.use_res_connect = self.stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # pw
                layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, momentum=0.1),
            ])
            self.conv = nn.Sequential(*layers)
            # torch.add를 floatfunctional로 교체
            self.skip_add = nn.quantized.FloatFunctional()

        def forward(self, x):
            if self.use_res_connect:
                return self.skip_add.add(x, self.conv(x))
            else:
                return self.conv(x)


    class MobileNetV2(nn.Module):
        def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
            """
            MobileNet V2 메인 클래스
            Args:
                num_classes (int): 클래스 숫자
                width_mult (float): 넓이 multiplier - 이 수를 통해 각 계층의 채널 개수를 조절
                inverted_residual_setting: 네트워크 구조
                round_nearest (int): 각 계층의 채널 숫를 이 숫자의 배수로 반올림
                1로 설정하면 반올림 정지
            """
            super(MobileNetV2, self).__init__()
            block = InvertedResidual
            input_channel = 32
            last_channel = 1280

            if inverted_residual_setting is None:
                inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ]

            # 사용자가 t,c,n,s를 필요하다는 것을 안다는 전제하에 첫 번째 요소만 확인
            if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
                raise ValueError("inverted_residual_setting should be non-empty "
                                 "or a 4-element list, got {}".format(inverted_residual_setting))

            # 첫 번째 계층 만들기
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            features = [ConvBNReLU(3, input_channel, stride=2)]
            # 역전된 잔차 블럭(inverted residual blocks) 만들기
            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            # 마지막 계층들 만들기
            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
            # nn.Sequential로 만들기
            self.features = nn.Sequential(*features)
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
            # 분류기(classifier) 만들기
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

            # 가중치 초기화
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            x = self.quant(x)
            x = self.features(x)
            x = x.mean([2, 3])
            x = self.classifier(x)
            x = self.dequant(x)
            return x

        # 양자화 전에 Conv+BN과 Conv+BN+Relu 모듈 결합(fusion)
        # 이 연산은 숫자를 변경하지 않음
        def fuse_model(self):
            for m in self.modules():
                if type(m) == ConvBNReLU:
                    torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                if type(m) == InvertedResidual:
                    for idx in range(len(m.conv)):
                        if type(m.conv[idx]) == nn.Conv2d:
                            torch.ao.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

2. 헬퍼(Helper) 함수
--------------------

다음으로 모델 평가를 위한 헬퍼 함수들을 만듭니다. 코드 대부분은
`여기 <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_ 에서 가져왔습니다.

.. code:: python

    class AverageMeter(object):
        """평균과 현재 값 계산 및 저장"""
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)


    def accuracy(output, target, topk=(1,)):
        """특정 k값을 위해 top k 예측의 정확도 계산"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    def evaluate(model, criterion, data_loader, neval_batches):
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        cnt = 0
        with torch.no_grad():
            for image, target in data_loader:
                output = model(image)
                loss = criterion(output, target)
                cnt += 1
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                print('.', end = '')
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))
                if cnt >= neval_batches:
                     return top1, top5

        return top1, top5

    def load_model(model_file):
        model = MobileNetV2()
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        model.to('cpu')
        return model

    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

3. Dataset과 DataLoader 정의하기
----------------------------------

마지막 주요 설정 단계로서 학습과 테스트 데이터를 위한 DataLoader를 정의합니다.

ImageNet 데이터
^^^^^^^^^^^^^^^

전체 ImageNet Dataset을 이용해서 이 튜토리얼의 코드를 실행시키기 위해, 첫번째로 `ImageNet Data <http://www.image-net.org/download>`_ 의 지시를 따라 ImageNet을 다운로드합니다. 다운로드한 파일의 압축을 'data_path'에 풉니다.

다운로드받은 데이터를 읽기 위해 아래에 정의된 DataLoader 함수들을 사용합니다.
이런 함수들 대부분은
`여기 <https://github.com/pytorch/vision/blob/master/references/detection/train.py>`_ 에서 가져왔습니다.


.. code:: python

    def prepare_data_loaders(data_path):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset = torchvision.datasets.ImageNet(
            data_path, split="train", transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        dataset_test = torchvision.datasets.ImageNet(
            data_path, split="val", transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size,
            sampler=train_sampler)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=eval_batch_size,
            sampler=test_sampler)

        return data_loader, data_loader_test


다음으로 사전에 학습된 MobileNetV2을 불러옵니다. 모델을 다운로드 받을 수 있는 URL을
`여기 <<https://download.pytorch.org/models/mobilenet_v2-b0353104.pth>>`_ 에서 제공합니다.

.. code:: python

    data_path = '~/.data/imagenet'
    saved_model_dir = 'data/'
    float_model_file = 'mobilenet_pretrained_float.pth'
    scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
    scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

    train_batch_size = 30
    eval_batch_size = 50

    data_loader, data_loader_test = prepare_data_loaders(data_path)
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir + float_model_file).to('cpu')

    # 다음으로 "모듈 결합"을 합니다. 모듈 결합은 메모리 접근을 줄여 모델을 빠르게 만들면서
    # 정확도 수치를 향상시킵니다. 모듈 결합은 어떠한 모델에라도 사용할 수 있지만,
    # 양자화된 모델에 사용하는 것이 특히나 더 일반적입니다.

    print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    float_model.eval()

    # 모듈 결합
    float_model.fuse_model()

    # Conv+BN+Relu와 Conv+Relu 결합에 유의
    print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)


마지막으로 "기준"이 될 정확도를 얻기 위해,
모듈 결합을 사용한 양자화되지 않은 모델의 정확도를 봅시다.

.. code:: python

    num_eval_batches = 1000

    print("Size of baseline model")
    print_size_of_model(float_model)

    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)


전체 모델은 50,000개의 이미지를 가진 eval 데이터셋에서 71.9%의 정확도를 보입니다.

이 값이 비교를 위한 기준이 될 것입니다. 다음으로 양자화된 모델을 봅시다.

4. 학습 후 정적 양자화(post-training static quantization)
--------------------------------------------------------

학습 후 정적 양자화는 동적 양자화처럼 가중치를 float에서 int로 변환하는 것뿐만 아니라
추가적인 단계도 수행합니다. 네트워크에 데이터 배치의 첫 번째 공급과 다른 활성값들의
분포 결과 계산이 이러한 단계입니다. (특히 이러한 추가적인 단계는 계산한 값을
기록하고 싶은 지점에 `observer` 모듈을 삽입합으로써 끝납니다.)
이러한 분포들은 추론 시점에 특정한 다른 활성값들이 어떻게 양자화되어야 하는지 결정하는데 사용됩니다.
(간단한 방법으로는 단순히 활성값들의 전체 범위를 256개의 단계로 나누는 것이지만,
좀 더 복잡한 방법도 제공합니다.) 특히, 이러한 추가적인 단계는 각 연산 사이사이의
양자화된 값을 float으로 변환 - 및 int로 되돌림 - 하는 것뿐만 아니라
양자화된 값을 모든 연산들끼리 주고 받는 것도 가능하게 하여 엄청난 속도 향상이 됩니다.

.. code:: python

    num_calibration_batches = 32

    myModel = load_model(saved_model_dir + float_model_file).to('cpu')
    myModel.eval()

    # Conv, bn과 relu 결합
    myModel.fuse_model()

    # 양자화 설정 명시
    # 간단한 min/max 범위 추정 및 텐서별 가중치 양자화로 시작
    myModel.qconfig = torch.ao.quantization.default_qconfig
    print(myModel.qconfig)
    torch.ao.quantization.prepare(myModel, inplace=True)

    # 첫 번째 보정
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

    # 학습 세트로 보정
    evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # 양자화된 모델로 변환
    torch.ao.quantization.convert(myModel, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(myModel)

    top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

양자화된 모델은 eval 데이터셋에서 56.7%의 정확도를 보여줍니다. 이는 양자화 파라미터를 결정하기 위해 단순 min/max Observer를 사용했기 때문입니다. 그럼에도 불구하고 모델의 크기를 3.6 MB 밑으로 줄였습니다. 이는 거의 4분의 1 로 줄어든 크기입니다.

이에 더해 단순히 다른 양자화 설정을 사용하기만 해도 정확도를 큰 폭으로 향상시킬 수 있습니다.
x86 아키텍처에서 양자화를 위한 권장 설정을 그대로 쓰기만 해도 됩니다.
이러한 설정은 아래와 같습니다:

- 채널별 기본 가중치 양자화
- 활성값을 수집해서 최적화된 양자화 파라미터를 고르는 히스토그램 Observer 사용

.. code:: python

    per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    print(per_channel_quantized_model.qconfig)

    torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
    torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)
    top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)


단순히 양자화 설정 방법을 변경하는 것만으로도 정확도가 67.3%를 넘을 정도로 향상이 되었습니다!
그럼에도 이 수치는 위에서 구한 기준값 71.9%에서 4퍼센트나 낮은 수치입니다.
이제 양자화 자각 학습을 시도해 봅시다.

5. 양자화 자각 학습(Quantization-aware training)
-------------------------------------------------

양자화 자각 학습(QAT)은 일반적으로 가장 높은 정확도를 제공하는 양자화 방법입니다.
모든 가중치화 활성값은 QAT로 인해 학습 도중에 순전파와 역전파를 도중 "가짜 양자화"됩니다.
이는 float값이 int8 값으로 반올림하는 것처럼 흉내를 내지만, 모든 계산은 여전히
부동소수점 숫자로 계산을 합니다. 그래서 결국 훈련 동안의 모든 가중치 조정은 모델이 양자화될
것이라는 사실을 "자각"한 채로 이루어지게 됩니다. 그래서 QAT는 양자화가 이루어지고 나면
동적 양자화나 학습 전 정적 양자화보다 대체로 더 높은 정확도를 보여줍니다.

실제로 QAT가 이루어지는 전체 흐름은 이전과 매우 유사합니다:

- 이전과 같은 모델을 사용할 수 있습니다. 양자화 자각 학습을 위한 추가적인 준비는 필요 없습니다.
- 가중치와 활성값 뒤에 어떤 종류의 가짜 양자화를 사용할 것인지 명시하는 ``qconfig`` 의 사용이 필요합니다.
  Observer를 명시하는 것 대신에 말이죠.

먼저 학습 함수부터 정의합니다:

.. code:: python

    def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
        model.train()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        avgloss = AverageMeter('Loss', '1.5f')

        cnt = 0
        for image, target in data_loader:
            start_time = time.time()
            print('.', end = '')
            cnt += 1
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            avgloss.update(loss, image.size(0))
            if cnt >= ntrain_batches:
                print('Loss', avgloss.avg)

                print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5))
                return

        print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=top1, top5=top5))
        return


이전처럼 모듈을 결합합니다.

.. code:: python

    qat_model = load_model(saved_model_dir + float_model_file)
    qat_model.fuse_model()

    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')

마지막으로 모델이 양자화 자각 학습을 준비하기 위해 ``prepare_qat`` 로 "가짜 양자화"를 수행합니다.

.. code:: python

    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)

높은 정확도의 양자화된 모델을 학습시키기 위해서는 추론 시점에서 정확한 숫자 모델링을 필요로 합니다.
그래서 양자화 자각 학습에서는 학습 루프를 이렇게 변경합니다:

- 추론 수치와 더 잘 일치하도록 학습이 끝날 때 배치 정규화를 이동 평균과 분산을 사용하는 것으로 변경합니다.
- 양자화 파라미터(크기와 영점)를 고정하고 가중치를 미세 조정(fine tune)합니다.

.. code:: python

    num_train_batches = 20

    # QAT는 시간이 걸리는 작업이며 몇 에폭에 걸쳐 훈련이 필요합니다.
    # 학습 및 각 에폭 이후 정확도 확인
    for nepoch in range(8):
        train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
        if nepoch > 3:
            # 양자화 파라미터 고정
            qat_model.apply(torch.ao.quantization.disable_observer)
        if nepoch > 2:
            # 배치 정규화 평균 및 분산 추정값 고정
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # 각 에폭 이후 정확도 확인
        quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))

양자화 자각 학습은 전체 ImageNet 데이터셋에서 71.5%의 정확도를 나타냅니다. 이 값은 기준값 71.9%에 소수점 수준으로 근접한 수치입니다.

양자화 자각 학습에 대한 더 많은 것들:

- QAT는 더 많은 디버깅을 가능하게 하는 학습 후 양자화 기술의 상위 집합입니다.
  예를 들어 모델의 정확도가 가중치나 활성 양자화로 인해 제한을 받아
  더 높아질 수 없는 상황인지 분석할 수 있습니다.
- 부동소수점을 사용한 양자화된 모델을 시뮬레이션 할 수도 있습니다.
  실제 양자화된 연산의 수치를 모델링하기 위해 가짜 양자화를 이용하고 있기 때문입니다.
- 학습 후 양자화 또한 쉽게 흉내낼 수 있습니다.

양자화를 통한 속도 향상
^^^^^^^^^^^^^^^^^^^^^^^^^

마지막으로 위에서 언급한 것들을 확인해 봅시다. 양자화된 모델이 실제로 추론도 더 빠르게 하는 걸까요?
시험해 봅시다:

.. code:: python

    def run_benchmark(model_file, img_loader):
        elapsed = 0
        model = torch.jit.load(model_file)
        model.eval()
        num_batches = 5
        # 이미지 배치들 이용하여 스크립트된 모델 실행
        for i, (images, target) in enumerate(img_loader):
            if i < num_batches:
                start = time.time()
                output = model(images)
                end = time.time()
                elapsed = elapsed + (end-start)
            else:
                break
        num_images = images.size()[0] * num_batches

        print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
        return elapsed

    run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

    run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)

맥북 프로의 로컬 환경에서 일반적인 모델 실행은 61ms, 양자화된 모델 실행은 20ms가 걸렸습니다.
이러한 결과는 부동소수점 모델과 양자화된 모델을 비교했을 때,
양자화된 모델에서 일반적으로 2-4x 속도 향상이 이루어진 것을 보여줍니다.

결론
----------

이 튜토리얼에서 학습 후 정적 양자화와 양자화 자각 학습이라는 두 가지 양자화 방법을 살펴봤습니다.
이 양자화 방법들이 "내부적으로" 어떻게 동작을 하는지와
PyTorch에서 어떻게 사용할 수 있는지도 보았습니다.

읽어주셔서 감사합니다. 언제나처럼 어떠한 피드백도 환영이니, 의견이 있다면
`여기 <https://github.com/pytorch/pytorch/issues>`_ 에 이슈를 남겨 주세요.
