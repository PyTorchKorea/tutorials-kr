"""
Torchaudio를 사용한 음성 명령 
******************************************

이 튜토리얼에서는 오디오 데이터셋을 올바르게 포맷한 다음
데이터셋의 오디오 분류기 네트워크를 train/test 하는 방법을 보여줍니다.

Colab은 GPU 옵션을 사용할 수 있습니다. 메뉴탭에서 “Runtime” 과
“Change runtime type” 을 차례로 선택합니다. 다음 팝업에서 GPU를 선택할 수 있습니다.
변경 후 런타임은 자동으로 재시작됩니다 (즉,
실행된 셀의 정보가 사라집니다).

먼저 홈페이지의 안내에 따라 설치할 수 있는
`torchaudio <https://github.com/pytorch/audio>`__ 같은 일반적인
torch 패키지를 가져옵니다.

"""

# Google Colab에서 실행할 "runtime type" 에 해당하는 줄의 주석 해제를 완료하십시오.

# CPU:
# !pip install pydub torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# GPU:
# !pip install pydub torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm


######################################################################
# CUDA GPU가 있는지 확인하고 장치를 선택합니다.
# GPU에서 네트워크를 실행하면 training/testing 실행 시간이 크게 줄어듭니다.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


######################################################################
# 데이터셋 가져오기
# ---------------------
#
# Torchaudio를 사용하여 데이터셋을 다운로드하고 표현합니다. 여기서는
# 서로 다른 사람이 말하는 35개 명령의 데이터셋인
# `SpeechCommands <https://arxiv.org/abs/1804.03209>`__를 사용합니다. 데이터셋
# ``SPEECHCOMMANDS`` 는 ``torch.utils.data.Dataset`` 버전입니다.
# 이 데이터셋에서 모든 오디오 파일의 길이는 약 1초(약 16000개의 시간 프레임)입니다.
#
# 실제 로딩 및 포맷 단계는 데이터 포인트에 접근할 때 발생하며,
# torchaudio는 오디오 파일을 텐서로 변환하는 작업을 담당합니다.
# 오디오 파일을 대신 직접 로드하려면 ``torchaudio.load()`` 를 
# 사용할 수 있습니다. 오디오 파일의 샘플링 주파수(SpeechCommands의 경우 16kHz)와
# 함께 새로 생성된 텐서가 포함된 튜플을 반환합니다.
#
# 데이터셋으로 돌아가면, 이를 표준 교육, 검증, 테스트 하위 세트로 나누는
# 하위 클래스를 만듭니다.
#

from torchaudio.datasets import SPEECHCOMMANDS
import os


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# 데이터의 training 및 testing 분할을 작성합니다. 이 튜토리얼에서는 검증을 사용하지 않습니다.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


######################################################################
# SPEECHCOMMANDS 데이터셋의 데이터 지점은 파형(오디오 신호), 샘플링 속도,
# 발화(라벨), 발화자의 ID, 발화 횟수 등으로 구성된 튜플입니다.
#

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.plot(waveform.t().numpy());


######################################################################
# 데이터셋에서 사용할 수 있는 레이블 목록을 찾아봅시다.
#

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
labels


######################################################################
# 35개의 오디오 라벨은 사용자가 말하는 명령입니다.
# 처음 몇 개의 파일은 사람들이 “marvin” 이라고 말하는 것입니다.
#

waveform_first, *_ = train_set[0]
ipd.Audio(waveform_first.numpy(), rate=sample_rate)

waveform_second, *_ = train_set[1]
ipd.Audio(waveform_second.numpy(), rate=sample_rate)


######################################################################
# 마지막 파일은 누군가가 “visual” 이라고 말하는 것입니다.
#

waveform_last, *_ = train_set[-1]
ipd.Audio(waveform_last.numpy(), rate=sample_rate)


######################################################################
# 데이터 포맷
# -------------------
#
# 이곳은 데이터에 변환을 적용하기에 좋은 장소입니다. 
# 파형의 경우 분류 전력 손실 없이 더 빠른 처리를 위해
# 오디오를 다운샘플링합니다.
#
# 여기에 다른 변형을 적용할 필요가 없습니다. 일부 데이터셋에서는
# 채널 치수를 따라 평균을 취하거나 채널 중 하나만 유지하여 채널 수 (예:
# stereo to mono)를 줄여야 하는 것이 일반적입니다.
# 음성 명령은 오디오에 단일 채널을 사용하므로
# 여기서는 필요하지 않습니다.
#

new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

ipd.Audio(transformed.numpy(), rate=new_sample_rate)


######################################################################
# 라벨 리스트의 인덱스를 사용하여 각 단어를 인코딩하고 있습니다.
#


def label_to_index(word):
    # 레이블에서 단어 위치 반환
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # 레이블에서 색인에 해당하는 단어 반환
    # 이것은 label_to_index의 역수이다.
    return labels[index]


word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)


######################################################################
# 오디오 녹음과 발언으로 구성된 데이터 포인트 목록을 모델에 대한 두 개의
# batched tensor로 변환하기 위해 배치별로 데이터셋을 반복할 수 있는
# PyTorch DataLoader에서 사용되는 collate 함수를 구현합니다.
# 정렬 함수 작업에 대한 자세한 내용은
# 문서 <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`__
# 를 참조하십시오.
#
# Collate 함수에서는 리샘플링과 텍스트 인코딩을 적용합니다.
#


def pad_sequence(batch):
    # 0으로 패딩을 하여 batch의 모든 tensor를 같은 길이로 만듭니다.
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # 데이터 튜플은 다음과 같은 형태를 갖습니다:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # 목록에 수집하고 레이블을 색인으로 인코딩합니다.
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Tensor 목록을 batched tensor로 그룹화합니다.
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


######################################################################
# 네트워크 정의
# ------------------
#
# 이 튜토리얼에서는 원시 오디오 데이터를 처리하기 위해 convolutional 신경망을
# 사용할 것입니다. 일반적으로 오디오 데이터에 더 고급 변환이 적용되지만
# CNN을 사용하여 원시 데이터를 정확하게 처리할 수 있습니다.
# 특정 아키텍처는 본 문서 <https://arxiv.org/pdf/1610.00087.pdf>`__에 기술된
# M5 네트워크 아키텍처를 본떠서 모델링되었습니다. 원시 오디오 데이터를 처리하는
# 모델의 중요한 측면은 첫 번째 계층의 필터의 수용 영역입니다.
# 우리 모델의 첫 번째 필터는 길이가 80이므로 8kHz에서 샘플링된 오디오를 처리할 때
# 수용 필드는 약 10ms (그리고 4kHz에서는 약 20 ms)입니다.
# 이 크기는 20ms 에서 40ms 사이의 수용 필드를 자주 사용하는 
# 음성 처리 애플리케이션과 유사합니다.
#


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


model = M5(n_input=transformed.shape[0], n_output=len(labels))
model.to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n = count_parameters(model)
print("Number of parameters: %s" % n)


######################################################################
# 우리는 논문에서 사용된 것과 동일한 최적화 기술인 무게 감소가 
# 0.0001로 설정된 Adam 최적화 도구를 사용할 것입니다. 처음에는 0.01의 학습률로 
# 훈련할 예정이지만 ``scheduler`` 를 이용해 전체 데이터를 20번 사용해 0.001로
# 낮추겠습니다.
#

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10


######################################################################
# 네트워크 Training과 Testing 
# --------------------------------
#
# 이제 모델에 training 데이터를 제공하는 training 함수를 정의하고 backward pass 및
# 최적화 단계를 수행하겠습니다. 훈련에서 우리가 사용할 손실은 음의 로그 가능성입니다.
# 그런 다음 각 epoch 이후에 네트워크를 테스트하여 교육 중에
# 정확도가 어떻게 달라지는지 확인합니다.
#


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # 변환 및 모델을 장치에서 직접 전체 batch에 적용
        data = transform(data)
        output = model(data)

        # Tensor의 크기에 대한 음의 로그 우도 (각각 x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


######################################################################
# Training 함수가 생겼으니 네트워크 정확도를 테스트하기 위한 testing을
# 만들어야 합니다. 모델을 ``eval()`` 모드로 설정한 다음 테스트 데이터셋에 대한
# 추론을 실행할 것입니다. ``eval()`` 을 호출하면 네트워크의 모든 모듈에 있는
# training 변수가 false로 설정됩니다. Batch 정규화와 드롭아웃 레이어와 같은
# 특정 레이어는 training 중에 다르게 동작하므로 이 단계는
# 올바른 결과를 얻는 데 중요합니다.
#


def number_of_correct(pred, target):
    # 정확한 예측 횟수 계산
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # Batch의 각 원소에 대해 가장 가능성이 높은 레이블 색인 찾기
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # 변환 및 모델을 장치에서 직접 전체 batch에 적용
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


######################################################################
# 마침내 우리는 네트워크를 훈련하고 테스트할 수 있습니다.
# 우리는 네트워크를 10 epochs로 훈련시키고 학습률을 낮추고 10 epochs 더 훈련시킬 것입니다.
# 훈련 중에 정확도가 어떻게 달라지는지 확인하기 위해 각 epochs 이후에
# 네트워크를 테스트합니다.
#

log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# 변환은 모델 및 데이터와 동일한 장치에서 진행되어야 합니다.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

# 훈련 손실과 반복 횟수를 그림으로 표시합시다.
# plt.plot(losses);
# plt.title("training loss");


######################################################################
# 네트워크는 2 epochs 이후 테스트셋에서 65%, 21 epochs 이후에서는
# 85% 이상 정확해야 합니다. 훈련셋에서 마지막 단어를 보고
# 모델이 어떻게 했는지 살펴봅시다.
#


def predict(tensor):
    # 모델을 사용하여 파형의 레이블 예측
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


waveform, sample_rate, utterance, *_ = train_set[-1]
ipd.Audio(waveform.numpy(), rate=sample_rate)

print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")


######################################################################
# 제대로 분류되지 않은 예가 있다면 찾아봅시다.
#

for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
    output = predict(waveform)
    if output != utterance:
        ipd.Audio(waveform.numpy(), rate=sample_rate)
        print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
        break
else:
    print("All examples in this dataset were correctly classified!")
    print("In this case, let's just look at the last data point")
    ipd.Audio(waveform.numpy(), rate=sample_rate)
    print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")


######################################################################
# 레이블 중 하나의 레코딩으로 자유롭게 시도해 보십시오!
# 예를 들어, Colab을 사용하여 아래 셀을 실행하는 동안 "Go"라고 말합니다.
# 이것은 1초의 오디오를 녹음하고 분류를 시도할 것입니다.
#


def record(seconds=1):

    from google.colab import output as colab_output
    from base64 import b64decode
    from io import BytesIO
    from pydub import AudioSegment

    RECORD = (
        b"const sleep  = time => new Promise(resolve => setTimeout(resolve, time))\n"
        b"const b2text = blob => new Promise(resolve => {\n"
        b"  const reader = new FileReader()\n"
        b"  reader.onloadend = e => resolve(e.srcElement.result)\n"
        b"  reader.readAsDataURL(blob)\n"
        b"})\n"
        b"var record = time => new Promise(async resolve => {\n"
        b"  stream = await navigator.mediaDevices.getUserMedia({ audio: true })\n"
        b"  recorder = new MediaRecorder(stream)\n"
        b"  chunks = []\n"
        b"  recorder.ondataavailable = e => chunks.push(e.data)\n"
        b"  recorder.start()\n"
        b"  await sleep(time)\n"
        b"  recorder.onstop = async ()=>{\n"
        b"    blob = new Blob(chunks)\n"
        b"    text = await b2text(blob)\n"
        b"    resolve(text)\n"
        b"  }\n"
        b"  recorder.stop()\n"
        b"})"
    )
    RECORD = RECORD.decode("ascii")

    print(f"Recording started for {seconds} seconds.")
    display(ipd.Javascript(RECORD))
    s = colab_output.eval_js("record(%d)" % (seconds * 1000))
    print("Recording ended.")
    b = b64decode(s.split(",")[1])

    fileformat = "wav"
    filename = f"_audio.{fileformat}"
    AudioSegment.from_file(BytesIO(b)).export(filename, format=fileformat)
    return torchaudio.load(filename)


# Google colab에서 노트북이 실행되는지 여부
if "google.colab" in sys.modules:
    waveform, sample_rate = record()
    print(f"Predicted: {predict(waveform)}.")
    ipd.Audio(waveform.numpy(), rate=sample_rate)


######################################################################
# 결론
# ----------
#
# 이 튜토리얼에서는 torchaudio를 사용하여 데이터셋을 로드하고 신호를 다시 샘플링했습니다. 
# 그런 다음 주어진 명령을 인식하도록 훈련한 신경망을 정의했습니다.
# 데이터셋의 크기를 줄일 수 있는 mel 주파수 세프스트랄 계수(MFCC)를
# 찾는 것과 같은 다른 데이터 전처리 방법도 있습니다.
# 이 변환은 torchaudio에서 ``torchaudio.transforms.MFCC`` 로도
# 사용할 수 있습니다.
#
