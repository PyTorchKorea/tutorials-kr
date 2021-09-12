"""
torchaudio를 사용한 음성 명령 인식 
******************************************

튜토리얼에선 규격에 맞게 오디오 데이터셋 구성을 하는법과 
음성 분류 신경망을 통해 학습/테스트 하는법에 대해 설명하겠습니다. 

Colab엔 GPU 옵션이 있습니다. 메뉴탭에 있는 “런타임” 에서 “런타임 유형 변경” 을 선택합니다. 
팝업창에서, GPU를 선택합니다. 변경 후, 런타임은 자동적으로 재시작합니다.(실행하고 있던 셀의 정보는 사라지게 됩니다)

먼저, 웹사이트 공지를 통해 설치한 `torchaudio <https://github.com/pytorch/audio>`__ 와 같은 공통으로 사용하고 있는 torch 패키지를 로드합니다.

"""

# Google Colab에서 "런타임 유형"에 해당하는 하기의 주석을 해제합니다.

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
# CUDA GPU를 고를 수 있는지 체크합니다. 
# GPU상에서 신경망을 실행하는 것은 학습/테스트 시간을 줄일 수 있습니다.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


######################################################################
# 데이터셋 가져오기
# ---------------------
#
# 데이터셋을 다운받고 보기 위해 torchaudio를 사용합니다.  
# 다화자가 35개의 명령어를 발화한 `SpeechCommands <https://arxiv.org/abs/1804.03209>`__
# 를 사용하겠습니다. ``SPEECHCOMMANDS`` 는 ``torch.utils.data.Dataset`` 의 한 데이터셋입니다.
# 데이터셋의 오디오 파일들의 길이는 대략 1초 정도(16000개 샘플)입니다.
#
# 데이터 로딩 및 포멧 단계는 데이터 포인트에 접근할 때 발생하고, torchaudio는 오디오 파일을
# 텐서로 변환합니다. 다른 오디오 파일을 사용하려면, ``torchaudio.load()`` 를 사용합니다. 
# 오디오 파일의 샘플링 주파수( SpeechCommands는 16kHz )에 따라 생성한 텐서를 튜플 형태로 반환 받게 됩니다.
#
# 데이터셋 설명으로 돌아와서, 학습, 검증, 테스트 데이터셋을 분리하는 데이터 클래스를 만들겠습니다.  
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


# 데이터를 학습셋과 테스트셋으로 나누어 줍니다. 본 튜토리얼에선 검증셋은 사용하지 않습니다.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


######################################################################
# SPEECHCOMMANDS 데이터셋에서 데이터 포인트는 wav(음성 신호), 
# 샘플링 주파수, 발화(텍스트), 화자의 ID, 발화 개수로 구성한 튜플입니다.  
#

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.plot(waveform.t().numpy());


######################################################################
# 데이터셋에서 사용할 발화를 확인해 봅시다.
#

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
labels


######################################################################
# 35개의 발화는 사용자가 말한 명령어들 입니다. 
# 첫째 몇개의 파일은 "marvin" 이라고 말합니다.
#

waveform_first, *_ = train_set[0]
ipd.Audio(waveform_first.numpy(), rate=sample_rate)

waveform_second, *_ = train_set[1]
ipd.Audio(waveform_second.numpy(), rate=sample_rate)


######################################################################
# 마지막 파일의 화자는 "visual" 이라고 말합니다.
#

waveform_last, *_ = train_set[-1]
ipd.Audio(waveform_last.numpy(), rate=sample_rate)


######################################################################
# 데이터 형식 지정하기
# -------------------
#
# 데이터 셋 구성을 마친 후, 데이터 변형을 적용합니다. 오디오 분류 특성을 줄이지 않는 선에서 
# 보다 빠른 처리를 위해 오디오를 다운 샘플링 합니다.

# 여기서 다른 데이터 변형은 필요하지 않습니다. 오디오 채널 수를 줄이기 위하여(스테테오 에서 모노)
# 채널간의 평균 값을 취하거나 단일 채널을 활용하여 채널 수를 감소하는 것이 보통이지만, 
# SpeechCommands 는 모노 채널을 사용하므로, 필요하지 않습니다.
#

new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

ipd.Audio(transformed.numpy(), rate=new_sample_rate)


######################################################################
# 발화 리스트의 인덱스를 사용하여 각 단어들을 인코딩 합니다.
#


def label_to_index(word):
    # 발화 리스트에 해당하는 인덱스를 반환합니다.
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # 인덱스에 해당하는 발화를 반환합니다. 
    # label_to_index 함수의 정반대입니다.
    return labels[index]


word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)


######################################################################
# 오디오 데이터와 발화로 구성하고 있는 데이터 포인트를 모델에 적용하기 위한 두 배치로 반환하기 위해 
# 배치를 반복해서 처리하는 PyTorch DataLoader 에서 사용하는 collate 함수를 구현합니다   
# collate 함수에 대해서 더 알고 싶은 사항은 `다음의 문서 <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`__ 를 참고 바랍니다.

# collate 함수에서, 샘플링을 재조정하거나, 텍스트 인코딩 방식을 적용합니다.  
#


def pad_sequence(batch):
    # 배치내에서 모든 텐서가 같은 길이를 가질 수 있도록 0으로 채웁니다. 
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

     # 튜플은 다음과 같은 포맷을 갖습니다:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # 리스트의 형태로 수집을 하고, 발화는 인덱스들로 인코딩합니다.
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # 텐서 리스트를 배치 텐서로 그룹화합니다.
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
# 신경망 정의
# ------------------
#
# 본 튜토리얼에선 합성곱 신경망을 사용하여 오디오 데이터를 처리합니다.
# 보통은 고급 변형 기법을 적용하여 오디오 데이터를 처리하지만, 합성곱 신경망은 오디오를 가공치 않고
# 정확하게 처리하는데 사용합니다. 자세한 신경망 구조는 M5 신경망 구조 이후에 설계되었고 `이 논문 <https://arxiv.org/pdf/1610.00087.pdf>`__ 에 자세한 설명을 해 놓았습니다.
# 오디오를 가공치 않고 처리하는 모델의 중요한 측면은 첫번째 층 필터들의 수용 영역입니다.
# 설계한 모델의 첫번째 길이는 80입니다. 그래서 8kHz으로 샘플링한 오디오 데이터를 처리할 때, 
# 수용 영역은 대략 10ms 입니다. (샘플링이 4kHz일 경우, 대략 20ms). 이 사이즈는
# 20ms 에서 40ms 길이의 수용 영역을 갖는 음성 어플리케이션과 유사합니다. 
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
# 논문에서 사용한, 가중치 감소 셋을 0.0001로 설정한 Adam 최적화를 사용할 것입니다. 
# 첫째로, 학습률을 0.01로 설정하지만, 20 에폭의 학습동안 ``scheduler`` 를 사용하여 0.001로 감소합니다. 
#

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 20에폭 후에 학습률은 10배 비율로 감소 합니다.


######################################################################
# 신경망 학습 및 테스트
# --------------------------------
#
# 학습 데이터를 모델에 적용하고 역전파 및 최적화를 수행하는 학습 함수를 정의 합니다. 
# 학습에서, 음수 로그 가능도 함수를 손실로 활용을 합니다.
# 신경망은 학습동안의 정확도가 어떻게 변하는지 체크하기 위해 각 에폭마다 테스트를 진행합니다. 
#


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # 데이터를 변형하고 device 에 올린 전체 배치를 모델에 적용합니다.
        data = transform(data)
        output = model(data)

        # 텐서크기 (batch x 1 x n_output)에 관한 음수 로그 가능도
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 학습 현황을 출력합니다.
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # 진행 표시를 업데이트 합니다.
        pbar.update(pbar_update)
        # 손실을 기록합니다.
        losses.append(loss.item())


######################################################################
# 현재, 학습 함수가 존재하므로 신경망의 정확도를 테스트 하기 위한 함수를 만들 필요가 있습니다. 
# 모델을 ``eval()`` 모드로 설정을 하고 테스트셋을 추론합니다.  
# ``eval()`` 은 신경망 안에 있는 모든 모듈의 학습 변수를 false로 설정합니다.   
# 배치 정규화 및 드롭아웃과 같은 특정 계층은 학습과는 다르게 동작하므로 
# 정확한 결과를 얻는데 중요한 과정이 됩니다. 
#


def number_of_correct(pred, target):
    # 정확한 예측 개수를 셉니다.
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # 배치의 각 텐서에서 가장 적합한 발화 인덱스를 찾습니다.
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # 데이터를 변형하고 device 에 올린 전체 배치를 모델에 적용합니다. 
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # 진행 표시를 업데이트 합니다.
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


######################################################################
# 마지막으로, 신경망을 학습하고 테스트 할 수 있게 되었습니다.
# 10개의 epoch 동안 신경망을 학습한 다음 학습률을 줄이고 10개의 epoch 동안 더 학습할 것입니다.
# 학습 하는 동안 정확도가 어떻게 달라지는지 확인하기 위해 각 에폭 후에 신경망을 테스트합니다. 
#

log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# 모델과 데이터가 같은 장치에 있어야 변형이 가능합니다.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

# 에폭 대비 학습 손실을 그래프로 출력합니다. 
# plt.plot(losses);
# plt.title("training loss");


######################################################################
# 신경망은 2에폭 이후에 65% 정확하며 21에폭 이후엔 85% 정확합니다. 
# 학습 셋의 마지막 단어를 확인 후에, 신경망이 어떻게 수행하는지 확인하겠습니다.
#


def predict(tensor):
    # 발화를 예측하기 위해 신경망을 사용합니다. 
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
# 정확하게 분류되지 않은 예가 있다면 찾아봅시다. 
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
# 발화중 하나를 자유롭게 녹음하십시오!
# 예를 들어, Colab을 사용할 경우 실행하고 있는 셀에서 “Go” 라고 말합니다.
# 1초 동안 녹음이 되고 어떤 발화에 해당하는지 분류를 진행할 것입니다. 
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


# 노트북이 Google colab에서 실행되는지 확인합니다.
if "google.colab" in sys.modules:
    waveform, sample_rate = record()
    print(f"Predicted: {predict(waveform)}.")
    ipd.Audio(waveform.numpy(), rate=sample_rate)


######################################################################
# 결론
# ----------
#
# 본 튜토리얼에선, 데이터셋을 로드하고 신호를 재조정 하기위해 torchaudio를 사용했습니다. 
# 주어진 명령어를 인식하게끔 학습한 신경망을 정의하였습니다.
# 또다른 데이터 전처리 방법도 존재합니다. 가령 mel frequency cepstral coefficients (MFCC)를 찾을 수 있고, 
# 데이터셋의 크기를 줄일 수 있습니다. 위와 같은 변형은 torchaudio에서 ``torchaudio.transforms.MFCC`` 으로 가능합니다.
#
