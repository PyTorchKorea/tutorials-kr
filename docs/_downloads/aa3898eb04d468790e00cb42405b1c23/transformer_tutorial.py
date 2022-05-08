"""
nn.Transformer 와 TorchText 로 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델링하기
=================================================================================

이 튜토리얼에서는
`nn.Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ 모듈을
이용하는 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델을 학습하는 방법을 배워보겠습니다.

PyTorch 1.2 버젼에는 `Attention is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`__ 논문에
기반한 표준 트랜스포머(transformer) 모듈을 포함하고 있습니다.
트랜스포머 모델은 다양한 시퀀스-투-시퀀스 문제들에서 더 병렬화(parallelizable)가 가능하면서도
순환 신경망(RNN; Recurrent Neural Network)과 비교하여 더 나은 성능을 보임이 입증되었습니다.
``nn.Transformer`` 모듈은 입력(input) 과 출력(output) 사이의 전역적인 의존성(global dependencies)
을 나타내기 위하여 (`nn.MultiheadAttention <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>`__ 으로
구현된) 어텐션(attention) 메커니즘에 전적으로 의존합니다.
현재 ``nn.Transformer`` 모듈은 모듈화가 잘 되어 있어
단일 컴포넌트 (예. `nn.TransformerEncoder <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html>`__ )
로 쉽게 적용 및 구성할 수 있습니다.

.. image:: ../_static/img/transformer_architecture.jpg

"""

######################################################################
# 모델 정의하기
# ----------------
#


######################################################################
# 이 튜토리얼에서, 우리는 ``nn.TransformerEncoder`` 모델을 언어 모델링(language modeling) 과제에 대해서 학습시킬 것입니다.
# 언어 모델링 과제는 주어진 단어 (또는 단어의 시퀀스) 가 다음에 이어지는 단어 시퀀스를 따를 가능성(likelihood)에 대한 확률을 할당하는 것입니다.
# 먼저, 토큰(token) 들의 시퀀스가 임베딩(embedding) 레이어로 전달되며, 이어서 포지셔널 인코딩(positional encoding) 레이어가 각 단어의 순서를 설명합니다.
# (더 자세한 설명은 다음 단락을 참고해주세요.)
# ``nn.TransformerEncoder`` 는 여러 개의
# `nn.TransformerEncoderLayer <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>`__
# 레이어로 구성되어 있습니다.
# ``nn.TransformerEncoder`` 내부의 셀프-어텐션(self-attention) 레이어들은 시퀀스 안에서의 이전 포지션에만 집중하도록 허용되기 때문에,
# 입력(input) 순서와 함께, 정사각 형태의 어텐션 마스크(attention mask) 가 필요합니다.
# 언어 모델링 과제를 위해서, 미래의 포지션에 있는 모든 토큰들은 마스킹 되어야(가려져야) 합니다.
# 실제 단어를 얻기 위해서, ``nn.TransformerEncoder`` 의 출력은 로그-소프트맥스(log-Softmax) 로 이어지는 최종 선형(Linear) 레이어로 전달 됩니다.
#

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


######################################################################
# ``PositionalEncoding`` 모듈은 시퀀스 안에서 토큰의 상대적인 또는 절대적인 포지션에 대한 어떤 정보를 주입합니다.
# 포지셔널 인코딩은 임베딩과 합칠 수 있도록 똑같은 차원을 가집니다.
# 여기에서, 우리는 다른 주파수(frequency) 의 ``sine`` 과 ``cosine`` 함수를 사용합니다.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


######################################################################
# 데이터 로드하고 배치 만들기
# -----------------------------
#


######################################################################
# 이 튜토리얼에서는 ``torchtext`` 를 사용하여 Wikitext-2 데이터셋을 생성합니다.
# torchtext 데이터셋에 접근하기 전에, https://github.com/pytorch/data 을 참고하여 torchdata를 설치하시기 바랍니다.
# 단어 오브젝트는 훈련 데이터셋(train dataset) 에 의하여 만들어지고, 토큰(token)을 텐서(tensor)로 수치화하는데 사용됩니다.
# Wikitext-2에서 보기 드믄 토큰(rare token)은 `<unk>` 로 표현됩니다.
#
# 주어진 1D 벡터의 시퀀스 데이터에서, ``batchify()`` 함수는 데이터를 ``batch_size`` 컬럼들로 정렬합니다.
# 만약 데이터가 ``batch_size`` 컬럼으로 나누어 떨어지지 않으면, 데이터를 잘라내서 맞춥니다.
# 예를 들어 (총 길이 26의) 알파벳을 데이터로 보고 ``batch_size=4`` 일 때, 알파벳은 길이가 6인 4개의 시퀀스로 나눠집니다:
#
# .. math::
#   \begin{bmatrix}
#   \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z}
#   \end{bmatrix}
#   \Rightarrow
#   \begin{bmatrix}
#   \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} &
#   \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} &
#   \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} &
#   \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix}
#   \end{bmatrix}
#
# 배치 작업(batching)은 더 많은 병렬 처리를 가능하게 하지만, 모델이 독립적으로 각 컬럼들을 취급해야 함을 뜻합니다;
# 예를 들어, 위 예제에서 ``G`` 와 ``F`` 의 의존성(dependance)은 학습되지 않습니다.
#

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


######################################################################
# 입력(input) 과 타겟(target) 시퀀스를 생성하기 위한 함수들
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# ``get_batch()`` 함수는 트랜스포머 모델을 위한 입력-타겟 시퀀스 쌍(pair)을 생성합니다.
# 이 함수는 소스 데이터를 ``bptt`` 길이를 가진 덩어리로 세분화 합니다.
# 언어 모델링 과제를 위해서, 모델은 다음 단어인 ``Target`` 이 필요 합니다.
# 예를 들어, ``bptt`` 의 값이 2 라면, 우리는 ``i`` = 0 일 때 다음의 2 개의 변수(Variable) 를 얻을 수 있습니다:
#
# .. image:: ../_static/img/transformer_input_target.png
#
# 변수 덩어리는 트랜스포머 모델의 ``S`` 차원과 일치하는 0 차원에 해당합니다.
# 배치 차원 ``N`` 은 1 차원에 해당합니다.
#

bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


######################################################################
# 인스턴스(instance) 초기화하기
# --------------------------------
#


######################################################################
# 모델의 하이퍼파라미터(hyperparameter)는 아래와 같이 정의됩니다.
# 단어 사이즈는 단어 오브젝트의 길이와 일치 합니다.
#

ntokens = len(vocab) # 단어 사전(어휘집)의 크기
emsize = 200 # 임베딩 차원
d_hid = 200 # nn.TransformerEncoder 에서 피드포워드 네트워크(feedforward network) 모델의 차원
nlayers = 2 # nn.TransformerEncoder 내부의 nn.TransformerEncoderLayer 개수
nhead = 2 # nn.MultiheadAttention의 헤드 개수
dropout = 0.2 # 드랍아웃(dropout) 확률
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


######################################################################
# 모델 실행하기
# -------------
#


######################################################################
# `CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`__ 를
# `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`__ (확률적 경사 하강법) 옵티마이저(optimizer)와
# 함께 사용하였습니다. 학습률(learning rate)는 5.0으로 초기화하였으며 `StepLR <https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR>`__
# 스케쥴을 따릅니다. 학습하는 동안, `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`__
# 을 사용하여 기울기(gradient)가 폭발(exploding)하지 않도록 합니다.
#

import copy
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # 학습률(learning rate)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # 학습 모드 시작
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != bptt:  # 마지막 배치에만 적용
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # 평가 모드 시작
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

######################################################################
# 에포크 내에서 반복됩니다. 만약 검증 오차(validation loss) 가 우리가 지금까지 관찰한 것 중 최적이라면 모델을 저장합니다.
# 매 에포크 이후에 학습률을 조절합니다.

best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)

    scheduler.step()


######################################################################
# 평가 데이터셋(test dataset)으로 모델을 평가하기
# -------------------------------------------------
#

test_loss = evaluate(best_model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
