"""
분산 데이터 병렬 처리와 병렬 처리 파이프라인을 사용한 트랜스포머 모델 학습
=======================================================================

**Author**: `Pritam Damania <https://github.com/pritamdamania87>`_
  **번역**: `백선희 <https://github.com/spongebob03>`_

이 튜토리얼은 `분산 데이터 병렬처리(Distributed Data Parallel) <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__ 와
`병렬 처리 파이프라인 <https://pytorch.org/docs/stable/pipeline.html>`__
를 사용하여 여러 GPU에 걸친 거대한 트랜스포머(Transformer) 모델을 어떻게 학습시키는지 보여줍니다.
이번 튜토리얼은 `nn.Transformer 와 TorchText 로 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델링하기 <https://tutorials.pytorch.kr/beginner/transformer_tutorial.html>`__ 의
확장판이며 분산 데이터 병렬 처리와 병렬 처리 파이프라인이 어떻게 트랜스포머 모델 학습에 쓰이는지 보여주기 위해 이전 튜토리얼에서의
모델 규모를 증가시켰습니다.

선수과목(Prerequisites):

    * `Pipeline Parallelism <https://pytorch.org/docs/stable/pipeline.html>`__
    * `nn.Transformer 와 TorchText 로 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델링하기 <https://tutorials.pytorch.kr/beginner/transformer_tutorial.html>`__
    * `분산 데이터 병렬 처리 시작하기 <https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html>`__
"""


######################################################################
# 모델 정의하기
# -------------
#

######################################################################
# ``PositionalEncoding`` 모듈은 시퀀스에서 토큰의 상대적, 절대적 위치에 대한
# 몇몇 정보를 주입합니다.
# 위치 인코딩은 임베딩과 같은 차원을 가지므로
# 둘을 합칠 수 있습니다. 여기서, 주파수가 다른 ``sine`` 과 ``cosine`` 함수를
# 사용합니다.

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


######################################################################
# 이번 튜토리얼에서는, 트랜스포머 모델을 두 개의 GPU에 걸쳐서 나누고
# 병렬 처리 파이프라인으로 학습시켜 보겠습니다. 추가로,
# `분산 데이터 병렬 처리 <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
# 를 사용하여 이 파이프라인의 두 복제를 훈련시킵니다. 한 프로세스는
# GPUs 0, 1에 거쳐 파이프를 구동하고 다른 프로세스는 GPUs 2, 3에서 파이프를 구동합니다. 그 다음, 이 두
# 프로세스는 분산 데이터 병렬처리로 두 복제본(replica)을 학습시킵니다.
# 모델은 바로 `nn.Transformer 와 TorchText 로 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델링하기
# <https://tutorials.pytorch.kr/beginner/transformer_tutorial.html>`__ 튜토리얼과
# 똑같은 모델이지만 두 단계로 나뉩니다. 대부분 파라미터(parameter)들은
# `nn.TransformerEncoder <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html>`__ 계층(layer)에 포함됩니다.
# `nn.TransformerEncoder <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html>`__ 는
# `nn.TransformerEncoderLayer <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>`__ 의 ``nlayers`` 로 구성되어 있습니다.
# 결과적으로, 이 튜토리얼에서는 ``nn.TransformerEncoder`` 에 중점을 두고 있으며
# ``nn.TransformerEncoderLayer`` 의 절반은 한 GPU에 두고
# 나머지 절반은 다른 GPU에 있도록 모델을 분할합니다. 이를 위해서 ``Encoder`` 와
# ``Decoder`` 섹션을 분리된 모듈로 빼낸 다음, 원본 트랜스포머 모듈을
# 나타내는 nn.Sequential을 빌드 합니다.


if sys.platform == 'win32':
    print('Windows platform is not supported for pipeline parallelism')
    sys.exit(0)
if torch.cuda.device_count() < 4:
    print('Need at least four GPU devices for this tutorial')
    sys.exit(0)

class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)

class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).permute(1, 0, 2)

######################################################################
# 학습을 위한 다중 프로세스 시작
# ------------------------------
#


######################################################################
# 각각 두 개의 GPU에서 자체 파이프라인을 구동하는 두 개의 프로세스를 시작합니다.
# ``run_worker`` 는 각 프로세스에 실행됩니다.

def run_worker(rank, world_size):


######################################################################
# 데이터 로드하고 배치 만들기
# ---------------------------
#

######################################################################
# 학습 프로세스는 ``torchtext`` 의 Wikitext-2 데이터셋을 사용합니다.
# torchtext 데이터셋에 접근하기 전에, https://github.com/pytorch/data 을 참고하여 torchdata를
# 설치하시기 바랍니다.
# 단어 오브젝트는 훈련 데이터셋으로 만들어지고, 토큰을 텐서(tensor)로 수치화하는데 사용됩니다.
# 시퀀스 데이터로부터 시작하여, ``batchify()`` 함수는 데이터셋을 열(column)들로 정리하고,
# ``batch_size`` 사이즈의 배치들로 나눈 후에 남은 모든 토큰을 버립니다.
# 예를 들어, 알파벳을 시퀀스(총 길이 26)로 생각하고 배치 사이즈를 4라고 한다면,
# 알파벳을 길이가 6인 4개의 시퀀스로 나눌 수 있습니다:
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
# 이 열들은 모델에 의해서 독립적으로 취급되며, 이는
# ``G`` 와 ``F`` 의 의존성이 학습될 수 없다는 것을 의미하지만, 더 효율적인
# 배치 프로세싱(batch processing)을 허용합니다.
#

# In 'run_worker'
    def print_with_rank(msg):
        print('[RANK {}]: {}'.format(rank, msg))

    from torchtext.datasets import WikiText2
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def data_process(raw_text_iter):
      data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
      return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    device = torch.device(2 * rank)

    def batchify(data, bsz, rank, world_size, is_train=False):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        # Divide the data across the ranks only for training data.
        if is_train:
            data_per_rank = data.size(0) // world_size
            data = data[rank * data_per_rank : (rank + 1) * data_per_rank]
        return data.to(device)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size, rank, world_size, True)
    val_data = batchify(val_data, eval_batch_size, rank, world_size)
    test_data = batchify(test_data, eval_batch_size, rank, world_size)


######################################################################
# 입력과 타겟 시퀀스를 생성하기 위한 함수들
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# ``get_batch()`` 함수는 트랜스포머 모델을 위한 입력과 타겟 시퀀스를
# 생성합니다. 이 함수는 소스 데이터를 ``bptt`` 길이를 가진 덩어리로 세분화합니다.
# 언어 모델링 과제를 위해서, 모델은
# 다음 단어인 ``Target`` 이 필요합니다. 예를 들어 ``bptt`` 의 값이 2라면,
# ``i`` = 0 일 때 다음의 2 개 변수(Variable)를 얻을 수 있습니다:
#
# .. image:: ../_static/img/transformer_input_target.png
#
# 청크가 차원 0에 속하며
# 트랜스포머 모델의 ``S`` 차원과 일치한다는 것을 유의해야 합니다.
# 배치 차원 ``N`` 은 1 차원에 해당합니다.
#

# In 'run_worker'
    bptt = 35
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        # Need batch dimension first for pipeline parallelism.
        return data.t(), target

######################################################################
# 모델 규모와 파이프 초기화
# -------------------------
#


######################################################################
# 병렬 처리 파이프라인을 활용한 대형 트랜스포머 모델 학습을 보이기 위해,
# 트랜스포머 계층 규모를 적절히 확장시킵니다.
# 4096차원의 임베딩 벡터, 4096의 은닉 사이즈, 16개의 어텐션 헤드(attention head)와 총 8 개의
# 트랜스포머 계층 (``nn.TransformerEncoderLayer``)를 사용합니다. 이는 최대
# **~1 억** 개의 파라미터를 갖는 모델을 생성합니다.
#
# `RPC 프레임워크 <https://pytorch.org/docs/stable/rpc.html>`__ 를 초기화해야 합니다.
# Pipe가  향후 호스트 파이프라인을 교차 확장할 수 있도록 하는 `RRef <https://pytorch.org/docs/stable/rpc.html#rref>`__ 를 통해
# RPC 프레임워크에 의존하기 때문입니다.
# 이때 RPC 프레임워크는 오직 하나의 하나의 worker로 초기화를 해야 하는데,
# 여러 GPU를 다루기 위해 프로세스 하나만 사용하고 있기 때문입니다.
#
# 그런 다음 파이프라인은 한 GPU에 8개의 트랜스포머와
# 다른 GPU에 8개의 트랜스포머 계층으로 초기화됩니다. 한 파이프는 GPU 0, 1에 거쳐 설정되고
# 다른 하나는 GPU 2, 3에 설정됩니다. 그런 다음 DistributedDataParallel을 사용하여 두 파이프가 모두 복제됩니다.

# In 'run_worker'
    ntokens = len(vocab) # the size of vocabulary
    emsize = 4096 # embedding dimension
    nhid = 4096 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 8 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 16 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value

    from torch.distributed import rpc
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name),
            # Specifying _transports and _channels is a workaround and we no longer
            # will have to specify _transports and _channels for PyTorch
            # versions >= 1.8.1
            _transports=["ibv", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )

    # Num gpus for model parallelism.
    num_gpus = 2
    partition_len = ((nlayers - 1) // num_gpus) + 1

    # Add encoder in the beginning.
    tmp_list = [Encoder(ntokens, emsize, dropout).cuda(2 * rank)]
    module_list = []

    # Add all the necessary transformer blocks.
    for i in range(nlayers):
        transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        if i != 0 and i % (partition_len) == 0:
            module_list.append(nn.Sequential(*tmp_list))
            tmp_list = []
        device = i // (partition_len)
        tmp_list.append(transformer_block.to(2 * rank + device))

    # Add decoder in the end.
    tmp_list.append(Decoder(ntokens, emsize).cuda(2 * rank + num_gpus - 1))
    module_list.append(nn.Sequential(*tmp_list))

    # Need to use 'checkpoint=never' since as of PyTorch 1.8, Pipe checkpointing
    # doesn't work with DDP.
    from torch.distributed.pipeline.sync import Pipe
    chunks = 8
    model = Pipe(torch.nn.Sequential(
        *module_list), chunks = chunks, checkpoint="never")

    # Initialize process group and wrap model in DDP.
    from torch.nn.parallel import DistributedDataParallel
    import torch.distributed as dist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
                backend="nccl", rank=rank, world_size=world_size)
    model = DistributedDataParallel(model)

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params

    print_with_rank('Total parameters in model: {:,}'.format(get_total_params(model)))

######################################################################
# 모델 실행하기
# -------------
#


######################################################################
# 손실(loss)을 추적하기 위해 `CrossEntropyLoss <https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__ 가
# 적용되며, 옵티마이저(optimizer)로서 `SGD <https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD>`__
# 는 확률적 경사하강법(stochastic gradient descent method)을 구현합니다. 초기
# 학습률(learning rate)은 5.0로 설정됩니다. `StepLR <https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR>`__ 는
# 에폭(epoch)에 따라서 학습률을 조절하는 데 사용됩니다. 학습하는 동안,
# 기울기 폭발(gradient exploding)을 방지하기 위해 모든 기울기를 함께 조정(scale)하는 함수
# `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_>`__
# 을 이용합니다.
#

# In 'run_worker'
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    import time
    def train():
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        ntokens = len(vocab)

        # Train only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, train_data.size(0) - 1)

        for batch, i in enumerate(range(0, nbatches, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            output = model(data).local_value()
            # Need to move targets to the device where the output of the
            # pipeline resides.
            loss = criterion(output.view(-1, ntokens), targets.cuda(2 * rank + 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 10
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print_with_rank('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, nbatches // bptt, scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(eval_model, data_source):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(vocab)
        # Evaluate only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, data_source.size(0) - 1)
        with torch.no_grad():
            for i in range(0, nbatches, bptt):
                data, targets = get_batch(data_source, i)
                output = eval_model(data).local_value()
                output_flat = output.view(-1, ntokens)
                # Need to move targets to the device where the output of the
                # pipeline resides.
                total_loss += len(data) * criterion(output_flat, targets.cuda(2 * rank + 1)).item()
        return total_loss / (len(data_source) - 1)

######################################################################
# 에폭을 반복합니다. 만약 검증 오차(validation loss)가 지금까지 관찰한 것 중 최적이라면
# 모델을 저장합니다. 각 에폭 이후에 학습률을 조절합니다.

# In 'run_worker'
    best_val_loss = float("inf")
    epochs = 3 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print_with_rank('-' * 89)
        print_with_rank('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print_with_rank('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()


######################################################################
# 평가 데이터셋으로 모델 평가하기
# -------------------------------
#
# 평가 데이터셋에서의 결과를 확인하기 위해 최고의 모델을 적용합니다.

# In 'run_worker'
    test_loss = evaluate(best_model, test_data)
    print_with_rank('=' * 89)
    print_with_rank('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print_with_rank('=' * 89)

# Main execution
import torch.multiprocessing as mp

if __name__=="__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
######################################################################
# Output
# ------
#


######################################################################
#.. code-block:: py
#
#    [RANK 0]: | epoch   1 |    10/   50 batches | lr 5.00 | ms/batch 778.97 | loss 43.31 | ppl 6432469059895903232.00
#    [RANK 1]: | epoch   1 |    10/   50 batches | lr 5.00 | ms/batch 778.90 | loss 44.50 | ppl 21245447128217366528.00
#    [RANK 0]: | epoch   1 |    20/   50 batches | lr 5.00 | ms/batch 699.89 | loss 44.50 | ppl 21176949187407757312.00
#    [RANK 1]: | epoch   1 |    20/   50 batches | lr 5.00 | ms/batch 699.87 | loss 44.62 | ppl 23975861229620961280.00
#    [RANK 0]: | epoch   1 |    30/   50 batches | lr 5.00 | ms/batch 698.86 | loss 41.62 | ppl 1193312915629888256.00
#    [RANK 1]: | epoch   1 |    30/   50 batches | lr 5.00 | ms/batch 698.87 | loss 40.69 | ppl 471605759847546240.00
#    [RANK 0]: | epoch   1 |    40/   50 batches | lr 5.00 | ms/batch 698.34 | loss 45.20 | ppl 42812308420836458496.00
#    [RANK 1]: | epoch   1 |    40/   50 batches | lr 5.00 | ms/batch 698.33 | loss 45.68 | ppl 68839569686012223488.00
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 1]: | end of epoch   1 | time: 40.08s | valid loss  0.80 | valid ppl     2.22
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | end of epoch   1 | time: 40.09s | valid loss  0.80 | valid ppl     2.22
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | epoch   2 |    10/   50 batches | lr 4.75 | ms/batch 768.51 | loss 36.34 | ppl 6063529544668166.00
#    [RANK 1]: | epoch   2 |    10/   50 batches | lr 4.75 | ms/batch 769.23 | loss 37.41 | ppl 17651211266236086.00
#    [RANK 0]: | epoch   2 |    20/   50 batches | lr 4.75 | ms/batch 699.57 | loss 28.97 | ppl 3798441739584.11
#    [RANK 1]: | epoch   2 |    20/   50 batches | lr 4.75 | ms/batch 699.56 | loss 29.28 | ppl 5203636967575.47
#    [RANK 0]: | epoch   2 |    30/   50 batches | lr 4.75 | ms/batch 699.04 | loss 28.43 | ppl 2212498693571.25
#    [RANK 1]: | epoch   2 |    30/   50 batches | lr 4.75 | ms/batch 699.05 | loss 28.33 | ppl 2015144761281.48
#    [RANK 0]: | epoch   2 |    40/   50 batches | lr 4.75 | ms/batch 699.10 | loss 23.30 | ppl 13121380184.92
#    [RANK 1]: | epoch   2 |    40/   50 batches | lr 4.75 | ms/batch 699.09 | loss 23.41 | ppl 14653799192.87
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | end of epoch   2 | time: 39.97s | valid loss  0.24 | valid ppl     1.27
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 1]: | end of epoch   2 | time: 39.98s | valid loss  0.24 | valid ppl     1.27
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | epoch   3 |    10/   50 batches | lr 4.51 | ms/batch 769.36 | loss 12.80 | ppl 361681.11
#    [RANK 1]: | epoch   3 |    10/   50 batches | lr 4.51 | ms/batch 768.97 | loss 12.57 | ppl 287876.61
#    [RANK 0]: | epoch   3 |    20/   50 batches | lr 4.51 | ms/batch 698.27 | loss 12.01 | ppl 164364.60
#    [RANK 1]: | epoch   3 |    20/   50 batches | lr 4.51 | ms/batch 698.30 | loss 11.98 | ppl 159095.89
#    [RANK 0]: | epoch   3 |    30/   50 batches | lr 4.51 | ms/batch 697.75 | loss 10.90 | ppl 54261.91
#    [RANK 1]: | epoch   3 |    30/   50 batches | lr 4.51 | ms/batch 697.72 | loss 10.89 | ppl 53372.39
#    [RANK 0]: | epoch   3 |    40/   50 batches | lr 4.51 | ms/batch 699.49 | loss 10.78 | ppl 47948.35
#    [RANK 1]: | epoch   3 |    40/   50 batches | lr 4.51 | ms/batch 699.50 | loss 10.79 | ppl 48664.42
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 0]: | end of epoch   3 | time: 39.96s | valid loss  0.38 | valid ppl     1.46
#    [RANK 0]: -----------------------------------------------------------------------------------------
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 1]: | end of epoch   3 | time: 39.96s | valid loss  0.38 | valid ppl     1.46
#    [RANK 1]: -----------------------------------------------------------------------------------------
#    [RANK 0]: =========================================================================================
#    [RANK 0]: | End of training | test loss  0.33 | test ppl     1.39
#    [RANK 0]: =========================================================================================
#    [RANK 1]: =========================================================================================
#    [RANK 1]: | End of training | test loss  0.33 | test ppl     1.39
#    [RANK 1]: =========================================================================================
#
