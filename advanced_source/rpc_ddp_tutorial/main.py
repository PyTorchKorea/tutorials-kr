import random

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP

NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16

# BEGIN hybrid_model
class HybridModel(torch.nn.Module):
    r"""
    The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """

    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(16, 8).cuda(device), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup.cuda(self.device))
# END hybrid_model

# BEGIN setup_trainer
def _run_trainer(remote_emb_module, rank):
    r"""
    각 트레이너는 매개변수 서버(parameter server)에서 임베딩 룩업(embedding lookup)을 포함하고
    전역에서 nn.Linear를 실행하는 전방 전달(forward pass)를 실행합니다.
    후방 전달(backward pass) 동안에 DDP는 조밀한 부분(nn.Linear)에 대한 그래디언트(gradient) 집계를 담당하고
    분산 autograd는 그래디언트 업데이트가 매개변수 서버로 전파되도록 합니다.
    """

    # 모델을 설정합니다.
    model = HybridModel(remote_emb_module, rank)

    # 모든 모델 매개변수를 DistributedOptimizer에 대한 rref로 검색합니다.

    # 임베딩 테이블에 대한 매개변수를 검색합니다.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    # model.fc.parameters()은 오직 지역 매개변수만 포함합니다.
    # 참고: 여기에서는 model.parameters()를 호출할 수 없습니다.
    # 왜냐하면 parameters()가 아니라 remote_parameters()를 지원하는
    # remote_emb_module.parameters()를 호출하기 때문입니다.
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Distributed optimizer을 설정합니다.
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()
    # END setup_trainer

    # BEGIN run_trainer
    def get_next_batch(rank):
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # 오프셋(offset)을 생성합니다.
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8).cuda(rank)
            yield indices, offsets_tensor, target

    # 100개의 에폭크(ephoc)에 대해 학습합니다.
    for epoch in range(100):
        # Distributed autograd context를 생성합니다.
        for indices, offsets, target in get_next_batch(rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # 분산 후방 전달(distributed backward pass)을 실행합니다.
                dist_autograd.backward(context_id, [loss])

                # Distributed optimizer를 실행시킵니다.
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
                # 반복할 때마다 다른 그래디언트를 호스팅하는 하나의 다른 distributed autograd context를 생성하므로
                # 그래디언트를 0으로 만들 필요가 없습니다.
        print("Training done for epoch {}".format(epoch))
        # END run_trainer

# BEGIN run_worker
def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """

    # 포트 충돌을 피하기 위해 init_rpc 및 init_process_group에 대해
    # TCP init_method에서 다른 포트 번호를 사용해야 합니다.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 2 는  master를 의미하고, 3은 is 매개변수 서버, 그리고 0과 1은 트레이너를 뜻한다.
    if rank == 2:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )

        # 트레이너에서 학습 루프를 실행합니다.
        futs = []
        for trainer_rank in [0, 1]:
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(remote_emb_module, rank)
            )
            futs.append(fut)

        # 모든 학습이 끝날 때까지 기다립니다.
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        # 트레이너에서 분산 데이터 병렬(Distributed DataParallel)에 대한 프로세스 그룹을 초기화합니다.
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # RPC를 초기화합니다.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # 트레이너는 마스터의 RPC를 기다리기만 합니다.
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # 매개변수 서버는 아무것도 하지 않습니다.
        pass

    # 모든 rpc가 끝날 때까지 차단합니다.
    rpc.shutdown()


if __name__ == "__main__":
    # 2개의 트레이너, 1개의 매개변수 서버, 1개의 마스터
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
# END run_worker
