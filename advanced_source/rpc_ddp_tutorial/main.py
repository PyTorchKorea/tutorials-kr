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
    하이브리드 모델은 희소 부분과 밀집 부분이 있는 모델입니다.
    1) 밀집 부분은 분산 데이터 병렬을 사용하여 모든 트레이너에 걸쳐 복제되는 nn.Linear 모듈입니다.
    2) 희소 부분은 매개변수 서버에서 nn.EmbeddingBag를 가지고 있는 원격 모듈입니다.
    이 원격 모델은 매개변수 서버의 임베딩 테이블에 대한 원격 참조(Remote Reference)를 얻을 수 있습니다.
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
    각 트레이너는 매개변수 서버에서 임베딩 찾기 작업(embedding lookup)을 포함하고
    전역에서 nn.Linear를 실행하는 순방향 전달(forward pass)를 실행합니다.
    역방향 전달(backward pass) 동안에, 분산 데이터 병렬은 밀집 부분(nn.Linear)에 대한 변화도 집계를 담당하고
    분산 autograd는 변화도 업데이트가 매개변수 서버로 전파되도록 합니다.
    """

    # 모델을 생성합니다.
    model = HybridModel(remote_emb_module, rank)

    # 모든 모델 매개변수를 분산 옵티마이저에 대한 rrefs로 검색합니다.

    # 임베딩 테이블에 대한 매개변수를 검색합니다.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    # model.fc.parameters()은 오직 지역 매개변수만 포함합니다.
    # 참고: 여기에서는 model.parameters()를 호출할 수 없습니다.
    # 왜냐하면 parameters()가 아니라 remote_parameters()를 지원하는
    # remote_emb_module.parameters()를 호출하기 때문입니다.
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # 분산 옵티마이저를 설정합니다.
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

    # 100개의 에폭(epoch)에 대해 학습합니다.
    for epoch in range(100):
        # 분산 autograd context를 생성합니다.
        for indices, offsets, target in get_next_batch(rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # 분산 역방향 전달(distributed backward pass)을 실행합니다.
                dist_autograd.backward(context_id, [loss])

                # 분산 옵티마이저를 갱신합니다.
                opt.step(context_id)

                # 반복될 때마다 다른 변화도를 호스팅하는 하나의 다른 분산 autograd context를 생성하므로
                # 변화도를 0으로 만들 필요가 없습니다.
        print("Training done for epoch {}".format(epoch))
        # END run_trainer

# BEGIN run_worker
def run_worker(rank, world_size):
    r"""
    이 함수는 RPC를 초기화하고, 함수를 호출하고,
    RPC를 종료하는 래퍼 함수(wrapper function)입니다.
    """

    # 포트 충돌을 피하기 위해 init_rpc 및 init_process_group에 대해
    # TCP init_method에서 다른 포트 번호를 사용해야 합니다.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # 순위(rank) 2는  master를 의미하고, 3은 매개변수 서버, 그리고 0과 1은 트레이너를 뜻합니다.
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
                trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank)
            )
            futs.append(fut)

        # 모든 학습이 끝날 때까지 기다립니다.
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        # 트레이너에서 분산 데이터 병렬에 대한 프로세스 그룹을 초기화합니다.
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
