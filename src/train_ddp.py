import os
import hydra
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig
import datetime

def worker_init_fn(worker_id, rank=0):
    st = np.random.get_state()[2]
    seed= st + worker_id + rank*100
    print(f"worker_init_fn {worker_id} rank {rank} st {st} seed {seed}")

    np.random.seed(seed)


def _main(rank, world_size, args):
    print(f"Rank {rank} is starting training")
    #setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if rank == 0:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

    train_set = hydra.utils.instantiate(args.dset.train)
    #train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.exp.batch_size,
                                               num_workers=args.exp.num_workers, pin_memory=True,
                                               worker_init_fn=lambda x: worker_init_fn(x, rank), timeout=0, prefetch_factor=20)
    train_loader = iter(train_loader)

    val_set_dict = {}
    if rank == 0:
        val_set = hydra.utils.instantiate(args.dset.validation)
        val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.exp.val_batch_size, num_workers=args.exp.num_workers,
                                             pin_memory=True, worker_init_fn=lambda x: worker_init_fn(x, rank=rank))
        val_set_dict[args.dset.validation.mode] = val_loader
    else:
        val_set = None
        val_loader = None
    

    try:
        if rank == 0:
            val_set = hydra.utils.instantiate(args.dset.validation_2)
            val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.exp.val_batch_size, num_workers=args.exp.num_workers,
                                                 pin_memory=True, worker_init_fn=lambda x: worker_init_fn(x, rank=rank))
            val_set_dict[args.dset.validation_2.mode] = val_loader
    except:
        print("Second validation set not found, using only first one")
        pass
    try:
        if rank == 0:
            val_set = hydra.utils.instantiate(args.dset.validation_3)
            val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.exp.val_batch_size, num_workers=args.exp.num_workers,
                                                 pin_memory=True, worker_init_fn=lambda x: worker_init_fn(x, rank=rank))
            val_set_dict[args.dset.validation_3.mode] = val_loader
    except:
        print("Second validation set not found, using only first one")
        pass

    try:
        if rank == 0:
            val_set = hydra.utils.instantiate(args.dset.validation_4)
            val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.exp.val_batch_size, num_workers=args.exp.num_workers,
                                                 pin_memory=True, worker_init_fn=lambda x: worker_init_fn(x, rank=rank))
            val_set_dict[args.dset.validation_4.mode] = val_loader
    except:
        print("Second validation set not found, using only first one")
        pass

    # Diffusion parameters
    diff_params = hydra.utils.instantiate(args.diff_params)

    # Network
    if args.network._target_ == 'networks.unet_octCQT.UNet_octCQT':
        network = hydra.utils.instantiate(args.network, sample_rate=args.exp.sample_rate, audio_len=args.exp.audio_len,
                                          device=device)
    else:
        network = hydra.utils.instantiate(args.network)

    network = network.to(device)

    # Tester
    args.tester.sampling_params.same_as_training = True
    args.tester.wandb.use = False

    from testing.tester import Tester
    import copy
    network_tester = copy.deepcopy(network).eval().requires_grad_(False)
    tester = Tester(args, network_tester, diff_params, device=device, in_training=True, test_set_dict=val_set_dict)

    # Trainer
    trainer = hydra.utils.instantiate(args.exp.trainer, args, train_loader, network, diff_params, tester, device, rank, world_size)

    # Print options
    if rank == 0:
        print()
        print('Training options:')
        print()
        print(f'Output directory:        {args.model_dir}')
        print(f'Network architecture:    {args.network._target_}')
        print(f'Dataset:    {args.dset.train._target_}')
        print(f'Diffusion parameterization:  {args.diff_params._target_}')
        print(f'Batch size:              {args.exp.batch_size}')
        print(f'Device:       {device}')
        print()

    trainer.training_loop()

def init_distributed_mode(rank, world_size):
    # Set the device for the current process
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

    # Initialize the process group with the specified device_id
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=3600),
    )

@hydra.main(config_path="conf", config_name="conf", version_base=None)
def main(cfg: DictConfig):

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    init_distributed_mode(rank, world_size)

    #dist.init_process_group(backend="nccl", init_method="env://")
    #rank = dist.get_rank()
    #world_size = dist.get_world_size()
    #print(f"Rank {rank}/{world_size} is running")
    dist.barrier()

    _main(rank,world_size, cfg)

if __name__ == "__main__":
    main()