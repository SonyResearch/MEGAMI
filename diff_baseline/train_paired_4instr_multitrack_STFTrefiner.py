import os
import sys
import re
import json
import hydra
import torch
import numpy as np


def worker_init_fn(worker_id, rank=0):
    st = np.random.get_state()[2]
    seed= st + worker_id + rank*100
    print(f"worker_init_fn {worker_id} rank {rank} st {st} seed {seed}")

    np.random.seed(seed)



def _main(args):

    print(f"Current Working Directory: {os.getcwd()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)


    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    train_set = hydra.utils.instantiate(args.dset.train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.exp.batch_size,
                                               num_workers=args.exp.num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn, timeout=0, prefetch_factor=5)
    train_loader = iter(train_loader)

    val_set_dict = {}
    val_set = hydra.utils.instantiate(args.dset.validation)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.exp.val_batch_size  )
    #val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.exp.val_batch_size, num_workers=args.exp.num_workers,
    #                                              pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=lambda x: x)
    val_set_dict[args.dset.validation.mode] = val_loader

    #try:
    val_set_2 = hydra.utils.instantiate(args.dset.validation_2)
    val_loader_2 = torch.utils.data.DataLoader(dataset=val_set_2, batch_size=args.exp.val_batch_size)
    #val_loader_2 = torch.utils.data.DataLoader(dataset=val_set_2, batch_size=args.exp.val_batch_size, num_workers=args.exp.num_workers,
    #                                              pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=lambda x: x)
    val_set_dict[args.dset.validation_2.mode] = val_loader_2
    #except:
    #    print("Second validation set not found, using only first one")
    #    pass

    try:
        val_set_3 = hydra.utils.instantiate(args.dset.validation_3)
        val_loader_3 = torch.utils.data.DataLoader(dataset=val_set_3, batch_size=args.exp.val_batch_size, num_workers=args.exp.num_workers,
                                                    pin_memory=True)
        val_set_dict[args.dset.validation_3.mode] = val_loader_3
    except:
        print("third validation set not found, using only first one")
        pass

    try:
        val_set_4 = hydra.utils.instantiate(args.dset.validation_4)
        val_loader_4 = torch.utils.data.DataLoader(dataset=val_set_4, batch_size=args.exp.val_batch_size )
        val_set_dict[args.dset.validation_4.mode] = val_loader_4
    except:
        print("fourth validation set not found, using only first one")
        pass

    print("Validation set keys:")
    print(val_set_dict.keys())

    print("path before diff params", sys.path)
    # Diffusion parameters
    print("Diffusion parameters:", args.diff_params)
    diff_params = hydra.utils.instantiate(args.diff_params)  # instantiate in trainer better

    # Network
    if args.network._target_ == 'networks.unet_octCQT.UNet_octCQT':
        network = hydra.utils.instantiate(args.network, sample_rate=args.exp.sample_rate, audio_len=args.exp.audio_len,
                                          device=device)  # instantiate

    else:
        network = hydra.utils.instantiate(args.network)  # instantiate in trainer better

    network = network.to(device)

    # Tester
    args.tester.sampling_params.same_as_training = True  # Make sure that we use the same HP for sampling as the ones used in training
    args.tester.wandb.use = False  # Will do that in training

    # tester=hydra.utils.instantiate(args.tester, args, network, diff_params)
    from testing.tester_multitrack import Tester
    import copy
    network_tester = copy.deepcopy(network).eval().requires_grad_(False)
    tester = Tester(args, network_tester, diff_params, device=device, in_training=True, test_set_dict=val_set_dict)

    # Trainer
    #print(args.exp.trainer)

    #trainer = hydra.utils.instantiate(args.exp.trainer, args, train_loader, network, diff_params, tester, device, distributed=False)  # This works
    print(f"Current Working Directory: {os.getcwd()}")

    print("path before training", sys.path)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from training.trainer_ddp_paired_4instr_multitrack_STFTrefiner import Trainer
    #print("args", args)
    trainer = Trainer( args=args, dset=train_loader, network=network, diff_params=diff_params, tester=tester, device=device, distributed=False)  # This works

    # Print options.
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


@hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
def main(args):
    _main(args)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------