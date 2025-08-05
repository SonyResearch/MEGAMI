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

    val_set_dict = {}
    test_set = hydra.utils.instantiate(args.dset.test)
    #test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1  )
    #val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.exp.val_batch_size, num_workers=args.exp.num_workers,
    #                                              pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=lambda x: x)
    val_set_dict["test"] = test_set


    print("Validation set keys:")
    print(val_set_dict.keys())

    print("path before diff params", sys.path)
    # Diffusion parameters

    # Network
    network = hydra.utils.instantiate(args.network)  # instantiate in trainer better

    network = network.to(device)

    # Tester
    args.tester.wandb.use = False  # Will do that in training

    # tester=hydra.utils.instantiate(args.tester, args, network, diff_params)
    from testing.tester_multitrack_WUN_4instr import Tester
    import copy
    network_tester = copy.deepcopy(network).eval().requires_grad_(False)
    tester = Tester(args, network_tester,  device=device, in_training=True, test_set_dict=val_set_dict)

    tester.load_checkpoint(args.tester.checkpoint)

    tester.do_test()



@hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
def main(args):
    _main(args)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------