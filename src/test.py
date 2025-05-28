import os
import hydra
#import click
import torch
#from utils.torch_utils import distributed as dist

def _main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #assert torch.cuda.is_available()
    #device="cuda"

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    #dirname = os.path.dirname(__file__)
    #args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
            raise Exception(f"Model directory {args.model_dir} does not exist")

    #################
    ## diff params ##
    #################

    diff_params=hydra.utils.instantiate(args.diff_params)

    #############
    ## Network ##
    #############

    # it prints some logs.
    network=hydra.utils.instantiate(args.network)
    network=network.to(device)

    ##############
    ## test set ##
    ##############

    # also dset log something
    #try:
    test_set=hydra.utils.instantiate(args.dset.test)
    #except:
    #    test_set=None


    #############
    ## Tester  ##
    #############

    #from testing.tester import Tester

    #tester = hydra.utils.instantiate(
    #    args.tester,  # Partially instantiated object
    #    args=args,  # Pass in args
    #    network=network,  # Pass in network
    #    diff_params=diff_params,  # Pass in diff_params
    #    inference_train_set=inference_train_set,
    #    inference_test_set=inference_test_set,
    #    device=device
    #)

    #tester=hydra.utils.instantiate(args.tester, args=args, network=network, diff_params=diff_params,  inference_train_set=inference_train_set, inference_test_set=inference_test_set, device=device)
    from testing.tester import Tester
    tester=Tester(args=args, network=network, diff_params=diff_params,test_set=test_set, device=device) #this will be used for making demos during training

    # Print optirain.
    print()
    print('Training options:')
    print()
    print(f'Output directory:           {args.model_dir}')
    print(f'Network architecture:       {args.network._target_}')
    print(f'Diffusion parameterization: {args.diff_params._target_}')
    print(f'Experiment:                 {args.exp.exp_name}')
    print(f'Tester:                     {args.tester.tester._target_}')
    print(f'Sampler:                    {args.tester.sampler._target_}')
    print(f'Checkpoint:                 {args.tester.checkpoint}')
    print(f'sample rate:                {args.exp.sample_rate}')
    # print(f'audio len:                  {args.exp.audio_len}')
    audio_len = args.exp.audio_len if not "audio_len" in args.tester.unconditional.keys() else args.tester.unconditional.audio_len
    print(f'audio len:                  {audio_len}')
    print()


    if args.tester.checkpoint != 'None':
        ckpt_path=os.path.join(args.model_dir, args.tester.checkpoint)
        tester.load_checkpoint(ckpt_path) 
    else:
        print("trying to load latest checkpoint")
        tester.load_latest_checkpoint()

    tester.do_test()

@hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
def main(args):
    #torch.cuda.set_device(args.gpu)
    _main(args)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------