
import omegaconf
import wandb
import os


class Sampler():

    def __init__(self, model, diff_params, args):

        self.model = model.eval() #is it ok to do this here?
        self.diff_params = diff_params #same as training, useful if we need to apply a wrapper or something
        self.args=args
        if self.args.tester.sampling_params.same_as_training:
            self.sde_hp = diff_params.sde_hp
        else:
            self.sde_hp = self.args.tester.sampling_params.sde_hp

        self.T = self.args.tester.sampling_params.T
        self.step_counter = 0


    #def setup_wandb(self):
    #     config=omegaconf.OmegaConf.to_container(
    #         self.args, resolve=True, throw_on_missing=True
    #     )
    #     self.wandb_run=wandb.init(project=self.args.logging.wandb.project, entity=self.args.logging.wandb.entity, config=config)
    #     self.wandb_run.name=self.args.tester.wandb.run_name +os.path.basename(self.args.model_dir)+"_"+self.args.exp.exp_name+"_"+self.wandb_run.id


