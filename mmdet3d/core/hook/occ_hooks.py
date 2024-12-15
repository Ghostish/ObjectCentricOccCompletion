import torch
from mmcv.runner.hooks import HOOKS, Hook, OptimizerHook

from mmcv.parallel import is_module_wrapper

@HOOKS.register_module()
class EnableAddableTrainingHook(Hook):
    """Enable addable training after a given epoch.
 
    Args:
        epoch (int):
    """

    def __init__(self, from_epoch=12,to_epoch=None):
        self.from_epoch = from_epoch
        self.to_epoch = to_epoch

    def before_train_epoch(self, runner) -> None:
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        
        assert hasattr(model, "start_add_train"), "model should have start_add_train attribute"
        if self.from_epoch == runner.epoch + 1:
            model.start_add_train = True
            runner.logger.info(f"Enable addable training from now.")
        
        runner.logger.info(f"Addable training is set as {model.start_add_train} at epoch {runner.epoch + 1}.")
    
    def after_train_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if self.to_epoch is not None and self.to_epoch == runner.epoch + 1:
            model.start_add_train = False
            runner.logger.info(f"Disable addable training from now.")

    # def after_train_iter(self, runner):
    #     if self.every_n_iters(runner, self.interval):
    #         print("checking")
    #         runner.logger.info(runner.outputs["loss"])


