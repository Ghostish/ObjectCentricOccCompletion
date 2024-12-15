import torch
from mmcv.runner.hooks import HOOKS, Hook, OptimizerHook


@HOOKS.register_module()
class CheckLossStatusHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=1):
        self.interval = interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            print("checking")
            runner.logger.info(runner.outputs["loss"])


@HOOKS.register_module()
class CheckParametersStatusHook(Hook):
    def after_train_iter(self, runner):
        # Check for unused parameters
        for name, param in runner.model.named_parameters():
            if (
                param.requires_grad
                # and "occ_head" not in name
                # and "module.roi_head.bbox_head.trans_enc.linear2.bias" in name
            ):
                if param.grad is None:
                    # print(f"parameter: {name},grad is None")
                    print(f"\n++++++++++++++++++++++++++++++++++parameter: {name},grad is {param.grad}\n")
