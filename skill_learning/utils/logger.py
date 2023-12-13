import os
from datetime import datetime

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, hps, global_step=0, best_metric=1e10, online=True, offline=True, model=None,):
        self.online = online
        self.offline = offline
        self._global_step = global_step
        self._best_metric = best_metric

        self._exp_log_name = hps['exp_name'] + datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
        self._log_path = os.path.join(hps['base_dir'], self._exp_log_name)
        os.makedirs(self._log_path, exist_ok=True)

        # print experiment summary
        print("---Experiment---")
        # print(yaml.dump(hps, indent=4))
        print(f"Msg: {hps['msg']}")
        print(f"Exp: {hps['exp_name']}")
        print(f"Log path: {self._log_path}")
        print("----------------")

        if self.offline: self.writer = SummaryWriter(self._log_path)
        if self.online:
            self.wandb = wandb.init(project='skill_learning',
                                    config=hps,
                                    name=self._exp_log_name,
                                    dir=self._log_path,
                                    reinit=True)

        if self.offline: self.writer.add_text('hyperparams', yaml.dump(hps, indent=4).replace(' ', '&nbsp;').replace('\n', '  \n'))
        if model is not None:
            if self.offline: self.writer.add_text("architecture", str(model).replace(' ', '&nbsp;').replace('\n', '  \n'))
            if self.online: self.wandb.watch(model)

        # Save configs to the log path
        with open(os.path.join(self._log_path, 'configs.yaml'), 'w') as f:
            yaml.dump(hps, f)

    def log(self, metrics, step=None, mode=None):
        step = self._global_step if step is None else step
        for k, v in metrics.items():
            metric_name = f'{mode}/{k}' if mode is not None else k
            self._log_individual(metric_name, v, step)

    def step(self, step=None):
        self._global_step = self._global_step + 1 if step is None else step

    def _log_individual(self, name, value, step):
        if self.offline: self.writer.add_scalar(name, value, global_step=step)
        if self.online: wandb.log({name: value})

    def log_histogram(self, name, data, step=None):
        step = self._global_step if step is None else step
        if self.offline: self.writer.add_histogram(name, data, step)
        if self.online: wandb.log({name: wandb.Histogram(data.cpu().detach())})

    def log_model(self, prefix, model, overwrite=False):
        model_name = f'{prefix}.pth' if overwrite else f'{prefix}_{self._global_step}.pth'
        log_path = os.path.join(self._log_path, model_name)
        if self.offline: torch.save(model.state_dict(), log_path)

    def is_better_loss(self, new_metric):
        result = False
        if new_metric < self._best_metric:
            result = True
            self._best_metric = new_metric
        return result

    def save_state(self, model, optimizer, scheduler):
        log_path = os.path.join(self._log_path, 'checkpoint.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': self._global_step,
            'best_loss': self._best_metric,
            'exp_log_name': self._exp_log_name,
        }, log_path)

