import torch
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR


class MultiStepRestartLR(_LRScheduler):
    """MultiStepLR with restarts.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        restarts (list): List of restart epoch indices. Default: None.
        restart_weights (list): List of weights for each restart.
            Default: [1].
    """

    def __init__(self, optimizer, milestones, gamma=0.1, restarts=None, restart_weights=None, last_epoch=-1):
        self.milestones = milestones
        self.gamma = gamma
        self.restarts = restarts if restarts else [0]
        self.restart_weights = restart_weights if restart_weights else [1]
        assert len(self.restarts) == len(self.restart_weights), 'restarts and restart_weights should have the same length.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self._get_closed_milestone(self.last_epoch, self.milestones)
                for group in self.optimizer.param_groups]

    def _get_closed_milestone(self, epoch, milestones):
        """Get the index of the most recent milestone that is <= epoch."""
        for i, m in enumerate(milestones):
            if epoch < m:
                return max(0, i - 1)
        return len(milestones) - 1


class CosineAnnealingRestartLR(_LRScheduler):
    """Cosine annealing with restarts.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        periods (list): List of period lengths for each restart.
        restart_weights (list): List of weights for each restart.
            Default: [1].
        eta_min (float): Minimum learning rate. Default: 0.
    """

    def __init__(self, optimizer, periods, restart_weights=None, eta_min=0, last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights if restart_weights else [1]
        self.eta_min = eta_min
        assert len(self.periods) == len(self.restart_weights), 'periods and restart_weights should have the same length.'
        self.cumulative_periods = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = self._get_closed_period(self.last_epoch, self.cumulative_periods)
        current_period = self.periods[idx]
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_epoch = self.last_epoch - nearest_restart

        return [self.eta_min + (base_lr * current_weight - self.eta_min) *
                (1 + torch.cos(torch.tensor(current_epoch / current_period * 3.141592653589793))) / 2
                for base_lr in self.base_lrs]

    def _get_closed_period(self, epoch, cumulative_periods):
        """Get the index of the most recent period that contains epoch."""
        for i, p in enumerate(cumulative_periods):
            if epoch < p:
                return i
        return len(cumulative_periods) - 1
