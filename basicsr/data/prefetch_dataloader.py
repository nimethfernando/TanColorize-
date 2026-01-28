import torch
from torch.utils.data import DataLoader


class CPUPrefetcher:
    """CPU prefetcher."""

    def __init__(self, loader):
        self.original_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.original_loader)


class CUDAPrefetcher:
    """CUDA prefetcher.

    Ref:
        https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, opt):
        self.original_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.opt = opt
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self._async_transfer_data(self.next_input)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input

    def _async_transfer_data(self, input):
        if isinstance(input, dict):
            return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in input.items()}
        elif isinstance(input, (list, tuple)):
            return [v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for v in input]
        elif isinstance(input, torch.Tensor):
            return input.cuda(non_blocking=True)
        else:
            return input

    def reset(self):
        self.loader = iter(self.original_loader)
        self.preload()
