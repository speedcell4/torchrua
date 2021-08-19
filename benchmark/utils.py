from datetime import datetime
from typing import Type


class Timer(object):
    def __init__(self) -> None:
        super(Timer, self).__init__()
        self.seconds = 0
        self.num_runs = 0

    def __enter__(self) -> None:
        self.start_tm = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.seconds += (datetime.now() - self.start_tm).total_seconds()
        self.num_runs += 1
        del self.start_tm

    @property
    def averaged_seconds(self) -> float:
        return self.seconds / max(1, self.num_runs)


class TimerSuit(object):
    def __init__(self) -> None:
        super(TimerSuit, self).__init__()
        self.rua_compile = Timer()
        self.rua_forward = Timer()
        self.rua_backward = Timer()

        self.naive_forward = Timer()
        self.naive_backward = Timer()

    def report(self) -> None:
        if self.rua_compile.num_runs == 0:
            naive_forward = self.naive_forward.averaged_seconds
            naive_backward = self.naive_backward.averaged_seconds
            print(f'PyTorch  ({naive_forward + naive_backward :.4f} sec) = '
                  f'forward ({naive_forward:.4f} sec) + '
                  f'backward ({naive_backward :.4f} sec)')

            rua_forward = self.rua_forward.averaged_seconds
            rua_backward = self.rua_backward.averaged_seconds
            print(f'TorchRua ({rua_forward + rua_backward :.4f} sec) = '
                  f'forward ({rua_forward:.4f} sec) + '
                  f'backward ({rua_backward :.4f} sec)')
        else:
            naive_forward = self.naive_forward.averaged_seconds
            naive_backward = self.naive_backward.averaged_seconds
            print(f'PyTorch  ({naive_forward + naive_backward :.4f} sec) = '
                  f'                       forward ({naive_forward:.4f} sec) + '
                  f'backward ({naive_backward :.4f} sec)')

            rua_compile = self.rua_compile.averaged_seconds
            rua_forward = self.rua_forward.averaged_seconds
            rua_backward = self.rua_backward.averaged_seconds
            print(f'TorchRua ({rua_compile + rua_forward + rua_backward :.4f} sec) = '
                  f'compile ({rua_compile:.4f} sec) + '
                  f'forward ({rua_forward:.4f} sec) + '
                  f'backward ({rua_backward :.4f} sec)')


def timeit(fn):
    def _timeit(func: Type[fn], num_runs: int = 100):
        timer = TimerSuit()
        for _ in range(num_runs):
            func(timer=timer)
        timer.report()

    _timeit.__name__ = fn.__name__
    return _timeit
