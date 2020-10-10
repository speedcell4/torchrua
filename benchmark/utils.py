from datetime import datetime


class Timer(object):
    def __init__(self):
        super(Timer, self).__init__()
        self.seconds = 0

    def __enter__(self):
        self.start_tm = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.seconds += (datetime.now() - self.start_tm).total_seconds()
        del self.start_tm
