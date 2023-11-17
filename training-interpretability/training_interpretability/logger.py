import abc
import sys


class Logger(abc.ABC):

    def print(self, txt: str):
        raise NotImplementedError
    
    def progress(self, *args, wrapper: bool = False):
        if wrapper:
            self.print("-" * 89)
        self.print("| " + " | ".join(args))
        if wrapper:
            self.print("-" * 89)
    
    def __call__(self, *args, **kwargs):
        self.print(*args, **kwargs)

class StdoutLogger(Logger):

    def print(self, txt: str):
        sys.stdout.write(txt)
        sys.stdout.flush()
