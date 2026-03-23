import functools
import logging
import os
import sys

import torch
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger_dist(output=None,
                      distributed_rank=0,
                      *,
                      color=True,
                      name="moco",
                      abbrev_name=None):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    logging.root = logger  # main logger.
    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


def format_dict(d, dec=4) -> str:
    s = []
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = float(v.detach().cpu().numpy())
        if isinstance(v, float):
            v = round(v, dec)
        v = str(v)
        s.append(f'\t{k:15}: {v:10}')
    return '\n'.join(s)


def format_list(l1, l2, dec=4) -> str:
    s = []
    for k, v in zip(l1, l2):
        if isinstance(v, torch.Tensor):
            v = float(v.detach().cpu().numpy())
        if isinstance(v, float):
            v = round(v, dec)
        v = str(v)
        s.append(f'\t{k:15}: {v:10}')
    return '\n'.join(s)


if __name__ == '__main__':
    classes = ['ceiling',
               'floor',
               'wall',
               'beam',
               'column',
               'window',
               'door',
               'chair',
               'table',
               'bookcase',
               'sofa',
               'board',
               'clutter']
    ious = [0.933, 0.977, 0.8481, 0.0005, 0.2784, 0.6131, 0.5104, 0.8172, 0.8357, 0.707, 0.7039, 0.4704, 0.5696]
    s = format_list(classes, ious, dec=2)

    s2 = format_dict({
        'loss': torch.tensor(0.89323),
        'lr': 0.000001,
        'diff': torch.tensor(1.24523),
        'time_cost': '12441s'
    })
    print(f'ious:\n{s}'
          + f'\ntrain:\n{s2}')
