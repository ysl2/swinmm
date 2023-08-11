import pysnooper
import datetime
import pathlib

TIMESTAMP = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
LOGDIR = pathlib.Path('/home/yusongli/Documents/swinmm/logs')
LOGDIR.mkdir(parents=True, exist_ok=True)

def snoop(**kwargs):
    return pysnooper.snoop(LOGDIR / f'{TIMESTAMP}.log', color=False, **kwargs)
