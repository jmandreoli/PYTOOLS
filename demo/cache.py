# File:                 demo/cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module

if __name__=='__main__':
    import sys, multiprocessing
    from time import sleep
    from cache import simplefunc, longfunc, process, demo
    DEMOS = (
        (simplefunc,('simplefunc(1,2)','simplefunc(1,y=2,z=3)')),
        (process,('process(ARG(1,b=2),ARG(3)).v','process(ARG(1,2),ARG(4)).v')),
        (longfunc,('longfunc(42,6)',)),
    )
    mp = multiprocessing.get_context('spawn') # not 'fork' because there are lots of threads around
    for f,L in DEMOS:
        print('-------------------------\nClearing',f)
        f.clear()
        w1 = mp.Process(target=demo,args=(1,L,))
        w2 = mp.Process(target=demo,args=(2,L,))
        w1.start(); sleep(2); w2.start()
        w1.join(); w2.join()
        try: input('RET: continue; Ctrl-C: stop')
        except: print(); break
        print()
    sys.exit(0)

#--------------------------------------------------------------------------------------------------

from pathlib import Path; DIR = Path(__file__).parent/'cache.dir'
from myutil.cache import lru_persistent_cache, lru_persistent_process_cache, ARG

@lru_persistent_cache(db=DIR,ignore=('z',))
def simplefunc(x,y=3,z=8): return x,y

@lru_persistent_cache(db=DIR,ignore=('delay',))
def longfunc(x,delay=10):
    from time import sleep
    sleep(delay)
    return x

def stepA(state,a,b):
    state.a = a
    state.b = b
    state.u = a+b
    return state

def stepB(state,c):
    state.v = c*state.u
    return state

process = lru_persistent_process_cache((stepA,dict(db=DIR)),(stepB,dict(db=DIR)))

def demo(r,L,q=None):
    import logging, logging.handlers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if q is None:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('Process %(process)d: %(message)s'))
    else:
        h = logging.handlers.QueueHandler(q)
    logger.addHandler(h)
    for x in L:
        logger.info('Computing [Run %s]: %s',r,x)
        logger.info('Result [Run %s]: %s = %s',r,x,eval(x))

