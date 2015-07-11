# File:                 demo/remote.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the remote package

import sys,os
from myutil.remote import shclients, sshclients
from myutil.remote.sge import sgeclients, sgesetup
sgesetup('spider-1','/usr/local/grid/XRCE/common/settings.sh')

if __name__=='__main__':
    from remote import f,h
    w = shclients(1,check=True)[0]
    try:
      w.declare(f); w.declare(h)
      print(tuple(h(3)))
    finally: w.shutdown()
    sys.exit(0)
    from remote import demo1, demo2
    HOSTS = (('spider-11',2),('spider-20',3))
    for d in demo1,demo2:
        for hosts,mode in (HOSTS,'SSH'), (None,'SGE'):
            print('----------------------------\n{} {}'.format(d.__name__,mode))
            d(hosts)
        try: input('Ret: continue; Ctrl-C: abort')
        except: print(); break
        else: print()
    sys.exit(0)

#--------------------------------------------------------------------------------------------------

def sclient(n,hosts=None,**ka):
    # launches *n* servers, using either SSH if *hosts* is not None or SGE otherwise
    # and returns corresponding proxies
    return sgeclients(n,**ka) if hosts is None else sshclients((n,hosts),**ka)

def demo1(hosts):
    ws = sclient(2,hosts)
    try:
        for w in ws:
            # assign methods 'getpid', 'uname', 'f' to *w*'s server
            # note that f is a generator function
            w.declare(os.getpid); w.declare(os.uname); w.declare(f)
        for w in ws:
            # now call these methods from proxy *w*
            print('\tprocess: {} on host: {}'.format(w.getpid(), w.uname().nodename.split('.')[0]))
            print('\tf(3)=',tuple(w.f(3)))
    finally: ws.shutdown()

def demo2(hosts):
    w = sclient(1,HOSTS,check=True)[0]
    try:
        # assign method 'g' to *w*'s server
        w.declare('remote.g')
        # that method itself launches servers and returns proxies
        ws = w.g(2,hosts)
        try:
            # call some methods from each returned proxy
            for w1 in ws: print('\tprocess: {} on host: {}'.format(w1.getpid(), w1.uname().nodename.split('.')[0]))
        finally:
            ws.shutdown()
    finally:
        w.shutdown()

def h(x): return (x,2*x,4*x)
def f(x):
    yield from (x,2*x,4*x)

def g(n,hosts):
    ws = sclient(n,hosts)
    for w in ws:
        w.declare(os.getpid)
        w.declare(os.uname)
    return ws


