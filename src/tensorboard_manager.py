# File:                 tensorboard_manager.py
# Creation date:        2020-10-20
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              A server of tensorboard servers
#

from __future__ import annotations
import falcon,time,psutil,sys,shutil,json
from pathlib import Path
from collections import namedtuple
from datetime import datetime
from subprocess import Popen
from string import Template

MAIN_TEMPLATE = Template(Path(__file__).with_name('tensorboard_manager-tmpl.html').read_text())
FAVICON = 'image/png',Path(__file__).with_name('tensorboard_manager.png').read_bytes()

#==================================================================================================
class ExperimentResource:
  r"""
A resource of this class represents a store of Tensorboard experiments.

:param store: the path to an existing directory, each member of which is an experiment, i.e. a set of tensorboard logs
  """
#==================================================================================================
  def __init__(self,store): self.store = store; self.procs = {}

#--------------------------------------------------------------------------------------------------
  def on_get(self,req,resp,exp=None):
    r"""
Starts or retrieves the tensorboard server associated with experiment *exp*, and returns a view of it (result: text/html).
    """
#--------------------------------------------------------------------------------------------------
    if exp:
      if not (self.store/exp).is_dir(): raise falcon.HTTPNotFound()
    else:
      cookies = req.get_cookie_values('tensorboard_last_visited_experiment')
      if cookies and (self.store/(exp:=cookies[0])).exists(): pass
      elif L:=[f.name for f in self.store.iterdir()]: exp = L[0]
      else: raise falcon.HTTPNotFound()
    proc = self.procs.get(exp)
    if proc is None:
      self.procs[exp] = proc = tensorboard_start(self.store/exp,req.env['SERVER_NAME'])
    url = proc.url
    resp.content_type = 'text/html'
    resp.body = MAIN_TEMPLATE.substitute(experiment=exp,tensorboard=url)
    resp.set_cookie('tensorboard_last_visited_experiment',exp)

#--------------------------------------------------------------------------------------------------
  def on_get_manage(self,req,resp,exp=None):
    r"""
Returns various aspects of experiment *exp* (result: text/json)
    """
#--------------------------------------------------------------------------------------------------
    op = req.params['op']
    path = self.store/exp
    resp.content_type = 'text/json'
    if op == 'experiments': # list of all the experiments in the store
      body = json.dumps(sorted([f.name for f in self.store.iterdir()]))
    elif op=='runs': # list of runs of *exp*
      body = json.dumps(sorted([f.name for f in path.iterdir()],reverse=True))
    else:
      raise falcon.HTTPBadRequest('Operation not recognised: '+op,'Allowed operations are: experiments,runs')
    resp.body = body

#--------------------------------------------------------------------------------------------------
  def on_post_manage(self,req,resp,exp=None):
    r"""
Performs various updates to experiment *exp* (result: text/plain).
    """
#--------------------------------------------------------------------------------------------------
    op = req.params['op']
    path = self.store/exp
    resp.content_type = 'text/plain'
    if op == 'restart': # forces restart of the tensorboard server of *exp*
      proc = self.procs[exp]
      proc.terminate()
      self.procs[exp] = tensorboard_start(path,req.env['SERVER_NAME'])
      body = f'Tensorboard for experiment "{exp}" has been restarted.'
    elif op == 'delete': # deletes experiment *exp* altogether
      proc = self.procs.pop(exp)
      proc.terminate()
      shutil.rmtree(str(path))
      body = f'Experiment "{exp}" has been deleted.'
    elif op == 'delruns': # deletes selected runs from *exp*
      L = req.get_param_as_list('remove',default=())
      n = 0
      for r in L:
        f = path/r
        if f.exists(): n += 1; shutil.rmtree(str(f))
      body = f'{n} runs have been removed from experiment "{exp}".'
    else:
      raise falcon.HTTPBadRequest('Operation not recognised: '+op,'Allowed operations are: restart,delete,delruns')
    resp.body = body

#==================================================================================================
def tensorboard_start(logdir:str,host:str,_Dummy=namedtuple('DummyTensorboard','url terminate')):
  r"""
Starts a tensorboard server listening to host *host* at a random free port. Returns the :class:`subprocess.Popen` instance, augmented with attribute :attr:`url` set to the URL for interaction with the tensorboard server.
  """
#==================================================================================================
  proc = Popen((str(Path(sys.executable).with_name('tensorboard')),'--host',host,'--port','0','--logdir',str(logdir)),text=True)
  p = psutil.Process(proc.pid)
  for _ in range(3):
    if (L:=p.connections()) and (a:=L[0]).status=='LISTEN':
      proc.url = f'http://{host}:{a.laddr.port}'
      return proc
    else: time.sleep(1)
  else:
    proc.terminate()
    return _Dummy(url=host+'/.dummytensorboard',terminate=lambda: None)

#==================================================================================================
class LoggingMiddleware:
  # A very basic logging middleware
#==================================================================================================
  def __init__(self,file): self.file = file
  def process_request(self,req,resp): print(datetime.now(),req.uri,file=self.file,flush=True)
  def process_response(self,req,resp,resource,req_succeeded): print(datetime.now(),req.uri,resp.status,file=self.file,flush=True)

#==================================================================================================
def main(store=None):
  r"""
Returns the main wsgi application object (to be used by the wsgi server).

:param store: the store of tensorboard experiments to manage. Default is ``~/.tensorboard/store``.
  """
#==================================================================================================
  if store is None: store = Path.home()/'.tensorboard'/'store'
  else: store = Path(store).resolve()
  for proc in psutil.process_iter():
    if proc.name() == 'tensorboard': proc.kill()
  app = falcon.App(middleware=[LoggingMiddleware(sys.stderr)])
  app.req_options.auto_parse_form_urlencoded = True
  rsc = ExperimentResource(store)
  class HardwiredResource:
    def __init__(self,val): self.val = val
    def on_get(self,req,resp): resp.content_type,resp.body = self.val
  app.add_route('/favicon.ico',HardwiredResource(FAVICON))
  app.add_route('/.dummytensorboard',HardwiredResource(('text/plain','The Tensorboard server could not be started')))
  app.add_route('/_{exp}',rsc,suffix='manage')
  app.add_route('/{exp}',rsc)
  return app

#==================================================================================================
def simple_serve():
  r"""
Launches an instance of the app through the basic python wsgi server.
  """
#==================================================================================================
  from wsgiref.simple_server import make_server
  from argparse import ArgumentParser
  p = ArgumentParser('Tensorboard manager')
  p.add_argument('--bind','-b',default=':6006',metavar='BIND',help='host:port on which to listen; host can be empty for localhost')
  p.add_argument('store',nargs='?',default=None,metavar='STORE',help='root path of the directory containing the tensorboard experiments to serve')
  args = p.parse_args(sys.argv[1:])
  bind = args.bind.split(':',1); bind.append('80'); host,port = bind[:2]
  with make_server(host,int(port),main(args.store)) as httpd:
    print(f'Tensorboard manager running at http://{host}:{port}')
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass

#==================================================================================================
if __name__=='__main__': simple_serve()
