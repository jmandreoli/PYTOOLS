# Author: Jean-Marc Andreoli
# Date: Oct 21, 2020

r"""
This module implements a wsgi application which manages of a set of tensorboard servers. It has been tested only with gunicorn.

* run for development: gunicorn --reload "tensorboard_manager_:main()"
* run for deployment: gunicorn tensorboard_manager:app
"""

import falcon,time,psutil,sys,subprocess,shutil,json,base64
from pathlib import Path
from collections import namedtuple
from functools import partial
from datetime import datetime
from subprocess import Popen
from string import Template

MAIN_TEMPLATE = Template(Path(__file__).with_name('tensorboard_manager-tmpl.html').read_text()) # INLINE
FAVICON = 'image/png',Path(__file__).with_name('tensorboard_manager.png').read_bytes() # INLINE

#==================================================================================================
class ExperimentResource:
  # A resource of this class represents an item in a store of Tensorboard experiments.
#==================================================================================================
  def __init__(self,store): self.store = store; self.procs = {}

#--------------------------------------------------------------------------------------------------
  def on_get(self,req,resp,exp=None):
    # Starts or retrieves the tensorboard server associated with the requested experiment,
    # and returns a view of it (results as html).
#--------------------------------------------------------------------------------------------------
    if exp:
      if not (self.store/exp).is_dir(): raise falcon.HTTPNotFound()
    else:
      cookies = req.get_cookie_values('tensorboard_last_visited_experiment')
      if cookies and (self.store/(exp:=cookies[0])).exists(): pass
      elif (L:=[f.name for f in self.store.iterdir()]): exp = L[0]
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
    # Returns various aspects of the requested experiment (results as json)
#--------------------------------------------------------------------------------------------------
    op = req.params['op']
    path = self.store/exp
    resp.content_type = 'text/json'
    if op == 'experiments': # list of all the experiments in the store
      body = json.dumps(sorted([f.name for f in self.store.iterdir()]))
    elif op=='runs': # list of runs of the requested experiment
      body = json.dumps(sorted([f.name for f in path.iterdir()],reverse=True))
    else:
      raise falcon.HTTPBadRequest('Operation not recognised: '+op,'Allowed operations are: experiments,runs')
    resp.body = body

#--------------------------------------------------------------------------------------------------
  def on_post_manage(self,req,resp,exp=None):
    # performs various updates to the requested experiment (results as plain text)
#--------------------------------------------------------------------------------------------------
    op = req.params['op']
    path = self.store/exp
    resp.content_type = 'text/plain'
    if op == 'restart': # restarts the tensorboard server of the requested experiment
      proc = self.procs[exp]
      proc.terminate()
      self.procs[exp] = tensorboard_start(path,req.env['SERVER_NAME'])
      body = f'Tensorboard for experiment "{exp}" has been restarted.'
    elif op == 'delete': # deletes the requested experiment altogether
      proc = self.procs.pop(exp)
      proc.terminate()
      shutil.rmtree(str(path))
      body = f'Experiment "{exp}" has been deleted.'
    elif op == 'delruns': # deletes selected runs from the requested experiment
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
def tensorboard_start(logdir,host,_Dummy=namedtuple('DummyTensorboard','url terminate')):
  # Starts a tensorboard server at logdir
#==================================================================================================
  proc = subprocess.Popen((str(Path(sys.executable).with_name('tensorboard')),'--host',host,'--port','0','--logdir',str(logdir)),text=True)
  p = psutil.Process(proc.pid)
  for _ in range(3):
    if (L:=p.connections()) and (a:=L[0]).status=='LISTEN':
      proc.url = f'http://{host}:{a.laddr.port}'
      return proc
    else: time.sleep(1)
  else:
    proc.terminate()
    proc = _Dummy(url=host+'/.dummytensorboard',terminate=lambda: None)

#==================================================================================================
class LoggingMiddleware:
  # A very basic logging middleware
#==================================================================================================
  def __init__(self,file): self.file = file
  def process_request(self,req,resp): print(datetime.now(),req.uri,file=self.file,flush=True)
  def process_response(self,req,resp,resource,req_succeeded): print(datetime.now(),req.uri,resp.status,file=self.file,flush=True)

#==================================================================================================
def main(store=None):
  # builds the main application (to be used by the wsgi server)
#==================================================================================================
  if store is None: store = Path.home()/'.tensorboard'/'store'
  else: store = Path(store).resolve()
  for proc in psutil.process_iter():
    if proc.name() == 'tensorboard': proc.kill()
  app = falcon.API(middleware=[LoggingMiddleware(sys.stderr)])
  app.req_options.auto_parse_form_urlencoded = True
  rsc = ExperimentResource(store)
  class HardResource:
    def __init__(self,val): self.val = val
    def on_get(self,req,resp): resp.content_type,resp.body = self.val
  app.add_route('/favicon.ico',HardResource(FAVICON))
  app.add_route('/.dummytensorboard',HardResource(('text/plain','The Tensorboard server could not be started')))
  app.add_route('/_{exp}',rsc,suffix='manage')
  app.add_route('/{exp}',rsc)
  return app

#==================================================================================================
def simple_serve(app,host='',port='6006'):
  # launches app through the basic python wsgi server
#==================================================================================================
  from wsgiref.simple_server import make_server
  with make_server(host,int(port),app) as httpd:
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass

#==================================================================================================
if __name__=='__main__': # produces a standalone file
  import re, stat
  this = Path(__file__).read_text()
  x = f'MAIN_TEMPLATE = Template(\'\'\'{MAIN_TEMPLATE.template}\'\'\')\n'
  this = re.sub('^MAIN_TEMPLATE = .+ # INLINE$',x,this,1,re.MULTILINE)
  x = f'FAVICON = \'{FAVICON[0]}\',base64.decodebytes(b\'\'\'\n{base64.encodebytes(FAVICON[1]).decode()}\'\'\')\n'
  this = re.sub('^FAVICON = .+ # INLINE$',x,this,1,re.MULTILINE)
  this = re.sub('^if __name__==\'__main__\':.*$','app = main()\nif __name__==\'__main__\': simple_serve(app,*sys.argv[1:])',this,1,re.MULTILINE|re.DOTALL)
  this = '# THIS IS A GENERATED FILE. DO NOT EDIT.'+30*'\n'+this
  src = Path(__file__).with_suffix('.new')
  with src.open('w') as v: v.write(this)
  trg = src.with_name('tensorboard_manager.py')
  src.rename(trg)
  trg.chmod(0o444) # to prevent editing
