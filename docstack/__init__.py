from .main import DocStack
import contextlib

#==================================================================================================
# Utilities
#==================================================================================================

def system(cmd,op,timeout):
  import sys, subprocess, traceback
  status = 'success'
  print('>>>OPERATION:',op,file=sys.stderr,flush=True)
  try: subprocess.check_call(cmd,stdout=sys.stderr,stderr=sys.stderr,timeout=timeout)
  except subprocess.TimeoutExpired as e: status = 'timeout'
  except subprocess.CalledProcessError as e: status = 'error[{}]'.format(e.returncode)
  except: status = 'syserror'; traceback.print_exc(file=sys.stderr)
  sys.stderr.flush()
  return status

@contextlib.contextmanager
def LogStep(logger,label,showdate=False):
  import time
  from datetime import datetime  
  tstart = time.perf_counter(), time.process_time()
  t = '[{}]'.format(datetime.now().isoformat()) if showdate else ''
  logger.info('%sBEGIN[%s]',t,label)
  yield
  t = '[{}]'.format(datetime.now().isoformat()) if showdate else ''
  logger.info('%sEND[%s] elapsed: %.1f cpu: %.1f',t,label,time.perf_counter()-tstart[0],time.process_time()-tstart[1])

