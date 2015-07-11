# File:                 cmdextra.py
# Creation date:        2014-06-19
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Extension of command line interpreters
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import re, traceback, functools
from pathlib import Path

#-----------------------------------------------------------------------------------------------------
class CmdArgParser:
#-----------------------------------------------------------------------------------------------------

  @staticmethod
  def INT(x): return int(x,10)
  @staticmethod
  def BOOL(x,B={'True':True,'False':False}): return B[x]
  @staticmethod
  def TOKEN(x): return x
  @staticmethod
  def PATH(x): return Path(x)

  def __init__(self,sig):
    self.signature = sig, dict((k,(dfl,scn)) for k,dfl,scn in sig)
    self.pname = ' '.join('{}:{}={}'.format(k,scn.__doc__ or scn.__name__,dfl) for k,dfl,scn in sig)

  def __call__(self,arg,regarg=re.compile(r"^(?:(?P<key>\w+)=)?(?:(?P<_val>'(?:[^']|'')*?')|(?P<val>\S+?))(?:\s+?(?P<tail>.*))?$")):
    arg = arg.strip()
    sig,dsig = self.signature
    pos = 0
    while arg:
      m = regarg.match(arg)
      assert m is not None, 'Parse error: "{}" "{}"'.format(arg,regarg.pattern)
      key,val,arg = m.group('key','val','tail')
      if val is None: val = m.group('_val')[1:-1].replace("''","'")
      if key is None:
        if pos<0: raise Exception('Positional arguments cannot occur after keyword ones')
        key,dfl,scn = sig[pos]
        pos += 1
      else:
        pos = -1
        dfl,scn = dsig[key]
      yield key,scn(val)

  def __str__(self):
    return self.pname

#-----------------------------------------------------------------------------------------------------
class MetaCmdParse (type):
  """
A metaclass adapted to :class:`cmd.Cmd` subclasses.
If a :class:`cmd.Cmd` subclasses is assigned this metaclass, then all do_ methods are preprocessed.
Each do_ method is expected to be defined by::

   def do_<name>(self,<signature>): ...

where <name> is the name of the command to execute and <signature> its signature.
The signature is a list of named parameters with annotation and default value, e.g.::

   def do_start(self,ind:INT=3,x:TOKEN=None): ...

The annotation of a parameter must be a function which converts a string into the intended tyoe.

The method definition will be transformed into::

   def do_<name>(self,arg):
     try:
       #parse *arg* according to the signature
       #call the original do_<name> with the parsed parameters
  """
#-----------------------------------------------------------------------------------------------------

  parser = CmdArgParser

  def __new__(cls,name,bases,namespace,**kwd):
    def wrapper(s,arg,func,parser):
      try: return func(s,**dict(parser(arg)))
      except:
        traceback.print_exc()
        s.lastcmderr = True
      else:
        s.lastcmderr = False
    for fnam,f in tuple(namespace.items()):
      if fnam[:3] == 'do_':
        sig = tuple((k,dfl,f.__annotations__.get(k)) for k,dfl in zip(f.__code__.co_varnames[1:f.__code__.co_argcount],(f.__defaults__ or ())))
        p = cls.parser(sig)
        namespace[fnam] = F = lambda s,arg,func=f,parser=p: wrapper(s,arg,func,parser)
        functools.update_wrapper(F,f)
        F.__doc__ = '{}\n{}'.format(F.__doc__,p)
    return super(MetaCmdParse,cls).__new__(cls,name,bases,namespace,**kwd)
