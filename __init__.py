# File:                 __init__.py
# Creation date:        2014-03-16
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities in Python
#

import os, collections, logging
logger = logging.getLogger(__name__)

#==================================================================================================
class ondemand:
  r"""
A decorator to declare, in a class, a computable attribute which is computed only once (its value is then cached). Example::

   class c:
     def __init__(self,u): self.u = u
     @ondemand
     def att(self): print('Computing att...'); return self.u+1
   x = c(3); print(x.att)
   #>>> Computing att...
   #>>> 4
   print(x.att)
   #>>> 4
   x.u = 6; print(x.att)
   #>>> 4
   del x.att; print(x.att)
   #>>> Computing att...
   #>>> 7
  """
#==================================================================================================

  __slots__ = ('get',)

  def __init__(self,get):
    from inspect import signature
    L = list(signature(get).parameters.values())
    if any(p.kind!=p.POSITIONAL_OR_KEYWORD for p in L) or any(p.default==p.empty for p in L[1:]):
      raise Exception('ondemand attribute definition must be a function with a single argument')
    self.get = get

  def __get__(self,instance,typ):
    if instance is None:
      return self
    else:
      val = self.get(instance)
      setattr(instance,self.get.__name__,val)
      return val

#==================================================================================================
class odict:
  r"""
Objects of this class present an attribute oriented interface to an underlying proxy mapping object. Keys in the proxy are turned into attributes of the instance. If a positional argument is passed, it must be the only argument and is taken as proxy, otherwise the proxy is built from the keyword arguments. Example::

   x = odict(a=3,b=6); print(x.a)
   #>>> 3
   del x.a; print(x)
   #>>> {'b':6}
   x.b += 7; x.__proxy__
   #>>> {'b':13}
   x == x.__proxy__
   #>>> True
  """
#==================================================================================================
  __slot__ = '__proxy__',
  def __init__(self,*a,**ka):
    if a:
      n = len(a)
      if n>1: raise TypeError('expected at most 1 positional argument, got {}'.format(n))
      if ka: raise TypeError('expected no keyword arguments, got {}'.format(len(ka)))
      r = a[0]
      while isinstance(r,odict): r = r.__proxy__
      assert isinstance(r,collections.abc.Mapping)
    else: r = dict(**ka)
    object.__setattr__(self,'__proxy__',r)
  def __eq__(self,other):
    return self.__proxy__ == (other.__proxy__ if isinstance(other,odict) else other)
  def __ne__(self,other):
    return self.__proxy__ != (other.__proxy__ if isinstance(other,odict) else other)
  def __getattr__(self,a):
    try: return self.__proxy__[a]
    except KeyError: raise AttributeError(a) from None
  def __setattr__(self,a,v):
    if a=='__proxy__': raise AttributeError(a)
    self.__proxy__[a] = v
  def __delattr__(self,a):
    try: del self.__proxy__[a]
    except KeyError: raise AttributeError(a) from None
  def __str__(self): return str(self.__proxy__)
  def __repr__(self): return repr(self.__proxy__)

#==================================================================================================
def config_xdg(rsc,deflt=None):
  r"""
:param rsc: the name of an XDG resource
:param deflt: a default value

Returns the content of the XDG resource named *rsc* or *deflt* if the resource is not found.
  """
#==================================================================================================
  try: from xdg.BaseDirectory import load_first_config
  except: return deflt
  p = load_first_config(rsc)
  if p is None: return deflt
  with open(p) as u: return u.read()

#==================================================================================================
def config_env(name,deflt=None,asfile=False):
  r"""
:param name: the name of an environment variable
:param deflt: a default value
:param asfile: whether to treat the value of the environment variable as a path to a file

Returns the string value of an environment variable (or the content of the file pointed by it if *asfile* is set) or *deflt* if the environment variable is not assigned.
  """
#==================================================================================================
  x = os.environ.get(name)
  if x is None: return deflt
  if asfile:
    with open(x) as u: x = u.read(x)
  return x

#==================================================================================================
def autoconfig(module,name,dflt=None,asfile=False):
  r"""
:param module: the name of a module
:param name: the name of a configuration parameter for that module
:param dflt: a default value for the configuration parameter
:param asfile: whether to treat the value of the environment variable as a path to a file

Returns an object obtained from an environment variable whose name is derived from *module* and *name*. For example, if *module* is ``mypackage.mymodule`` and *name* is ``myparam`` then the environment variable is ``MYPACKAGE_MYMODULE_MYPARAM``. The value of that variable (or the content of the file pointed by it if *asfile* is set) is executed in an empty dictionary and the value attached to key *name* is returned. If the variable is not assigned, *dflt* is returned.
  """
#==================================================================================================
  x = config_env('{}_{}'.format(module.replace('.','_'),name).upper(),asfile=asfile)
  if x:
    d = {}
    exec(x,d)
    return d[name]
  else: return dflt

#==================================================================================================
class Config (collections.abc.Mapping):
  r"""
Instances of this class represent configurations which can be consistently setup either from ipython widgets or from command line arguments.

:param conf: the specification of each argument passed to the :meth:`add_argument` method
:param initv: an argument assignment applied at initialisation
  """
#==================================================================================================
  class Pconf:
    __slots__ = 'name','value','helper','widget','cparser'
    def __init__(s,*a):
      for k,v in zip(s.__slots__,a): setattr(s,k,v)
    def __repr__(s): return 'Pconf<{}>'.format(','.join('{}={}'.format(k,repr(getattr(s,k))) for k in s.__slots__))

#--------------------------------------------------------------------------------------------------
  def __init__(self,*conf,**initv):
#--------------------------------------------------------------------------------------------------
    from collections import OrderedDict
    self.pconf = OrderedDict()
    self.initial = {}
    self.widget_style = dict(width='15cm')
    self.label_style = dict(width='2cm',padding='0cm',align_self='flex-start')
    self.button_style = dict()
    for a,ka in conf: self.add_argument(*a,**ka)
    self.reset(**initv)

  def __getitem__(self,k): return self.pconf[k].value
  def __setitem__(self,k,v): self.pconf[k].value = v
  def __iter__(self): return iter(self.pconf)
  def __len__(self): return len(self.pconf)
  def __repr__(self): return 'Config<{}>'.format(','.join('{}={}'.format(k,repr(e.value)) for k,e in self.pconf.items()))

#--------------------------------------------------------------------------------------------------
  def fromipyui(self):
    r"""
Instantiates the arguments through a ipython user interface.
    """
#--------------------------------------------------------------------------------------------------
    from IPython.display import display
    display(self.widget)
#--------------------------------------------------------------------------------------------------
  def fromcmdline(self,args):
    r"""
Instantiates the arguments by parsing *args*.
    """
#--------------------------------------------------------------------------------------------------
    self.cparser.parse_args(args,odict(self))

#--------------------------------------------------------------------------------------------------
  def reset(self,**ka):
    r"""
Reinitialises the argument values.
    """
#--------------------------------------------------------------------------------------------------
    for k in ka:
      if k not in self.initial: raise KeyError(k)
    for k,v in ka.items(): self.initial[k] = self.pconf[k].value = v
    return self

#--------------------------------------------------------------------------------------------------
  def add_argument(self,name,value,cat=None,helper='',widget=None,cparser=None):
    r"""
:param name: name of the argument
:type name: :class:`str`
:param value: the initial value of the argument
:type value: :class:`object`
:param cat: the category of the argument
:type cat: :class:`NoneType`\|\ :class:`slice`\|\ :class:`tuple`\|\ :class:`list`\|\ :class:`dict`
:param helper: the helper for the argument
:type helper: :class:`str`\|\ tuple(\ :class:`str`\|\ pair(\ :class:`str`,\ :class:`str`))
:param widget: the widget specification for the argument
:type widget: :class:`NoneType`\|\ :class:`str`\|\ :class:`dict`
:param cparser: the command line parser specification for the argument
:type cparser: :class:`NoneType`\|\ :class:`str`\|\ :class:`dict`

* If *cat* is a :class:`slice`, the argument ranges over a subset of numbers.

  - If the ``stop`` attribute of *cat* is an integer, the target range is the set of integers between the values of *cat* attributes ``start`` (default 0)  and ``stop`` minus 1. The ``step`` attribute (default 1) only specifies a sampling step.
  - If the ``stop`` attribute of *cat* is real, the target range is the set of reals between the values of *cat* attributes ``start`` (default 0) and ``stop``. The ``step`` attribute (default 100) specifies a sampling step, directly if it is a real or adjusted so as to obtain exactly that number of samples if it is an integer.

* If *cat* is a :class:`tuple` or :class:`list`, it is turned into a :class:`collections.OrderedDict` whose keys are the string values of its elements.
* If *cat* is a :class:`dict` (which includes :class:`collections.OrderedDict`), the range is the set of values of *cat*. The keys are simply string names to denote these values. If the special key ``__multiple__`` is set to :const:`True`, it is deleted and the target range is in fact the set of tuples of the other values of *cat*.

If *widget* is :const:`None`, it is replaced by an empty dict, and if it is a string, it is replaced with a dict with a single key ``type`` assigned that string. The ``type`` key, if present, must be the name of a widget constructor in module :mod:`ipywidgets` and must be compatible with *cat* and *value*. If not present, a default value is guessed from *cat* and *value*. The ``layout`` key, if present, must be a dict passed to :class:`ipywidgets.Layout`, itself passed to the widget constructor. The other key-value pairs of *widget* are passed to the widget constructor as keyword arguments.

If *cparser* is :const:`None`, the argument is ignored by the command line parser. If it is a string, it is replaced by a dict with a single key ``spec`` assigned that string. The ``spec`` key holds the name of the argument as it appears in the command line. The other key-value pairs of *cparser* are passed to the :meth:`ArgumentParser.add_argument` method of the constructed parser.
    """
#--------------------------------------------------------------------------------------------------
    from collections import OrderedDict
    from functools import partial
    def checkbounds(x,typ,vmin,vmax,name):
      try: x = typ(x)
      except: raise ConfigException('Bounds constraint violated',name) from None
      if not (vmin<=x and x<=vmax): raise ConfigException('Bounds constraint violated',name) from None
      return x
    def checkchoice(x,options,name):
      try: return options[x]
      except KeyError: raise ConfigException('Choice constraint violated',name) from None
    type2widget = {
      bool:('Checkbox','ToggleButton'),
      int:('IntText',),
      float:('FloatText',),
      str:('Text','Textarea',),
    }
    mult2widget = {
      True:('SelectMultiple',),
      False:('Dropdown','Select','RadioButtons','ToggleButtons',),
    }
    def set_widget_type(*L):
      if 'type' in widget:
        if widget['type'] not in L: raise ConfigException('Inconsistent widget for value type',name)
      else: widget.update(type=L[0])
    if isinstance(helper,str): helper = helper,helper
    elif isinstance(helper,tuple): helper = tuple(map(''.join,zip(*(((x,x) if isinstance(x,str) else x) for x in helper))))
    else: raise ConfigException('Invalid helper spec',name)
    if widget is None: widget = {}
    elif isinstance(widget,str): widget = dict(type=widget)
    elif isinstance(widget,dict): widget = widget.copy()
    else: raise ConfigException('Invalid widget spec',name)
    if cparser is not None:
      if isinstance(cparser,str): cparser = dict(spec=(cparser,))
      elif isinstance(cparser,dict): cparser = cparser.copy()
      else: raise ConfigException('Invalid cparser spec',name)
    if cat is None:
      typ = type(value)
      L = type2widget.get(typ)
      if L is None: raise ConfigException('Inconsistent cat for value type',name)
      set_widget_type(*L)
      if cparser is not None:
        if typ is bool: cparser.update(action=('store_false' if value else 'store_true'))
        else: cparser.update(type=(lambda x: eval(x,{})))
    elif isinstance(cat,slice):
      start,stop,step = (cat.start or 0),cat.stop,cat.step
      if isinstance(stop,int):
        if not (isinstance(start,int) and start<stop and (step is None or (isinstance(step,int) and step>0))): raise ConfigException('Invalid slice cat',name)
        if step is None: step = 1
        slider,typ = 'IntSlider',int
      elif isinstance(stop,float):
        if not (isinstance(start,(int,float)) and start<stop and (step is None or (isinstance(step,(int,float)) and step>0))): raise ConfigException('Invalid slice cat',name)
        if step is None: step = (stop-start)/100
        elif isinstance(step,int): step = (stop-start)/step
        slider,typ = 'FloatSlider',float
      else: raise ConfigException('Invalid slice cat',name)
      set_widget_type(slider)
      widget.update(min=start,max=stop,step=step)
      if cparser is not None:
        cparser.update(type=partial(checkbounds,typ=typ,vmin=start,vmax=stop,name=name))
    elif isinstance(cat,(tuple,list,dict)):
      if isinstance(cat,(tuple,list)): cat = OrderedDict((str(x),x) for x in cat)
      else: cat = cat.copy()
      multiple = bool(cat.pop('__multiple__',False))
      set_widget_type(*mult2widget[multiple])
      cvalues = tuple(cat.values())
      values = value if multiple else (value,)
      if any(v not in cvalues for v in values):
        raise ConfigException('Inconsistent value for cat',name)
      widget.update(options=cat)
      if cparser is not None:
        cparser.update(type=partial(checkchoice,options=cat,name=name))
        if multiple: cparser.update(nargs='*')
    else: raise ConfigException('Unrecognised cat',name)
    self.pconf[name] = self.Pconf(name,value,helper,widget,cparser)
    self.initial[name] = value

#--------------------------------------------------------------------------------------------------
  @ondemand
  def widget(self):
#--------------------------------------------------------------------------------------------------
    from functools import partial
    from collections import ChainMap
    import ipywidgets
    def upde(e,w): e.value = w.value
    def updw(w,x): w.value = x
    def row(e):
      ka = e.widget.copy()
      w = getattr(ipywidgets,ka.pop('type'))
      style = ka.pop('layout',None)
      w = w(value=e.value,layout=(widget_layout if style is None else ipywidgets.Layout(**ChainMap(style,self.widget_style))),**ka)
      hb = ipywidgets.Button(icon='fa-undo',tooltip='Reset to default',layout=rbutton_layout)
      hb.on_click(lambda but,w=w,x=e.value: updw(w,x))
      lb = ipywidgets.HTML('<span title="{}">{}</span>'.format(e.helper[0],e.name),layout=label_layout)
      w.observe((lambda evt,e=e,w=w: upde(e,w)),'value')
      return (e.name,w),ipywidgets.HBox(children=(hb,lb,w))
    widget_layout = ipywidgets.Layout(**self.widget_style)
    label_layout = ipywidgets.Layout(**self.label_style)
    rbutton_layout = ipywidgets.Layout(width='0.5cm',padding='0cm')
    W,L = zip(*(row(e) for e in self.pconf.values()))
    header,buttons,footer = self.widget_context()
    if buttons: buttons = [ipywidgets.HBox(children=buttons)]
    w = ipywidgets.VBox(children=header+list(L)+buttons+footer)
    w.pconf = dict(W)
    return w
#--------------------------------------------------------------------------------------------------
  def widget_context(self):
    r"""
Returns a triple of a list of prolog widgets, a list of buttons and a list of epilog widgets, put around the argument widgets. This implementations returns no prolog, no epilog and a single button resetting the configuration to its initial value. Subclasses can refine that behaviour.
    """
#--------------------------------------------------------------------------------------------------
    return [],[self.make_widget_reset_button(self.initial,icon='fa-undo',description='reset')],[]
#--------------------------------------------------------------------------------------------------
  def make_widget_reset_button(self,data,**ka):
#--------------------------------------------------------------------------------------------------
    from collections import ChainMap
    import ipywidgets
    def click(b):
      for k,w in self.widget.pconf.items():
        if k in data: w.value = data[k]
    style = ka.pop('layout',None)
    b = ipywidgets.Button(layout=ipywidgets.Layout(**(self.button_style if style is None else ChainMap(style,self.button_style))),**ka)
    b.on_click(click)
    return b

#--------------------------------------------------------------------------------------------------
  @ondemand
  def cparser(self):
#--------------------------------------------------------------------------------------------------
    from argparse import ArgumentParser
    header,footer = self.cparser_context()
    p = ArgumentParser(description=header,epilog=footer)
    for e in self.pconf.values():
      if e.cparser is not None:
        ka = e.cparser.copy()
        spec = ka.pop('spec',())
        p.add_argument(*spec,dest=e.name,default=e.value,help=e.helper[1],**ka)
    return p
#--------------------------------------------------------------------------------------------------
  def cparser_context(self):
    r"""
Returns a pair of a prolog and an epilog (both :class:`str` or :class:`NoneType`) for the command line parser. This implemetation returns :const:`None` for both prolog and epilog. Subclasses can refine that behaviour.
    """
#--------------------------------------------------------------------------------------------------
    return None, None

class ConfigException (Exception): pass

#==================================================================================================
class HtmlPlugin:
  r"""
Instances of this class have an ipython HTML representation based on :func:`html_incontext`.
  """
#==================================================================================================
  _html_limit = 50
  def _repr_html_(self):
    from lxml.etree import tostring
    return tostring(html_incontext(self),encoding='unicode').replace('&gt;','>')

#==================================================================================================
class ARG (tuple):
  r"""
Instances of this (immutable) class are pairs of a tuple of positional arguments and a dict of keyword arguments. Useful to manipulate function invocation arguments without making the invocation.

Methods:
  """
#==================================================================================================
  def __new__(cls,*a,**ka):
    return super().__new__(cls,(a,ka))
  def __getnewargs_ex__(self): return tuple(self)

  def variant(self,*a,**ka):
    r"""
Returns a variant of *self* where *a* is appended to the positional arguments and *ka* is updated into the keyword arguments.
    """
    a0,ka0 = self
    a = a0+a
    ka1 = ka0.copy(); ka1.update(ka); ka = ka1
    return ARG(*a,**ka)

  def __repr__(self):
    a,ka = self
    a = ','.join(repr(v) for v in a)
    ka = ','.join('{}={}'.format(k,repr(v)) for k,v in ka.items())
    return 'ARG({}{}{})'.format(a,(',' if a and ka else ''),ka)

#==================================================================================================
def zipaxes(L,fig,sharex=False,sharey=False,**ka):
  r"""
:param L: an arbitrary sequence
:param fig: a figure
:type fig: :class:`matplotlib.figure.Figure`
:param sharex: whether all the axes share the same x-axis scale
:type sharex: :class:`bool`
:param sharey: whether all the axes share the same y-axis scale
:type sharey: :class:`bool`
:param ka: passed to :meth:`add_subplot` generating new axes

Yields the pair of *o* and an axes on *fig* for each item *o* in sequence *L*. The axes are spread more or less uniformly.
  """
#==================================================================================================
  from math import sqrt,ceil
  L = list(L)
  N = len(L)
  nc = int(ceil(sqrt(N)))
  nr = int(ceil(N/nc))
  axrefx,axrefy = None,None
  for i,o in enumerate(L,1):
    ax = fig.add_subplot(nr,nc,i,sharex=axrefx,sharey=axrefy,**ka)
    if sharex and axrefx is None: axrefx = ax
    if sharey and axrefy is None: axrefy = ax
    yield o,ax

#==================================================================================================
class Expr (HtmlPlugin):
  r"""
Instances of this class implement closed symbolic expressions.

:param func: the function of the symbolic expression
:param a: list of positional arguments of the symbolic expression
:param ka: dictionary of keyword arguments of the symbolic expression

The triple *func*, *a*, *ka* forms the configuration of the :class:`Expr` instance. Its value is defined as the result of calling function *func* with positional and keyword arguments *a* and *ka*. The value is actually computed only once (and cached), and only when method :meth:`incarnate` is invoked. Subclasses should define automatic triggers of incarnation (see e.g. classes :class:`MapExpr` and :class:`CallExpr`). The incarnation cache can be reset by invoking method :meth:`reset`.

Initially, a :class:`Expr` instance is mutable, and its configuration can be changed. It becomes immutable (frozen) after any of the following operations: incarnation (even after it is reset), hashing, and pickling. Incarnation is not saved on pickling, hence lost on unpickling, but can of course be restored using method :meth:`incarnate`. Thus, when receiving a foreign :class:`Expr` instance, a process can decide whether it wants to access its value or only inspect its configuration. If recomputing the value is costly, use a persistent cache for *func*.

Caveat: function *func* should be defined at the top-level of its module, and the values in *a* and *ka* should be deterministically picklable and hashable (in particular: no dicts nor lists). Hence, any :class:`Expr` instance is itself deterministically picklable and hashable, and can thus be used as argument in the configuration of another :class:`Expr` instance.
  """
#==================================================================================================

  def __init__(self,func,*a,**ka):
    assert callable(func)
    self.reset()
    self.config = [func,a,ka]
    self.key = None

  def rearg(self,*a,**ka):
    r"""
Updates the config arguments of this instance: *a* is appended to the list of positional arguments and *ka* is updated into the dictionary of keyword arguments. Raises an error if the instance is frozen.
    """
    assert self.key is None, 'attempt to update a frozen Expr instance'
    self.config[1] += a
    self.config[2].update(ka)

  def refunc(self,f):
    r"""
Set *f* as the config function of this instance. Raises an error if the instance is frozen.
    """
    assert self.key is None, 'attempt to update a frozen Expr instance'
    self.config[0] = f

  def incarnate(self):
    if self.incarnated: return
    self.incarnated = True
    self.freeze()
    func,a,ka = self.config
    self.value = func(*a,**ka)

  def reset(self):
    self.incarnated = False
    self.value = None

  def freeze(self):
    k = self.key
    if k is None:
      func,a,ka = self.config
      self.key = k = func,a,tuple(sorted(ka.items()))
    return k

  def as_html(self,incontext):
    from lxml.builder import E
    func,a,ka = self.config
    x = html_parlist(a,sorted(ka.items()),incontext,deco=('[|','|',']'))
    x.insert(0,E.B('::'))
    x.insert(0,(E.SPAN if self.incarnated else E.EM)(str(func),style='padding:5px;'))
    return x

  def __getstate__(self): return self.freeze()
  def __setstate__(self,key):
    self.reset()
    func,a,ka = self.key = key
    self.config = [func,a,dict(ka)]
  def __hash__(self): return hash(self.freeze())
  def __eq__(self,other): return isinstance(other,Expr) and self.config == other.config
  def __repr__(self):
    if self.incarnated: return repr(self.value)
    func,a,ka = self.config
    sep = ',' if (a and ka) else ''
    a = ','.join(repr(v) for v in a)
    ka = ','.join('{}={}'.format(k,repr(v)) for k,v in sorted(ka.items()))
    return '{}({}{}{})'.format(func,a,sep,ka)

class MapExpr (Expr,collections.abc.Mapping):
  r"""
Symbolic expressions of this class are also (read-only) mappings and trigger incarnation on all the mapping operations, then delegate such operations to their value (expected to be a mapping).
  """
  def __getitem__(self,k): self.incarnate(); return self.value[k]
  def __len__(self): self.incarnate(); return len(self.value)
  def __iter__(self): self.incarnate(); return iter(self.value)

class CallExpr (Expr):
  r"""
Symbolic expressions of this class are also callables, and trigger incarnation on invocation, then delegate the invocation to their value (expected to be a callable).
  """
  def __call__(self,*a,**ka): self.incarnate(); return self.value(*a,**ka)

#==================================================================================================
def ipybrowse(D,start=1,pgsize=10):
  r"""
:param D: a sequence object (must have :func:`len`\ ).
:param start: the index of the initial page
:param pgsize: the size of the pages

A simple utility to browse sliceable objects page per page in IPython.
  """
#==================================================================================================
  import ipywidgets
  from IPython.display import display, clear_output
  P = (len(D)-1)//pgsize + 1
  if P==1: display(D)
  else:
    def show():
      with wout:
        clear_output(wait=True)
        display(D[(w.value-1)*pgsize:w.value*pgsize])
    wout = ipywidgets.Output()
    w = ipywidgets.IntSlider(description='page',value=(1 if start<1 else P if start>P else start),min=1,max=P,layout=dict(width='20cm'))
    w.observe((lambda c: show()),'value')
    show()
    return ipywidgets.VBox(children=(w,wout))

#==================================================================================================
def ipyfilebrowse(path,start=None,step=50,period=1.,context=(10,5),**ka):
  r"""
:param path: a path to an existing file
:type path: :class:`str`\|\ :class:`pathlib.Path`
:param start: index of start pointed
:type start: :class:`NoneType`\|\ :class:`int`\|\ :class:`float`
:param step: step size in bytes
:type step: :class:`int`
:param period: period in sec between refreshing attempts
:type period: :class:`float`
:param context: pair of number of lines before and after to display around current position
:type context: (\ :class:`int`,\ :class:`int`)

A simple utility to browse a byte file object, possibly while it expands. If *period* is :const:`None`, no change tracking is performed. If *start* is :const:`None`, the start position is end-of-file. If *start* is of type :class:`int`, it denotes the exact start position in bytes. If *start* is of type :class:`float`, it must be between :const:`0.` and :const:`1.`, and the start position is set (approximatively) at that position relative to the whole file.
  """
#==================================================================================================
  import ipywidgets
  from pathlib import Path
  from threading import Thread
  from time import sleep
  def track():
    nonlocal fsize
    c = fsize
    while period:
      fsize = path.stat().st_size
      if fsize != c:
        n = wctrl.value
        wctrl.max = fsize
        if n==c: wctrl.value = fsize
        else: setpos(n)
        c = fsize
      sleep(period)
  def setpos(n,nbefore=context[0],d=200*context[0],nafter=context[1]):
    m = n-d
    if m<0: m,d = 0,n
    file.seek(m)
    x = file.read(d)
    y = x.rsplit(b'\n',nbefore)
    if len(y)>nbefore: x = b'\n'.join(y[1:])
    file.seek(n)
    x += b''.join(file.readline() for i in range(nafter))
    wwin.value = x.decode()
  if isinstance(path,str): path = Path(path)
  else: assert isinstance(path,Path)
  file = path.open('rb')
  fsize = path.stat().st_size or 1
  if start is None: start = fsize
  elif isinstance(start,float):
    assert 0<=start and start<=1
    start = int(start*fsize)
  else:
    assert isinstance(start,int)
    if start<0:
      start += fsize
      if start<0: start = 0
  wwin = ipywidgets.Textarea(rows=context[0]+context[1],layout=ka)
  wctrl = ipywidgets.IntSlider(min=0,max=fsize,step=step,value=start,layout=dict(width=wwin.layout.width))
  wctrl.observe((lambda c: setpos(c.new)),'value')
  setpos(start)
  if period: Thread(target=track,daemon=True).start()
  w = ipywidgets.VBox(children=(wctrl,wwin))
  def close(wclose=w.close):
    nonlocal period
    period = 0.
    file.close()
    wclose()
  w.close = close
  return w

#==================================================================================================
class ipytoolbar:
  r"""
A simple utility to build a toolbar of buttons in IPython. Keyword arguments in the toolbar constructor are used as default values for all the buttons created. To create a new button, use method :meth:`add` with one positional argument: *callback* (a callable with no argument to invoke when the action is activated). If keyword arguments are present, they are passed to the button constructor. The button widget is returned. Method :meth:`display` displays the toolbar. The toolbar widget is available as attribute :attr:`widget`.
  """
#==================================================================================================
  bstyle=dict(padding='0cm')
  def __init__(self,**ka):
    import ipywidgets
    from IPython.display import display
    for k,v in self.bstyle.items(): ka.setdefault(k,v)
    bstyle = ka.items()
    self.widget = widget = ipywidgets.HBox(children=())
    def add(callback,**ka):
      s = ka.pop('layout',None)
      s = {} if s is None else s.copy()
      for k,v in bstyle: s.setdefault(k,v)
      b = ipywidgets.Button(layout=s,**ka)
      b.on_click(lambda b: callback())
      widget.children += (b,)
      return b
    self.add = add
    self.display = lambda: display(widget)

#==================================================================================================
def exploredb(spec):
  r"""
:param spec: an sqlalchemy url or engine or metadata structure, defining the database to explore
:type spec: :class:`sqlalchemy.engine.Engine`\|\ :class:`str`\|\ :class:`sqlalchemy.MetaData`

Display an IPython widget for basic database exploration. If a metadata structure is specified, it must be bound to an existing engine and reflected.
  """
#==================================================================================================
  import ipywidgets
  from functools import lru_cache
  from pandas import read_sql_query
  from sqlalchemy import select, func, MetaData, create_engine
  from sqlalchemy.engine import Engine
  from IPython.display import display, clear_output
  # Content retrieval
  def schemag():
    def fstr(x): return x
    def fbool(x): return 'x' if x else ''
    def fany(x): return str(x).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;') if x else ''
    def row(c):
      def val(x):
        try: return x[1](getattr(c,x[0]))
        except: return '*'
      return '<tr>{}</tr>'.format(''.join('<td class="field-{}"><span>{}</span></td>'.format(x[0],val(x)) for x in schema))
    schema = (
      ('name',fstr,'name',4),
      ('type',fany,'type',4),
      ('primary_key',fbool,'P',.5),
      ('nullable',fbool,'N',.5),
      ('unique',fbool,'U',.5),
      ('default',fany,'default',4),
      ('constraints',fany,'constraints',4),
      ('foreign_keys',fany,'foreign',4),
    )
    wstyle = '\n'.join('td.field-{} {{ min-width:{}cm; max-width:{}cm; }}'.format(x[0],x[3],x[3]) for x in schema)
    thead = '<tr>{}</tr>'.format(''.join('<td title="{}" class="field-{}">{}</td>'.format(x[0],x[0],x[2]) for x in schema))
    fmt = '<div><style scoped="scoped">{}{}</style><table><thead>{}</thead><tbody>'.format(exploredb.style['schema'],wstyle,thead),'</tbody></table></div>'
    return lambda table,pre=fmt[0],suf=fmt[1]: pre+''.join(map(row,tables[table].columns))+suf
  schema = lru_cache(None)(schemag())
  def size(table):
    return engine.execute(select([func.count()]).select_from(tables[table])).fetchone()[0]
  def sample(table,offset,nsample):
    sql = select(wscol.value,limit=nsample,offset=offset,order_by=(list(tables[table].primary_key) or tables[table].columns.values()[:1]))
    r = read_sql_query(sql,engine)
    r.index = list(range(offset,offset+min(nsample,len(r))))
    return r
  if isinstance(spec,MetaData):
    meta = spec
    if not meta.is_bound(): raise ValueError('Argument of type {} must be bound to an existing engine'.format(MetaData))
    no_table_msg = '{} object has no table (perhaps it was not reflected)'.format(MetaData) 
  else:
    if isinstance(spec,str): spec = create_engine(spec)
    elif not isinstance(spec,Engine):
      raise TypeError('Expected {}|{}|{}; Found {}'.format(str,Engine,MetaData,type(spec)))
    meta = MetaData(bind=spec)
    meta.reflect(views=True)
    no_table_msg = 'Database is empty'
  tables = meta.tables
  if not tables: return no_table_msg
  engine = meta.bind
  active = True
  scol_ = dict((table,[(c.name,c) for c in t.columns]) for table,t in tables.items())
  scol = dict((table,tuple(t.columns)) for table,t in tables.items())
  # widget creation
  wtitle = ipywidgets.HTML('<div style="{}">{}</div>'.format(exploredb.style['title'],engine))
  wtable = ipywidgets.Select(options=sorted(tables),layout=dict(width='10cm'))
  wsize = ipywidgets.Text(value='',tooltip='Number of rows',disabled=True,layout=dict(width='2cm'))
  wschema = ipywidgets.HTML()
  wscol = ipywidgets.SelectMultiple(options=[],layout=dict(flex_flow='column'))
  wdetail = ipywidgets.Tab(children=(wschema,wscol),layout=dict(display='none'))
  wdetail.set_title(0,'Column definitions'); wdetail.set_title(1,'Column selection')
  wdetailb = ipywidgets.Button(tooltip='toggle detail display (red border means some columns are hidden)',icon='fa-info-circle',layout=dict(width='.4cm',padding='0'))
  wreloadb = ipywidgets.Button(tooltip='reload table',icon='fa-refresh',layout=dict(width='.4cm',padding='0'))
  woffset = ipywidgets.IntSlider(description='offset',min=0,step=1,layout=dict(width='12cm'))
  wnsample = ipywidgets.IntSlider(description='nsample',min=1,max=50,step=1,layout=dict(width='10cm'))
  wout = ipywidgets.Output()
  # widget updaters
  def show():
    nonlocal active
    active = False
    woffset.value = 0
    wnsample.value = 5
    wscol.value = ()
    wscol.options =  opts = scol_[wtable.value]
    wscol.value = scol[wtable.value]
    active = True
    wscol.rows = min(len(opts),20)
    sz = size(wtable.value)
    woffset.max = max(sz-1,0)
    wsize.value = str(sz)
    wschema.value = schema(wtable.value)
    showc()
  def showc():
    if active:
      with wout:
        clear_output(wait=True)
        display(sample(wtable.value,woffset.value,wnsample.value))
  def setscol():
    if active: scol[wtable.value] = wscol.value
    wdetailb.layout.border = 'none' if len(scol[wtable.value]) == len(scol_[wtable.value]) else 'thin solid red'
    showc()
  def toggledetail(inv={'inline':'none','none':'inline'}): wdetail.layout.display = inv[wdetail.layout.display]
  # callback attachments
  wdetailb.on_click((lambda b: toggledetail()))
  wtable.observe((lambda c: show()),'value')
  wreloadb.on_click((lambda b: show()))
  woffset.observe((lambda c: showc()),'value')
  wnsample.observe((lambda c: showc()),'value')
  wscol.observe((lambda c: setscol()),'value')
  # initialisation
  show()
  return ipywidgets.VBox(children=(wtitle,ipywidgets.HBox(children=(wtable,wsize,wdetailb,wreloadb)),wdetail,ipywidgets.HBox(children=(woffset,wnsample,)),wout))

exploredb.style = dict(
  schema='''
table { border-collapse: collapse; }
table > thead { display:block; }
table > tbody { display: block; max-height: 10cm; overflow-y: auto; padding-right: 10cm; }
table > thead > tr > td { padding: 1mm; text-align: center; font-weight: bold; color: white; background-color: navy; border: thin solid white; }
table > tbody > tr > td { padding: 1mm; border: thin solid blue; overflow: hidden; }
table > tbody > tr > td > span { position: relative; background-color: white; white-space: nowrap; color:black; z-index: 0; }
table > tbody > tr > td:hover { overflow: visible; }
table > tbody > tr > td:hover > span { color:purple; z-index: 1; }
  ''',
  title='background-color:gray; color:white; font-weight:bold; padding:.2cm',
)

#==================================================================================================
def html_incontext(x):
  r"""
:param x: an arbitrary python object

Returns an HTML object (etree as understood by package :mod:`lxml`) representing object *x*. The representation is by default simply the string representation of *x* (enclosed in a SPAN element), but can be customised if *x* supports method :meth:`as_html`.

Method :meth:`as_html` should only be defined for hashable objects. It takes as input a function *incontext* and returns the base HTML representation of the object. If invoked on a compound object, it should not recursively invoke method :meth:`html_incontext` nor :meth:`as_html` on its components to obtain their HTML representations, because that would produce representations of unmanageable size in case of recursions or repetitions (if two components share a common sub-component). Instead, the representation of a component object should be a "pointer" in the form of a string `?`\ *n* (within a SPAN element) where *n* is a unique reference number. The pointer for a component can be obtained by calling the argument function *incontext* with that object as argument. The scope of such pointers is the toplevel call of function :func:`html_incontext`, and two occurrences of equal objects will have the same pointer.

When pointers are created, the result of calling function :func:`html_incontext` on a object *x* is the base HTML representation of *x* including its pointers, followed by a TABLE mapping each pointer reference *n* to the base HTML representation of the reference object (possibly containing pointers itself, recursively). In the following example, the base HTML representation of a node in a graph is given by its label followed by the sequence of pointers to its successor nodes::

   class N: # class of nodes in a (possibly cyclic) directed graph
     def __init__(self,tag,*succ): self.tag,self.succ = tag,succ
     def as_html(self,incontext):
       from lxml.builder import E
       return E.DIV(E.B(self.tag),'[|',*(y for x in self.succ for y in (incontext(x),'|')),']')
   # Let's build a trellis DAG
   n = N('')
   na,nb,nc = N('a',n),N('b',n),N('c',n)
   nab,nbc,nac = N('ab',na,nb),N('bc',nb,nc),N('ac',na,nc)
   nabc = N('abc',nab,nbc,nac)
   html_incontext(nabc)

produces (up to some attributes)::

   <TABLE>
     <!-- THEAD: base HTML representation (with pointers) of the initial object -->
     <THEAD><TR><TD colspan="2"> <DIV><B>abc</>[|<SPAN>?1</>|<SPAN>?5</>|<SPAN>?7</>|]</DIV> </TD></TR></THEAD>
     <!-- TBODY: mapping each pointer to the base HTML representation of its reference -->
     <TBODY>
       <TR> <TH>?1</TH> <TD> <DIV><B>ab</>[|<SPAN>?2</>|<SPAN>?4</>|]</DIV> </TD> </TR>
       <TR> <TH>?2</TH> <TD> <DIV><B>a</>[|<SPAN>?3</>|]</DIV> </TD> </TR>
       <TR> <TH>?3</TH> <TD> <DIV><B></>[|]</DIV> </TD> </TR>
       <TR> <TH>?4</TH> <TD> <DIV><B>b</>[|<SPAN>?3</>|]</DIV> </TD> </TR>
       <TR> <TH>?5</TH> <TD> <DIV><B>bc</>[|<SPAN>?4</>|<SPAN>?6</>|]</DIV> </TD> </TR>
       <TR> <TH>?6</TH> <TD> <DIV><B>c</>[|<SPAN>?3</>|]</DIV> </TD> </TR>
       <TR> <TH>?7</TH> <TD> <DIV><B>ac</>[|<SPAN>?2</>|<SPAN>?6</>|]</DIV> </TD> </TR>
     </TBODY>
   </TABLE>
  """
#==================================================================================================
  from lxml.builder import E
  from lxml.etree import ElementTextIterator
  def hformat(L):
    return E.DIV(E.STYLE(html_incontext.style,scoped='scoped'),E.TABLE(E.THEAD(E.TR(E.TD(E.DIV(L[0][1],**{'class':'main'}),colspan="2"))),E.TBODY(*(E.TR(E.TD(E.SPAN('?{}'.format(k),**{'class':'pointer'})),E.TD(x)) for k,x in L[1:])),**{'class':'toplevel'}))
  def incontext(v):
    try: q = ctx.get(v)
    except: return E.SPAN(str(v)) # for unhashable objects
    if q is None:
      if hasattr(v,'as_html'):
        k,ref = len(ctx),'py_{}'.format(id(v))
        ctx[v] = q = [k,ref,None]
        try: x = v.as_html(incontext)
        except: x = E.SPAN(repr(v)) # should not fail, but just in case
        L.append((k,E.DIV(x,id=ref,**{'class':'main'})))
        tit = q[2] = ' '.join(ElementTextIterator(x))
      else: return E.SPAN(str(v))
    else: k,ref,tit = q
    js = lambda x,ref=ref: 'document.getElementById(\'{}\').style.outline=\'{}\''.format(ref,('thick solid red' if x else 'inherit'))
    return E.SPAN('?{}'.format(k),title=tit,onmouseenter=js(True),onmouseleave=js(False),**{'class':'pointer'})
  L = []
  ctx = {}
  e = incontext(x)
  n = len(L)
  return e if n==0 else L[0][1] if n==1 else hformat(sorted(L))

html_incontext.style = '''
table.toplevel { border-collapse: collapse; }
table.toplevel > thead > tr > td, table.toplevel > tbody > tr > td { border: thin solid black; background-color:white; text-align:left; }
table.toplevel > thead > tr > td { padding:0; border-bottom-width: thick; }
table.toplevel span.pointer { color: blue; background-color: #e0e0e0; font-weight:bold; }
table.toplevel div.main { padding:0; max-height: 5cm; overflow-y: auto; }
'''

#==================================================================================================
def html_parlist(La,Lka,incontext,deco=('','',''),padding='5px'):
#==================================================================================================
  from lxml.builder import E
  def h():
    opn,sep,cls = deco
    stl = 'padding: {}'.format(padding)
    yield opn
    for v in La: yield E.SPAN(incontext(v),style=stl); yield sep
    for k,v in Lka: yield E.SPAN(E.B(k),'=',incontext(v),style=stl); yield sep
    yield cls
  return E.DIV(*h(),style='padding:0')

#==================================================================================================
def html_table(irows,fmts,hdrs=None,opening=None,closing=None,htmlclass='default'):
  r"""
:param irows: a generator of pairs of an object (key) and a tuple of objects (value)
:param fmts: a tuple of format functions matching the length of the value tuples
:param hdrs: a tuple of strings matching the length of the value tuples
:param opening,closing: strings at head and foot of table

Returns an HTML table object with one row for each pair generated from *irow*. The key of each pair is in a column of its own with no header, and the value must be a tuple whose length matches the number of columns. The format functions in *fmts*, one for each column, are expected to return HTML objects (as understood by :mod:`lxml`). *hdrs* may specify headers as a tuple of strings, one for each column.
  """
#==================================================================================================
  from lxml.builder import E
  def thead():
    if opening is not None: yield E.TR(E.TD(opening,colspan=str(1+len(fmts))))
    if hdrs is not None: yield E.TR(E.TD(),*(E.TH(hdr) for hdr in hdrs))
  def tbody():
    for ind,row in irows:
      yield E.TR(E.TH(str(ind)),*(E.TD(fmt(v)) for fmt,v in zip(fmts,row)))
  def tfoot():
    if closing is not None: yield E.TR(E.TD(),E.TD(closing,colspan=str(len(fmts))))
  return E.DIV(E.STYLE(html_table.style,scoped='scoped'),E.TABLE(E.THEAD(*thead()),E.TBODY(*tbody()),E.TFOOT(*tfoot()),**{'class':htmlclass}))

html_table.style = '''
  table.default { border-collapse: collapse; }
  table.default > thead > tr > th, table.default > thead > tr > td, table.default > tbody > tr > th, table.default > tbody > tr > td, table.default > tfoot > tr > td  { background-color: white; text-align: left; vertical-align: top; border: thin solid black; }
  table.default > thead > tr > th, table.default > thead > tr > td { background-color: gray; color: white }
  table.default > tfoot > tr > td { background-color: #f0f0f0; color: navy; }
'''

#==================================================================================================
def html_stack(*a,**ka):
  r"""
:param a: a list of (lists of) HTML objects (as understood by :mod:`lxml`)
:param ka: a dictionary of HTML attributes for the DIV encapsulating each object

Merges the list of HTML objects into a single HTML object, which is returned.
  """
#==================================================================================================
  from lxml.builder import E
  return E.DIV(*(E.DIV(x,**ka) for x in a))

#==================================================================================================
class SQliteStack:
  r"""
Objects of this class are aggregation functions which simply collect results in a list, for use with a SQlite database. Example::

   with sqlite3.connect('/path/to/db') as conn:
     rstack = SQliteStack.setup(conn,'stack',2)
     for school,x in conn.execute('SELECT school,stack(age,height) FROM PupilsTable GROUP BY school'):
       print(school,rstack(x))

prints pairs *school*, *L* where *L* is a list of pairs *age*, *height*.
  """
#==================================================================================================
  contents = {}
  def __init__(self): self.content = []
  def step(self,*a): self.content.append(a)
  def finalize(self): n = id(self.content); self.contents[n] = self.content; return n
  @staticmethod
  def setup(conn,name,n):
    conn.create_aggregate(name,n,SQliteStack)
    return SQliteStack.contents.pop

#==================================================================================================
def SQliteNew(path,schema):
  r"""
Makes sure the file at *path* is a SQlite3 database with schema exactly equal to *schema*.
  """
#==================================================================================================
  import sqlite3
  if isinstance(schema,str): schema = list(sql.strip() for sql in schema.split('\n\n'))
  with sqlite3.connect(path,isolation_level='EXCLUSIVE') as conn:
    S = list(sql for sql, in conn.execute('SELECT sql FROM sqlite_master WHERE name NOT LIKE \'sqlite%\''))
    if S:
      if S!=schema: raise Exception('database has a version conflict')
    else:
      for sql in schema: conn.execute(sql)

#==================================================================================================
def gitcheck(path):
  r"""
:param path: a path to a git repository
:type path: :class:`str`

Assumes that *path* denotes a git repository (target) which is a passive copy of another repository (source) on the same file system. If that is not the case, returns :const:`None`. Checks that the target repository is up-to-date, and updates it if needed using the git pull operation. Use the ``GIT_PYTHON_GIT_EXECUTABLE`` environment variable to set the Git executable if it is not the default ``/usr/bin/git``.
  """
#==================================================================================================
  from git import Repo
  trg = Repo(path)
  try: r = trg.remote()
  except ValueError: return # no remote
  if not os.path.isdir(r.url): return # remote not on file system
  src = Repo(r.url)
  if src.is_dirty(): raise GitException('source-dirty',trg,src)
  if trg.is_dirty(): raise GitException('target-dirty',trg,src)
  if not all((f.flags & f.HEAD_UPTODATE) for f in r.pull()):
    logger.info('Synched (git pull) %s',trg)
    if src.commit()!=trg.commit(): raise GitException('synch-failed',trg,src)
    return True
class GitException (Exception): pass

#==================================================================================================
def gitcheck_package(pkgname):
  r"""
:param pkgname: full name of a package
:type pkgname: :class:`str`

Assumes that *pkgname* is the name of a python regular (non namespace) package and invokes :meth:`gitcheck` on its path. Reloads the package if git update was performed.
  """
#==================================================================================================
  from importlib.util import find_spec
  from importlib import reload
  from sys import modules
  try: path, = find_spec(pkgname).submodule_search_locations
  except: raise ValueError('Not a regular package',pkgname)
  if gitcheck(path):
    m = modules.get(pkgname)
    if m is not None: logger.warning('Reloading %s ...',pkgname); reload(m)
    return True

#==================================================================================================
def SQLinit(engine,meta):
  r"""
:param engine: a sqlalchemy engine (or its url)
:param meta: a sqlalchemy metadata structure

* When the database is empty, a ``Metainfo`` table with a single row matching exactly the :attr:`info` attribute of *meta* is created, then the database is populated using *meta*.

* When the database is not empty, it must contain a ``Metainfo`` table with a single row matching exactly the :attr:`info` attribute of *meta*, otherwise an exception is raised.
  """
#==================================================================================================
  from datetime import datetime
  from sqlalchemy import MetaData, Table, Column, create_engine, event
  from sqlalchemy.types import DateTime, Text, Integer
  from sqlalchemy.sql import select, insert, update, delete, and_
  if isinstance(engine,str): engine = create_engine(engine)
  if engine.driver == 'pysqlite':
    # fixes a bug in the pysqlite driver; to be removed if fixed
    # see http://docs.sqlalchemy.org/en/rel_1_0/dialects/sqlite.html
    def do_connect(conn,rec): conn.isolation_level = None
    def do_begin(conn): conn.execute('BEGIN')
    event.listen(engine,'connect',do_connect)
    event.listen(engine,'begin',do_begin)
  meta_ = MetaData(bind=engine)
  meta_.reflect()
  if meta_.tables:
    try:
      metainfo_table = meta_.tables['Metainfo']
      metainfo = dict(engine.execute(select((metainfo_table.c))).fetchone())
      del metainfo['created']
    except: raise SQLinitMetainfoException('Not found')
    for k,v in meta.info.items():
      if metainfo.get(k) != str(v):
        raise SQLinitMetainfoException('{}[expected:{},found:{}]'.format(k,v,metainfo.get(k)))
  else:
    metainfo_table = Table(
      'Metainfo',meta_,
      Column('created',DateTime()),
      *(Column(key,Text()) for key in meta.info)
    )
    meta_.create_all()
    engine.execute(insert(metainfo_table).values(created=datetime.now(),**meta.info))
    meta.create_all(bind=engine)
  return engine

class SQLinitMetainfoException (Exception): pass

#==================================================================================================
class SQLHandler (logging.Handler):
  r"""
:param engine: a sqlalchemy engine (or its url)
:param label: a text label 

A logging handler class which writes the log messages into a database.
  """
#==================================================================================================
  def __init__(self,engine,label,*a,**ka):
    from datetime import datetime
    from sqlalchemy.sql import select, insert, update, delete, and_
    meta = SQLHandlerMetadata()
    engine = SQLinit(engine,meta)
    session_table = meta.tables['Session']
    log_table = meta.tables['Log']
    with engine.begin() as conn:
      if engine.dialect.name == 'sqlite': conn.execute('PRAGMA foreign_keys = 1')
      conn.execute(delete(session_table).where(session_table.c.label==label))
      c = conn.execute(insert(session_table).values(label=label,started=datetime.now()))
      session = c.inserted_primary_key[0]
      c.close()
    def dbrecord(rec,log_table=log_table,session=session):
      with engine.begin() as conn:
        conn.execute(insert(log_table).values(session=session,level=rec.levelno,created=datetime.fromtimestamp(rec.created),module=rec.module,funcName=rec.funcName,message=rec.message))
    self.dbrecord = dbrecord
    super().__init__(*a,**ka)

  def emit(self,rec):
    self.format(rec)
    self.dbrecord(rec)

def SQLHandlerMetadata(info=dict(origin=__name__+'.SQLHandler',version=1)):
  from sqlalchemy import Table, Column, ForeignKey, MetaData
  from sqlalchemy.types import DateTime, Text, Integer
  meta = MetaData(info=info)
  Table(
    'Session',meta,
    Column('oid',Integer(),primary_key=True,autoincrement=True),
    Column('started',DateTime()),
    Column('label',Text(),index=True,unique=True,nullable=False),
    )
  Table(
    'Log',meta,
    Column('oid',Integer(),primary_key=True,autoincrement=True),
    Column('session',ForeignKey('Session.oid',ondelete='CASCADE'),index=True,nullable=False),
    Column('level',Integer(),nullable=False),
    Column('created',DateTime(),nullable=False),
    Column('module',Text(),index=True,nullable=False),
    Column('funcName',Text(),nullable=False),
    Column('message',Text(),nullable=False),
    )
  return meta

#==================================================================================================
class ormsroot (collections.abc.MutableMapping,HtmlPlugin):
  r"""
Instances of this class implement very simple persistent object managers based on the sqlalchemy ORM. This class should not be instantiated directly.

Each subclass *C* of this class should define a class attribute :attr:`base` assigned an sqlalchemy declarative persistent class *B*. Once this is done, a specialised session maker can be obtained by invoking class method :meth:`sessionmaker` of class *C*, with the url of an sqlalchemy supported database as first argument, the other arguments being the same as those for :meth:`sqlalchemy.orm.sessionmaker`. This sessionmaker will produce sessions with the following characteristics:

* they are attached to an sqlalchemy engine at this url (note that engines are reused across multiple sessionmakers with the same url)

* they have a :attr:`root` attribute, pointing to an instance of *C*, which acts as a dictionary where the keys are the primary keys of *B* and the values the corresponding ORM entries. The root object has a convenient ipython HTML representation. Direct update to the dictionary is not supported, except deletion, which is made persitent on session commit.

Class *C* should provide a method to insert new objects in the persistent class *B*, and they will then be reflected in the session root (and made persitent on session commit). Example::

   from sqlalchemy.ext.declarative import declarative_base; Base = declarative_base()
   from sqlalchemy import Column, Text, Integer

   class Entry (Base): # example of sqlalchemy declarative persistent class definition
     __tablename__ = 'Entry'
     oid = Column(Integer(),primary_key=True)
     name = Column(Text())
     age = Column(Integer())
     def __str__(self): return 'Entry<{},{}>'.format(self.oid,self.name)

   class Root(ormsroot): # simple manager for class Entry
     base = Entry
     def new(self,name,age):
       r = Entry(name=name,age=age)
       self.session.add(r)
       return r

   sessionmaker = Root.sessionmaker

Example of use (assuming :func:`sessionmaker` as above has been imported)::

   Session = sessionmaker('sqlite://') # memory db for the example

   from contextlib import contextmanager
   @contextmanager
   def mysession(): # sessions as simple sequences of instructions, for the example
     s = Session()
     try: yield s
     else: s.commit()
     s.close()

   with mysession() as s: # first session, define two entries
     jack = s.root.new('jack',45); joe = s.root.new('joe',29)
     print('Listing:',*s.root.values()) # s.root used as a mapping
   #>>> Listing: Entry<1,jack> Entry<2,joe>

   with mysession() as s: # second session, possibly on another process (if not memory db)
     jack = s.root.pop(1) # remove jack (directly from mapping)
     print('Deleted: {}'.format(jack),'; Listing',*s.root.values())
   #>>> Deleted: Entry<1,jack> ; Listing: Entry<2,joe>

   with mysession() as s: # But of course direct sqlalchemy operations are available
     for x in s.query(Entry.name).filter(Entry.age>25): print(*x)
   #>>> joe
  """
#==================================================================================================

  def __init__(self,session,dclass):
    self.dclass = dclass
    self.pk = pk = self.dclass.__table__.primary_key.columns.values()
    self.pkindex = (lambda r,k=pk[0].name: getattr(r,k)) if len(pk)==1 else (lambda r,pk=pk: tuple(getattr(r,c.name) for c in pk))
    self.session = session

  def __getitem__(self,k):
    r = self.session.query(self.dclass).get(k)
    if r is None: raise KeyError(k)
    return r

  def __delitem__(self,k):
    self.session.delete(self[k])

  def __setitem__(self,k,v):
    raise Exception('Direct create/update not permitted')

  def __iter__(self):
    for r in self.session.query(*self.pk): yield self.pkindex(r)

  def __len__(self):
    return self.session.query(self.base).count()

  def __hash__(self): return hash((self.session,self.dclass))

  def as_html(self,incontext):
    return html_stack(*(v.as_html(incontext) for k,v in sorted(self.items())))

  cache = {}
  @classmethod
  def sessionmaker(cls,url,execution_options={},*a,**ka):
    from sqlalchemy import event
    from sqlalchemy.orm import sessionmaker
    engine = cls.cache.get(url)
    if engine is None: cls.cache[url] = engine = SQLinit(url,cls.base.metadata)
    if execution_options:
      trace = execution_options.pop('trace',{})
      engine = engine.execution_options(**execution_options)
      for evt,log in trace.items():
        if isinstance(log,int): log = lambda _lvl=log,_fmt='SQA:{}%s'.format(evt),**ka:logger.log(_lvl,_fmt,ka)
        event.listen(engine,evt,log,named=True)
    Session_ = sessionmaker(engine,*a,**ka)
    def Session(**x):
      s = Session_(**x)
      s.root = cls(s,cls.base)
      return s
    return Session

#==================================================================================================
class spark:
  r"""
If a resource ``spark/pyspark.py`` exists in an XDG configuration file, that resource is executed (locally) and should define a function :func:`init` used as class method :meth:`init` and meant to initialise Spark. The initialisation must assign a :class:`dict` (even an empty one) to attribute :attr:`conf`.
  """
#==================================================================================================
  conf = None

  @classmethod
  def display_monitor_link(cls,sc):
    r"""
Displays a link to the monitor of :class:`pyspark.SparkContext` *sc*. Recall that the monitor is active only between the creation of *sc* and its termination (when method :meth:`stop` is invoked).
    """
    from IPython.display import display_html
    display_html('<a target="_blank" href="{}">SparkMonitor[{}@{}]</a>'.format(sc.uiWebUrl,sc.appName,sc.master),raw=True)

  @classmethod
  def SparkContext(cls,display=True,debug=False,**ka):
    r"""
Returns an instance of :class:`pyspark.SparkContext` created with the predefined configuration held in attribute :attr:`conf` of this class, possibly partially overridden if key `conf` is in *ka* (its value must then be a :class:`dict` of :class:`str`). If *debug* is :const:`True`, prints the exact configuration used. If *display* is :const:`True`, displays a link to the monitor of the created context.
    """
    from pyspark import SparkContext, SparkConf
    cfg = SparkConf().setAll(cls.conf.items())
    for k,v in ka.pop('conf',{}).items(): cfg.set(k,v)
    if debug: print(cfg.toDebugString())
    sc = SparkContext(conf=cfg,**ka)
    if display: cls.display_monitor_link(sc)
    return sc

  init = config_xdg('spark/pyspark.py')
  if init is None: init = classmethod(lambda cls,**ka: None)
  else:
    D = {}
    exec(init,D)
    init = classmethod(D['init'])
    del D

#==================================================================================================
class basic_stats:
  r"""
Instances of this class maintain basic statistics about a group of values.

:param weight: total weight of the group
:param avg: weighted average of the group
:param var: weighted variance of the group
  """
#==================================================================================================
  def __init__(self,weight=0.,avg=0.,var=0.): self.weight = weight; self.avg = avg; self.var = var
  def __add__(self,other):
    if isinstance(other,basic_stats): w,a,v = other.weight,other.avg,other.var
    else: w,a,v = 1.,other,0.
    W = self.weight+w; r_self = self.weight/W; r_other = w/W; d = a-self.avg
    return basic_stat(weight=W,avg=r_self*self.avg+r_other*a,var=r_self*self.var+r_other*v+(r_self*d)*(r_other*d))
  def __iadd__(self,other):
    if isinstance(other,basic_stats): w,a,v = other.weight,other.avg,other.var
    else: w,a,v = 1.,other,0.
    self.weight += w; r = w/self.weight; d = a-self.avg
    self.avg += r*d; self.var += r*(v-self.var+(1-r)*d*d)
    return self
  def __repr__(self): return 'basic_stats<weight:{},avg:{},var:{}>'.format(repr(self.weight),repr(self.avg),repr(self.var))

#==================================================================================================
def iso2date(iso):
  r"""
:param iso: triple as returned by :meth:`datetime.date.isocalendar`

Returns the :class:`datetime.date` instance for which the :meth:`datetime.date.isocalendar` method returns *iso*::

   from datetime import datetime
   d = datetime.now().date(); assert iso2date(d.isocalendar()) == d
  """
#==================================================================================================
  from datetime import date,timedelta
  isoyear,isoweek,isoday = iso
  jan4 = date(isoyear,1,4)
  iso1 = jan4-timedelta(days=jan4.weekday())
  return iso1+timedelta(days=isoday-1,weeks=isoweek-1)

#==================================================================================================
def size_fmt(size,binary=True,precision=4,suffix='B'):
  r"""
:param size: a positive number representing a size
:type size: :class:`int`
:param binary: whether to use IEC binary or decimal convention
:type binary: :class:`bool`
:param precision: number of digits displayed (at least 4)
:type precision: :class:`int`

Returns the representation of *size* with IEC prefix. Each prefix is *K* times the previous one for some constant *K* which depends on the convention: *K*\ =1024 with the binary convention (marked with an ``i`` after the prefix); *K*\ =1000 with the decimal convention. Example::

   print(size_fmt(2**30), size_fmt(5300), size_fmt(5300,binary=False), size_fmt(42897.3,binary=False,suffix='m')
   #>>> 1GiB 5.176KiB 5.3KB 42.9Km
  """
#==================================================================================================
  thr,mark = (1024.,'i') if binary else (1000.,'')
  fmt = '{{:.{}g}}{{}}{}{}'.format(precision,mark,suffix).format
  if size<thr: return str(size)+suffix
  size /= thr
  for prefix in 'KMGTPEZ':
    if size < thr: return fmt(size,prefix)
    size /= thr
  return fmt(size,'Y') # :-)

#==================================================================================================
def time_fmt(time,precision=2):
  r"""
:param time: a number representing a time in seconds
:type time: :class:`int`\|\ :class:`float`
:param precision: number of digits displayed
:type precision: :class:`int`

Returns the representation of *time* in one of days,hours,minutes,seconds (depending on magnitude). Example::

   print(time_fmt(100000,4),time_fmt(4238.45),time_fmt(5.35,0))
   #>>> 1.1574day 1.18hr 5sec
  """
#==================================================================================================
  fmt = '{{:.{}f}}'.format(precision).format
  if time < 60.: return '{}sec'.format(fmt(time))
  time /= 60.
  if time < 60.: return '{}min'.format(fmt(time))
  time /= 60.
  if time < 24.: return '{}hr'.format(fmt(time))
  time /= 24.
  return '{}day'.format(fmt(time))

#==================================================================================================
def versioned(v):
  r"""
A decorator which assigns attribute ``version`` of the target function to *v*. The function must be defined at the toplevel of its module. The version must be a simple value.
  """
#==================================================================================================
  def transf(f):
    from inspect import isfunction
    assert isfunction(f)
    f.version = v; return f
  return transf

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
class pickleclass:
  r"""
This namespace class defines class methods :meth:`load`, :meth:`loads`, :meth:`dump`, :meth:`dumps`, similar to those of the standard :mod:`pickle` module, but with class specific Pickler/Unpickler which must be defined in subclasses.
  """
#--------------------------------------------------------------------------------------------------

  @classmethod
  def dump(cls,obj,v): cls.Pickler(v).dump(obj)

  @classmethod
  def load(cls,u): return cls.Unpickler(u).load()

  @classmethod
  def dumps(cls,obj):
    from io import BytesIO
    with BytesIO() as v: cls.Pickler(v).dump(obj); return v.getvalue()

  @classmethod
  def loads(cls,s):
    from io import BytesIO
    with BytesIO(s) as u: return cls.Unpickler(u).load()
