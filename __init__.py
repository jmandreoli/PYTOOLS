import collections
import logging
logger = logging.getLogger(__name__)

#==================================================================================================
class ondemand:
  r"""
Use as a decorator to declare, in a class, a computable attribute which is computed only once (its value is then cached). Example::

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
    if any(p for p in signature(get).parameters.values() if p.kind!=p.POSITIONAL_OR_KEYWORD):
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
Objects of this class present an attribute oriented interface to an underlying proxy mapping object. Keys in the proxy are turned into attributes of the instance. Example::

   x = odict(a=3,b=6); print(x.a)
   #>>> 3
   del x.a; print(x)
   #>>> {'b':6}
   x.b += 7; x.__proxy__
   #>>> {'b':13}
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
def configurable_decorator(decorator):
  r"""
Turns *decorator* into a configurable decorator. Example::

   @configurable_decorator
   def trace(f,message='hello'):
     @wraps(f)
     def F(*a,**ka): print(message); return f(*a,**ka)
     return F

   @trace
   def g(x): return x+1
   g(3)
   #>>> hello
   #>>> 4

   @trace(message='hello world')
   def h(x): return x+1
   h(4)
   #>>> hello word
   #>>> 5   
  """
#==================================================================================================
  from functools import update_wrapper, partial
  def D(*a,**ka):
    return decorator(*a,**ka) if a else partial(D,**ka)
  return update_wrapper(D,decorator)

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
  import os
  x = os.environ.get(name)
  if x is None: return deflt
  if asfile:
    with open(x) as u: x = u.read(x)
  return x

#==================================================================================================
def autoconfig(module,name,dflt=None,asfile=False):
  r"""
:param module: the name of a module
:param name: the name of a configuration parameter
:param dflt: a default value for the configuration parameter
:param asfile: whether to treat the value of the environment variable as a path to a file

Returns an object obtained from an environment variable with a name derived from *module* and *name*. For example, if *module* is ``mypackage.mymodule`` and *name* is ``myparam`` then the environment variable is ``MYPACKAGE_MYMODULE_MYPARAM``. The value of that variable (or the content of the file pointed by it if *asfile* is set) is executed in an empty dictionary and the value attached to key *name* is returned. If the variable is not assigned, *dflt* is returned.
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

:param initv: an argument assignment applied at initialisation
:type initv: dict(:class:`str`,:class:`object`)|list(pair(:class:`str`,:class:`object`))
:param preset: a list of argument assignments controlled by button (widgets only)
:type preset: list(pair(:class:`str`,(dict(:class:`str`,:class:`object`)|\ :class:`str`)))
:param title: an html string
:type title: :class:`str`
:param conf: the specification of each argument passed to the :meth:`add_argument` method
  """
#==================================================================================================

  iwstyle = dict(width='20cm')
  hbstyle = dict(width='.5cm',padding='0cm')
  class Pconf:
    __slots__ = 'name','value','helper','widget','cparse'
    def __init__(s,*a):
      for k,v in zip(s.__slots__,a): setattr(s,k,v)
  @staticmethod
  def checkbounds(x,typ,vmin,vmax,name):
    try: x = typ(x)
    except: raise ConfigException('Bounds constraint violated',name) from None
    if not (vmin<=x and x<=vmax): raise ConfigException('Bounds constraint violated',name) from None
    return x
  @staticmethod
  def checkchoice(x,options,name):
    try: return options[x]
    except KeyError: raise ConfigException('Choice constraint violated',name) from None
  type2widget = {
    bool:('Checkbox','ToggleButton'),
    int:('IntText',),
    float:('FloatText',),
    str:('Text','TextArea',),
  }
  mult2widget = {
    True:('SelectMultiple','ToggleButtons',),
    False:('Dropdown','Select','RadioButtons',),
  }
  def __init__(self,*conf,initv=(),preset=(),title=''):
    from collections import OrderedDict
    self.pconf = p = OrderedDict()
    self.obj = None
    self.widget_ = None
    self.cparser_ = None
    self.initial = None
    for a,ka in conf: self.add_argument(*a,**ka)
    if isinstance(initv,dict): initv = initv.items()
    for k,v in initv:
      e = p.get(k)
      if e is None: self.add_argument(k,v)
      else: e.value = v
    self.preset = preset
    self.title = title
    self.initial = dict((e.name,e.value) for e in p.values())

  def __getitem__(self,k): return self.pconf[k].value
  def __iter__(self): return iter(self.pconf)
  def __len__(self): return len(self.pconf)
  def __repr__(self): return 'Config<{}>'.format(','.join('{}={}'.format(k,repr(e.value)) for k,e in self.pconf.items()))

  def fromipyui(self):
    r"""
Instantiates the arguments through a ipython user interface.
    """
    from IPython.display import display
    display(self.widget)
  def fromcmdline(self,args):
    r"""
Instantiates the arguments by parsing *args*.
    """
    u = self.cparser.parse_args(args)
    for k,e in self.pconf.items():
      if hasattr(u,k): e.value = getattr(u,k)

  def reset(self):
    r"""
Resets any argument instantiations and reinitialise the argument values.
    """
    self.widget_ = None
    self.cparser_ = None
    for k,v in self.initial.items(): self.pconf[k].value = v

  def add_argument(self,name,value,cat=None,helper='',widget=None,cparse=None):
    r"""
Instances of this class are argument specifications for :class:`Config` instances.

:param name: name of the argument
:type name: :class:`str`
:param value: the initial value of the argument
:type value: :class:`object`
:param cat: the category of the argument
:type cat: :class:`NoneType`\ |\ :class:`slice`\ |\ :class:`tuple`\ |\ :class:`list`\ |\ :class:`dict`
:param helper: the helper for the argument
:type helper: :class:`str`
:param widget: the widget specification for the argument (see below)
:param cparse: the command line parser for the argument (see below)
    """
    from collections import OrderedDict
    from functools import partial
    if isinstance(helper,str): helper = helper,helper
    elif isinstance(helper,tuple): helper = tuple(map(''.join,zip(*(((x,x) if isinstance(x,str) else x) for x in helper))))
    else: raise ConfigException('Invalid helper spec',name)
    if widget is None: widget = {}
    elif isinstance(widget,str): widget = dict(type=widget)
    elif isinstance(widget,dict): widget = widget.copy()
    else: raise ConfigException('Invalid widget spec',name)
    if cparse is not None:
      if isinstance(cparse,str): cparse = dict(spec=(cparse,))
      elif isinstance(cparse,tuple): cparse = dict(spec=cparse)
      elif isinstance(cparse,dict): cparse = cparse.copy()
      else: raise ConfigException('Invalid cparse spec',name)
    if cat is None:
      typ = type(value)
      L = self.type2widget.get(typ)
      if L is None: raise ConfigException('Inconsistent cat for value type',name)
      if 'type' in widget:
        if widget['type'] not in L: raise ConfigException('Inconsistent widget for value type',name)
      else: widget.update(type=L[0])
      if cparse is not None: cparse.update(type=typ)
    elif isinstance(cat,slice):
      start,stop,step = (cat.start or 0),cat.stop,cat.step
      if isinstance(stop,int):
        assert isinstance(start,int)
        if step is None: step = 1
        else: assert isinstance(step,int) and step>0
        slider = 'IntSlider'
        typ = int
      elif isinstance(stop,float):
        assert isinstance(start,(int,float)) and start<stop
        if step is None: step = (stop-start)/100
        elif isinstance(step,int): assert step>0; step = (stop-start)/step
        else: assert isinstance(step,float) and step>0
        slider = 'FloatSlider'
        typ = float
      if 'type' in widget:
        if widget['type']!=slider: raise ConfigException('Inconsistent widget for cat',name)
      else: widget.update(type=slider)
      widget.update(min=start,max=stop,step=step)
      if cparse is not None:
        cparse.update(type=partial(self.checkbounds,typ=typ,vmin=start,vmax=stop,name=name))
    elif isinstance(cat,(tuple,list,dict)):
      if isinstance(cat,(tuple,list)): cat = OrderedDict((str(x),x) for x in cat); multiple = False
      else: cat = cat.copy(); multiple = cat.pop('__multiple__',False)
      L = self.mult2widget[multiple]
      if 'type' in widget:
        if widget['type'] not in L: raise ConfigException('Inconsistent widget for cat',name)
      else: widget.update(type=L[0])
      cvalues = cat.values()
      values = value if multiple else (value,)
      if any(v not in cvalues for v in values): raise ConfigException('Inconsistent value for cat',name)
      widget.update(options=cat)
      if cparse is not None:
        if multiple: cparse.update(type=(lambda x,D=cat: tuple(D[y] for y in x)))
        else: cparse.update(type=partial(self.checkchoice,options=cat,name=name),choices=cvalues)
    else: raise ConfigException('Unrecognised cat',name)
    self.pconf[name] = self.Pconf(name,value,helper,widget,cparse)
    if self.initial is not None: self.widget_ = self.cparser_ = None; self.initial[name] = value

  @property
  def widget(self):
    if self.widget_ is not None: return self.widget_
    from functools import partial
    import ipywidgets
    def upd(ev,D=self.pconf): w = ev['owner']; D[w.description].value = w.value
    def updw(w,x): w.value = x
    def upda(data):
      for w in W:
        k = w.description
        if k in data: w.value = data[k]
    def Action(action,**ka):
      b = ipywidgets.Button(**ka); b.on_click(lambda b: action()); return b
    def row(e):
      ka = e.widget.copy()
      w = getattr(ipywidgets,ka.pop('type'))
      style = ka.pop('layout',None)
      w = w(description=e.name,value=e.value,layout=(iwlayout if style is None else ipywidgets.Layout(**style)),**ka)
      W.append(w)
      w.observe(upd,'value')
      return ipywidgets.HBox(children=(Action(partial(updw,w,e.value),description='?',tooltip=e.helper[0],layout=hblayout),w))
    W = []
    iwlayout = ipywidgets.Layout(**self.iwstyle)
    hblayout = ipywidgets.Layout(**self.hbstyle)
    header,footer = [],[]
    if self.preset:
      preset = [(k,(getattr(self,v) if isinstance(v,str) else v)) for k,v in self.preset]
      footer.append(ipywidgets.HBox(children=[Action(partial(upda,data),description=label) for label,data in preset]))
    if self.title:
      header.append(ipywidgets.HTML(self.title))
    self.widget_ = ipywidgets.VBox(children=header+[row(e) for e in self.pconf.values()]+footer)
    return self.widget_

  @property
  def cparser(self):
    if self.cparser_ is not None: return self.cparser_
    from argparse import ArgumentParser
    self.cparser_ = p = ArgumentParser(self.title)
    for e in self.pconf.values():
      if e.cparse is not None:
        ka = e.cparse.copy()
        spec = ka.pop('spec',())
        p.add_argument(*spec,dest=e.name,default=e.value,help=e.helper[1],**ka)
    return p

class ConfigException (Exception): pass

#==================================================================================================
class HtmlPlugin:
  r"""
Instances of this class have an ipython HTML representation based on :func:`html_incontext`.
  """
#==================================================================================================
  def _repr_html_(self):
    from lxml.etree import tostring
    return tostring(html_incontext(self),encoding='unicode')

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

The triple *func*, *a*, *ka* forms the configuration of the :class:`Expr` instance. Its value is defined as the result of calling function *func* with positional and keyword arguments *a* and *ka*. The value is actually computed only once (and cached), and only when method :meth:`incarnate` is invoked. Subclasses should define automatic triggers of incarnation (see e.g. class :class:`MapExpr`). The incarnation cache can be reset by invoking method :meth:`reset`.

Initially, a :class:`Expr` instance is mutable, and its configuration can be changed. It becomes immutable (frozen) after any of the following operations: incarnation (even after it is reset), hashing, and pickling. Incarnation is not saved on pickling, hence lost on unpickling, but can of course be restored using method :meth:`incarnate`. Thus, when receiving a foreign :class:`Expr` instance, a process can decide whether it wants to access its value or only inspect its definition. If recomputing the value is costly, use a persistent cache for *func*.

Caveat: function *func* should be defined at the top-level of its module, and the values in *a* and *ka* should be picklable and hashable (in particular: no dicts nor lists). Hence, any :class:`Expr` instance is itself picklable and hashable. In particular, it can be used as argument in the configuration of another :class:`Expr` instance.
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
  def __repr__(self): return repr(self.value) if self.incarnated else super().__repr__()

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
  from IPython.display import display
  from ipywidgets.widgets.interaction import interact
  P = (len(D)-1)//pgsize + 1
  if start<1: start = 1
  if start>P: start = P
  if P==1: display(D)
  else: interact((lambda page=start:display(D[(page-1)*pgsize:page*pgsize])),page=(1,P))

#==================================================================================================
def ipylist(name,columns):
  r"""
:param name: name of the type
:param columns: a tuple of column names

Returns a subclass of :class:`list` with an IPython pretty printer for columns. Each element of the list is 
  """
#==================================================================================================
  from lxml.etree import tostring, _Element
  from lxml.builder import E
  def parse(col):
    if isinstance(col,str): return col.split(':')[0], '{{{}}}'.format(col).format
    cn,cd = col; assert isinstance(cn,str) and callable(cd); return col
  colnames,coldefns = zip(*map(parse,columns))
  t = type(name,(list,),{})
  fmts = len(columns)*((lambda s: s),)
  def html(x): return x if isinstance(x,_Element) else E.SPAN(str(x))
  t._repr_html_ = lambda self: tostring(html_table(enumerate((html(col(x)) for col in coldefns) for x in self),fmts,hdrs=colnames),encoding='unicode')
  t.__getitem__ = lambda self,s,t=t: t(super(t,self).__getitem__(s)) if isinstance(s,slice) else super(t,self).__getitem__(s)
  return t

#==================================================================================================
def html_incontext(x,refstyle='color: blue; background-color: #e0e0e0;'):
  r"""
:param x: an arbitrary python object

Returns an HTML object (as understood by :mod:`lxml`) representing object *x*. The representation is by default simply the string representation of *x* (enclosed in a SPAN element), but can be customised if *x* supports method :meth:`as_html`. In that case, the result is that of invoking that method.

Method :meth:`as_html` should only be defined for hashable objects. When invoked on an object, it may recursively compute the HTML representations of its components. To avoid duplication and infinite recursion, the HTML representation of a component should be obtained by passing it to the (sole) argument of method :meth:`as_html`, which is a function which returns the normal HTML representation of its argument on the first call, but returns, on subsequent calls, a "pointer" to that representation in the form of a string `?`\ *n* (within a SPAN element) where *n* is a unique reference number. The scope of such pointers is one global call of function :func:`html_incontext` (which should never be invoked recursively).

When pointers are created, the result of calling function :func:`html_incontext` on a object *x* is the normal HTML representation of *x* including its pointers, followed by a TABLE mapping each pointer reference *n* to its own HTML representation. Example::

   class T:
     def __init__(self,*children): self.children = children
     def as_html(self,incontext):
       from lxml.builder import E
       return E.DIV(*(y for x in self.children for y in (incontext(x),'|')))
   # Let's build a trellis DAG
   x = T('')
   xa,xb,xc = T('a',x),T('b',x),T('c',x)
   xab,xbc,xac = T('ab',xa,xb),T('bc',xb,xc),T('ac',xa,xc)
   xabc = T('abc',xab,xbc,xac)
   html_incontext(xabc)

produces (up to some attributes)::

   <TABLE>
     <THEAD><TR><TD colspan="2"> <DIV><SPAN>abc</>|<SPAN>?1</>|<SPAN>?5</>|<SPAN>?7</>|</DIV> </TD></TR></THEAD>
     <TBODY>
       <TR> <TH>?1</TH> <TD> <DIV><SPAN>ab</>|<SPAN>?2</>|<SPAN>?4</>|</DIV> </TD> </TR>
       <TR> <TH>?2</TH> <TD> <DIV><SPAN>a</>|<SPAN>?3</>|</DIV> </TD> </TR>
       <TR> <TH>?3</TH> <TD> <DIV><SPAN></>|</DIV> </TD> </TR>
       <TR> <TH>?4</TH> <TD> <DIV><SPAN>b</>|<SPAN>?3</>|</DIV> </TD> </TR>
       <TR> <TH>?5</TH> <TD> <DIV><SPAN>bc</>|<SPAN>?4</>|<SPAN>?6</>|</DIV> </TD> </TR>
       <TR> <TH>?6</TH> <TD> <DIV><SPAN>c</>|<SPAN>?3</>|</DIV> </TD> </TR>
       <TR> <TH>?7</TH> <TD> <DIV><SPAN>ac</>|<SPAN>?2</>|<SPAN>?6</>|</DIV> </TD> </TR>
     </TBODY>
   </TABLE>
  """
#==================================================================================================
  from lxml.builder import E
  from lxml.etree import ElementTextIterator
  def hformat(L):
    return E.TABLE(E.THEAD(E.TR(E.TD(L[0][1],colspan="2",style='padding:0; border-bottom: thick solid black;'))),E.TBODY(*(E.TR(E.TH('?{}'.format(k),style=refstyle),E.TD(x)) for k,x in L[1:])))
  def incontext(v):
    try: q = ctx.get(v)
    except: return E.SPAN(str(v)) # for unhashable objects
    if q is None:
      if hasattr(v,'as_html'):
        k,ref = len(ctx),'py_{}'.format(id(v))
        ctx[v] = q = [k,ref,None]
        try: x = v.as_html(incontext)
        except: x = E.SPAN(repr(v)) # should not fail, but just in case
        L.append((k,E.DIV(x,id=ref)))
        tit = q[2] = ' '.join(ElementTextIterator(x))
      else: return E.SPAN(str(v))
    else: k,ref,tit = q
    js = lambda x,ref=ref: 'document.getElementById(\'{}\').style.outline=\'{}\''.format(ref,('thick solid red' if x else 'inherit'))
    return E.SPAN('?{}'.format(k),style=refstyle,title=tit,onmouseenter=js(True),onmouseleave=js(False))
  L = []
  ctx = {}
  e = incontext(x)
  n = len(L)
  return e if n==0 else L[0][1] if n==1 else hformat(sorted(L))

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
def html_safe(x):
  r"""
Returns an HTML safe representation of *x*.
  """
#==================================================================================================
  return x._repr_html_() if hasattr(x,'_repr_html_') else str(x).replace('<','&lt;').replace('>','&gt;')

#==================================================================================================
def html_table(irows,fmts,hdrs=None,title=None):
  r"""
:param irows: a generator of pairs of an object (key) and a tuple of objects (value)
:param fmts: a tuple of format functions matching the length of the value tuples
:param hdrs: a tuple of strings matching the length of the value tuples
:param title: a string

Returns an HTML table object with one row for each pair generated from *irow*. The key of each pair is in a column of its own with no header, and the value must be a tuple whose length matches the number of columns. The format functions in *fmts*, one for each column, are expected to return HTML objects (as understood by :mod:`lxml`). *hdrs* may specify headers as a tuple of strings, one for each column.
  """
#==================================================================================================
  from lxml.builder import E
  def thead():
    if title is not None: yield E.TR(E.TD(title,colspan=str(1+len(fmts))),style='background-color: gray; color: white')
    if hdrs is not None: yield E.TR(E.TD(),*(E.TH(hdr) for hdr in hdrs))
  def tbody():
    for ind,row in irows:
      yield E.TR(E.TH(str(ind)),*(E.TD(fmt(v)) for fmt,v in zip(fmts,row)))
  return E.TABLE(E.THEAD(*thead()),E.TBODY(*tbody()))

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
def gitcheck(pkgname):
  r"""
:param pkgname: full name of a package
:type pkgname: :class:`str`

Assumes that *pkgname* is the name of a python package contained in a git repository which is a local, passive copy of a remote repository. Checks that the package is up-to-date, and updates it if needed using the git pull operation. Call before loading the package... Use the ``GIT_PYTHON_GIT_EXECUTABLE`` environment variable to set the Git executable if it is not the default ``/usr/bin/git``.
  """
#==================================================================================================
  from git import Repo
  from importlib.machinery import PathFinder
  from importlib import reload
  from sys import modules
  for path in PathFinder.find_spec(pkgname).submodule_search_locations:
    try: r = Repo(path)
    except OSError: continue
    break
  else:
    raise GitException('target-not-found')
  if r.is_dirty(): raise GitException('target-dirty',r)
  rr = Repo(r.remote().url)
  if rr.is_dirty(): raise GitException('source-dirty',r,rr)
  if r.commit() != rr.commit():
    logger.info('Synching (git pull) %s ...',r)
    r.remote().pull()
    if r.commit() != rr.commit(): raise GitException('synch-failed',r,rr)
    m = modules.get(pkgname)
    if m is not None:
      logger.warn('Reloading %s ...',pkgname)
      reload(m)
class GitException (Exception): pass

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
By default, class method :meth:`init` of this class is identical to :meth:`default_init`. If a resource ``spark/pyspark.py`` exists in an XDG configuration file, that resource is executed (locally) and should define a function :func:`init` used as class method :meth:`init` instead. The defined function can of course invoke method :meth:`default_init`.
  """
#==================================================================================================
  sc = None

  @classmethod
  def default_init(cls,**ka):
    r"""
Creates a :class:`pyspark.context.SparkContext` instance with keyword arguments *ka*, and stores it as class attribute :attr:`sc`. The value of key ``conf`` in *ka*, if present, is converted to an instance of :class:`SparkConf`, if not already one, by applying method :meth:`SparkConf.setAll`.
    """
    import atexit
    from pyspark import SparkContext, SparkConf
    if cls.sc is not None: cls.sc.stop()
    conf = ka.get('conf')
    if conf is not None: ka['conf'] = oconf = SparkConf(); oconf.setAll(conf)
    cls.sc = sc = SparkContext(**ka)
    atexit.register(sc.stop)

  init = config_xdg('spark/pyspark.py')
  if init is None: init = default_init
  else:
    D = {}
    exec(init,D)
    init = classmethod(D['init'])
    del D

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

Returns the representation of *size* with IEC prefix. Each prefix is ``K`` times the previous one for some constant ``K`` which depends on the convention: ``K``\ =1024 with the binary convention (marked with an ``i`` before the prefix); ``K``\ =1000 with the decimal convention. Example::

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
def time_fmt(time,precision=3):
  r"""
:param time: a number representing a time in seconds
:type time: :class:`int`\|\ :class:`float`
:param precision: number of digits displayed
:type precision: :class:`int`

Returns the representation of *time* in one of days,hours,minutes,seconds (depending on magnitude). Example::

   print(time_fmt(100000,4),time_fmt(4238.45,2),time_fmt(5.35))
   #>>> 1.157day 1.2hr 5sec
  """
#==================================================================================================
  fmt = '{{:.{}g}}'.format(precision).format
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
A decorator which assigns a version attribute to a function. The function must be defined at the toplevel of its module. The version must be a simple value.
  """
#==================================================================================================
  def transf(f):
    from inspect import isfunction
    assert isfunction(f)
    f.version = v; return f
  return transf

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
