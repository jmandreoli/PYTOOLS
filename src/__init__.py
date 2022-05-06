# File:                 __init__.py
# Creation date:        2014-03-16
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities in Python
#

r"""
:mod:`PYTOOLS` (top level) --- Generic utilities
================================================

Available types and functions
-----------------------------
"""

import os, re, pickle, collections
from typing import Any, Union, Callable, Iterable, Mapping, MutableMapping, Tuple, Optional
import logging; logger = logging.getLogger(__name__)

#==================================================================================================
class owrap:
  r"""
Objects of this class present an attribute oriented proxy interface to an underlying mapping object.

:param __ref__: a reference to a (possibly mutable) mapping object, of which this instance is to be a proxy

Keys in the reference are turned into attributes of the proxy. If *__ref__* is :const:`None`, the reference is a new empty dictionary, otherwise *__ref__* must be a mapping object, and *__ref__* (or its own reference if it is itself of class :class:`owrap`) is taken as reference. The reference is first updated with the keyword arguments *ka*. Example::

   r = dict(a=3,b=6); x = owrap(r)
   assert x.a == 3
   del x.a; x.b += 7
   assert r == {'b':13}
   r['c'] = 42
   assert x.c == 42
   assert x == r
  """
#==================================================================================================
  __slot__ = '__ref__',
  __ref__:Mapping[str,Any]
  r"""The proxy object"""
  def __init__(self,__ref__:Mapping[str,Any]=None,**ka):
    if __ref__ is None: r = dict(ka)
    else:
      r:Mapping[str,Any] = __ref__
      if isinstance(r,owrap): r = r.__ref__
      else: assert isinstance(r,collections.abc.Mapping)
      if ka: r.update(ka)
    super().__setattr__('__ref__',r)
  def __eq__(self,other):
    return self.__ref__ == (other.__ref__ if isinstance(other,owrap) else other)
  def __ne__(self,other):
    return self.__ref__ != (other.__ref__ if isinstance(other,owrap) else other)
  def __hash__(self): return hash(self.__ref__)
  def __getattr__(self,a):
    try: return self.__ref__[a]
    except KeyError: raise AttributeError(a) from None
  def __setattr__(self,a,v):
    self.__ref__[a] = v
  def __delattr__(self,a):
    try: del self.__ref__[a]
    except KeyError: raise AttributeError(a) from None
  def __str__(self): return str(self.__ref__)
  def __repr__(self): return repr(self.__ref__)

#==================================================================================================
class fmtwrap(collections.abc.MutableMapping):
  r"""
Instances of this class provide a formatting wrapper to an underlying mapping object. Each instance keeps a list of visible keys in the reference object.

* accessing a key raises an error if it is not visible, otherwises returns the formatted value of that same key in the reference object
* assigning a key to a value forwards the assignment to the reference object, and also memorises the key as visible. If the key starts with `_`, that character is first dropped and the key is, on the contrary, removed from the visible list if present.
* deleting a key deletes it from both the reference object and the visible list.

:param ref: the reference object (a possibly mutable mapping), of which this instance is to be a partial proxy

E.g.::

   r = {}; x = fmtwrap(r)
   x.fmt.update(pi='{:.3f}'.format,title=str.title)
   x.update(pi=3.14,title='the value of pi',a='something',_b='another')
   assert x==dict(pi='3.140',title='The Value Of Pi',a="'something'")
   assert r==dict(pi=3.14,title='the value of pi',a='something',b='another')
   del x['title']
   assert x==dict(pi='3.140',a="'something'") and r==dict(pi=3.14,a='something',b='another')
   x.update(pi=3.141592,_a='boo')
   assert x==dict(pi='3.142') and r==dict(pi=3.141592,a='boo',b='another')
   x.update(z=42)
   x.fmt.update(z='{:x}'.format)
   assert x==dict(pi='3.142',z='2a') and r==dict(pi=3.141592,a='boo',b='another',z=42)
  """
#==================================================================================================
  ref:MutableMapping[str,Any]
  r"""The reference mapping object"""
  fmt:MutableMapping[str,Callable[[Any],str]]
  r"""The formatters (default is function :func:`repr`)"""

  def __init__(self,ref): self.fmt = {}; self.visible = {}; self.ref = ref
  def __getitem__(self,k):
    if self.visible.get(k): return self.fmt.get(k,repr)(self.ref[k])
    else: raise KeyError(k)
  def __setitem__(self,k,v):
    if k.startswith('_'): k = k[1:]; self.visible.pop(k,None)
    else: self.visible[k] = True
    self.ref[k] = v
  def __delitem__(self,k):
    del self.visible[k]
    del self.ref[k]
  def __iter__(self): yield from iter(self.visible)
  def __len__(self): return len(self.visible)
#--------------------------------------------------------------------------------------------------
  def describe(self,tab:str='',lim:int=120,file=None):
    r"""Pretty-prints out this instance. Each of the possibly multiple lines of output are indented by a prefix given by parameter *tab* and clipped at a number of characters (including the prefix) given by parameter *lim*."""
#--------------------------------------------------------------------------------------------------
    n = max(map(len,self))
    lim = lim-len(tab)-n-3
    if lim<10: print(tab,'...',file=file)
    else:
      def clip(s:str,lim:int=lim):
        n = len(s)
        if n<=lim: return s
        n = (n-3)//2
        return f'{s[:n]}...{s[-n:]}'
      for k,v in self.items():
        for i,s in enumerate(v.split('\n')):
          print(tab,(k if i==0 else '').ljust(n),(' = ' if i==0 else '.. '),clip(s),sep='',file=file)
    print(end='',file=file,flush=True)

#==================================================================================================
class forward:
  r"""
This object allows to name callable members of module/packages without loading them. They are loaded only on actual call. Example::

   from sys import modules
   array = forward.numpy.array
   assert 'numpy' not in modules # numpy is not loaded
   x = array((1,2,3))
   assert 'numpy' in modules and x.shape == (3,) # now numpy is loaded and x is computed
  """
#==================================================================================================
  __slot__ = '__spec__','__value__'
  def __init__(self,s=()): self.__spec__ = s; self.__value__ = None
  def __getattr__(self,a): return self.__class__(self.__spec__+(a,))
  def __call__(self,*a,**ka):
    from importlib import import_module
    f = self.__value__
    if f is None:
      self.__value__ = f = getattr(import_module('.'.join(self.__spec__[:-1])),self.__spec__[-1])
    return f(*a,**ka)
  def __getstate__(self): return self.__spec__
  def __setstate__(self,s): self.__spec__ = s; self.__value__ = None
  def __repr__(self): return 'forward<{}{}>'.format('.'.join(self.__spec__),('' if self.__value__ is None else ':incarnated'))
forward = forward()

#==================================================================================================
def config_xdg(rsc:str,dflt:Any=None):
  r"""
:param rsc: the name of an XDG resource
:param dflt: a default value

Returns the content of the XDG resource named *rsc* or *dflt* if the resource is not found.
  """
#==================================================================================================
  try: from xdg.BaseDirectory import load_first_config
  except: return dflt
  p = load_first_config(rsc)
  if p is None: return dflt
  with open(p) as u: return u.read()

#==================================================================================================
def config_env(name:str,dflt:str=None,asfile:bool=False):
  r"""
:param name: the name of an environment variable
:param dflt: a default value
:param asfile: whether to treat the value of the environment variable as a path to a file

Returns the string value of an environment variable (or the content of the file pointed by it if *asfile* is set) or *dflt* if the environment variable is not assigned.
  """
#==================================================================================================
  x = os.environ.get(name)
  if x is None: return dflt
  if asfile:
    with open(x) as u: x = u.read(x)
  return x

#==================================================================================================
def autoconfig(module:str,name:str,dflt:Any=None,asfile:bool=False):
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
class HtmlPlugin:
  r"""
Instances of this class have an ipython HTML representation. The representation is by default simply the string representation of the instance (enclosed in a ``span`` element), but can be customised if it supports method :meth:`as_html`.

Method :meth:`as_html` should only be defined for hashable objects. It takes as input a function *_* and returns the base HTML representation of the object (as understood by module :mod:`lxml.html`). If invoked on a compound object, it should not recursively invoke method :meth:`as_html` on its components to obtain their HTML representations, because that would be liable to unmanageable repetitions (if two components share a common sub-component) or infinite recursions. Instead, the representation of a component object should be obtained by calling function *_* with that object as argument. It returns a "pointer" in the form of a string ``?``\ *n* (within a ``span`` element) where *n* is a unique reference number. The scope of such pointers is the toplevel call of method :meth:`_repr_html_`, which guarantees that two occurrences of equal objects will have the same pointer.

The :class:`HtmlPlugin` representation of an object is a ``table`` element whose head row is the base HTML representation of that object (including its pointers), and whose body rows align each pointer reference to the base HTML representation of its referenced object (possibly containing pointers itself, recursively). In the following example, the base HTML representation of a node in a graph is given by its label followed by the sequence of pointers to its successor nodes::

   class N (HtmlPlugin): # class of nodes in a (possibly cyclic) directed graph
     def __init__(self,tag,*succ): self.tag,self.succ = tag,list(succ)
     def as_html(self,_):
       from lxml.html.builder import E
       return E.div(E.b(self.tag),'[|',*(y for x in self.succ for y in (_(x),'|')),']')
   # Let's build a trellis DAG
   n = N('')
   na,nb,nc = N('a',n),N('b',n),N('c',n)
   nab,nbc,nac = N('ab',na,nb),N('bc',nb,nc),N('ac',na,nc)
   nabc = N('abc',nab,nbc,nac)
   # n.succ.append(nabc) # makes the graph cyclic
   from IPython.display import display
   display(nabc)

produces (up to some attributes):

.. code-block:: html

   <table>
     <!-- thead: base HTML representation (with pointers) of the initial object -->
     <thead><tr><td colspan="2"> <div><b>abc</b>[|<span>?1</span>|<span>?5</span>|<span>?7</span>|]</div> </td></tr></thead>
     <!-- tbody: mapping each pointer to the base HTML representation of its reference -->
     <tbody>
       <tr> <th>?1</th> <td> <div><b>ab</b>[|<span>?2</span>|<span>?4</span>|]</div> </td> </tr>
       <tr> <th>?2</th> <td> <div><b>a</b>[|<span>?3</span>|]</div> </td> </tr>
       <tr> <th>?3</th> <td> <div><b></b>[|]</div> </td> </tr>
       <tr> <th>?4</th> <td> <div><b>b</b>[|<span>?3</span>|]</div> </td> </tr>
       <tr> <th>?5</th> <td> <div><b>bc</b>[|<span>?4</span>|<span>?6</span>|]</div> </td> </tr>
       <tr> <th>?6</th> <td> <div><b>c</b>[|<span>?3</span>|]</div> </td> </tr>
       <tr> <th>?7</th> <td> <div><b>ac</b>[|<span>?2</span>|<span>?6</span>|]</div> </td> </tr>
     </tbody>
   </table>

which displays roughly as:

   +----+----------------------+
   | **abc**\[\|?1\|?5\|?7\|\] |
   +----+----------------------+
   | ?1 | **ab**\[\|?2\|?4\|\] |
   +----+----------------------+
   | ?2 | **a**\[\|?3\|\]      |
   +----+----------------------+
   | ?3 | \[\|\]               |
   +----+----------------------+
   | ?4 | **b**\[\|?3\|\]      |
   +----+----------------------+
   | ?5 | **bc**\[\|?4\|?6\|\] |
   +----+----------------------+
   | ?6 | **c**\[\|?3\|\]      |
   +----+----------------------+
   | ?7 | **ac**\[\|?2\|?6\|\] |
   +----+----------------------+
  """
#==================================================================================================
  _html_style = '''
#toplevel { border-collapse: collapse; }
#toplevel > thead > tr > td, #toplevel > tbody > tr > td { border: thin solid; text-align:left; }
#toplevel > thead > tr { border-bottom: thick solid; }
#toplevel > thead > tr > td > div, #toplevel > tbody > tr > td > div { padding:0; max-height: 5cm; overflow-y: auto; }
#toplevel span.pointer { padding: 0; color: blue; background-color: #e0e0e0; font-weight:bold; }
'''
  _html_limit = 50
  def _repr_html_(self):
    from lxml.html.builder import E
    from lxml.html import tostring
    from collections import OrderedDict
    def hformat(p,*L,style=self._html_style):
      if L:
        table = E.table(E.thead(E.tr(E.td(p.html,colspan='2'))),E.tbody(*(E.tr(E.td(p.element()),E.td(p.html)) for p in L)),id=tid)
        return E.div(E.style(style.replace('#toplevel','#'+tid),scoped='scoped'),table)
      else: return p.html
    def _(v):
      try: p = ctx.get(v)
      except: return E.span(repr(v)) # for unhashable objects
      if p is None:
        if isinstance(v,HtmlPlugin):
          ctx[v] = p = HtmlPluginPointer(tid,len(ctx))
          try: x = v.as_html(_)
          except: x = E.span(repr(v)) # should not fail, but just in case
          p.html = E.div(x)
        else: return E.span(repr(v))
      return p.element(asref=True)
    tid = unid('htmlplugin')
    ctx = OrderedDict()
    e = _(self)
    return tostring(hformat(*ctx.values()) if ctx else e,encoding=str)
class HtmlPluginPointer:
  __slots__ = 'name','attrs','html'
  def __init__(self,tid,k):
    self.name = '?'+str(k)
    tref = f'document.getElementById(\'{tid}\').rows[{k}]'
    self.attrs = dict(
      onmouseenter=tref+'.style.outline=\'thin solid red\'',
      onmouseleave=tref+'.style.outline=\'\'',
      onclick=tref+'.scrollIntoView()',
      )
  def element(self,asref=False,**ka):
    from lxml.html.builder import E
    if asref: ka.update(self.attrs)
    ka['class'] = 'pointer'
    return E.span(self.name,**ka)

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

  incarnated: bool
  value: Any

  def __init__(self,func:Callable,*a,**ka):
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

  def refunc(self,f:Callable):
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

  def as_html(self,_):
    from lxml.html.builder import E
    func,a,ka = self.config
    opn,clo = ((E.span if self.incarnated else E.em)(str(func),style='padding:5px;'),E.b('['),), (E.b(']'),)
    return html_parlist(_,a,sorted(ka.items()),opening=opn,closing=clo)

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
    ka = ','.join(f'{k}={repr(v)}' for k,v in sorted(ka.items()))
    return f'{func}({a}{sep}{ka})'

#--------------------------------------------------------------------------------------------------
class MapExpr (Expr,collections.abc.Mapping):
  r"""
Symbolic expressions of this class are also (read-only) mappings and trigger incarnation on all the mapping operations, then delegate such operations to their value (expected to be a mapping).
  """
#--------------------------------------------------------------------------------------------------
  def __getitem__(self,k): self.incarnate(); return self.value[k]
  def __len__(self): self.incarnate(); return len(self.value)
  def __iter__(self): self.incarnate(); return iter(self.value)

#--------------------------------------------------------------------------------------------------
class CallExpr (Expr):
  r"""
Symbolic expressions of this class are also callables, and trigger incarnation on invocation, then delegate the invocation to their value (expected to be a callable).
  """
#--------------------------------------------------------------------------------------------------
  def __call__(self,*a,**ka): self.incarnate(); return self.value(*a,**ka)

#==================================================================================================
def html_parlist(
  html:Callable[[Any],'lxml.html.Element'],La:Iterable[Any],Lka:Iterable[Tuple[str,Any]],
  opening:Iterable['lxml.html.Element']=(),closing:Iterable['lxml.html.Element']=(),style:str='padding: 5px'
  )->'lxml.html.HtmlElement':
  r"""
:param html: callable to use on components to get their HTML representation
:param La: anonymous components
:param Lka: named components
:param opening: lists of HTML elements used as prolog
:param closing: lists of HTML elements used as epilog

Returns a default HTML representation of a compound object, where *La,Lka* are the lists of unnamed and named components.The representation consists of the HTML elements in *opening* followed by the representation of the components in *La* and *Lka* (the latter are prefixed with their names in bold), followed by the HTML elements in *closing*.
  """
#==================================================================================================
  from lxml.html.builder import E
  def content():
    for v in La: yield E.span(html(v),style=style)
    for k,v in Lka: yield E.span(E.b(str(k)),'=',html(v),style=style)
  return E.div(*opening,*content(),*closing,style='padding:0')

#==================================================================================================
def html_table(
  irows:Iterable[Tuple[Any,Tuple[Any,...]]],fmts:Tuple[Callable[[Any],str],...],
  hdrs:Tuple[str,...]=None,opening:str=None,closing:str=None,encoding:type|str=None
  )->str|'lxml.html.HtmlElement':
  r"""
:param irows: a generator of pairs of an object (key) and a tuple of objects (value)
:param fmts: a tuple of format functions matching the length of the value tuples
:param hdrs: a tuple of strings matching the length of the value tuples
:param opening: lists of HTML elements used as head of table
:param closing: lists of HTML elements used as foot of table
:param encoding: encoding of the result

Returns an HTML table object (as understood by :mod:`lxml`) with one row for each pair generated from *irow*. The key of each pair is in a column of its own with no header, and the value must be a tuple whose length matches the number of columns. The format functions in *fmts*, one for each column, are expected to return HTML objects. *hdrs* may specify headers as a tuple of strings, one for each column. If *encoding* is :const:`None`, the result is returned as an :mod:`lxml.html` object, otherwise it is returned as a string with the specified encoding.
  """
#==================================================================================================
  from lxml.html.builder import E
  from lxml.html import tostring
  def thead():
    if opening is not None: yield E.tr(E.td(opening,colspan=str(1+len(fmts))))
    if hdrs is not None: yield E.tr(E.td(),*(E.th(hdr) for hdr in hdrs))
  def tbody():
    for ind,row in irows:
      yield E.tr(E.th(str(ind)),*(E.td(fmt(v)) for fmt,v in zip(fmts,row)))
  def tfoot():
    if closing is not None: yield E.tr(E.td(),E.td(closing,colspan=str(len(fmts))))
  tid = unid('table')
  t = E.div(E.style(html_table.style.replace('#toplevel','#'+tid),scoped='scoped'),E.table(E.thead(*thead()),E.tbody(*tbody()),E.tfoot(*tfoot()),id=tid))
  return t if encoding is None else tostring(t,encoding=encoding)

html_table.style = '''
  #toplevel { border-collapse: collapse; }
  #toplevel > thead > tr > th, #toplevel > thead > tr > td, #toplevel > tbody > tr > th, #toplevel > tbody > tr > td, #toplevel > tfoot > tr > td  { background-color: white; color:black; text-align: left; vertical-align: top; border: thin solid black; }
  #toplevel > thead > tr > th, #toplevel > thead > tr > td { background-color: gray; color: white }
  #toplevel > tfoot > tr > td { background-color: #f0f0f0; color: navy; }
'''

#==================================================================================================
def html_stack(*a,**ka):
  r"""
:param a: a list of (lists of) HTML objects (as understood by :mod:`lxml.html`)
:param ka: a dictionary of HTML attributes for the DIV encapsulating each object

Merges the list of HTML objects into a single HTML object, which is returned.
  """
#==================================================================================================
  from lxml.html.builder import E
  return E.div(*(E.div(x,**ka) for x in a))

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
def SQliteNew(path:str,schema:Union[str,Iterable[str]]):
  r"""
:param path: a path to an sqlite database
:param schema: the schema specification as a list of SQL queries (possibly joined by ``\n\n``)

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
def gitcheck(path:str,update:bool=False):
  r"""
:param path: a path to a git repository
:param update: whether to update if stale

Checks directory at *path*, assumed to be a git repository. If *update* is true, attempts to synch with its origin. Returns a tuple of the status  (a string) followed by a list of details. The details are fields ``ref``, ``flags`` and ``note`` from the :class:`git.remote.FetchInfo` object returned by the pull operation. Status is

* ``dirty``: repository is dirty (no details)
* ``uptodate``: repository is up to date (details only if remote exists)
* ``uptodate-now``: repository was stale but has been successfully updated (when *update* is :const:`True`)
* ``stale``: repository is stale but has not been touched (when *update* is :const:`False`)

Use the ``GIT_PYTHON_GIT_EXECUTABLE`` environment variable to set the Git executable if it is not the default ``/usr/bin/git``.
  """
#==================================================================================================
  from git import Repo
  r = Repo(path)
  if r.is_dirty(): return 'dirty',
  try: rm = r.remote()
  except ValueError: return 'uptodate', # clean and no remote
  i, = rm.pull(dry_run=not update)
  d = str(i.ref),i.flags,i.note # detail
  if i.flags & i.HEAD_UPTODATE: return 'uptodate',*d
  if update:
    if i.commit!=r.commit(): raise Exception('Git synch failed',d)
    logger.info('git pull run on %s',r)
    return 'uptodate-now',*d
  return 'stale',*d

#==================================================================================================
def gitcheck_package(pkgname:str,update=False):
  r"""
:param pkgname: full name of a package
:param update: whether to update (git pull) if stale

Assumes that *pkgname* is the name of a python regular (non namespace) package and invokes :meth:`gitcheck` on its path. Reloads the package if ``uptodate-now`` is returned.
  """
#==================================================================================================
  from importlib.util import find_spec
  from importlib import reload
  from sys import modules
  try: path, = find_spec(pkgname).submodule_search_locations
  except: raise ValueError('Not a regular package',pkgname)
  c = gitcheck(path,update)
  if c[0] == 'uptodate-now':
    m = modules.get(pkgname)
    if m is not None: logger.warning('Reloading %s',pkgname); reload(m)
  return c

#==================================================================================================
def SQLinit(engine:Union[str,'sqlalchemy.Engine'],meta:'sqlalchemy.MetaData')->'sqlalchemy.Engine':
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
      metainfo = dict(engine.execute(select(metainfo_table.c)).fetchone())
      del metainfo['created']
    except: raise SQLinitMetainfoException('Not found')
    for k,v in meta.info.items():
      if metainfo.get(k) != str(v):
        raise SQLinitMetainfoException(f'{k}[expected:{v},found:{metainfo.get(k)}]')
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
SQLinit.MetainfoException = SQLinitMetainfoException

#==================================================================================================
class SQLHandler (logging.Handler):
  r"""
:param engine: a sqlalchemy engine (or its url)
:param label: a text label

A logging handler class which writes the log messages into a database.
  """
#==================================================================================================
  def __init__(self,engine:Union[str,'sqlalchemy.Engine'],label:str,*a,**ka):
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

#--------------------------------------------------------------------------------------------------
def SQLHandlerMetadata(info:Mapping[str,Any]=dict(origin=__name__+'.SQLHandler',version=1))->'sqlalchemy.MetaData':
#--------------------------------------------------------------------------------------------------
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

* they have a :attr:`root` attribute, pointing to an instance of *C*, which acts as a mapping where the keys are the primary keys of *B* and the values the corresponding ORM entries. The root object has a convenient ipython HTML representation.

Class *C* should provide a method to insert new objects in the persistent class *B*, and they will then be reflected in the session root (and made persitent on session commit). Example::

   from sqlalchemy.ext.declarative import declarative_base; Base = declarative_base()
   from sqlalchemy import Column, Text, Integer

   class Entry (Base): # example of sqlalchemy declarative persistent class definition
     __tablename__ = 'Entry'
     oid = Column(Integer(),primary_key=True)
     name = Column(Text())
     age = Column(Integer())
     def __repr__(self): return 'Entry<{0.oid},{0.name},{0.age}>'.format(self)

   class Root(ormsroot): # manager for class Entry
     base = Entry

   sessionmaker = Root.sessionmaker

Example of use (assuming :func:`sessionmaker` as above has been imported)::

   Session = sessionmaker('sqlite://') # memory db for the example

   from contextlib import contextmanager
   @contextmanager
   def mysession(): # basic sessions as simple block of instructions
     s = Session(autocommit=True)
     try: yield s
     finally: s.close()

   with mysession() as s: assert len(s.root)==0

   with mysession() as s:
     # first session, define two entries. Use key None in assignment because Entry is autokey
     self.root[None] = Entry(name='jack',age=42); self.root[None] = Entry(name='jill',age=29)
     assert len(s.root) == 2 and s.root[1].name=='jack' and s.root[2].name=='jill'

   with mysession() as s: # second session, possibly on another process (if not memory db)
     jack = s.root.pop(1) # also removes from db
     assert len(s.root) == 1 and jack.name=='jack'

   with mysession() as s: # But of course direct sqlalchemy operations are available
     assert len(s.root) == 1
     x, = list(s.query(Entry.name).filter(Entry.age>25))
   assert x.name == 'jill'
    """
#==================================================================================================

  base = None # must be defined in subclasses, NOT at the instance level

  def __init__(self,session):
    self.pk = pk = self.base.__table__.primary_key.columns.values()
    if len(pk) == 1:
      getpk = lambda r,kn=pk[0].name: getattr(r,kn)
      def setpk(k,r,kn=pk[0].name):
        if k is not None: r[kn] = k # does nothing is k is None (autokeys)
        return r
    else:
      getpk = lambda r,kns=tuple(c.name for c in pk): tuple(getattr(r,kn) for kn in kns)
      def setpk(ks,r,kns=tuple(c.name for c in pk)):
        assert len(ks) in (0,)
        for kn,k in zip(kns,ks): r[kn] = k # does nothing if ks==() (autokeys)
        return r
    self.getpk,self.setpk = getpk,setpk
    self.session = session

  def __getitem__(self,k):
    r = self.session.query(self.base).get(k)
    if r is None: raise KeyError(k)
    return r

  def __delitem__(self,k):
    self.session.delete(self[k])

  def __setitem__(self,k,r):
    assert isinstance(r,self.base)
    self.session.add(self.setpk(k,r))

  def __iter__(self):
    for r in self.session.query(*self.pk): yield self.getpk(r)
  def items(self): return ((self.getpk(r),r) for r in self.session.query(self.base))

  def __len__(self):
    return self.session.query(self.base).count()

  def __hash__(self): return hash((self.session,self.base))

  def as_html(self,_):
    from itertools import islice
    n = len(self)-self._html_limit
    L = self.items(); closing = None
    if n>0: L = islice(L,self._html_limit); closing = f'{n} more'
    return html_table(sorted((k,(v,)) for k,v in L),fmts=(repr,),opening=repr(self),closing=closing)
  def __repr__(self): return f'{self.__class__.__name__}<{self.base.__name__}>'

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
        if isinstance(log,int): log = lambda _lvl=log,_fmt=f'SQA:{evt}%s',**ka:logger.log(_lvl,_fmt,ka)
        event.listen(engine,evt,log,named=True)
    Session_ = sessionmaker(engine,*a,**ka)
    def Session(**x):
      s = Session_(**x)
      s.root = cls(s)
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
    display_html(f'<a target="_blank" href="{sc.uiWebUrl}">SparkMonitor[{sc.appName}@{sc.master}]</a>', raw=True)

  @classmethod
  def SparkContext(cls,display=True,debug=False,conf={},**ka):
    r"""
Returns an instance of :class:`pyspark.SparkContext` created with the predefined configuration held in attribute :attr:`conf` of this class, updated by *conf* (its value must then be a :class:`dict` of :class:`str`). If *debug* is :const:`True`, prints the exact configuration used. If *display* is :const:`True`, displays a link to the monitor of the created context.
    """
    cls._init()
    from pyspark import SparkContext, SparkConf
    cfg = SparkConf().setAll(cls.conf.items()).setAll(conf.items())
    if debug: print(cfg.toDebugString())
    sc = SparkContext(conf=cfg,**ka)
    if display: cls.display_monitor_link(sc)
    return sc

  @classmethod
  def _init(cls):
    if cls.conf is None:
      cls.init()
      assert isinstance(cls.conf,dict) and all((isinstance(k,str) and isinstance(v,str)) for k,v in cls.conf.items()), f'Class method {cls.__module__}.{cls.__name__}.init must assign a str-str dictionary to class attribute \'conf\''

  init = config_xdg('spark/pyspark.py')
  if init is None: init = classmethod(lambda cls,**ka: None)
  else:
    D = {}
    exec(init,D)
    init = classmethod(D['init'])
    del D

#==================================================================================================
class ImapFolder:
  r"""
An instance of this class is an iterable enumerating the unread messages in a folder of an IMAP server. On calling method :meth:`commit`, the messages which have been enumerated since the last commit or rollback (triggered by method :meth:`rollback`) are marked as read. The total number of messages in the folder is held in attribute :attr:`total`, and the number of unread message is given by function :func:`len`.
  """
# ==================================================================================================
  def __init__(self,server:'imaplib.IMAP4',folder:str):
    from email import message_from_bytes
    from email.message import EmailMessage
    from email.policy import default as DefaultEmailPolicy
    def message(n):
      t,v = server.fetch(n,'(RFC822)')
      assert t=='OK'
      x = message_from_bytes(v[0][1],_class=EmailMessage,policy=DefaultEmailPolicy)
      seen.append(n)
      return x
    t,v = server.select(folder)
    assert t=='OK'
    self.total = v[0].decode()
    t,v = server.search(None,'UNSEEN')
    assert t=='OK'
    selected = v[0].decode().split()
    self.content_len = len(selected)
    self.content = (message(n) for n in selected)
    seen = []
    def commit():
      server.store(','.join(seen),'+FLAGS',r'\Seen')
      self.content_len -= len(seen)
      seen.clear()
    self.commit = commit
    self.rollback = seen.clear
  def __iter__(self): yield from self.content
  def __len__(self): return self.content_len

#==================================================================================================
class basic_stats:
  r"""
Instances of this class maintain basic statistics about a group of values.

:param weight: total weight of the group
:param avg: weighted average of the group
:param var: weighted variance of the group
  """
#==================================================================================================
  def __init__(self,weight:Union[int,float,complex]=1.,avg:Union[int,float,complex]=0.,var:Union[int,float,complex]=0.):
    self.weight = weight; self.avg = avg; self.var = var
  def __add__(self,other):
    w,a,v = (other.weight,other.avg,other.var) if isinstance(other,basic_stats) else (1.,other,0.)
    W = self.weight+w; r_self = self.weight/W; r_other = w/W; d = a-self.avg
    return basic_stats(weight=W,avg=r_self*self.avg+r_other*a,var=r_self*self.var+r_other*v+(r_self*d)*(r_other*d))
  def __iadd__(self,other):
    w,a,v = (other.weight,other.avg,other.var) if isinstance(other,basic_stats) else (1.,other,0.)
    self.weight += w; r = w/self.weight; d = a-self.avg
    self.avg += r*d; self.var += r*(v-self.var+(1-r)*d*d)
    return self
  def __repr__(self): return f'basic_stats<weight:{repr(self.weight)},avg:{repr(self.avg)},var:{repr(self.var)}>'
  @property
  def std(self):
    from math import sqrt
    return sqrt(self.var)

#==================================================================================================
def iso2date(iso:Tuple[int,int,int])->'datetime.date':
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
def size_fmt(size:int,binary:bool=True,precision:int=4,suffix:str='B')->str:
  r"""
:param size: a positive number representing a size
:param binary: whether to use IEC binary or decimal convention
:param precision: number of digits displayed (at least 4)

Returns the representation of *size* with IEC prefix. Each prefix is *K* times the previous one for some constant *K* which depends on the convention: *K* =1024 with the binary convention (marked with an ``i`` after the prefix); *K* =1000 with the decimal convention. Example::

   print(size_fmt(2**30), size_fmt(5300), size_fmt(5300,binary=False), size_fmt(42897.3,binary=False,suffix='m'))
   #> 1GiB 5.176KiB 5.3KB 42.9Km
  """
#==================================================================================================
  thr,mark = (1024.,'i') if binary else (1000.,'')
  fmt = f'{{:.{precision}g}}{{}}{mark}{suffix}'.format
  if size<thr: return str(size)+suffix
  size /= thr
  for prefix in 'KMGTPEZ':
    if size < thr: return fmt(size,prefix)
    size /= thr
  return fmt(size,'Y') # :-)

#==================================================================================================
def time_fmt(time:Union[int,float],precision:int=2)->str:
  r"""
:param time: a number representing a time in seconds
:param precision: number of digits displayed

Returns the representation of *time* in one of days,hours,minutes,seconds,milli-seconds (depending on magnitude). Example::

   print(time_fmt(100000,4),time_fmt(4238.45),time_fmt(5.35,0),time_fmt(.932476))
   #> 1.1574day 1.18hr 5sec 932msec
  """
#==================================================================================================
  fmt = f'{{:.{precision}f}}'.format
  if time < 1.: return f'{1000 * time:3.3g}msec'
  if time < 60.: return f'{fmt(time)}sec'
  time /= 60.
  if time < 60.: return f'{fmt(time)}min'
  time /= 60.
  if time < 24.: return f'{fmt(time)}hr'
  time /= 24.
  return f'{fmt(time)}day'

#==================================================================================================
def versioned(v)->Callable[[Callable],Callable]:
  r"""
A decorator which assigns attribute :attr:`version` of the target function to *v*. The function must be defined at the toplevel of its module. The version must be a simple value.
  """
#==================================================================================================
  def transform(f:Callable):
    from inspect import isfunction
    assert isfunction(f)
    f.version = v; return f
  return transform

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def unid(post:str='',pre:str=__name__)->str:
  r"""
Returns a "unique" id for miscellanous uses.
  """
#--------------------------------------------------------------------------------------------------
  from time import time
  return (pre+str(time())+post).replace('.','_')

#--------------------------------------------------------------------------------------------------
def printast(x:Union[str,'ast.AST']):
  r"""
:param x: an AST or string (parsed into an AST)

Pretty-prints the AST object (python abstract syntax tree).
  """
#--------------------------------------------------------------------------------------------------
  import ast
  def pp(x,pre,indent):
    if isinstance(x,ast.AST):
      print(indent,pre,':',x.__class__.__name__)
      indent += ' | '
      for k,y in ast.iter_fields(x): pp(y,k,indent)
    elif isinstance(x,list):
      for i,y in enumerate(x): pp(y, f'{pre}[{i}]', indent)
    else: print(indent,pre,'=',x)
  if isinstance(x,str): x = ast.parse(x)
  else: assert isinstance(x,ast.AST)
  pp(x,'top','')

#--------------------------------------------------------------------------------------------------
class pickleclass:
  r"""
This namespace class defines class methods :meth:`load`, :meth:`loads`, :meth:`dump`, :meth:`dumps`, similar to those of the standard :mod:`pickle` module, but with class specific Pickler/Unpickler which must be defined in subclasses.
  """
#--------------------------------------------------------------------------------------------------

  Pickler: Callable[[Any],pickle.Pickler]
  Unpickler: Callable[[Any],pickle.Unpickler]

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

#--------------------------------------------------------------------------------------------------
def einsum_nd(sgn:str,*a,pat=re.compile(r'(\w+)\.?')):
  r"""
Equivalent of numpy.einsum with named dimensions (limited to 26 like the original!). Dimension names must be words in the sense of the :mod:`re` module (unicode allowed) and must be separated by '.' in operand shapes.
  """
#--------------------------------------------------------------------------------------------------
  from numpy import einsum
  D = collections.defaultdict(lambda A=iter('abcdefghijklmnopqrstuvwxyz'): next(A))
  return einsum(pat.sub((lambda m,D=D:D[m[1]]),sgn),*a)
