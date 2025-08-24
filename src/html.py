# File:                 html.py
# Creation date:        2025-08-21
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some ipython utilities using the html display mechanism
#

from __future__ import annotations
import logging; logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple
import lxml
from lxml.html.builder import E
from lxml.html import tostring
from . import unid

#==================================================================================================
_html_style = '''
#toplevel { border-collapse: collapse; }
#toplevel > thead > tr > td, #toplevel > tbody > tr > td { border: thin solid; text-align:left; }
#toplevel > thead > tr { border-bottom: thick solid; }
#toplevel > thead > tr > td > div, #toplevel > tbody > tr > td > div { padding:0; max-height: 5cm; overflow-y: auto; }
#toplevel span.pointer { padding: 0; color: cyan; background-color: gray; font-weight:bold; }
'''
def repr_html(obj,style=_html_style):
  r"""
Produces an ipython HTML representation. The representation is by default simply the string representation of the instance (enclosed in a ``span`` element), but can be customised if it supports method :meth:`as_html`.

Method :meth:`as_html` should only be defined for hashable objects. It takes as input a function *_* and returns the base HTML representation of the object (as understood by module :mod:`lxml.html`). If invoked on a compound object, it should not recursively invoke method :meth:`as_html` on its components to obtain their HTML representations, because that would be liable to unmanageable repetitions (if two components share a common sub-component) or infinite recursions. Instead, the representation of a component object should be obtained by calling function *_* with that object as argument. It returns a "pointer" in the form of a string ``?``\ *n* (within a ``span`` element) where *n* is a unique reference number. The scope of such pointers is the toplevel call of method :meth:`_repr_html_`, which guarantees that two occurrences of equal objects will have the same pointer.

The :class:`HtmlPlugin` representation of an object is a ``table`` element whose head row is the base HTML representation of that object (including its pointers), and whose body rows align each pointer reference to the base HTML representation of its referenced object (possibly containing pointers itself, recursively). In the following example, the base HTML representation of a node in a graph is given by its label followed by the sequence of pointers to its successor nodes::

   class N: # class of nodes in a (possibly cyclic) directed graph
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
   from IPython.display import display_html
   display_html(repr_html(nabc))

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
  class Pointer:
    _slots__ = 'element', 'element_', 'html'
    def __init__(self,tid,k):
      name = f'_{k}'
      tref = f'document.getElementById(\'{tid}\').rows[{k}]'
      cls = {'class': 'pointer'}
      cls_ = dict(cls,onmouseenter=f'{tref}.style.outline=\'thin solid red\'',onmouseleave=f'{tref}.style.outline=\'\'',onclick=f'{tref}.scrollIntoView()')
      self.element = lambda: E.span(name, **cls)
      self.element_ = lambda: E.span(name, **cls_)
  def hformat(p,*L,style=style):
    if L:
      table = E.table(E.thead(E.tr(E.td(p.html,colspan='2'))),E.tbody(*(E.tr(E.td(p.element()),E.td(p.html)) for p in L)),id=tid)
      return E.div(E.style(style.replace('#toplevel','#'+tid),scoped='scoped'),table)
    else: return p.html
  def _(v):
    try: p = ctx.get(v)
    except TypeError: return E.span(repr(v)) # for unhashable objects
    if p is None:
      if hasattr(v,'as_html'):
        ctx[v] = p = Pointer(tid,len(ctx))
        p.html = E.div(v.as_html(_))
      else: return E.span(repr(v))
    return p.element_()
  tid = unid('htmlplugin')
  ctx = {}
  e = _(obj)
  return tostring(hformat(*ctx.values()) if ctx else e,encoding=str)

#==================================================================================================
def html_parlist(
  html:Callable[[Any],lxml.html.Element],La:Iterable[Any],Lka:Iterable[Tuple[str,Any]],
  opening:Iterable[lxml.html.Element]=(),closing:Iterable[lxml.html.Element]=(),style:str='padding: 5px'
  )->lxml.html.HtmlElement:
  r"""
:param html: callable to use on components to get their HTML representation
:param La: anonymous components
:param Lka: named components
:param opening: lists of HTML elements used as prolog
:param closing: lists of HTML elements used as epilog

Returns a default HTML representation of a compound object, where *La,Lka* are the lists of unnamed and named components.The representation consists of the HTML elements in *opening* followed by the representation of the components in *La* and *Lka* (the latter are prefixed with their names in bold), followed by the HTML elements in *closing*.
  """
#==================================================================================================
  def content():
    for v in La: yield E.span(html(v),style=style)
    for k,v in Lka: yield E.span(E.b(str(k)),'=',html(v),style=style)
  return E.div(*opening,*content(),*closing,style='padding:0')

#==================================================================================================
def html_table(
    irows:Iterable[Tuple[Any,Tuple[Any,...]]],fmts:Tuple[Callable[[Any],str],...],
    hdrs:Tuple[str,...]=None,opening:str=None,closing:str=None,encoding:type|str=None
  )->str|lxml.html.HtmlElement:
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
  return E.div(*(E.div(x,**ka) for x in a))
