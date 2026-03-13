# File:                 html.py
# Creation date:        2025-08-21
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some ipython utilities using the html display mechanism
#

import logging; logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
from functools import partial
import lxml
from lxml.html.builder import E
from lxml.html import tostring, fromstring
from . import unid

#==================================================================================================
class scoped_style:
  # A utility class to incorporate a (configurable) scoped style in html elements
  # value (str): the current value of the style
#==================================================================================================
  __slots__ = 'value',
  def __init__(self,default_style:str): self.value = f'{default_style.strip()}\n'
  def __call__(self,html:lxml.html.HtmlElement):
    if (eid:=html.attrib.get('id')) is None: eid = html.attrib['id'] = unid('_scoped_style')
    return E.div(E.style(self.value.replace('#toplevel','#'+eid),scoped='scoped'),html)
  def __iadd__(self,other:str): self.value += f'{other.strip()}\n'; return self
  def __radd__(self, other:str): return scoped_style(f'{self.value}{other.strip()}\n')
  def bind(self,obj): obj.style = self; return obj # for use as a decorator

#==================================================================================================
@scoped_style('''
#toplevel { border-collapse: collapse; }
#toplevel > thead > tr > td, #toplevel > tbody > tr > td { border: thin solid; text-align:left; }
#toplevel > thead > tr { border-bottom: thick solid; }
#toplevel > thead > tr > td, #toplevel > tbody > tr > td { padding:1mm; }
#toplevel > tbody > tr > td { max-height: 5cm; overflow-y: auto; }
#toplevel span.pointer { padding: 0; color: cyan; background-color: gray; font-weight:bold; }
''').bind
def repr_html(obj:Any)->str:
  r"""
:param obj: object to display (represent in html)
:returns: a HTML ``table`` element whose head row is the base HTML representation of *obj* (including its pointers), and whose body rows map each pointer name to the base HTML representation of its referenced object (possibly containing pointers itself, recursively)

Produces an HTML representation of *obj* formatted as a string. The representation is by default simply the string representation of *obj* enclosed in a ``span`` element, but can be customised if *obj* supports method :meth:`_repr_html_` with parameter *tail*.

Method :meth:`_repr_html_` with parameter *tail* should only be defined for hashable objects. It should always have the following implementation::

   def _repr_html_(self,tail=None):
     if tail is None: return repr_html(self) # return type: str
     ...
     return ... # return type must be lxml.html.Element

The returned value when *tail* is not :const:`None` must be the base HTML representation (as understood by module :mod:`lxml.html`) of the invoking object *self*. If that object is compound, the representation of each component object should be obtained by invoking function *tail* on it. This avoids unmanageable repetitions (if two components share a common sub-component) or infinite recursions. The *tail* function always returns a "pointer" in the form of a string ``?``\ *n* (within a ``span`` element) where *n* is a unique reference number. The scope of such pointers is the toplevel call of function :func:`repr_html`, which guarantees that two occurrences of equal objects will have the same pointer.
  """
#==================================================================================================
  from inspect import signature
  def tail(v):
    try:
      k = ctx.get(v)
      if k is None:
        f = v._repr_html_
        if signature(f).parameters.get('tail') is None: return fromstring(f())
        f = partial(robust,f,v)
        ctx[v] = k = len(ptab); ptab.append(f)
    except: return E.span(repr(v))
    else: return ref(k)
  def robust(f,v):
    try: h = f(tail); assert isinstance(h,lxml.html.HtmlElement)
    except: return E.span(repr(v))
    else: return h
  def ref(k,point=True)->lxml.html.HtmlElement:
    attr = { 'class': 'pointer' }
    if point:
      tref = f'document.getElementById(\'{tid}\').rows[{k}]'
      attr |= {'onmouseenter':f'{tref}.style.outline=\'thick solid red\'','onmouseleave':f'{tref}.style.outline=\'\'','onclick':f'{tref}.scrollIntoView()'}
    return E.span(f'_{k}',**attr)
  tid = unid('_repr_html'); ctx = {}; ptab = []; html = tail(obj)
  if ptab:
    (ref0,top),*L = ((ref(k,False),f()) for k,f in enumerate(ptab)) # invocation f() can add new items in ptab
    html = repr_html.style(E.table(E.thead(E.tr(E.td(top,colspan='2'))),E.tbody(*(E.tr(E.td(k),E.td(v)) for k,v in L)),id=tid))
  return tostring(html,encoding=str)

#==================================================================================================
@scoped_style('''
  #toplevel { border-collapse: collapse; }
  #toplevel > thead > tr > th, #toplevel > thead > tr > td, #toplevel > tbody > tr > th, #toplevel > tbody > tr > td, #toplevel > tfoot > tr > td  { background-color: white; color:black; text-align: left; vertical-align: top; border: thin solid black; }
  #toplevel > thead > tr > th, #toplevel > thead > tr > td { background-color: gray; color: white }
  #toplevel > tfoot > tr > td { background-color: #f0f0f0; color: navy; }
''').bind
def html_table(
  irows:Iterable[tuple[str,lxml.html.Element,...]],hdrs:tuple[str,...],
  opening:Iterable[lxml.html.Element]=(),closing:Iterable[lxml.html.Element]=(),
  encoding:type|str=None
)->str|lxml.html.HtmlElement:
  r"""
:param irows: a generator of pairs of an object (key) and a tuple of objects (value)
:param hdrs: a tuple of strings matching the length of the value tuples
:param opening: list of HTML elements used as head of table
:param closing: list of HTML elements used as foot of table
:param encoding: encoding of the result

Returns an HTML table object (as understood by :mod:`lxml`) with one row for each pair generated from *irows*. The key of each pair is in a column of its own with no header, and the value must be a tuple whose length matches that of *hdrs*, which specify headers as a tuple of strings, one for each column. If *encoding* is :const:`None` (default), the result is returned as an :mod:`lxml.html` object, otherwise it is returned as a string with the specified encoding.
  """
#==================================================================================================
  n = str(1+len(hdrs))
  if opening: opening = (E.tr(E.th(*opening,colspan=n)),)
  thead = (*opening,E.tr(E.td(),*(E.th(hdr) for hdr in hdrs)))
  tbody = [E.tr(E.th(ind),*(E.td(v) for v in row)) for ind,*row in irows]
  if closing: closing = (E.tfoot(E.tr(E.td(*closing,colspan=n))),)
  t = html_table.style(E.table(E.thead(*thead),E.tbody(*tbody),*closing))
  return t if encoding is None else tostring(t,encoding=encoding)

#==================================================================================================
def html_parlist(
  La:Iterable[lxml.html.Element],Lka:Iterable[tuple[str,lxml.html.Element]],
  opening:Iterable[lxml.html.Element]=(),closing:Iterable[lxml.html.Element]=(),style:str='padding: 5px',encoding:type|str=None
  )->str|lxml.html.HtmlElement:
  r"""
:param La: anonymous components
:param Lka: named components
:param opening: list of HTML elements used as opening
:param closing: list of HTML elements used as closing

Returns a default HTML representation of a compound object, where *La,Lka* are the lists of unnamed and named components.The representation consists of the HTML elements in *opening* followed by the representation of the components in *La* and *Lka* (the latter are prefixed with their names in bold), followed by the HTML elements in *closing*. If *encoding* is :const:`None` (default), the result is returned as an :mod:`lxml.html` object, otherwise it is returned as a string with the specified encoding.
  """
#==================================================================================================
  t = E.div(*opening,*(E.span(v,style=style) for v in La),*(E.span(E.b(str(k)),'=',v,style=style) for k,v in Lka),*closing,style='padding:0')
  return t if encoding is None else tostring(t,encoding=encoding)
