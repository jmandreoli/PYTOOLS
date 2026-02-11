# File:                 html.py
# Creation date:        2025-08-21
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some ipython utilities using the html display mechanism
#

import logging; logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import lxml
from lxml.html.builder import E
from lxml.html import tostring
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

Produces an HTML representation of *obj* formatted as a string. The representation is by default simply the string representation of *obj* enclosed in a ``span`` element, but can be customised if *obj* supports method :meth:`as_html`.

Method :meth:`as_html` should only be defined for hashable objects. It must take as input a function *_* and return the base HTML representation of its invoking object (as understood by module :mod:`lxml.html`). If invoked on a compound object, it should not recursively invoke method :meth:`as_html` on its components to obtain their HTML representations, because that would be liable to unmanageable repetitions (if two components share a common sub-component) or infinite recursions. Instead, the representation of a component object should be obtained by invoking function *_* with that object as sole argument. It returns a "pointer" in the form of a string ``?``\ *n* (within a ``span`` element) where *n* is a unique reference number. The scope of such pointers is the toplevel call of function :func:`repr_html`, which guarantees that two occurrences of equal objects will have the same pointer.
  """
#==================================================================================================
  def _(v):
    try: k = ctx.get(v)
    except TypeError: return E.span(repr(v)) # for unhashable objects
    if k is None:
      if (tail:=getattr(v,'as_html',None)) is None: return E.span(repr(v))
      ctx[v] = k = len(ptab); ptab.append(lambda: tail(_))
    return ref(k)
  def ref(k,point=True)->lxml.html.HtmlElement:
    attr = { 'class': 'pointer' }
    if point:
      tref = f'document.getElementById(\'{tid}\').rows[{k}]'
      attr |= {'onmouseenter':f'{tref}.style.outline=\'thick solid red\'','onmouseleave':f'{tref}.style.outline=\'\'','onclick':f'{tref}.scrollIntoView()'}
    return E.span(f'_{k}',**attr)
  tid = unid('_repr_html'); ctx = {}; ptab = []; html = _(obj)
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
    irows:Iterable[tuple[Any,tuple[Any,...]]],fmts:tuple[Callable[[Any],str],...],
    hdrs:tuple[str,...]=None,opening:Iterable[lxml.html.Element]=(),closing:Iterable[lxml.html.Element]=(),
    encoding:type|str=None
  )->str|lxml.html.HtmlElement:
  r"""
:param irows: a generator of pairs of an object (key) and a tuple of objects (value)
:param fmts: a tuple of format functions matching the length of the value tuples
:param hdrs: a tuple of strings matching the length of the value tuples
:param opening: list of HTML elements used as head of table
:param closing: list of HTML elements used as foot of table
:param encoding: encoding of the result

Returns an HTML table object (as understood by :mod:`lxml`) with one row for each pair generated from *irows*. The key of each pair is in a column of its own with no header, and the value must be a tuple whose length matches the number of columns. The format functions in *fmts*, one for each column, are expected to return HTML objects. *hdrs* may specify headers as a tuple of strings, one for each column. If *encoding* is :const:`None` (default), the result is returned as an :mod:`lxml.html` object, otherwise it is returned as a string with the specified encoding.
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
  t = html_table.style(E.table(E.thead(*thead()),E.tbody(*tbody()),E.tfoot(*tfoot())))
  return t if encoding is None else tostring(t,encoding=encoding)

#==================================================================================================
def html_parlist(
  html:Callable[[Any],lxml.html.Element],La:Iterable[Any],Lka:Iterable[tuple[str,Any]],
  opening:Iterable[lxml.html.Element]=(),closing:Iterable[lxml.html.Element]=(),style:str='padding: 5px',encoding:type|str=None
  )->str|lxml.html.HtmlElement:
  r"""
:param html: callable to use on components to get their HTML representation
:param La: anonymous components
:param Lka: named components
:param opening: list of HTML elements used as prolog
:param closing: list of HTML elements used as epilog

Returns a default HTML representation of a compound object, where *La,Lka* are the lists of unnamed and named components.The representation consists of the HTML elements in *opening* followed by the representation of the components in *La* and *Lka* (the latter are prefixed with their names in bold), followed by the HTML elements in *closing*. If *encoding* is :const:`None` (default), the result is returned as an :mod:`lxml.html` object, otherwise it is returned as a string with the specified encoding.
  """
#==================================================================================================
  def content():
    for v in La: yield E.span(html(v),style=style)
    for k,v in Lka: yield E.span(E.b(str(k)),'=',html(v),style=style)
  t = E.div(*opening,*content(),*closing,style='padding:0')
  return t if encoding is None else tostring(t,encoding=encoding)
