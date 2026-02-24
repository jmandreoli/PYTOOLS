# File:                 demo/demo_html.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the html module

from PYTOOLS.html import repr_html

class Node: # class of nodes in a (possibly cyclic) directed graph

  tag:str
  succ:list[Node]

  def __init__(self,tag:str,*succ:Node):
    self.tag,self.succ = tag,list(succ)

  def _repr_html_(self,tail=None):
    # for use with IPython smart display
    from lxml.html.builder import E
    if tail is None: return repr_html(self)
    succ = (y for x in self.succ for y in (tail(x),'|'))
    return E.div(E.b(self.tag),'[|',*succ,']')

# A trellis graph
def trellis(recurs=False):
  n = Node('')
  na,nb,nc = Node('a', n),Node('b', n),Node('c', n)
  nab,nbc,nac = Node('ab', na, nb),Node('bc', nb, nc),Node('ac', na, nc)
  nabc = Node('abc', nab, nbc, nac)
  if recurs: n.succ.append(nabc)
  return nabc

p = RUN.path('.html'); p.write_text(repr_html(trellis()))

print(f'''
**Tabular representation of a trellis graph**
(hover over pointers to highlight their mapped HTML representation)

.. raw:: html
   :file: {p.name}
''')
