# File:                 genshi.py
# Creation date:        2014-03-16
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities in Python
#

r"""
This module is only meant to be used as a command line.

.. code:: shell

   python -m genshi SOURCE

``SOURCE`` must be a Genshi template file consistent with its extension (Markup for .html, .xml, .xhtml and Text otherwise). It is applied to the input, which must be a json encoded dictionary, and the result is sent to the output.
"""
import sys, json
from pathlib import Path
from genshi.template import NewTextTemplate as TextTemplate, MarkupTemplate

if __name__=='__main__':
  source, = sys.argv[1:]
  source = Path(source); suffix = source.suffix
  Template,rendering = (MarkupTemplate,suffix[1:]) if suffix in ('.html','.xhtml','.xml') else (TextTemplate,'text')
  t = Template(source.read_text())
  inp = sys.stdin.read()
  data = json.loads(inp)
  out = t.generate(**data).render(rendering)
  sys.stdout.write(out)
