# File:                 genshi.py
# Creation date:        2014-03-16
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities in Python
#

r"""
This module is meant to be used as a script. It takes a source file path as single argument. The file must contain a :mod:`genshi` template, either Markup or Text depending on the path extension. The script produces on its output the rendering of the template given the input data, which must be a json encoded dictionary assigning the variable names used in the template.
"""

import sys, json
from pathlib import Path
from genshi.template import NewTextTemplate as TextTemplate, MarkupTemplate

if __name__=='__main__':
  source, = sys.argv[1:]
  p_source = Path(source); suffix = p_source.suffix
  Template,rendering = (MarkupTemplate,suffix[1:]) if suffix in ('.html','.xhtml','.xml') else (TextTemplate,'text')
  t = Template(p_source.read_text())
  inp = sys.stdin.read()
  data = json.loads(inp)
  out = t.generate(**data).render(rendering)
  sys.stdout.write(out)
