:mod:`PYTOOLS.genshi` --- A template application script
=======================================================

This module is only meant to be used as a command line.

.. code:: shell

   python -m PYTOOLS.genshi SOURCE

``SOURCE`` must be a Genshi template file consistent with its extension (Markup for .html, .xml, .xhtml and Text otherwise). It is applied to the input, which must be a json encoded dictionary assigning the variable names used in the template, and the result is sent to the output.
