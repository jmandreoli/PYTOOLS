:mod:`PYTOOLS.html` --- Utilities for ipython html display
==========================================================

This module provides some utilities to display object as html (esp. useful with the ipython advanced display facilities).

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: mod_html.py
   :lines: 6-
   :language: python
   :tab-width: 2

Here, a compound object with recursive structure represents a graph as an instance of class :class:`Node`. For each :class:`Node` instance, attribute :attr:`tag` holds its tag and attribute :attr:`succ` holds the list of its successor nodes. Method :meth:`as_html` of class :class:`Node` allows the customisation of function :func:`repr_html` (from this module), so that a graph is displayed as a table of its nodes. Note that this method is not invoked recursively on the successor nodes, which would result in repetitions, and a potentially infinite recursion. Instead, the argument of method :meth:`as_html` is invoked with each successor node passed as argument.

.. admonition:: Typical output

   .. include:: _resource/mod_html.out

Available types and functions
-----------------------------

.. automodule:: PYTOOLS.html
   :members:
   :member-order: bysource
   :show-inheritance:
