:mod:`remote` --- A remote object manager
=========================================

This module provides functionalities for remote method invocation.
It consists of a thin layer around :mod:`pyro4`.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/remote.py
   :language: python
   :tab-width: 2

Typical output:

.. literalinclude:: ../demo/remote.out

Available types and functions
-----------------------------

.. automodule:: myutil.remote.main
   :members:
   :member-order: bysource
   :show-inheritance:

.. automodule:: myutil.remote.sge
   :members:
   :member-order: bysource
   :show-inheritance:

