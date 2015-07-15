:mod:`monitor` --- Generic loop monitoring
==========================================

This module provides basic functionalities to monitor loops.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/monitor.py
   :language: python
   :tab-width: 2

In :func:`demo1`, the monitored iterable is infinite, but the monitor stops it by setting a threshold on the number of iterations and the total cpu time. Adjust *maxiter* and *maxcpu* to see how far your kernel can go.

In :func:`demo2`, the monitored iterable is also infinite, and produces a regular sample of the coordinates of a point moving at constant speed on a regular curve (here an epi-cycloid). A monitor (named :attr:`my_monitor`) is used to collect these points into a list *L*, which is itself displayed on a :mod:`matplotlib` figure as an animation. When *tailn* is non null, only the *tailn* last points are shown at any time.

Typical output:

.. literalinclude:: ../demo/monitor.out

.. |output| image:: ../demo/monitor.png

|output|

Available types and functions
-----------------------------

.. automodule:: myutil.monitor
   :members:
   :member-order: bysource
   :show-inheritance:

