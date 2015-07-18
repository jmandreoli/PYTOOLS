:mod:`monitor` --- Generic loop monitoring
==========================================

This module provides basic functionalities to monitor loops.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/monitor.py
   :language: python
   :tab-width: 2

In :func:`demo1`, the monitored iterable is unbounded, but the monitor stops it by setting a threshold on the number of iterations and the total cpu time. Adjust *maxiter* and *maxcpu* to see how far your kernel can go.

In :func:`demo2`, the monitored iterable is also unbounded, and produces a regular sample of the coordinates of a point moving at constant speed on a regular curve (here a cycloid). A :func:`averaging_monitor` computes statistics on the first coordinate of the sample points. A :func:`buffer_monitor` is used to collect the sample points into a list buffer (add a *size* argument to bound the size of the buffer). A :func:`display_monitor` displays it dynamically on a :mod:`matplotlib` figure.

Typical output:

.. literalinclude:: ../demo/monitor.out

.. |output| image:: ../demo/monitor.png
   :width: 400px

|output|

Available types and functions
-----------------------------

.. automodule:: myutil.monitor
   :members:
   :member-order: bysource
   :show-inheritance:

