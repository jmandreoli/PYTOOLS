:mod:`chrono` --- recording persistent event flows
==================================================

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/chrono.py
   :language: python
   :tab-width: 2

The generator function :func:`weather` retrieves weather data from a remote site. Instances of :class:`OpenWeatherFlow` represent persistent flows polled from that site. Similarly, instances of :class:`ProcFlow` represent persistent flows of process data (obtained using :mod:`psutil`) from the local machine.

Typical output:

.. literalinclude:: ../demo/chrono.out

Available types and functions
-----------------------------

.. automodule:: myutil.chrono
   :members:
   :member-order: bysource
   :show-inheritance:

