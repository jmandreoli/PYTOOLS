:mod:`chrono` --- recording persistent event flows
==================================================

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/chrono.py
   :language: python
   :tab-width: 2

The generator :func:`weather` retrieves weather data from a remote site. Instances of :func:`OpenWeatherFlow` represent  persistent flows polled from that site.

Typical output:

.. literalinclude:: ../demo/chrono.out

Available types and functions
-----------------------------

.. automodule:: myutil.chrono
   :members:
   :member-order: bysource
   :show-inheritance:

