:mod:`PYTOOLS.animation` --- Animation utilities
================================================

This module provides basic utilities to play animations in :mod:`matplotlib`. An animation is any callback which can be passed a simulation time, enumerated by a player object. The set of simulation times is split into a sequence of contiguous intervals called tracks. A track map is a function of simulation time which returns the bounds of its track interval if the simulation time is valid, and :const:`None` otherwise.

Available types and functions
-----------------------------

.. automodule:: PYTOOLS.animation
   :members:
   :member-order: bysource
   :show-inheritance:
