:mod:`cache` --- A persistent cache mechanism
=============================================

This module provides basic persistent cache functionalities.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/cache.py
   :language: python
   :tab-width: 2

* Function :func:`simplefunc` is turned into a persistent cache, using the :func:`lru_persistent_cache` decorator. Invocations of the function will use the cache. The cache is on disk (in folder *DIR*) and is shared across processes/threads. Observe that argument *z* is ignored, so calls differing only on that argument will hit the same cache cell.

* Function :func:`longfunc` is also turned into a persistent cache. Unlike :func:`simplefunc`, it takes some time to complete and it may raise an exception. If a call is ongoing and a concurrent call sharing the same cache cell is invoked, the latter is suspended until the former completes. If a call results in an exception, that exception is cached and both that call and all those waiting on it eventually raise the exception.

* Function :const:`proc` builds a process (closed symbolic expression) which chains function :func:`stepA` and :func:`stepB`, both persistently cached. The process consists of 4 steps:

  * `P_ini`: calls :func:`stepA` to get a dictionary with keys `a`, `b`, `c`
  * `P_ab`: calls :func:`stepB` to chain a dictionary with key `ab` assigned the sum of values of `a` and `b` plus some constant `rab`
  * `P_bc`: calls :func:`stepB` to chain a dictionary with key `bc` assigned the sum of values of `b` and `c` plus some constant `rbc`
  * `P_abc`: calls :func:`stepB` to chain a dictionary with key `abc` assigned the sum of values of `ab` and `bc` plus some constant `rabc`

  Of course, simple sums are not very exciting, but the purpose is to illustrate the dependencies between arbitrary tasks which can be much more complex (and computationaly heavy). The logged trace of the computation illustrates both the evaluation mechanism of processes (called incarnation) and its subtle interaction with cacheing.

Typical output:

.. literalinclude:: ../demo/cache.out

Available types and functions
-----------------------------

.. automodule:: myutil.cache
   :members:
   :member-order: bysource
   :show-inheritance:

