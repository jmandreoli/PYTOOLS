:mod:`cache` --- A persistent cache mechanism
=============================================

This module provides basic persistent cache functionalities.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/cache.py
   :language: python
   :tab-width: 2

* Using the :func:`lru_persistent_cache` decorator, function :func:`simplefunc` is turned into a persistent cache, and invocations of that function use the cache. The cache is on disk (in folder *DIR*) and is shared across processes/threads.

* Function :func:`longfunc` is also turned into a persistent cache. Unlike :func:`simplefunc`, it takes some time to complete and may raise an exception. If a call is ongoing and a concurrent call sharing the same cache cell is invoked, the latter is suspended until the former completes. If a call results in an exception, that exception is cached and both that call and all those waiting on it eventually raise the exception.

* Function :const:`proc` builds a closed symbolic expression which chains function :func:`stepA` and :func:`stepB`, both persistently cached. The process consists of 4 steps:

  * `P_ini`: calls :func:`stepA` to get a dictionary with keys `a`, `b`, `c`
  * `P_ab`: calls :func:`stepB` to chain a dictionary with key `ab` assigned the sum of values of `a` and `b` plus some constant `rab`
  * `P_bc`: calls :func:`stepB` to chain a dictionary with key `bc` assigned the sum of values of `b` and `c` plus some constant `rbc`
  * `P_abc`: calls :func:`stepB` to chain a dictionary with key `abc` assigned the sum of values of `ab` and `bc` plus some constant `rabc`

  Of course, cacheing such simple operations is not very interesting, but the purpose of the example is to illustrate dependencies between arbitrary tasks which could be much more complex (and computationaly heavy). The logged trace of the computation illustrates both the evaluation mechanism of symbolic expressions (called incarnation, see :class:`Expr`) and its subtle interaction with cacheing.

Typical output:

.. literalinclude:: ../demo/cache.out

Available types and functions
-----------------------------

.. automodule:: myutil.cache
   :members:
   :member-order: bysource
   :show-inheritance:

