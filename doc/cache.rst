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

* Callable :const:`proc` chains function :func:`stepA` and :func:`stepB`. Thus e.g., the following are equivalent:

  .. code:: python

    proc(s_A=ARG(1,b=2,z=36),s_B=ARG(3,24))
    stepB(stepA(1,b=2,z=36),3,24)

  All the prefix chains, here :func:`stepA` and :func:`stepA`;\ :func:`stepB`, are cached.

Typical output:

.. literalinclude:: ../demo/cache.out

Available types and functions
-----------------------------

.. automodule:: myutil.cache
   :members:
   :member-order: bysource
   :show-inheritance:

