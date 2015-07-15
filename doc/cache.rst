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

* Function :func:`longfunc` is also turned into a persistent cache. Unlike :func:`simplefunc`, it takes some time to complete. If a call is ongoing and a concurrent call sharing the same cache cell is invoked, the latter is suspended until the former completes.

* Callable :const:`process` chains function :func:`stepA` and :func:`stepB`. Thus e.g.

  .. code:: python

    process(ARG(1,b=2),ARG(3))

  first creates an empty assignable object *o*, then calls, in sequence

  .. code:: python

    stepA(o,1,b=2) ; stepB(o,3)

  and returns *o*. All the prefix chains (here :func:`stepA`; and :func:`stepA`;\ :func:`stepB`;) are cached.

Typical output:

.. literalinclude:: ../demo/cache.out

Available types and functions
-----------------------------

.. automodule:: myutil.cache
   :members:
   :member-order: bysource
   :show-inheritance:

