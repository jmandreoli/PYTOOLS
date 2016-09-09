:mod:`cache` --- A persistent cache mechanism
=============================================

This module provides basic persistent cache functionalities.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/cache.py
   :language: python
   :tab-width: 2

The code at the bottom runs each of the demos defined in `DEMOS` twice, in two distinct processes started within 2 seconds of each other.

* Using the :func:`lru_persistent_cache` decorator, function :func:`simplefunc` is turned into a persistent cache, and invocations of that function reuse the cache from one run to the other. The cache is on disk (in folder *DIR*) and is shared across processes/threads.

* Function :func:`longfunc` is also turned into a persistent cache. Unlike :func:`simplefunc`, it takes some time to complete and may raise an exception. The second run hits the same cache cell before the first run has finished to compute it. In that case, the former waits until the latter completes to reuse the cached value. In one demo, that value is an exception, which is raised in both runs.

* Function :func:`vfunc` is assigned a version (here the process id, just to show the behaviour when the version changes). Therefore, the persistent cache creates distinct cache blocks for the distinct versions, and there is no reuse of the cached values from one run to the other.

* Function :const:`proc` builds a closed symbolic expression which chains function :func:`stepA` and :func:`stepB`, both persistently cached. The process consists of 4 steps:

  * `P_ini`: calls :func:`stepA` which returns a dictionary with keys `a`, `b`, `c`
  * `P_ab`: calls :func:`stepB` which chains to its first argument a dictionary with key `ab` assigned the sum of values of keys `a` and `b` plus some constant `rab`
  * `P_bc`: calls :func:`stepB` which chains to its first argument a dictionary with key `bc` assigned the sum of values of keys `b` and `c` plus some constant `rbc`
  * `P_abc`: calls :func:`stepB` which chains to its first argument a dictionary with key `abc` assigned the sum of values of keys `ab` and `bc` plus some constant `rabc`

  Of course, cacheing such simple operations is not very interesting, but the purpose of the example is to illustrate dependencies between arbitrary tasks which could be much more complex (and computationaly heavy). The logged trace of the computation illustrates both the evaluation mechanism of symbolic expressions (called incarnation, see :class:`Expr`) and its subtle interaction with cacheing.

When a cache cell is created by an invocation, any versioned function which appears in any of its arguments at any level is memorised together with its version, so the cache cell is later invalidated if the version of any of these functions is updated. For example, in the last demo, the result of :func:`proc` is a symbolic invocation of :func:`stepB` whose first argument embeds two other symbolic invocations of :func:`stepB` and one of :func:`stepA`. If :func:`stepA` were versioned, and its version changed between two incarnations of the result of :func:`proc`, the cell for the first incarnation (attached to :func:`stepB`) would be invalidated, even if the version of :func:`stepB` has not changed.

Typical output:

.. literalinclude:: ../demo/cache.out

Available types and functions
-----------------------------

.. automodule:: myutil.cache
   :members:
   :member-order: bysource
   :show-inheritance:

