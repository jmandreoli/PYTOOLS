:mod:`PYTOOLS.cache` --- A persistent cache mechanism
=====================================================

This module provides basic persistent cache functionalities.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/cache.py
   :language: python
   :tab-width: 2

To illustrate the cross-process capability of the cache, function :func:`demo` runs each of the demos defined in variable `DEMOS` twice, in two distinct processes started within 2 seconds of each other. When applicable, the cache produced by the first run is reused in the second.

* Using the :func:`persistent_cache` decorator, function :func:`simplefunc` is turned into a persistent cache, and invocations of that function reuse the cache from one run to the other. Indeed, the cache is on disk (in folder *DIR*) and is shared across processes/threads.

* Function :func:`longfunc` is also turned into a persistent cache. Unlike :func:`simplefunc`, it takes some time to complete and may raise an exception. The second run hits the same cache cell before the first run has finished to compute it. In that case, the former waits until the latter completes to reuse the cached value. In one demo, that value is an exception, which is raised in both runs (whether waiting occured or not).

* Function :func:`vfunc` is assigned a version (here the process id, just to show the behaviour when the version changes). Therefore, the persistent cache creates distinct cache blocks for the distinct versions, and there is no reuse of the cached values from one run to the other.

* Function :func:`proc` builds a closed symbolic expression (instance of :class:`Expr`) which implements the following workflow based on the two cached functions :func:`stepI` and :func:`stepK`.

  .. image:: cache.png
     :scale: 65%
     :alt: workflow representation of function :func:`proc`

  It consists of 4 tasks:

  * `P_ini`: calls :func:`stepI` which returns an input dictionary with keys `'a'`, `'b'`, `'c'` modified by some constant `d`
  * `P_ab`: calls :func:`stepK` which chains to the result of the first task (`P_ini`) a dictionary with key `'ab'` assigned the sum of values of keys `'a'` and `'b'` plus some constant `rab`
  * `P_bc`: calls :func:`stepK` which chains to the result of the first task (`P_ini`) a dictionary with key `'bc'` assigned the sum of values of keys `'b'` and `'c'` plus some constant `rbc`
  * `P_abc`: calls :func:`stepK` which chains to the chained results of the last two tasks (`P_ab,P_bc`) a dictionary with key `'abc'` assigned the sum of values of keys `'ab'` and `'bc'` plus some constant `rabc`

  Of course, cacheing such simple operations is not very interesting, but the purpose of the example is to illustrate cacheing in the presence of dependencies between arbitrary tasks which could be much more complex (and computationaly heavy). The logged trace of the computation illustrated in the diagram below shows how the evaluation mechanism of symbolic expressions (called incarnation) in class :class:`MapExpr` and that of map chaining in class :class:`collections.ChainMap` interact with persistent cacheing to provide a flexible workflow implementation.

  .. figure:: cache-diag.png
     :scale: 65%

  Note that this diagram assumes a full garbage collection before each demo, otherwise, some cache accesses are skipped. Indeed, within a process, a persistent cache keeps weak references to all its past accesses (when their values are amenable to weak reference) and reuses them as long as they are not collected.

Typical output:

.. literalinclude:: ../demo/cache.out

Discussion
----------

Persistent cacheing and versioned functions
...........................................
When a cache cell is created by an invocation, any versioned function which appears in any of its arguments at any level is memorised together with its version, so the cache cell is later missed (but not removed) if the version of any of these functions has changed. For example, suppose a function ``f`` is persistently cached and has a cell obtained by an invocation ``f(3,g)`` where ``g`` is a versioned function. Another invocation of the same ``f(3,g)`` later in another process may miss the cell for any of the following reasons:

* the calling function ``f`` has a new version;
* the argument function ``g`` has a new version.

In the example above, the first argument of the persistently cached function :func:`stepK` is typically a symbolic expression (instance of class :class:`Expr`) which contains references at different depths to other functions, in particular :func:`stepI`. So if the latter were versioned and its version changed, the corresponding cache cells for function :func:`stepK` would be missed, even if the version of :func:`stepK` were not changed.

Persistent cacheing with symbolic expressions for workflow task execution
.........................................................................
In a typical workflow task executor (see e.g. function :func:`stepK` in the example above), the first argument *E* is typically an instance of :class:`MapExpr` whose configuration describes all the previous tasks in the workflow. Its incarnation is the set of key-value pairs computed by the previous tasks. The new task typically seeks to enrich this set with some new key-value pairs, based on those of *E*. Since *E* is immutable, it is not possible to directly update it. There are two possibilities: convert *E* into a proper dictionary, or keep *E* as such and use function :func:`collections.ChainMap`::

   def task_exec(E,...):
     ... # compute some update dictionary D from E and the other arguments
     return dict(E,**D) # first solution
     return ChainMap(E,**D) # second solution

In both cases, the same key-value pairs are returned. There is an important difference though.

* In the first solution, *E* is recursively incarnated by its conversion to a dict, so the cached value contains all the key-value pairs already typically assigned by the previous tasks, to which the new key-value pairs from *D* are appended. The advantage of that solution is that a single cache lookup gives access to the computed results of all the tasks up to the current one. On the other hand, the drawback is that this cached value is mostly redundant with the cached values of the previous tasks, and these may be very large.
* In the second solution, the cached value essentially holds the new key-value pairs of *D* together with the configuration of *E*, not its recursive incarnation, and the former is usually much smaller than the latter. The price to pay is that access to the results of previous tasks now requires re-incarnating *E* hence re-accessing the cache, but that overhead is often small and worth the economy in overall cache redundancy.

Available types and functions
-----------------------------

.. automodule:: PYTOOLS.cache
   :members:
   :member-order: bysource
   :show-inheritance:
