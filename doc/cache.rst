:mod:`cache` --- A persistent cache mechanism
=============================================

This module provides basic persistent cache functionalities.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/cache.py
   :language: python
   :tab-width: 2

To illustrate the cross-process capability of the cache, the code at the bottom (functions :func:`demo` and :func:`demo_`) runs each of the demos defined in `DEMOS` twice, in two distinct processes started within 2 seconds of each other.

* Using the :func:`persistent_cache` decorator, function :func:`simplefunc` is turned into a persistent cache, and invocations of that function reuse the cache from one run to the other. Indeed, the cache is on disk (in folder *DIR*) and is shared across processes/threads.

* Function :func:`longfunc` is also turned into a persistent cache. Unlike :func:`simplefunc`, it takes some time to complete and may raise an exception. The second run hits the same cache cell before the first run has finished to compute it. In that case, the former waits until the latter completes to reuse the cached value. In one demo, that value is an exception, which is raised in both runs (whether waiting occured or not).

* Function :func:`vfunc` is assigned a version (here the process id, just to show the behaviour when the version changes). Therefore, the persistent cache creates distinct cache blocks for the distinct versions, and there is no reuse of the cached values from one run to the other.

* Function :func:`proc` builds a closed symbolic expression (instance of :class:`Expr`) which implements the following workflow based on the two cached functions :func:`stepA` and :func:`stepB`.

  .. image:: cache.png
     :scale: 65%
     :alt: workflow representation of function :func:`proc`

  It consists of 4 tasks:

  * `P_ini`: calls :func:`stepA` which returns an input dictionary with keys `'a'`, `'b'`, `'c'` modified by some constant `d`
  * `P_ab`: calls :func:`stepB` which chains to the result of the first task (`P_ini`) a dictionary with key `'ab'` assigned the sum of values of keys `'a'` and `'b'` plus some constant `rab`
  * `P_bc`: calls :func:`stepB` which chains to the result of the first task (`P_ini`) a dictionary with key `'bc'` assigned the sum of values of keys `'b'` and `'c'` plus some constant `rbc`
  * `P_abc`: calls :func:`stepB` which chains to the chained results of the last two tasks (`P_ab,P_bc`) a dictionary with key `'abc'` assigned the sum of values of keys `'ab'` and `'bc'` plus some constant `rabc`

  Of course, cacheing such simple operations is not very interesting, but the purpose of the example is to illustrate dependencies between arbitrary tasks which could be much more complex (and computationaly heavy). The logged trace of the computation illustrated below shows how the evaluation mechanism of symbolic expressions (called incarnation) in class :class:`MapExpr` and that of map chaining in class :class:`collections.ChainMap` interact with persistent cacheing.

  .. figure:: cache-diag.png
     :scale: 65%
     

Typical output:

.. literalinclude:: ../demo/cache.out

Discussion
----------

Persistent cacheing and versioned functions
...........................................
When a cache cell is created by an invocation, any versioned function which appears in any of its arguments at any level is memorised together with its version, so the cache cell is later missed (but not removed) if the version of any of these functions has changed. For example, suppose a function ``f`` is persistently cached and has a cell obtained by an invocation ``f(3,g)`` where ``g`` is a versioned function. Another invocation of the same ``f(3,g)`` later in another process may miss the cell for any of the following reasons:

* the calling function ``f`` has a new version (recall that persistently cached functions are always versioned);
* the argument function ``g`` has a new version.

In the example above, the first argument of the persistently cached function :func:`stepB` is typically a symbolic expression (instance of class :class:`Expr`) which contains references at different depths to other functions, in particular :func:`stepA`. So if the latter were versioned and its version changed, the corresponding cache cells for function :func:`stepB` would be missed, even if the version of :func:`stepB` were not changed.

Persistent cacheing with symbolic expressions for workflow task execution
.........................................................................
A typical workflow task executor (see e.g. function :func:`stepB` in the example above) looks like this::

   def task_exec(E,...):
     ...
     return ChainMap({...},E)

Typically *E* is an instance of :class:`MapExpr` whose configuration describes all the previous tasks in the workflow. Its incarnation is the set of key-value pairs computed by the previous tasks. The new task enriches this set of key-value pairs with some new items. Since *E* is immutable, it is not possible to directly update it, but a functionally equivalent alternative to the last line could still be::

   E = dict(E); E.update({...}); return E

In both cases, the same key-value pairs are returned. There is an important difference though. The cached value in the alternative is a regular dict containing all the keys already present in *E* (recursively incarnated by its conversion to a dict), typically assigned by the previous tasks, plus those computed by the new task. The advantage of that solution is that a single cache lookup gives access to the computed results of all the tasks up to the current one. On the other hand, the drawback is that this cached value might be large, and mostly redundant with the cached values of the previous tasks. With the :func:`ChainMap` solution, only the configuration of *E*, not its incarnation, is stored, and it is usually of negligible size compared to its incarnation. The price to pay is that access to the results of previous tasks now requires re-incarnating *E* hence re-accessing the cache, but that overhead is often small and worth the economy in overall cache redundancy.

Available types and functions
-----------------------------

.. automodule:: myutil.cache
   :members:
   :member-order: bysource
   :show-inheritance:
