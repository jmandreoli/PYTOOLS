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

  Of course, cacheing such simple operations is not very interesting, but the purpose of the example is to illustrate dependencies between arbitrary tasks which could be much more complex (and computationaly heavy). The logged trace of the computation illustrates both the evaluation mechanism of symbolic expressions (called incarnation, see :class:`Expr`) and its subtle interaction with cacheing.

Typical output:

.. literalinclude:: ../demo/cache.out

Discussion
----------
When a cache cell is created by an invocation, any versioned function which appears in any of its arguments at any level is memorised together with its version, so the cache cell is later invalidated (but not removed) if the version of any of these functions is updated. For example, in the last demo, the result of :func:`proc` is a symbolic invocation of :func:`stepB` whose first argument embeds two other symbolic invocations of :func:`stepB` and one of :func:`stepA`. If :func:`stepA` were versioned, and its version had changed between two incarnations of the result of :func:`proc` in distinct processes, the top cell for the first incarnation (attached to :func:`stepB`) would not be hit by the second incarnation, even though the version of :func:`stepB` has not changed.

Note that function :func:`stepB` has the structure of a generic task executor::

   def task_exec(E,...):
     ...
     return ChainMap({...},E)

Keeping in mind that *E*, which captures all the previous tasks, is typically an instance of :class:`MapExpr`, hence immutable, it is not possible to directly update *E*, but an alternative to the last line could still be::

   E = dict(E); E.update({...}); return E

The cached value would then be a regular dict containing all the keys already present in *E*, typically assigned by previous tasks, plus those computed by that task. The advantage would be that a single cache lookup gives access to the computed results of all the tasks. On the other hand, the drawback is that the cached value might be large, and mostly redundant with the cached values of the previous tasks. With the :func:`ChainMap` solution, the cached value contains *E* directly, the storage of which only involves its configuration, which is typically of negligible size compared to its value. The price to pay is that access to the results of previous tasks requires re-accessing the cache, but that overhead is often small and worth the economy in total cache size.

Available types and functions
-----------------------------

.. automodule:: myutil.cache
   :members:
   :member-order: bysource
   :show-inheritance:

