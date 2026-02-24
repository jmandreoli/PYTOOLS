:mod:`PYTOOLS.cache` --- A persistent cache mechanism
=====================================================

This module provides basic persistent cache functionalities. Persistence means that cached values are stored and retrieved outside the main memory, hence can live longer than the python process where they are declared (in contrast with what function :func:`functools.cache` already provides). A default implementation of a persistent cache store is provided using the local file system. A protocol,

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: mod_cache.py
   :lines: 6-
   :language: python
   :tab-width: 2

To illustrate the cross-process capability of the cache, function :func:`demo` runs each of the demos defined in variable `DEMOS` twice, in two distinct processes started within 2 seconds of each other. When applicable, the cache produced by the first run is reused in the second.

* Using the :func:`persistent_cache` decorator, function :func:`simplefunc` is turned into a persistent cache, and invocations of that function reuse the cache from one run to the other. Indeed, the cache is on disk (in folder *_dbpath*) and is shared across processes/threads.

* Function :func:`longfunc` is also turned into a persistent cache. Unlike :func:`simplefunc`, it takes some time to complete and may raise an exception. The second run hits the same cache cell before the first run has finished to compute it. In that case, the former waits until the latter completes to reuse the cached value. In one demo, that value is an exception, which is raised in both runs (whether waiting occurred or not).

* Function :func:`vfunc` is assigned a different version in the two runs (just to show the behaviour when the version changes). The persistent cache creates distinct cache blocks for the distinct versions, and there is no reuse of the cached values from one run to the other.

* Function :func:`proc` builds a closed symbolic expression (instance of :class:`Symbolic`) which implements the following workflow based on the two cached functions :func:`stepI` and :func:`stepK`.

  .. image:: _build/resource/mod_cache-workflow.png
     :scale: 65%
     :alt: workflow representation of function :func:`proc`

  It consists of 4 tasks:

  * `P_ini`: calls :func:`stepI` which returns an input dictionary with keys `'a'`, `'b'`, `'c'` modified by some constant `d`
  * `P_ab`: calls :func:`stepK` which chains to the result of the first task (`P_ini`) a dictionary with key `'ab'` assigned the sum of values of keys `'a'` and `'b'` plus some constant `rab`
  * `P_bc`: calls :func:`stepK` which chains to the result of the first task (`P_ini`) a dictionary with key `'bc'` assigned the sum of values of keys `'b'` and `'c'` plus some constant `rbc`
  * `P_abc`: calls :func:`stepK` which chains to the chained results of the last two tasks (`P_ab,P_bc`) a dictionary with key `'abc'` assigned the sum of values of keys `'ab'` and `'bc'` plus some constant `rabc`

  Of course, the purpose of cacheing such simple operations is to illustrate cacheing in the presence of dependencies between arbitrary tasks which could be much more complex (and computationally heavy). The logged trace of the computation illustrated in the diagram below shows how the evaluation mechanism of symbolic expressions interacts with persistent cacheing to provide a flexible workflow implementation.

  .. figure:: _build/resource/mod_cache-timeline.png
     :scale: 65%

  Note that this diagram assumes a full garbage collection before each demo, otherwise, some cache accesses are skipped. Indeed, within a process, a persistent cache keeps weak references to all its past accesses (when their values are amenable to weak reference) and reuses them as long as they are not collected.

.. topic:: Typical output:

   .. literalinclude:: _build/resource/mod_cache.out

Discussion
----------

Persistent cacheing and versioned functions
...........................................
When a cache cell is created by an invocation, any versioned function which appears in any of its arguments at any level is memorised together with its version, so the cache cell is later invalidated (but not removed) if the version of any of these functions has changed. For example, suppose a function ``f`` is persistently cached and has a cell obtained by an invocation ``f(3,g)`` where ``g`` is a versioned function. Another invocation of the same ``f(3,g)`` later in another process may miss the cell for any of the following reasons:

* the calling function ``f`` has a new version;
* the argument function ``g`` has a new version.

In the example above, the first argument of the persistently cached function :func:`stepK` is typically a symbolic expression (instance of class :class:`Symbolic`) which contains references at different depths to other functions, in particular :func:`stepI`. So if the latter were versioned and its version changed, the corresponding cache cells for function :func:`stepK` would be invalidated, even if the version of :func:`stepK` were not changed.

Persistent cacheing with symbolic expressions for workflow task execution
.........................................................................
In a typical workflow task executor (see e.g. function :func:`stepK` in the example above), the first argument *E* is typically an instance of :class:`Symbolic` whose configuration describes all the previous tasks in the workflow. Its incarnation is a workflow state given by a set of key-value pairs, computed by the previous tasks. The new task typically seeks to enrich this state with some new key-value pairs given as a dictionary *D*.::

   @persistent_cache(...)
   def task_exec(E,...):
     ... # compute some update dictionary D from E and the other arguments
     return E|D

Note that, because *E* is symbolic, the output is itself symbolic, and the new workflow state is available only through its incarnation, e.g. when retrieving a key-value pair from that state. Since the task is persistently cached, the new state *E*``|``*D* is pickled as a symbolic expression. When retrieving that same state in a different process, it is unpickled, so that *D* is restored but the incarnation of *E* is lost (although it was available on pickling). It needs to be re-incarnated when retrieving a key-value pair in *E*. To minimise this cost, all the tasks in the construction of *E* should be persistently cached. Alternatively, if access to keys in *D* are more likely than in *E*, the returned value could be replaced by ``Symbolic(collections.ChainMap,D,E)`` using standard module :mod:`collections`.

Database schema and abstract classes
------------------------------------

.. automodule:: PYTOOLS.cache_v1
   :members:
   :member-order: bysource
   :show-inheritance:

Available types and functions
-----------------------------

.. automodule:: PYTOOLS.cache
   :members:
   :member-order: bysource
   :show-inheritance:
