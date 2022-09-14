:mod:`PYTOOLS.tensorboard_manager` --- A server of tensorboard servers
======================================================================
This module implements a wsgi application which launches, then manages interactions with, a set of tensorboard servers. Note that the app kills all existing tensorboard processes at startup.

* With the default wsgi server included in the python distribution, the tensorboard manager is started by:

  .. code-block:: sh

     python -m PYTOOLS.tensorboard_manager [ARGS]

  Run the following to obtain information on ARGS:

  .. code-block:: sh

     python -m PYTOOLS.tensorboard_manager --help

* With gunicorn, the command is:

  .. code-block:: sh

     gunicorn [SERVER-OPTIONS] "PYTOOLS.tensorboad_manager.main([APP-ARGS])"

  See function :func:`main` for the supported APP-ARGS.
  Refer to the documentation of other wsgi servers for a similar command.

Available types and functions
-----------------------------

.. automodule:: PYTOOLS.tensorboard_manager
   :members:
   :member-order: bysource
   :show-inheritance:
