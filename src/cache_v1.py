# File:                 cache.py
# Creation date:        2026-02-10
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Sqlalchemy schema for persistent cache management
#

from typing import List, Iterable, Any
from datetime import datetime
from sqlalchemy import MetaData, Text, Integer, Boolean, DateTime, LargeBinary, Float, ForeignKey, Index, create_engine
from sqlalchemy.orm import mapped_column, relationship, sessionmaker, DeclarativeBase, Mapped
from .sql import SQLinit

__all__ = 'Block', 'Cell', 'AbstractFunctor', 'AbstractStorage'

__version__ = 1
class Base (DeclarativeBase): metadata = MetaData(info={'repository':f'{__package__.title()}.Cache','version':__version__})

def get_sessionmaker(url:str): return sessionmaker(SQLinit(create_engine(url,isolation_level='SERIALIZABLE'),Base.metadata))

class Block (Base):
  r"""Reflection of the table of blocks"""
  __tablename__ = 'Block'
  oid:Mapped[int]          = mapped_column(Integer(),primary_key=True,autoincrement=True)
  functor:Mapped[bytes]    = mapped_column(LargeBinary(),nullable=False,unique=True,index=True)
  r"""functor of the block, as a pickled instance of :class:`AbstractFunctor`"""
  # relations, constraints
  cells:Mapped[List[Cell]] = relationship('Cell',back_populates='block',cascade='delete')
  r"""list of child cells"""

class Cell (Base):
  r"""Reflection of the table of cells"""
  __tablename__ = 'Cell'
  oid:Mapped[int]             = mapped_column(Integer(),primary_key=True,autoincrement=True)
  block_oid:Mapped[int]       = mapped_column(Integer(),ForeignKey('Block.oid'),nullable=False,index=True)
  ckey:Mapped[bytes]          = mapped_column(LargeBinary(),nullable=False)
  r"""ckey of the cell, as returned by method :meth:`AbstractFunctor.getkey`"""
  tstamp:Mapped[datetime]     = mapped_column(DateTime(),nullable=False,index=True)
  r"""time stamp of creation or last hit of the cell"""
  hits:Mapped[int]            = mapped_column(Integer(),default=0)
  r"""total number of hits the cell"""
  size:Mapped[int]            = mapped_column(Integer(),default=0)
  r"""size of the cell value, negative if error, and null if still pending"""
  duration:Mapped[float|None] = mapped_column(Float())
  r"""duration of the interval between the invocation which created the cell and the completion of its value"""
  # relations, constraints
  block:Mapped[Block]         = relationship('Block',back_populates='cells')
  r"""parent block"""
  Index('block-ckey',block_oid,ckey,unique=True)

import abc

#==================================================================================================
class AbstractFunctor (metaclass=abc.ABCMeta):
  r"""
An instance of this class defines a type of (single argument) call to be cached.
  """
#==================================================================================================

  @abc.abstractmethod
  def getkey(self,arg:Any)->bytes:
    r"""
:param arg: an arbitrary python object.

Returns a byte string which represents *arg* uniquely.
    """

  @abc.abstractmethod
  def getval(self,arg:Any):
    r"""
:param arg: an arbitrary python object.

Returns the result of calling this functor with argument *arg*.
    """

  @abc.abstractmethod
  def html(self,ckey:bytes,_)->str:
    r"""
:param ckey: a byte string as returned by invocation of method :meth:`getkey`

Returns an HTML formatted representation of the argument of that invocation.
    """

  @abc.abstractmethod
  def obsolete(self,tol)->bool:
    r"""
:param tol: a value indicating the tolerance for non obsolescence.

Returns a boolean indicating whether the functor is obsolete.
    """

#==================================================================================================
class AbstractStorage (metaclass=abc.ABCMeta):
  r"""
An instance of this class stores cached values on a persistent support.
  """
#==================================================================================================

  db_url:str
  r"""sqlalchemy url of the index database"""

  @abc.abstractmethod
  def setval(self,oid:int,val:Any)->int:
    r"""
:param oid: the identifier of a cell

Stores the cell value. This method is called inside the transaction which inserts a new cell into a cache index, hence exactly once overall for a given cell.
    """

  @abc.abstractmethod
  def getval(self,oid:int,wait:bool)->Any:
    r"""
:param oid: the identifier of a cell
:param wait: whether the cell value is currently being computed by a concurrent thread/process

Retrieves the cell value, possibly waiting for it to be stored. This method is called inside the transaction which looks up a cell from a cache index, which may happens multiple times in possibly concurrent threads/processes for a given cell.
    """

  @abc.abstractmethod
  def remove(self,L:Iterable[int]):
    r"""
:param L: an iterable of cell identifiers

Frees the storage resources associated with the cells.
    """
