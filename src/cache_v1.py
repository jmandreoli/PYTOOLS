# File:                 cache.py
# Creation date:        2026-02-10
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Sqlalchemy schema for persistent cache management
#

from typing import List
from datetime import datetime
from sqlalchemy import MetaData, Text, Integer, Boolean, DateTime, LargeBinary, Float, ForeignKey, create_engine, Index
from sqlalchemy.orm import mapped_column, relationship, sessionmaker, DeclarativeBase, Mapped
from .sql import SQLinit

__all__ = 'Block', 'Cell'

__version__ = 1
class Base (DeclarativeBase): metadata = MetaData(info={'repository':'Application','version':__version__})

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
