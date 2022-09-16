# File:                 torch.py
# Creation date:        2020-01-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for pytorch
#

from typing import Sequence, Any, Callable, Iterable, Mapping, Tuple
import logging; logger = logging.getLogger(__name__)
import torch

#--------------------------------------------------------------------------------------------------
class MultiHeadAttention(torch.nn.Module):
  r"""
An instance of this class is a multi-head attention module. Parameters are

.. math::

   \begin{array}{ll|ll}
   \Lambda^{\textrm{(x)}} & \langle K,P',D' \rangle & \Theta^{\textrm{(x)}} & \langle K,P,D \rangle \\
   \Lambda^{\textrm{(y)}} & \langle K,Q',D' \rangle & \Theta^{\textrm{(y)}} & \langle K,Q,D \rangle \\
   \Lambda^{\textrm{(o)}} & \langle K,D' \rangle & \Theta^{\textrm{(o)}} & \langle K,D \rangle
   \end{array}

:param K: number of heads
:param dim: triple :math:`(P,Q,D)` with component defaults :math:`D=\left\lfloor\frac{Q}{K}\right\rfloor` and :math:`Q=P`
:param dimʹ: triple :math:`(P',Q',D')` with same component defaults as *dim*, and default to *dim*
:param bias: where to use bias (possible values are ``Lambda``, ``Theta``, ``both`` (or :const:`True`), ``none`` (or :const:`False`)
  """
#--------------------------------------------------------------------------------------------------
  def __init__(self,K:int,dim:tuple[int],dimʹ:tuple[int]=None,bias:bool|str=False):
    super().__init__()
    P,Q,D = self._parsedim(dim,K); Pʹ,Qʹ,Dʹ = (P,Q,D) if dimʹ is None else self._parsedim(dimʹ,K)
    bias_:Tuple[bool,bool] = {'none':(False,False),False:(False,False),'Lambda':(True,False),'Λ':(True,False),'Theta':(False,True),'ϴ':(False,True),'both':(True,True),True:(True,True)}[bias]
    self.K,self.P,self.Pʹ,self.Q,self.Qʹ,self.D,self.Dʹ = K,P,Pʹ,Q,Qʹ,D,Dʹ
    self.Λ = torch.nn.ParameterList(torch.nn.Parameter(torch.FloatTensor(K,R,Dʹ)) for R in (Pʹ,Qʹ))
    self.ϴ = torch.nn.ParameterList(torch.nn.Parameter(torch.FloatTensor(K,R,D)) for R in (P,Q))
    self.Λₒ = torch.nn.Parameter(torch.FloatTensor(1,1,K,Dʹ)) if bias_[0] is True else None
    self.ϴₒ = torch.nn.Parameter(torch.FloatTensor(1,1,K,D)) if bias_[1] is True else None
    self._reset_params()
    self.temperature = torch.sqrt(torch.tensor(self.Dʹ)) # applied to softmax

  @staticmethod
  def _parsedim(dim,K):
    dim = tuple(dim) if isinstance(dim,Iterable) else (dim,)
    assert 1 <= len(dim) <= 3 and all(isinstance(n,int) and n>1 for n in dim), 'Incorrect dim specification. Format (P,Q,D); default Q=P, default D=Q//K.'
    if len(dim) == 2: P,Q = dim; dim = P,Q,Q//K
    elif len(dim) == 1: P, = dim; dim = P,P,P//K
    return dim

  def _reset_params(self):
    for Λ in self.Λ: torch.nn.init.xavier_uniform_(Λ)
    for ϴ in self.ϴ: torch.nn.init.xavier_uniform_(ϴ)
    if self.Λₒ is not None: torch.nn.init.xavier_uniform_(self.Λₒ)
    if self.ϴₒ is not None: torch.nn.init.zeros_(self.ϴₒ)

  def forward(self,yʹ,xʹ=None,x=None,mask=None):
    r"""
Formula:

.. math::

   \begin{align*}
   y_b & = \sum_k \left[A_{bk}^\top x_b \Theta_k^{\textrm{(x)}}+\mathbf{1}_N\otimes\Theta_k^{\textrm{(o)}}\right]\Theta_k^{\textrm{(y)}\top}\\
   A_{bk} & = \textrm{softmax}_{\textrm{row}}(\tilde{x}_{bk}\tilde{y}_{bk}^\top)\\
   \tilde{x}_{bk} & = x'_b\Lambda_k^{\textrm{(x)}}+\mathbf{1}_M\otimes\Lambda_k^{\textrm{(o)}}\\
   \tilde{y}_{bk} & = y'_b\Lambda_k^{\textrm{(y)}}
   \end{align*}

:param x: :math:`x` tensor of shape :math:`\langle B,M,P \rangle` (a.k.a. value input)
:param xʹ: :math:`x'` tensor of shape :math:`\langle B,M,P' \rangle` (a.k.a. key input)
:param yʹ: :math:`y'` tensor of shape :math:`\langle B,N,Q' \rangle` (a.k.a. query input)
:param mask: tensor of shape :math:`\langle M,N\rangle` or :math:`\langle B,M,N\rangle` (in log domain: possible values include :math:`-\infty`)
:return: :math:`y` tensor of shape :math:`\langle B,N,Q \rangle` (output)
    """
    if xʹ is None: xʹ = yʹ
    if x is None: x = xʹ
    r = torch.einsum('bmp,kpd->bmkd',xʹ,self.Λ[0]) # r: B,M,K,D'
    if self.Λₒ is not None: r = r + self.Λₒ
    r = torch.einsum('bmkd,kqd,bnq->kbmn',r,self.Λ[1],yʹ) # r: K,B,M,N
    if mask is not None:
      if len(mask.shape)==2: r = r + mask[None,None,...]
      else: assert len(mask.shape)==3; r = r + mask[None,...]
    r = torch.softmax(r/self.temperature,dim=-2)
    r = torch.einsum('kbmn,bmp,kpd->bnkd',r,x,self.ϴ[0]) # r: B,N,K,D
    if self.ϴₒ is not None: r = r + self.ϴₒ
    r = torch.einsum('bnkd,kqd->bnq',r,self.ϴ[1]) # r: B,N,Q
    return r

#--------------------------------------------------------------------------------------------------
class Bilinear (torch.nn.Module):
  r"""
An instance of this class is a bilinear product module. Parameters are

.. math::

   \begin{array}{ll|ll|ll}
   \Lambda & \langle P,Q \rangle &
   \Lambda^{\textrm{(x)}} & \langle P \rangle & \Lambda^{\textrm{(y)}} & \langle Q \rangle
   \end{array}

:param P: first input dimension
:param Q: second input dimension (defaults to the first one)
:param bias: whether to add biases
  """
#--------------------------------------------------------------------------------------------------
  def __init__(self,P:int,Q:int=None,bias:bool=True):
    super().__init__()
    assert isinstance(P,int) and P>0
    if Q is None: Q=P
    else: assert isinstance(Q,int) and Q>0
    self.P,self.Q = P,Q
    self.Λ = torch.nn.Parameter(torch.FloatTensor(P,Q))
    self.Λₒ = torch.nn.ParameterList(torch.nn.Parameter(torch.FloatTensor(R,1)) for R in (P,Q)) if bias is True else None
    self._reset_params()

  def _reset_params(self):
    torch.nn.init.xavier_uniform_(self.Λ)
    if self.Λₒ is not None:
      for Λₒ in self.Λₒ: torch.nn.init.zeros_(Λₒ)

  def forward(self,x,y=None,mask=None):
    r"""
Formula:

.. math::

   \begin{align*}
   z_b & = x_b\Lambda y_b^\top + x_b\Lambda^{\textrm{(x)}}\otimes\mathbf{1}_N + \mathbf{1}_M\otimes y_b\Lambda^{\textrm{(y)}}
   \end{align*}

:param x: :math:`x` tensor of shape :math:`\langle B,M,P \rangle`
:param y: :math:`y` tensor of shape :math:`\langle B,N,Q \rangle`
:param mask: tensor of shape :math:`\langle M,N\rangle` or :math:`\langle B,M,N\rangle`
:return: :math:`z` tensor of shape :math:`\langle B,M,N \rangle`
    """
    if y is None: y = x
    r = torch.einsum('bmp,pq,bnq->bmn',x,self.Λ,y) # r: B,M,N
    if self.Λₒ is not None: r = (r + x@self.Λₒ[0]) + (y@self.Λₒ[1]).transpose(1,2)
    if mask is not None:
      if len(mask.shape)==2: r = r + mask[None,...]
      else: assert len(mask.shape)==3; r = r + mask
    return r
