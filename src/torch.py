# File:                 torch.py
# Creation date:        2020-01-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for pytorch
#

from typing import Sequence, Any, Callable, Iterable, Mapping, Tuple, Optional
import logging; logger = logging.getLogger(__name__)
import torch

#==================================================================================================
class GeneralisedConvolution(torch.nn.Module):
  r"""
An instance of this class is a generalised convolution module.
  """
#==================================================================================================

  temperature = None
  """Temperature of the softmax applied to scores"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,K:int,P:int,Q:Optional[int]=None,D:Optional[int]=None,bias:bool=True):
    r"""
Model parameters are

.. math::

   \Theta^{\textrm{(x)}}{:}\langle K,P,D \rangle \hspace{.5cm} \Theta^{\textrm{(y)}}{:}\langle K,Q,D \rangle \hspace{.5cm} \Theta^{\textrm{(o)}}{:}\langle K,D \rangle \hspace{.5cm} \Theta^{\textrm{(oo)}}{:}\langle Q \rangle

:param K: number of heads
:param P: value-input dimension :math:`P`
:param Q: output dimension :math:`Q`, default :math:`P`
:param D: convolution head dimension :math:`D`, default :math:`\left\lfloor\frac{Q}{K}\right\rfloor`
:param bias: whether to use the biases :math:`\Theta^{\textrm{(o),(oo)}}`
    """
#--------------------------------------------------------------------------------------------------
    super().__init__()
    K=self._dim(K); P=self._dim(P); Q=self._dim(Q,P); D=self._dim(D,Q//K); self.K,self.P,self.Q,self.D=K,P,Q,D
    self.ϴ = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(K,d,D)) for d in (P,Q)) # ϴx,ϴy
    self.ϴₒ = torch.nn.ParameterList((torch.nn.Parameter(torch.empty(1,K,1,D)),torch.nn.Parameter(torch.empty(1,1,Q)))) if bias is True else None # ϴo,ϴoo

#--------------------------------------------------------------------------------------------------
  def _reset_params(self):
#--------------------------------------------------------------------------------------------------
    for ϴ in self.ϴ: torch.nn.init.xavier_uniform_(ϴ)
    if self.ϴₒ is not None:
      for ϴ in self.ϴₒ: torch.nn.init.zeros_(ϴ)

#--------------------------------------------------------------------------------------------------
  def forward(self,score,x,mask=None,process_attn:Callable=(lambda a:None)):
    r"""
Generalised Convolution formula:

.. math::

   \begin{align*}
   y_b & = \sum_k A_{bk}^\top\left[x_b \Theta_k^{\textrm{(x)}}+\mathbf{1}_M\otimes\Theta_k^{\textrm{(o)}}\right]\Theta_k^{\textrm{(y)}\top}+\mathbf{1}_N\otimes\Theta^{\textrm{(oo)}}\\
   \textrm{where } & A_{bk}{:}\langle M,N \rangle\\
   A_{bk} & = \textrm{softmax}_{\textrm{col}}\frac{1}{\textrm{temp}}(\bar{A}_{bk}+\textrm{mask}_b)
   \end{align*}

:param score: tensor :math:`\bar{A}{:}\langle B,K,M,N \rangle` (attention scores)
:param x: tensor :math:`x{:}\langle B,M,P \rangle` (value-input)
:param mask: tensor :math:`\langle B,M,N \rangle` or :math:`\langle M,N \rangle` (in log domain: possible values include :math:`-\infty`)
:param process_attn: applied to the attention tensor :math:`\langle B,K,M,N \rangle` (caution: large) to produce the auxiliary output; default returns :const:`None`
:return: pair of output tensor :math:`y{:}\langle B,N,Q \rangle` and auxiliary output (see *process_attn*)
    """
#--------------------------------------------------------------------------------------------------
    r = score # r: B,K,M,N
    del score
    if mask is not None:  # mask: M,N or B,M,N
      if len(mask.shape) == 2: r = r + mask[None,None,:,:]
      else: assert len(mask.shape) == 3; r = r + mask[:,None,:,:]
    r = torch.softmax(r/self.temperature,dim=-2) # attention tensor
    aux = process_attn(r)
    xt = torch.einsum('bmp,kpd->bkmd',x,self.ϴ[0]) # xt: B,K,M,D
    if self.ϴₒ is not None: xt = xt + self.ϴₒ[0]
    r = torch.einsum('bkmn,bkmd,kqd->bnq',r,xt,self.ϴ[1])  # r: B,N,Q
    if self.ϴₒ is not None: r = r + self.ϴₒ[1]
    return r,aux

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def _dim(d,default=None):
#--------------------------------------------------------------------------------------------------
    if default is None: assert isinstance(d,int) and d>1; return d
    if d is None: return default
    assert isinstance(d,int) and d>1; return d

#==================================================================================================
class MultiHeadAttention(GeneralisedConvolution):
  r"""
An instance of this class is a Vanilla multi-head attention module.
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,*args,Pʹ:Optional[int]=None,Qʹ:Optional[int]=None,Dʹ:Optional[int]=None,bias:bool=False,**kargs):
    r"""
Additional model parameters for Vanilla attention are:

.. math::

   \Lambda^{\textrm{(x)}}{:}\langle K,P',D' \rangle \hspace{.5cm} \Lambda^{\textrm{(y)}}{:}\langle K,Q',D' \rangle \hspace{.5cm} \Lambda^{\textrm{(o)}}{:}\langle K,D' \rangle

:param Pʹ: key-input dimension :math:`P'`, default :math:`P`
:param Qʹ: query-input dimension :math:`Q'`, default :math:`Q`
:param Dʹ: attention head dimension :math:`D'`, default :math:`\left\lfloor\frac{Q'}{K}\right\rfloor`
:param bias: whether to use the bias  :math:`\Lambda^{\textrm{(o)}}` + the convolution biases
:param args: passed to :class:`GeneralisedConvolution` constructor
:param kargs: passed to :class:`GeneralisedConvolution` constructor
    """
#--------------------------------------------------------------------------------------------------
    super().__init__(*args,bias=bias,**kargs)
    Pʹ=self._dim(Pʹ,self.P); Qʹ=self._dim(Qʹ,self.Q); Dʹ=self._dim(Dʹ,Qʹ//self.K); self.Pʹ,self.Qʹ,self.Dʹ=Pʹ,Qʹ,Dʹ
    self.Λ = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(self.K,d,Dʹ)) for d in (Pʹ,Qʹ)) # Λx,Λy
    self.Λₒ = torch.nn.Parameter(torch.empty(1,self.K,1,Dʹ)) if bias is True else None # Λo
    self.temperature = torch.sqrt(torch.tensor(Dʹ))
    self._reset_params()

#--------------------------------------------------------------------------------------------------
  def _reset_params(self):
#--------------------------------------------------------------------------------------------------
    super()._reset_params()
    for Λ in self.Λ: torch.nn.init.xavier_uniform_(Λ)
    if self.Λₒ is not None: torch.nn.init.zeros_(self.Λₒ)

  #--------------------------------------------------------------------------------------------------
  def forward(self,yʹ,xʹ=None,**kargs):
    r"""
Computes the attention scores passed to generalised convolution:

.. math::

   \begin{align*}
   \bar{A}_{bk} & = \bar{x}_{bk}\bar{y}_{bk}^\top\\
   \textrm{where } & \bar{x}_{bk}{:}\langle M,D' \rangle,\; \bar{y}_{bk}{:}\langle N,D' \rangle\\
   \bar{x}_{bk} & = x'_b\Lambda_k^{\textrm{(x)}}\\
   \bar{y}_{bk} & = y'_b\Lambda_k^{\textrm{(y)}}+\mathbf{1}_N\otimes\Lambda_k^{\textrm{(o)}}
   \end{align*}

:param yʹ: tensor :math:`y'{:}\langle B,N,Q' \rangle` (query-input)
:param xʹ: tensor :math:`x'{:}\langle B,M,P' \rangle` (key-input)
:param kargs: passed to :meth:`GeneralisedConvolution.forward` with key ``x`` set to :math:`x'` if absent
:return: see :meth:`GeneralisedConvolution.forward`
    """
#--------------------------------------------------------------------------------------------------
    if xʹ is None: xʹ = yʹ
    kargs.setdefault('x',xʹ)
    r = torch.einsum('bnq,kqd->bknd',yʹ,self.Λ[1]) # r: B,K,N,D'
    if self.Λₒ is not None: r = r + self.Λₒ
    r = torch.einsum('bknd,bmp,kpd->bkmn',r,xʹ,self.Λ[0]) # r: B,K,M,N
    return super().forward(r,**kargs)

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def torch_convert(a:torch.nn.MultiheadAttention):
    r"""
Converts a :class:`torch.nn.MultiheadAttention` instance into an instance of this class with (almost) same behaviour.

:param a: the instance to convert
    """
#--------------------------------------------------------------------------------------------------
    assert isinstance(a,torch.nn.MultiheadAttention)
    assert a.bias_k is None and a.bias_v is None, 'Extra biases not supported (no idea what they do)'
    self = MultiHeadAttention(K=a.num_heads,P=a.vdim,Q=a.embed_dim,Pʹ=a.kdim,Qʹ=a.embed_dim,bias=a.in_proj_bias is not None)
    assert self.D == self.Dʹ == a.head_dim # sanity check
    q_weight_d,k_weight_d,v_weight_d = (a.q_proj_weight.data,a.k_proj_weight.data,a.v_proj_weight.data) if a.in_proj_weight is None else torch.chunk(a.in_proj_weight.data,3)
    L = (self.Λ[1],q_weight_d.T),(self.Λ[0],k_weight_d.T),(self.ϴ[0],v_weight_d.T),(self.ϴ[1],a.out_proj.weight.data)
    for w_,w in L: w_.data[...] = torch.stack(w.chunk(a.num_heads,dim=1))
    if a.in_proj_bias is not None:
      q_bias_d,_,v_bias_d = torch.chunk(a.in_proj_bias.data,3) # ignore useless k_bias_d
      L = (self.Λₒ,q_bias_d),(self.Θₒ[0],v_bias_d)
      for b_,b in L: b_.data[0,:,0,:] = torch.stack(b.chunk(a.num_heads))
      self.Θₒ[1].data[0,0,:] = a.out_proj.bias.data
    return self

#--------------------------------------------------------------------------------------------------
  def torch_compare(self,a:torch.nn.MultiheadAttention,B:int,M:int,N:int):
    r"""
Compares a :class:`torch.nn.MultiheadAttention` instance with this instance on a sample input.

:param a: the instance to compare
:param B: batch size
:param M: input size
:param N: output size
:return: pair of aligned output and dictionary of aligned gradients per parameter
    """
#--------------------------------------------------------------------------------------------------
    def grads(y_,y):
      for p in (*self.parameters(),*a.parameters()): p.grad = None # reset all gradients
      torch.mean(y_).backward(); torch.mean(y).backward() # launch back-propagation on sample
      q_weight_g,k_weight_g,v_weight_g = (a.q_proj_weight.grad,a.k_proj_weight.grad,a.v_proj_weight.grad) if a.in_proj_weight is None else torch.chunk(a.in_proj_weight.grad,3)
      L = ('Λ[1]',self.Λ[1],q_weight_g.T),('Λ[0]',self.Λ[0],k_weight_g.T),('ϴ[0]',self.ϴ[0],v_weight_g.T),('ϴ[1]',self.ϴ[1],a.out_proj.weight.grad)
      yield from ((p,w_.grad,torch.stack(w.chunk(a.num_heads,dim=1))) for p,w_,w in L)
      if a.in_proj_bias is not None:
        q_bias_g,_,v_bias_g = torch.chunk(a.in_proj_bias.grad,3) # ignore useless k_bias_g (always null)
        L = ('Λₒ',self.Λₒ,q_bias_g),('Θₒ[0]',self.Θₒ[0],v_bias_g)
        yield from ((p,b_.grad[0,:,0,:],torch.stack(b.chunk(a.num_heads))) for p,b_,b in L)
        yield 'Θₒ[1]',self.Θₒ[1].grad[0,0],a.out_proj.bias.grad
    yʹ = torch.rand(B,N,a.embed_dim)  # query
    xʹ = torch.rand(B,M,a.kdim)  # key
    x = torch.rand(B,M,a.vdim)  # value
    y,_ = a(yʹ,xʹ,x,need_weights=False)
    y_,_ = self(yʹ,xʹ,x=x)
    return (y_,y),{p:(u_,u) for p,u_,u in grads(y_,y)}

#==================================================================================================
class MultiHeadMixedAttention(GeneralisedConvolution):
  r"""
An instance of this class is a multi-head Mixed attention module.
  """
#==================================================================================================
#--------------------------------------------------------------------------------------------------
  def __init__(self,*args,Rʹ:Optional[int]=None,Pʹ:Optional[int]=None,Qʹ:Optional[int]=None,Dʹ:Optional[int]=None,bias:bool=False,**kargs):
    r"""
Additional model parameters for Mixed attention scores are:

.. math::

   \begin{array}{l}
   \Lambda^{\textrm{(x)}}{:}\langle K,P',D' \rangle \hspace{.5cm} \Lambda^{\textrm{(y)}}{:}\langle K,Q',D' \rangle \hspace{.5cm} \Lambda^{\textrm{(zx)}},\Lambda^{\textrm{(zy)}}{:}\langle K,R',D' \rangle\\
   \Lambda^{\textrm{(ox)}},\Lambda^{\textrm{(oy)}},\Lambda^{\textrm{(o)}}{:}\langle K,D' \rangle
   \end{array}

:param Rʹ: matrix-input dimension :math:`R'`, default :math:`\left\lfloor\sqrt{PQ}\right\rfloor`
:param Pʹ: key-input dimension :math:`P'`, default :math:`P`
:param Qʹ: query-input dimension :math:`Q'`, default :math:`Q`
:param Dʹ: attention head dimension :math:`D'`, default :math:`\left\lfloor\frac{Q'}{K}\right\rfloor`
:param bias: whether to use the biases :math:`\Lambda^{\textrm{(ox),(oy),(o)}}` + the convolution biases
:param args: passed to :class:`GeneralisedConvolution` constructor
:param kargs: passed to :class:`GeneralisedConvolution` constructor
    """
#--------------------------------------------------------------------------------------------------
    super().__init__(*args,bias=bias,**kargs)
    P,Q=self.P,self.Q; Rʹ=self._dim(Rʹ,int((P*Q)**.5)); Pʹ=self._dim(Pʹ,P); Qʹ=self._dim(Qʹ,Q); Dʹ=self._dim(Dʹ,Qʹ//self.K); self.Rʹ,self.Pʹ,self.Qʹ,self.Dʹ=Rʹ,Pʹ,Qʹ,Dʹ
    self.Λ = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(self.K,d,Dʹ)) for d in (Pʹ,Qʹ,Rʹ,Rʹ)) # Λx, Λy, Λzx, Λzy
    self.Λₒ = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(1,self.K,*n*(1,),Dʹ)) for n in (1,1,2)) if bias is True else None # Λox,Λoy,Λo
    self.temperature = torch.sqrt(torch.tensor(Dʹ))
    self._reset_params()

#--------------------------------------------------------------------------------------------------
  def _reset_params(self):
#--------------------------------------------------------------------------------------------------
    super()._reset_params()
    for Λ in self.Λ: torch.nn.init.xavier_uniform_(Λ)
    if self.Λₒ is not None:
      for Λ in self.Λₒ: torch.nn.init.zeros_(Λ)

#--------------------------------------------------------------------------------------------------
  def forward(self,zʹ,yʹ,xʹ=None,**kargs):
    r"""
Computes the attention scores passed to generalised convolution:

.. math::

   \begin{align*}
   \bar{A}_{bk} & = \bar{x}_{bk}\bar{y}_{bk}^\top+E_{\frac{md,mnd}{mn}}(\bar{x}_{bk},\bar{z}^{\textrm{(y)}}_{bk})+E_{\frac{nd,mnd}{mn}}(\bar{y}_{bk},\bar{z}^{\textrm{(x)}}_{bk})\\
   \textrm{where } & \bar{x}_{bk}{:}\langle M,D' \rangle,\; \bar{y}_{bk}{:}\langle N,D' \rangle,\; \bar{z}_{bk}{:}\langle M,N,D' \rangle\\
   \bar{x}_{bk} & = x'_b\Lambda_k^{\textrm{(x)}}+\mathbf{1}_M\otimes\Lambda_k^{\textrm{(ox)}}\\
   \bar{y}_{bk} & = y'_b\Lambda_k^{\textrm{(y)}}+\mathbf{1}_N\otimes\Lambda_k^{\textrm{(oy)}}\\
   \bar{z}^{\textrm{(x)}}_{bk} & = z'_b\Lambda_k^{\textrm{(zx)}}\\
   \bar{z}^{\textrm{(y)}}_{bk} & = z'_b\Lambda_k^{\textrm{(zy)}}+\mathbf{1}_M\otimes\mathbf{1}_N\otimes\Lambda_k^{\textrm{(o)}}
   \end{align*}

:param zʹ: tensor :math:`z'{:}\langle B,M,N,R' \rangle` (matrix-input)
:param yʹ: tensor :math:`y'{:}\langle B,N,Q' \rangle` (query-input)
:param xʹ: tensor :math:`x'{:}\langle B,M,P' \rangle` (key-input)
:param kargs: passed to :meth:`GeneralisedConvolution.forward` with key ``x`` set to :math:`x'` if absent
:return: see :meth:`GeneralisedConvolution.forward`
    """
#--------------------------------------------------------------------------------------------------
    if xʹ is None: xʹ = yʹ
    kargs.setdefault('x',xʹ)
    xt = torch.einsum('bmp,kpd->bkmd',xʹ,self.Λ[0]) # xt: B,K,M,D'
    if self.Λₒ is not None: xt = xt + self.Λₒ[0]
    yt = torch.einsum('bnq,kqd->bknd',yʹ,self.Λ[1]) # yt: B,K,N,D'
    if self.Λₒ is not None: yt = yt + self.Λₒ[1]
    r = torch.einsum('bkmd,bknd->bkmn',xt,yt) # standard attention, r: B,K,M,N
    zy = torch.einsum('bmnr,krd->bkmnd',zʹ,self.Λ[3]) # zy: B,K,M,N,D'
    if self.Λₒ is not None: zy = zy + self.Λₒ[2]
    r = r + torch.einsum('bkmnd,bkmd->bkmn',zy,xt)
    # no need for zy-bias (redundant)
    r = r + torch.einsum('bknd,bmnr,krd->bkmn',yt,zʹ,self.Λ[2])
    return super().forward(r,**kargs)
