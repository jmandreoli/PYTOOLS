# File:                 torch.py
# Creation date:        2020-01-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for pytorch
#

from collections.abc import Sequence, Callable, Generator
from typing import Optional, Any
import re
import logging; logger = logging.getLogger(__name__)
import torch

#==================================================================================================
class GeneralisedConvolution(torch.nn.Module):
  r"""
An instance of this class is a generalised convolution module. Ref:

  Andreoli, Jean-Marc. 2019. ‘`Convolution, Attention and Structure Embedding <https://arxiv.org/abs/1905.01289>`_’. In Proc. of NeurIPS Workshop on Graph Representation Learning, 8. Vancouver, BC, Canada.
  """
#==================================================================================================

  temperature:torch.Tensor
  r"""Temperature of the softmax applied to scores"""
  K:int
  r"""Number of heads"""
  P:int; Q:int; D:int
  projx:'Einsum'; projy:'Einsum'

#--------------------------------------------------------------------------------------------------
  def __init__(self,K:int,P:int,Q:Optional[int]=None,D:Optional[int]=None,bias:bool=True):
    r"""
Model parameters are

.. math::

   \Theta^{\textrm{(x)}}{:}\langle K,P,D \rangle\; \Theta^{\textrm{(y)}}{:}\langle K,Q,D \rangle\; \eta{:}\langle K,D \rangle\; \eta^{\textrm{(o)}}{:}\langle Q \rangle

:param K: number of heads
:param P: input dimension :math:`P`
:param Q: output dimension :math:`Q`, default :math:`P`
:param D: convolution head dimension :math:`D`, default :math:`\lfloor\frac{Q}{K}\rfloor`
:param bias: whether to use the biases :math:`\eta,\eta^{\textrm{(o)}}`
    """
#--------------------------------------------------------------------------------------------------
    super().__init__()
    K=self._dim(K); P=self._dim(P); Q=self._dim(Q,P); D=self._dim(D,Q//K); self.K,self.P,self.Q,self.D=K,P,Q,D
    self.projx = Einsum('kpd,bmp->bkmd',K,P,D,bias=bias) # ϴx,η
    self.projy = Einsum('kqd,bknd->bnq',K,Q,D,bias=bias) # ϴy,ηₒ

#--------------------------------------------------------------------------------------------------
  def forward(self,score:torch.Tensor,x:torch.Tensor,mask:Optional[torch.Tensor]=None,process_attn:Callable[[torch.Tensor],Any]=(lambda a:None))->tuple[torch.Tensor,Any]:
    r"""
Generalised Convolution formula (with biases):

.. math::

   \begin{align*}
   y_b & = \sum_k A_{bk}^\top\bar{x}_{bk}\Theta_k^{\textrm{(y)}\top}+\mathbf{1}_N\otimes\eta^{\textrm{(o)}}\\
   \textrm{where } & A_{bk}{:}\langle M,N \rangle\; \bar{x}_{bk}{:}\langle M,D \rangle\\
   A_{bk} & = \textrm{softmax}_{\textrm{col}}\frac{1}{\textrm{temp}}(\bar{A}_{bk}+\textrm{mask}_b)\\
   \bar{x}_{bk} &= x_b \Theta_k^{\textrm{(x)}}+\mathbf{1}_M\otimes\eta_k
   \end{align*}

:param score: tensor :math:`\bar{A}{:}\langle B,K,M,N \rangle` (attention scores)
:param x: tensor :math:`x{:}\langle B,M,P \rangle` (input)
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
    x_ = self.projx(x) # x_: B,K,M,D
    r = torch.einsum('bkmn,bkmd->bknd',r,x_)  # r: B,K,N,D
    r = self.projy(r) # r: B,N,Q
    return r,aux

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def _dim(d:Optional[int],default:Optional[int]=None)->int:
#--------------------------------------------------------------------------------------------------
    if d is None: assert default is not None; return default
    if not isinstance(d,int) or d<=1: raise TypeError('dim')
    return d

#==================================================================================================
class GeneralisedMultiHeadAttention(GeneralisedConvolution):
  r"""
This class regroups a number of subclasses in which the scores are computed from projections of the inputs.
  """
#==================================================================================================

  Pʹ:int; Qʹ:int; Dʹ:int
  projxʹ:'Einsum'; projyʹ:'Einsum'

#--------------------------------------------------------------------------------------------------
  def __init__(self,*args,Pʹ:Optional[int]=None,Qʹ:Optional[int]=None,Dʹ:Optional[int]=None,bias:bool=True,_bias:Optional[bool]=None,**kargs):
    r"""
Additional model parameters for generalised multi-head attention are:

.. math::

   \Lambda^{\textrm{(x)}}{:}\langle K,P',D' \rangle\; \Lambda^{\textrm{(y)}}{:}\langle K,Q',D' \rangle\; \beta^{\textrm{(x)}},\beta^{\textrm{(y)}}{:}\langle K,D' \rangle

:param Pʹ: key-input dimension :math:`P'`, default :math:`P`
:param Qʹ: query-input dimension :math:`Q'`, default :math:`Q`
:param Dʹ: attention head dimension :math:`D'`, default :math:`\lfloor\frac{Q'}{K}\rfloor`
:param bias: whether to use the bias :math:`\beta^{\textrm{(x),(y)}}` + the convolution biases
:param args: passed to :class:`GeneralisedConvolution` constructor
:param kargs: passed to :class:`GeneralisedConvolution` constructor
    """
#--------------------------------------------------------------------------------------------------
    super().__init__(*args,bias=bias,**kargs)
    Pʹ=self._dim(Pʹ,self.P); Qʹ=self._dim(Qʹ,self.Q); Dʹ=self._dim(Dʹ,Qʹ//self.K); self.Pʹ,self.Qʹ,self.Dʹ=Pʹ,Qʹ,Dʹ
    self.projxʹ = Einsum('kpd,bmp->bkmd',self.K,Pʹ,Dʹ,bias=(bias if _bias is None else _bias)) # Λx,βx
    self.projyʹ = Einsum('kqd,bnq->bknd',self.K,Qʹ,Dʹ,bias=bias) # Λy,βy
    self.temperature = torch.sqrt(torch.tensor(Dʹ))

#--------------------------------------------------------------------------------------------------
  def forward(self,yʹ:torch.Tensor,xʹ:Optional[torch.Tensor]=None,x:Optional[torch.Tensor]=None,ctx:tuple[torch.Tensor,...]=(),**kargs)->tuple[torch.Tensor,Any]:
    r"""
Uses generalised attention scores obtained from projection of the (key and query) inputs.

  .. math::

   \begin{align*}
   & \bar{x}_{bk}{:}\langle M,D' \rangle,\; \bar{y}_{bk}{:}\langle N,D' \rangle\\
   \bar{x}_{bk} & = x'_b\Lambda_k^{\textrm{(x)}}+\mathbf{1}_M\otimes\beta^{\textrm{(x)}}_k\\
   \bar{y}_{bk} & = y'_b\Lambda_k^{\textrm{(y)}}+\mathbf{1}_N\otimes\beta^{\textrm{(y)}}_k
   \end{align*}

:param yʹ: tensor :math:`y'{:}\langle B,N,Q' \rangle` (query-input)
:param xʹ: tensor :math:`x'{:}\langle B,M,P' \rangle` (key-input), default :math:`y'`
:param x: passed to :meth:`GeneralisedConvolution.forward` (value-input), default :math:`x'`
:param ctx: context for scoring
:param kargs: passed to :meth:`GeneralisedConvolution.forward`
:return: see :meth:`GeneralisedConvolution.forward`
    """
#--------------------------------------------------------------------------------------------------
    if xʹ is None: xʹ = yʹ
    if x is None: x = xʹ
    x̄ = self.projxʹ(xʹ) # x_: B,K,M,Dʹ
    ȳ = self.projyʹ(yʹ) # y_: B,K,N,Dʹ
    return super().forward(self.score(x̄,ȳ,*ctx),x,**kargs)

#==================================================================================================
class MultiHeadAttention(GeneralisedMultiHeadAttention):
  r"""
An instance of this class is a Vanilla multi-head attention module. Ref:

  Vaswani et al. 2017. ‘`Attention Is All You Need <http://arxiv.org/abs/1706.03762>`_’. arXiv: 1706.03762
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,*args,**kargs):
    r"""
Same as :class:`GeneralisedMultiHeadAttention` constructor, but disables bias :math:`\beta^{\textrm{(x)}}` which is redundant.

:param args: passed to :class:`GeneralisedMultiHeadAttention` constructor
:param kargs: passed to :class:`GeneralisedMultiHeadAttention` constructor
    """
#--------------------------------------------------------------------------------------------------
    super().__init__(*args,_bias=False,**kargs)

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def score(x̄:torch.Tensor,ȳ:torch.Tensor)->torch.Tensor:
    r"""
Computes the attention scores used in :meth:`forward`. Conforms to :class:`torch.nn.MultiheadAttention`

:param x̄: tensor :math:`\bar{x}{:}\langle B,K,M,D' \rangle` (key-proj)
:param ȳ: tensor :math:`\bar{y}{:}\langle B,K,N,D' \rangle` (query-proj)
:return: tensor :math:`\langle B,K,M,N \rangle`
    """
#--------------------------------------------------------------------------------------------------
    r = torch.einsum('bkmd,bknd->bkmn',x̄,ȳ)
    return r

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def torch_convert(a:torch.nn.MultiheadAttention)->tuple['MultiHeadAttention',Callable[[int,int,int],dict[str,tuple[torch.Tensor,torch.Tensor]]]]:
    r"""
Converts a :class:`torch.nn.MultiheadAttention` instance into an instance of this class with same behaviour (up to numerical instabilities). An additional output is a function :func:`test` which draws a random sample of the module input given batch size and two sequence lengths, and returns a comparison of the two modules on that sample. Example::

   mod = torch.nn.MultiheadAttention(embed_dim=12,num_heads=4,batch_first=True,kdim=17,vdim=19)
   mod_,test = MultiHeadAttention.torch_convert(mod)
   cmp = test(B=64,M=100,N=80) # generates sample(batch:64,in-length:100,out-length:80) and returns aligned comparisons
   assert all(torch.allclose(u,v,atol=1e-5) for u,v in cmp.values()) # increase atol if it fails

:param a: the instance to convert
:return: an equivalent :class:`MultiHeadAttention` instance and its test function
    """
#--------------------------------------------------------------------------------------------------
    def set_data():
      q_weight_d,k_weight_d,v_weight_d = (a.q_proj_weight.data,a.k_proj_weight.data,a.v_proj_weight.data) if a.in_proj_weight is None else torch.chunk(a.in_proj_weight.data,3)
      L = (self.projyʹ,q_weight_d.T),(self.projxʹ,k_weight_d.T),(self.projx,v_weight_d.T),(self.projy,a.out_proj.weight.data)
      for proj,w in L: proj.weight.data[...] = torch.stack(torch.chunk(w,a.num_heads,dim=1))
      if a.in_proj_bias is not None:
        q_bias_d,_,v_bias_d = torch.chunk(a.in_proj_bias.data,3) # ignore useless k_bias_d
        L = (self.projyʹ,q_bias_d),(self.projx,v_bias_d)
        for proj,b in L: proj.bias.data[0,:,0,:] = torch.stack(torch.chunk(b,a.num_heads))
        self.projy.bias.data[0,0,:] = a.out_proj.bias.data
    def get_grad()->Generator[tuple[str,torch.Tensor,torch.Tensor],None,None]:
      q_weight_g,k_weight_g,v_weight_g = (a.q_proj_weight.grad,a.k_proj_weight.grad,a.v_proj_weight.grad) if a.in_proj_weight is None else torch.chunk(a.in_proj_weight.grad,3)
      L = ('Λy',self.projyʹ,q_weight_g.T),('Λx',self.projxʹ,k_weight_g.T),('ϴx',self.projx,v_weight_g.T),('ϴy',self.projy,a.out_proj.weight.grad)
      yield from ((p,proj.weight.grad,torch.stack(torch.chunk(w,a.num_heads,dim=1))) for p,proj,w in L)
      if a.in_proj_bias is not None:
        q_bias_g,_,v_bias_g = torch.chunk(a.in_proj_bias.grad,3) # ignore useless k_bias_g (theoretically always null)
        L = ('βy',self.projyʹ,q_bias_g),('η',self.projx,v_bias_g)
        yield from ((p,proj.bias.grad[0,:,0,:],torch.stack(torch.chunk(b,a.num_heads))) for p,proj,b in L)
        yield 'ηₒ',self.projy.bias.grad[0,0],a.out_proj.bias.grad
    assert isinstance(a,torch.nn.MultiheadAttention), 'Argument must be a torch.nn.MultiheadAttention instance'
    assert a.batch_first is True, 'Option batch_first=False not supported (too lazy although easy)'
    assert a.bias_k is None and a.bias_v is None, 'Extra biases not supported (no idea what they do)'
    self = MultiHeadAttention(K=a.num_heads,P=a.vdim,Q=a.embed_dim,Pʹ=a.kdim,Qʹ=a.embed_dim,bias=a.in_proj_bias is not None)
    assert self.D == self.Dʹ == a.head_dim # sanity check
    set_data()
    def test(B:int,M:int,N:int):
      yʹ = torch.rand(B,N,a.embed_dim)  # query
      xʹ = torch.rand(B,M,a.kdim)  # key
      x = torch.rand(B,M,a.vdim)  # value
      y,_ = a(yʹ,xʹ,x,need_weights=False); y_,_ = self(yʹ,xʹ,x) # forward
      a.zero_grad(); self.zero_grad() # reset all gradients
      torch.mean(y).backward(); torch.mean(y_).backward() # back-propagation
      return {'':(y,y_)}|{p:(u,u_) for p,u_,u in get_grad()}
    return self,test

#==================================================================================================
class MultiHeadMixedAttention(GeneralisedMultiHeadAttention):
  r"""
An instance of this class is a multi-head Mixed attention module, inspired by various papers in the literature.
  """
#==================================================================================================

  Rʹ:int
  projzx:'Einsum'; projzy:'Einsum'

#--------------------------------------------------------------------------------------------------
  def __init__(self,Rʹ:int,*args,**kargs):
    r"""
Additional model parameters for Mixed attention scores are:

.. math::

   \begin{array}{l}
   \Lambda^{\textrm{(zx)}},\Lambda^{\textrm{(zy)}}{:}\langle K,R',D' \rangle
   \end{array}

:param Rʹ: edge-input dimension :math:`R'`
:param args: passed to :class:`GeneralisedMultiHeadAttention` constructor
:param kargs: passed to :class:`GeneralisedMultiHeadAttention` constructor
    """
#--------------------------------------------------------------------------------------------------
    super().__init__(*args,**kargs) # Λx,βx  Λy,βy
    self.Rʹ=Rʹ=self._dim(Rʹ)
    self.projzx = Einsum('krd,bmnr->bkmnd',self.K,Rʹ,self.Dʹ,bias=False) # Λzx
    self.projzy = Einsum('krd,bmnr->bkmnd',self.K,Rʹ,self.Dʹ,bias=False) # Λzy

#--------------------------------------------------------------------------------------------------
  def forward(self,zʹ:torch.Tensor,*args,**kargs):
    r"""
Invokes :meth:`GeneralisedMultiHeadAttention.forward` with edge input as scoring context.

:param zʹ: tensor :math:`z'{:}\langle B,M,N,R' \rangle` (edge-input)
    """
#--------------------------------------------------------------------------------------------------
    return super().forward(*args,ctx=(zʹ,),**kargs)

#--------------------------------------------------------------------------------------------------
  def score(self,x̄:torch.Tensor,ȳ:torch.Tensor,zʹ:torch.Tensor)->torch.Tensor:
    r"""
Computes the attention scores passed to generalised convolution:

.. math::

   \begin{align*}
   \bar{A}_{bk} & = E_{\frac{nmd,mnd}{mn}}(\;\mathbf{1}_N{\otimes}\bar{x}_{bk}{+}E_{\frac{mnr,rd}{nmd}}(z'_b,\Lambda_k^{\textrm{(zx)}})\;,\;\mathbf{1}_M{\otimes}\bar{y}_{bk}{+}E_{\frac{mnr,rd}{mnd}}(z'_b,\Lambda_k^{\textrm{(zy)}})\;)
   \end{align*}

:param x̄: tensor :math:`\bar{x}{:}\langle B,K,M,D' \rangle` (key-proj)
:param ȳ: tensor :math:`\bar{y}{:}\langle B,K,N,D' \rangle` (query-proj)
:param zʹ: tensor :math:`z'{:}\langle B,M,N,R' \rangle` (edge-input)
:return: see :meth:`GeneralisedMultiHeadAttention.score`
    """
#--------------------------------------------------------------------------------------------------
    z̄ = self.projzx(zʹ) # z̄: B,K,M,N,D'
    r = torch.einsum('bkmd,bkmnd->bkmn',x̄,z̄)
    z̄ = self.projzy(zʹ) # z̄: B,K,M,N,D'
    r = r + torch.einsum('bknd,bkmnd->bkmn',ȳ,z̄)
    return r

#==================================================================================================
class MultiHeadMixedAttentionAlt1(GeneralisedMultiHeadAttention):
  r"""
An instance of this class is a multi-head Mixed attention module. Ref:

  Henderson et al. 2023. ‘`Transformers as Graph-to-Graph Models <https://doi.org/10.18653/v1/2023.bigpicture-1.8>`_’. In Proc. of the Big Picture Workshop, 93–107. Singapore: Association for Computational Linguistics.
  """
#==================================================================================================

  Rʹ:int
  projzx:'Einsum'; projzy:'Einsum'

#--------------------------------------------------------------------------------------------------
  def __init__(self,Rʹ:int,*args,**kargs):
    r"""
Additional model parameters for Mixed attention scores are:

.. math::

   \begin{array}{l}
   \Lambda^{\textrm{(zx)}},\Lambda^{\textrm{(zy)}}{:}\langle K,R',D' \rangle
   \end{array}

:param Rʹ: edge-input dimension :math:`R'`
:param args: passed to :class:`GeneralisedMultiHeadAttention` constructor
:param kargs: passed to :class:`GeneralisedMultiHeadAttention` constructor
    """
#--------------------------------------------------------------------------------------------------
    super().__init__(*args,**kargs) # Λx,βx  Λy,βy
    self.Rʹ=Rʹ=self._dim(Rʹ)
    self.projzx = Einsum('krd,bmnr->bkmnd',self.K,Rʹ,self.Dʹ,bias=False) # Λzx
    self.projzy = Einsum('krd,bmnr->bkmnd',self.K,Rʹ,self.Dʹ,bias=False) # Λzy

#--------------------------------------------------------------------------------------------------
  def forward(self,zʹ:torch.Tensor,*args,**kargs)->tuple[torch.Tensor,Any]:
    r"""
Invokes :meth:`GeneralisedMultiHeadAttention.forward` with edge input as scoring context.

:param zʹ: tensor :math:`z'{:}\langle B,M,N,R' \rangle` (edge-input)
    """
#--------------------------------------------------------------------------------------------------
    return super().forward(*args,ctx=(zʹ,),**kargs)

#--------------------------------------------------------------------------------------------------
  def score(self,x̄:torch.Tensor,ȳ:torch.Tensor,zʹ:torch.Tensor)->torch.Tensor:
    r"""
Computes the attention scores passed to generalised convolution:

.. math::

   \begin{align*}
   \bar{A}_{bk} & = \bar{x}_{bk}\bar{y}_{bk}^\top+E_{\frac{md,mnr,rd}{mn}}(\bar{x}_{bk},z'_b,\Lambda_k^{\textrm{(zy)}})+E_{\frac{nd,mnr,rd}{mn}}(\bar{y}_{bk},z'_b,\Lambda_k^{\textrm{(zx)}})
   \end{align*}

:param x̄: tensor :math:`\bar{x}{:}\langle B,K,M,D' \rangle` (key-proj)
:param ȳ: tensor :math:`\bar{y}{:}\langle B,K,N,D' \rangle` (query-proj)
:param zʹ: tensor :math:`z'{:}\langle B,M,N,R' \rangle` (edge-input)
:return: see :meth:`GeneralisedMultiHeadAttention.score`
    """
#--------------------------------------------------------------------------------------------------
    r = torch.einsum('bkmd,bknd->bkmn',x̄,ȳ) # standard attention, r: B,K,M,N
    z̄ = self.projzx(zʹ) # z̄: B,K,M,N,D'
    r = r + torch.einsum('bknd,bkmnd->bkmn',ȳ,z̄)
    z̄ = self.projzy(zʹ) # z̄: B,K,M,N,D'
    r = r + torch.einsum('bkmd,bkmnd->bkmn',x̄,z̄)
    return r

#==================================================================================================
class MultiHeadMixedAttentionAlt2(GeneralisedMultiHeadAttention):
  r"""
An instance of this class is a multi-head Mixed attention module. Ref:

  Kwon et al. 2023. ‘`Matrix Encoding Networks for Neural Combinatorial Optimization <https://arxiv.org/abs/2106.11113>`_’. In Proc. of 37-th Annual Conference on Neural Information Processing Systems (NeurIPS), New Orleans, LA, U.S.A.
  """
#==================================================================================================

  Rʹ:int; Dʺ:int
  projzʹ:'Einsum'; proj1:'Einsum'; proj2:'Einsum'

#--------------------------------------------------------------------------------------------------
  def __init__(self,Rʹ:int,*args,Dʺ=None,bias:bool=True,mlpd:float=3.,**kargs):
    r"""
Additional model parameters for Mixed attention scores are:

.. math::

   \begin{array}{l}
   \Lambda^{\textrm{(z)}}{:}\langle K,R',D'' \rangle\; \alpha,\alpha'{:}\langle K,D'' \rangle
   \end{array}

:param Rʹ: edge-input dimension :math:`R'`
:param Dʺ: internal multi-layer perceptron dimension :math:`D''` (default see below)
:param mlpd: the default value for :math:`D''` is given by :math:`\lfloor R'{\times}\textrm{mlpd}\rfloor`
:param args: passed to :class:`GeneralisedMultiHeadAttention` constructor
:param kargs: passed to :class:`GeneralisedMultiHeadAttention` constructor
    """
#--------------------------------------------------------------------------------------------------
    super().__init__(*args,**kargs) # Λx,βx  Λy,βy
    self.Rʹ=Rʹ=self._dim(Rʹ); assert isinstance(mlpd,float); self.Dʺ=Dʺ=self._dim(Dʺ,int(mlpd*Rʹ))
    self.projzʹ = Einsum('krd,bmnr->bkmnd',self.K,Rʹ,Dʺ,bias=False) # Λz
    self.proj1 = Einsum('kd,bkmn->bkmnd',self.K,Dʺ,bias=False) # α
    self.proj2 = Einsum('kd,bkmnd->bkmn',self.K,Dʺ,bias=False) # α'
    self.temperature = torch.tensor(1.)

#--------------------------------------------------------------------------------------------------
  def forward(self,zʹ:torch.Tensor,*args,**kargs)->tuple[torch.Tensor,Any]:
    r"""
Invokes :meth:`GeneralisedMultiHeadAttention.forward` with edge input as scoring context.

:param zʹ: tensor :math:`z'{:}\langle B,M,N,R' \rangle` (edge-input)
    """
#--------------------------------------------------------------------------------------------------
    return super().forward(*args,ctx=(zʹ,),**kargs)

#--------------------------------------------------------------------------------------------------
  def score(self,x̄:torch.Tensor,ȳ:torch.Tensor,zʹ:torch.Tensor)->torch.Tensor:
    r"""
Computes the attention scores passed to generalised convolution:

.. math::

   \begin{align*}
   \bar{A}_{bk} & = \textrm{relu}(\bar{x}_{bk}\bar{y}_{bk}^\top\otimes\alpha_k+z'_b\Lambda_k^{\textrm{(z)}})\alpha'_k
   \end{align*}

:param x̄: tensor :math:`\bar{x}{:}\langle B,K,M,D' \rangle` (key-proj)
:param ȳ: tensor :math:`\bar{y}{:}\langle B,K,N,D' \rangle` (query-proj)
:param zʹ: tensor :math:`z'{:}\langle B,M,N,R' \rangle` (edge-input)
:return: see :meth:`GeneralisedMultiHeadAttention.score`
    """
#--------------------------------------------------------------------------------------------------
    r = torch.einsum('bkmd,bknd->bkmn',x̄,ȳ) # r: B,K,M,N
    z̄ = self.projzʹ(zʹ) # z̄: B,K,M,N,D''
    r = self.proj1(r) + z̄ # r: B,K,M,N,D''
    torch.relu_(r)
    r = self.proj2(r) # r: B,K,M,N
    return r

#==================================================================================================
class Einsum (torch.nn.Module):
  r"""
An instance of this class is essentially a Linear module allowing more flexibility in indices using the einsum notation.
  """
#==================================================================================================
  sig_pattern = re.compile(r'([a-z]+),([a-z]+)->([a-z]+)')
  r"""Pattern for allowed signatures (the 3 groups specify the weight, input and output components)"""

  weight:torch.Tensor; bias:Optional[torch.Tensor]

#--------------------------------------------------------------------------------------------------
  def __init__(self,sig:str,*dims:int,bias:bool=True):
    r"""
Model parameters are

* a weight tensor :math:`\Lambda` matching the weight component of *sig* and
* a bias tensor :math:`\beta` matching the intersection of the output component and the weight component of *sig*.

For example with :code:`Einsum('pq,bnp->bnq',P,Q)` where :code:`P,Q` are integers, the model parameters are:

.. math::

   \Lambda{:}\langle P,Q\rangle\; \beta{:}\langle Q\rangle

:param sig: an einsum-like signature conforming to :attr:`sig_pattern`
:param dims: shape of the weight tensor
:param bias: whether to use a bias
    """
#--------------------------------------------------------------------------------------------------
    super().__init__()
    self.sig,bias_dims = sig,self._parse(sig,dims,bias)
    self.weight =  torch.nn.Parameter(torch.empty(*dims)) # Λ
    self.bias = None if bias_dims is None else torch.nn.Parameter(torch.empty(*bias_dims)) # β
    self._reset_params()

#--------------------------------------------------------------------------------------------------
  def _reset_params(self):
#--------------------------------------------------------------------------------------------------
    torch.nn.init.xavier_uniform_(self.weight)
    if self.bias is not None: torch.nn.init.zeros_(self.bias)

#--------------------------------------------------------------------------------------------------
  def forward(self,x:torch.Tensor)->torch.Tensor:
    r"""
Computes :math:`y` defined as follows:

.. math::

   y = E_{\textrm{sig}}(\Lambda,x) + \textrm{reshape}(\beta)

where :math:`E_{\textrm{sig}}` denotes the Einsum operator with the signature given by attribute :attr:`sig`, and parameter :math:`\beta` is reshaped to match the output component of the signature (in the example, it becomes :math:`\mathbf{1}_B\otimes\mathbf{1}_N\otimes\beta` where dimensions :math:`B,N` come from the input :math:`x`).

:param x: tensor :math:`x` matching the input component of the signature (in the example :math:`x{:}\langle B,N,P\rangle`)
:return: tensor :math:`y` matching the output component of the signature (in the example :math:`y{:}\langle B,N,Q\rangle`)
    """
#--------------------------------------------------------------------------------------------------
    r = torch.einsum(self.sig,self.weight,x)
    if self.bias is not None: r = r + self.bias
    return r

#--------------------------------------------------------------------------------------------------
  def __repr__(self):
#--------------------------------------------------------------------------------------------------
    return f'Einsum({self.sig!r},{','.join(map(str,self.weight.shape))})'

#--------------------------------------------------------------------------------------------------
  def _parse(self,sig:str,dims:Sequence[int],bias:bool):
#--------------------------------------------------------------------------------------------------
    for culprit,check in (('signature',isinstance(sig,str)),('dims',all(isinstance(d,int) for d in dims)),('bias',isinstance(bias,bool))):
      if check is False: raise TypeError(culprit)
    m = self.sig_pattern.fullmatch(sig)
    assert m is not None, f'Signature must conform to "{self.sig_pattern}"'
    p_,i_,o_ = m.groups()
    assert len(p_)==len(set(p_)) == len(dims) and len(o_)==len(set(o_)) and all(c in p_ or c in i_ for c in o_), 'Wrong signature'
    if bias is True:
      bias_ = [c for c in o_ if c in p_]
      assert len(bias_)>0, 'The given signature does not allow bias; set bias=False'
      dims_ = {c:dims[p_.index(c)] for c in bias_}
      return tuple(dims_.get(c,1) for c in o_)

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def torch_convert(a:torch.nn.Linear)->tuple['Einsum',Callable[[int,int],dict[str,tuple[torch.Tensor,torch.Tensor]]]]:
    r"""
Converts a :class:`torch.nn.Linear` instance into an instance of this class with same behaviour (up to numerical instabilities). An additional output is a function :func:`test` which draws a random sample of the module input given batch size and sequence length, and returns a comparison of the two modules on that sample. Example::

   mod = torch.nn.Linear(42,17); mod_,test = Einsum.torch_convert(mod)
   assert mod_.sig == 'pq,bnp->bnq' and mod_.weight.shape == (42,17)
   cmp = test(64,100) # generates sample(batch:64,length:100) and returns aligned fwd and bwd values
   assert all(torch.allclose(u,v,atol=1e-6) for u,v in cmp.values()) # increase atol if it fails

:param a: the instance to convert
:return: an equivalent :class:`Einsum` instance and its test function
    """
#--------------------------------------------------------------------------------------------------
    def set_data():
      self.weight.data[...] = a.weight.data.T
      if a.bias is not None: self.bias.data[0,0,:] = a.bias.data
    def get_grad()->Generator[tuple[str,torch.Tensor,torch.Tensor],None,None]:
      yield 'weight',self.weight.grad,a.weight.grad.T
      if a.bias is not None: yield 'bias',self.bias.grad[0,0],a.bias.grad
    assert isinstance(a,torch.nn.Linear)
    self = Einsum('pq,bnp->bnq',a.in_features,a.out_features,bias=a.bias is not None)
    set_data()
    def test(B:int,M:int):
      x = torch.rand(B,M,a.in_features)
      y = a(x); y_ = self(x) # forward
      a.zero_grad(); self.zero_grad() # reset all gradients
      torch.mean(y).backward(); torch.mean(y_).backward() # back-propagation
      return {'':(y,y_)}|{p:(u,u_) for p,u_,u in get_grad()}
    return self,test
