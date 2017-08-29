import logging
logger = logging.getLogger(__name__)

from functools import partial
from numpy import array, sin, cos, arccos, square, sqrt, infty, pi
from scipy.integrate import quad
from ipyshow.odesimu import System

#==================================================================================================
class Pendulum (System):
#==================================================================================================

  shadowshape = (2,)

  def __init__(self,L,G):
    """
:param G: gravitational acceleration, in m/s^2
:param L: length of pendulum in m
    """
    self.L, self.G = L, G
    def main(t,state,a=-G/L):
      theta,dtheta = state
      ddtheta = a*sin(theta)
      return array((dtheta, ddtheta))
    self.main = main
    def jac(t,state,a=-G/L):
      theta,dtheta = state
      return array(((0,1),(a*cos(theta),0)))
    self.jacobian = jac
    def analytics(state,a=-G/L):
      theta,dtheta = state
      K = square(dtheta)+2*a*cos(theta)
      return K
    self.analytics = analytics
    def fordisplay(state):
      theta,w = state
      live = L*sin(theta), -L*cos(theta)
      return live, live
    self.fordisplay = fordisplay

#--------------------------------------------------------------------------------------------------
  def display(self,ax,ini=None,refsize=50.,**ka):
    """
:param refsize: size in points^2 of the pendulum (and rotation axe)
    """
#--------------------------------------------------------------------------------------------------
    L = 1.05*self.L
    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    ax.set_aspect('equal')
    ax.scatter((0.,),(0.,),c='k',marker='o',s=refsize)
    diag_l, = ax.plot((),(),'k')
    diag_s = ax.scatter((),(),s=refsize,marker='o',c='r')
    tail_l, = ax.plot((),(),'y')
    a = -self.G/self.L
    K = self.analytics(ini)
    c = .5*K/a
    if c<-1: q = 1; alpha = pi
    elif c==-1: q = nan; alpha = pi
    else: q = 2; alpha = arccos(c)
    if alpha<.1: T = pi/sqrt(2)
    else: T, err = quad((lambda t,c=c: 1/sqrt(cos(t)-c)),0,alpha)
    T *= q*sqrt(-2/a)
    ax.set_title(r'trajectory:CircularSegment($R$={:.2f},$\alpha$={:.2f}) period:{:.2f}'.format(self.L,alpha*180/pi,T))
    def disp(t,live,tail):
      x,y = live
      diag_l.set_data((0,x),(0,y))
      diag_s.set_offsets(((x,y),))
      tail_l.set_data(tail[:,0],tail[:,1])
    return super().display(ax,disp,ini=ini,**ka)

  @staticmethod
  def makestate(theta=0.,w=0.): return array((theta,w),float)*pi/180.

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def run():
#--------------------------------------------------------------------------------------------------
  from animate import launch
  syst = Pendulum(1,9.81)
  animate=dict(
    repeat=False,
    interval=40,
  )
  launch(
    syst,animate,
    srate=25.,
    taild=1.,
    ini=syst.makestate(179.),
  )

#--------------------------------------------------------------------------------------------------
def runui():
#--------------------------------------------------------------------------------------------------
  from animate import launchui
  def config(*animate):
    from animate import cfg_anim
    from myutil import quickui
    ffmt = '{:.1f}'.format
    Q = quickui.cbook
    return quickui.configuration(
      Q.Ctransf(
        partial(
          dict,
          srate=25.,
          taild=1.,
          maxtime=infty,
        ),
        multi=True,
      ),
      ('srate',dict(sel=None,accept=Q.i,tooltip='sample rate in Hz'),
       Q.Cbase.LinScalar(vmin=10,vmax=50,nval=1000,vtype=float,fmt=ffmt)
      ),
      ('taild',dict(sel=None,accept=Q.i,tooltip='shadow length in s'),
       Q.Cbase.LinScalar(step=1,vmin=0,vmax=10,vtype=float,fmt=ffmt)
      ),
      ('syst',dict(sel=None),
       Q.Ctransf(
         partial(
           Pendulum,
           L=1.,
           G=9.81,
         )
       ),
       ('G',dict(accept=Q.i,tooltip='gravity acceleration in m/s^2'),
        Q.Cbase.LinScalar(vmin=1.,vmax=20.,nval=1000,vtype=float,fmt=ffmt)
       ),
       ('L',dict(accept=Q.i,tooltip='length of pendulum in m'),
        Q.Cbase.LinScalar(vmin=.1,vmax=10.,nval=1000,vtype=float,fmt=ffmt)
       ),
      ),
      ('ini',dict(),
       Q.Ctransf(
         partial(
           Pendulum.makestate,
           theta=179.,
         )
       ),
       ('theta',dict(accept=Q.i,tooltip='initial angle of pendulum in deg'),
        Q.Cbase.LinScalar(vmin=-180.,vmax=180.,nval=1000,vtype=float,fmt=ffmt)
       ),
      ),
      ('maxtime',dict(accept=Q.i,tooltip='duration of simulation in s'),
       Q.Cbase.LinScalar(step=1.,vmin=10.,vmax=3600.,vtype=float,fmt=ffmt)
      ),
      ('animate',dict(sel=None))+
      cfg_anim(
        *animate,
        modifier=dict(
          timer=dict(interval=40.,),
          save=dict(
            filename='pendulum1.mp4',
            metadata=dict(
              title='A simple pendulum simulation',
            ),
          ),
        ),
      ),
    )
  launchui(config,width=900)

#--------------------------------------------------------------------------------------------------

if __name__=='__main__':
  logging.basicConfig(level=logging.INFO)
  runui()
