import logging
logger = logging.getLogger(__name__)

from numpy import sin, cos, pi, array, square, sqrt, clip, infty
from functools import partial
from ipyshow.odesimu.system import System

#==================================================================================================
class DoublePendulum (System):
#==================================================================================================

    def __init__(self,L1,L2,M1,M2,G):
        """
:param G: gravitational acceleration, in m/s^2
:param L1: length of pendulum 1 in m
:param L2: length of pendulum 2 in m
:param M1: mass of pendulum 1 in kg
:param M2: mass of pendulum 2 in kg
        """
        self.L1, self.L2, self.M1, self.M2, self.G = L1, L2, M1, M2, G
        def main(t,state,a=1+M1/M2,b=L1/L2,c=G/L2):
            theta1,dtheta1,theta2,dtheta2 = state
            delta = theta1-theta2
            cosdelta, sindelta = cos(delta), sin(delta)
            u = -square(dtheta2)*sindelta-a*c*sin(theta1)
            v = b*square(dtheta1)*sindelta-c*sin(theta2)
            D = a-square(cosdelta)
            ddtheta1 = (u-v*cosdelta)/(D*b)
            ddtheta2 = (a*v-u*cosdelta)/D
            return array((dtheta1, ddtheta1, dtheta2, ddtheta2))
        self.main = main
        def fordisplay(state):
            theta1,w1,theta2,w2 = state
            x1 = L1*sin(theta1)
            y1 = -L1*cos(theta1)
            x2 = x1 + L2*sin(theta2)
            y2 = y1 - L2*cos(theta2)
            live = (x1,y1),(x2,y2)
            return live, live[1]
        self.fordisplay = fordisplay

    def display(self,ax,refsize=100.,**ka):
        """
:param refsize: average size in points^2 of the pendulum (and size of rotation axe)

The actual size of the two pendulums are computed around this average to reflect their mass ratio.
        """
        L = 1.05*(self.L1+self.L2)
        ax.set_xlim(-L,L)
        ax.set_ylim(-L,L)
        ax.set_aspect('equal')
        ax.scatter((0.,),(0.,),c='k',marker='o',s=refsize)
        m1,m2 = self.M1,self.M2
        r = clip(sqrt(m1/m2),1./refsize,refsize)
        sz = (refsize*r,refsize/r)
        diag_l, = ax.plot((),(),'k')
        diag_s = ax.scatter((),(),s=sz,marker='o',c=('b','r'))
        tail_l, = ax.plot((),(),'y')
        ax.set_title(r'trajectory:cahotic')
        def disp(t,live,tail):
            (x1,y1),(x2,y2) = live
            diag_l.set_data((0,x1,x2),(0,y1,y2))
            diag_s.set_offsets(((x1,y1),(x2,y2)))
            tail_l.set_data(tail[:,0],tail[:,1])
        return super(DoublePendulum,self).display(ax,disp,**ka)

    @staticmethod
    def makestate(theta1=None,w1=0.,theta2=None,w2=0.): return array((theta1,w1,theta2,w2))*pi/180.

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def run():
#--------------------------------------------------------------------------------------------------
    from animate import launch
    syst = DoublePendulum(1.,.5,4.,1.,9.81)
    animate=dict(
        repeat=False,
        interval=40,
        )
    launch(syst,animate,
        srate=25.,
        taild=1.,
        ini=syst.makestate(theta1=180.,theta2=1.),
        )

def runui():
    from animate import launchui
    def test(*animate):
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
             Q.Cbase.LinScalar(step=1,vmin=0,vmax=10,vtype=float)
             ),
            ('syst',dict(sel=None),
             Q.Ctransf(
                partial(
                  DoublePendulum,
                  M1=5.,
                  L1=2.,
                  M2=.1,
                  L2=1.,
                  G=9.81,
                  )
                ),
             ('G',dict(accept=Q.i,tooltip='acceleration due to gravity in m/s^2'),
              Q.Cbase.LinScalar(vmin=1.,vmax=20.,nval=1000,vtype=float,fmt=ffmt)
              ),
             ('M1',dict(accept=Q.i,tooltip='mass of pendulum 1 in kg'),
              Q.Cbase.LinScalar(vmin=.05,vmax=20.,nval=1000,vtype=float,fmt=ffmt)
              ),
             ('L1',dict(accept=Q.i,tooltip='length of pendulum 1 in m'),
              Q.Cbase.LinScalar(vmin=.1,vmax=10.,nval=1000,vtype=float,fmt=ffmt)
              ),
             ('M2',dict(accept=Q.i,tooltip='mass of pendulum 2 in kg'),
              Q.Cbase.LinScalar(vmin=.05,vmax=20.,nval=1000,vtype=float,fmt=ffmt)
              ),
             ('L2',dict(accept=Q.i,tooltip='length of pendulum 2 in m'),
              Q.Cbase.LinScalar(vmin=.1,vmax=10.,nval=1000,vtype=float,fmt=ffmt)
              ),
             ),
            ('ini',dict(),
              Q.Ctransf(
                  partial(
                    DoublePendulum.makestate,
                    theta1=180.,
                    theta2=1., 
                    )
                  ),
              ('theta1',dict(accept=Q.i,tooltip='initial angle of pendulum 1 in deg'),
               Q.Cbase.LinScalar(vmin=-180.,vmax=180.,nval=1000,vtype=float,fmt=ffmt)
               ),
              ('theta2',dict(accept=Q.i,tooltip='initial angle of pendulum 2 in deg'),
               Q.Cbase.LinScalar(vmin=-180.,vmax=180.,nval=1000,vtype=float,fmt=ffmt)
               ),
              ),
            ('maxtime',dict(accept=Q.i,tooltip='duration of simulation in s'),
              Q.Cbase.LinScalar(step=1.,vmin=10.,vmax=3600.,vtype=float,fmt=ffmt)
              ),
            ('animate',dict(sel=False))+
            cfg_anim(
              *animate,
               modifier=dict(
                timer=dict(
                  interval=40.,
                  ),
                save=dict(
                  filename='pendulum2.mp4',
                  extra_args = ('-cache','1000000'),
                  metadata=dict(
                    title='A double pendulum simulation',
                    ),
                  ),
                )
               ),
            )
    launchui(test,width=600)

#--------------------------------------------------------------------------------------------------

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    runui()

