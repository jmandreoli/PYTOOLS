import logging
logger = logging.getLogger(__name__)

from numpy import array, zeros, ones, empty, arange, newaxis, abs, sum, square, sqrt, log, exp, argmin, argmax, amin, amax, nonzero, all, any, nan, isnan, dot, mean, average, std
from numpy import concatenate, cumsum, maximum, infty, ceil, where
from numpy.random import multivariate_normal
from functools import partial

#==================================================================================================
class RandomWalk (object):
#==================================================================================================

    def __init__(self,stepg,N):
        assert isinstance(N,int)
        trip = stepg(N)
        trip = concatenate((zeros((1,trip.shape[1])),trip),axis=0)
        self.trip = cumsum(trip,axis=0)
        self.norm = sqrt(sum(square(self.trip),axis=1))

    def display(self,axes,animate=None):
        ax,axnorm = axes
        N = len(self.trip)
        ax.set_aspect('equal')
        ax.set_title('A normal random walk')
        ax.scatter(0.,0.,c='r',zorder=1)
        a_trip = ax.plot((),(),'b')[0]
        a_end = ax.scatter(0.,0.,c='r',marker='^',zorder=1)
        axnorm.set_title('Distance to the origin')
        time = arange(N)
        a_norm = axnorm.plot((),(),c='k')[0]
        clock = axnorm.text(.05,.95,'*',ha='center',transform=ax.transAxes,bbox=dict(edgecolor='k',facecolor='none'),fontsize='x-small')
        btrip = arrround(abs(self.trip))
        bnorm = arrround(self.norm)
        def disp(t):
            a_trip.set_data(self.trip[:t+1,0],self.trip[:t+1,1])
            a_end.set_offsets((self.trip[t,0],self.trip[t,1]))
            a_norm.set_data(time[:t+1],self.norm[:t+1])
            b = btrip[t]
            if t==0 or any(btrip[t-1]<b): ax.set_xlim(-b[0],b[0]); ax.set_ylim(-b[1],b[1])
            if t%20==0: axnorm.set_xlim(0,t+20)
            b = bnorm[t]
            if t==0 or bnorm[t-1]<b: axnorm.set_ylim(0,b)
            clock.set_text(str(t))
        if animate is None: disp(N-1)
        else: return animate(ax.figure,func=disp,init_func=(lambda: ()),frames=N)

#==================================================================================================
# Utilities
#==================================================================================================

def bvn(vx=1.,vy=1.,cor=0.,mx=0.,my=0.):
    vxy=cor*sqrt(vx*vy)
    mean = array((mx,my))
    cov = array(((vx,vxy),(vxy,vy)))
    return lambda n, g=multivariate_normal: g(mean,cov,(n,))

def arrround(a,axis=0,r=5.,s=log(1.5)):
    a = maximum.accumulate(a,axis=axis)
    m = amin(where(a!=0,a,infty),axis=axis)
    a = where(a!=0,a,m)
    return r*exp(ceil(log(a/r)/s)*s)

#--------------------------------------------------------------------------------------------------
def run():
#--------------------------------------------------------------------------------------------------
    from animate import launch
    syst = RandomWalk(
        stepg=bvn(vx=3.1,vy=2.8,cor=.5),
        N=400,
        )
    animate=dict(
        repeat=False,
        interval=50,
        )
    def axes(fig):
        fig.set_size_inches((8,12),forward=True)
        return fig.add_subplot(2,1,1),fig.add_subplot(2,1,2)
    launch(syst,animate,axes=axes)

#--------------------------------------------------------------------------------------------------
def runui():
#--------------------------------------------------------------------------------------------------
    from animate import launchui
    from myutil import set_qtbinding; set_qtbinding('pyqt4')  
    def config(*animate):
        from animate import cfg_anim
        from myutil import quickui
        ffmt = '{0:.3f}'.format
        Q = quickui.cbook
        return quickui.configuration(
            Q.Ctransf(dict,multi=True),
            ('syst',dict(sel=None),
             Q.Ctransf(
                partial(
                  RandomWalk,
                  N=400,
                  )
                ),
             ('stepg',dict(tooltip='Specification of the normal 0-mean random step'),
              Q.Ctransf(
                  partial(
                    bvn,
                    vx=3.1,
                    vy=2.8,
                    cor=.5
                    )
                  ),
              ('vx',dict(accept=Q.i),
               Q.Cbase.LinScalar(vmin=.1,vmax=1.,nval=1000,vtype=float,fmt=ffmt)
               ),
              ('vy',dict(accept=Q.i),
               Q.Cbase.LinScalar(vmin=.1,vmax=1.,nval=1000,vtype=float,fmt=ffmt)
               ),
              ('cor',dict(accept=Q.i),
               Q.Cbase.LinScalar(vmin=0.,vmax=1.,nval=1000,vtype=float,fmt=ffmt)
               )
              ),
             ('N',dict(accept=Q.i),
              Q.Cbase.LinScalar(step=1,vmin=100,vmax=1000,vtype=int)
              ),
             ),
            ('animate',dict(sel=False))+
            cfg_anim(
              *animate,
               modifier=dict(
                save=dict(
                  filename='randwalk.mp4',
                  metadata=dict(
                    title='A random walk',
                    ),
                  ),
                )
               ),
            )
    def axes(view):
        view.figure.set_size_inches((8,12),forward=True)
        view.make_grid(2,1)
        return view[0,0].make_axes(),view[1,0].make_axes()
    launchui(config,axes=axes,width=600)

#--------------------------------------------------------------------------------------------------

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    runui()

