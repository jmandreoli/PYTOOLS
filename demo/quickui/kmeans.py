import logging
logger = logging.getLogger(__name__)

from numpy import array, zeros, ones, empty, arange, newaxis, abs, sum, square, sqrt, log, exp, argmin, argmax, amin, amax, nonzero, all, any, nan, isnan, dot, mean, average, std
from numpy import linalg, concatenate
from numpy.random import uniform, multinomial, multivariate_normal, rand
from functools import partial
from collections import defaultdict

#==================================================================================================
class Kmeans:
#==================================================================================================

    def __init__(self,data,K):
        """
:param data: array of points in the space
:param K: number of expected clusters
        """
        N,D = data.shape
        assert isinstance(K,int)
        self.data = data
        self.K = K
        self.centroids = zeros((K,D))
        for i in (0,1):
            dmin,dmax = amin(data[:,i]),amax(data[:,i])
            self.centroids[:,i] = uniform(dmin,dmax,(K,))
        self.clusters = zeros((K,N),bool)

    def update_centroids(self):
        centroids = self.centroids.copy()
        for centroid,cluster in zip(self.centroids,self.clusters):
            centroid[...] = mean(self.data[cluster],axis=0)
        return all(self.centroids==centroids)

    def update_clusters(self):
        dist = array([[linalg.norm(x-c) for c in self.centroids] for x in self.data])
        assignment = argmin(dist,axis=1)
        active = ones((self.K,),bool)
        for k,cluster in enumerate(self.clusters):
            q = assignment==k
            if any(q): cluster[...] = q
            else: active[k] = False
        if any(~active):
            self.K = sum(active)
            self.clusters = self.clusters[active]
            self.centroids = self.centroids[active]

    def run(self):
        t = 0
        while True:
            self.update_clusters()
            t += 1
            if self.update_centroids(): break
        return t

    def runstep(self):
        t = 0
        while True:
            self.update_clusters()
            yield t,0
            t += 1
            if self.update_centroids(): yield t,-1; break
            else: yield t,1

#--------------------------------------------------------------------------------------------------
    def display(self,ax,animate=None):
        """
:param ax: matplotlib axes on which to display
:param animate: animation function
        """
#--------------------------------------------------------------------------------------------------
        ax.scatter(self.data[:,0],self.data[:,1],c='b',marker='.')
        ax.set_aspect('equal')
        clock = ax.text(.01,.99,'*',fontsize='xx-small',ha='left',va='top',transform=ax.transAxes,bbox=dict(edgecolor='k',facecolor='none'))
        a_centroids = ax.scatter((),(),c='r',marker='o')
        a_clusters = tuple(ax.plot((),(),c='r')[0] for cluster in self.clusters)
        def disp_centroids():
            a_centroids.set_offsets(self.centroids)
        def disp_clusters():
            for a in a_clusters: a.set_data((),())
            for a,cluster in zip(a_clusters,self.clusters):
                if sum(cluster)>=2:
                    p = array(cvxhull(self.data[cluster]))
                    p = concatenate((p,p[0:1]),axis=0)
                    a.set_data(p[:,0],p[:,1])
        if animate is None:
            t = self.run()
            clock.set_text(u'{}:end'.format(t))
            disp_centroids()
            disp_clusters()
        else:
            def animate_disp(tphase,Tag=('cluster','centroids','end')):
                t,phase = tphase
                clock.set_text('{0}:{1}'.format(t,Tag[phase]))
                return (disp_clusters,disp_centroids)[phase]()
            return animate(ax.figure,func=animate_disp,init_func=disp_centroids,frames=self.runstep)

#==================================================================================================
# Utilities
#==================================================================================================

def cvxhull(c):
    assert len(c)>1
    def det(p,q,r): # determinant of pq,qr
        return (q[0]-p[0])*r[1] + (r[0]-q[0])*p[1] + (p[0]-r[0])*q[1]
    points = list(map(tuple,c))
    points.sort()
    upper = [points[0], points[1]]
    for p in points[2:]:
        upper.append(p)
        while len(upper) > 2 and det(*upper[-3:])<=0: del upper[-2]
    del upper[-1]
    points.reverse()
    lower = [points[0], points[1]]
    for p in points[2:]:
        lower.append(p)
        while len(lower) > 2 and det(*lower[-3:])<=0: del lower[-2]
    del lower[-1]
    upper.extend(lower)
    return upper

def mixture(N=1000,weights=None,comp=None):
    return concatenate(tuple(c(n) for c,n in zip(comp,multinomial(N,weights))),axis=0)

def bvn(mx=0.,my=0.,vx=1.,vy=1.,cor=0.):
    vxy=cor*sqrt(vx*vy)
    mean = array((mx,my))
    cov = array(((vx,vxy),(vxy,vy)))
    return lambda n, g=multivariate_normal: g(mean,cov,(n,))

#--------------------------------------------------------------------------------------------------
def run():
#--------------------------------------------------------------------------------------------------
    from animate import launch
    syst = Kmeans(
        data=mixture(
            N=1000,
            weights=array((.2,.4,.4)),
            comp=(
                bvn(10.,0.,5.,3.,.2),
                bvn(0.,0.,6.,7.,.5),
                bvn(0.,10.,3.,8.,.7),
                ),
            ),
        K=5,
        )
    animate=dict(
        repeat=False,
        interval=300,
        )
    launch(syst,animate)

#--------------------------------------------------------------------------------------------------
def runui():
#--------------------------------------------------------------------------------------------------
    from myutil import set_qtbinding; set_qtbinding('pyqt4')
    from animate import launchui
    def config(*animate):
        from animate import cfg_anim
        from myutil import quickui
        krange = range(1,4)
        ffmt = '{0:.2f}'.format
        Q = quickui.cbook
        def normalise(a): return a/sum(a)
        def c_component():
            return (
                Q.Ctransf(
                    partial(
                        bvn,
                        mx=10*rand(),
                        my=10*rand(),
                        vx=10*rand(),
                        vy=10*rand(),
                        cor=rand(),
                        )
                    ),
                ('mx',dict(accept=Q.i),
                    Q.Cbase.LinScalar(vmin=0.,vmax=10.,nval=100,vtype=float,fmt=ffmt)
                    ),
                ('my',dict(accept=Q.i),
                    Q.Cbase.LinScalar(vmin=0.,vmax=10.,nval=100,vtype=float,fmt=ffmt)
                    ),
                ('vx',dict(accept=Q.i),
                    Q.Cbase.LinScalar(vmin=0.,vmax=10.,nval=100,vtype=float,fmt=ffmt)
                    ),
                ('vy',dict(accept=Q.i),
                    Q.Cbase.LinScalar(vmin=0.,vmax=10.,nval=100,vtype=float,fmt=ffmt)
                    ),
                ('cor',dict(accept=Q.i),
                    Q.Cbase.LinScalar(vmin=-1.,vmax=1.,nval=100,vtype=float,fmt=ffmt)
                    ),
                )
        def c_weight():
            return (
                Q.Cbase.LinScalar(vmin=0.,vmax=10.,nval=100,vtype=float,fmt=ffmt),
                )
        return quickui.configuration(
            Q.Ctransf(dict,multi=True),
            ('syst',dict(sel=None),
             Q.Ctransf(
                partial(
                  Kmeans,
                  K=5,
                  )
                ),
             ('data',dict(sel=None),
              Q.Ctransf(
                  partial(
                    mixture,
                    N=1000,
                    )
                  ),
              ('N',dict(accept=Q.i,tooltip='Number of points'),
               Q.Cbase.LinScalar(vmin=50,vmax=20000,nval=1000,vtype=int)
               ),
              ('weights',dict(tooltip='Weights of the generating mixture'),
               Q.Ccomp(lambda l: normalise(array(Q.selectv(l))),
                       defaultdict(lambda: 10*uniform())
                       )
               )+tuple(('w'+str(k),dict(accept=Q.i))+c_weight() for k in krange),
              ('comp',dict(tooltip='Components of the generating mixture'),
               Q.Ccomp(Q.selectv,None)
               )+tuple(('c'+str(k),dict(anchor=True))+c_component() for k in krange),
              ),
             ('K',dict(accept=Q.i,tooltip='Number of clusters'),
              Q.Cbase.LinScalar(vmin=2,vmax=20,step=1,vtype=int)
              ),
             ),
            ('animate',dict(sel=False))+
            cfg_anim(
              *animate,
               modifier=dict(
                save=dict(
                  filename='kmeans.mp4',
                  metadata=dict(
                    title='An execution of the kmeans algorithm',
                    ),
                  ),
                )
               ),
            )
    launchui(config)

#--------------------------------------------------------------------------------------------------

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    runui()

