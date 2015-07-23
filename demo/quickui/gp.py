import logging
logger = logging.getLogger(__name__)

from functools import partial
from collections import namedtuple
from numpy import amin, amax, linspace, newaxis, sqrt, dot, concatenate
from scipy.special import erfinv
from sklearn.gaussian_process import GaussianProcess

#==================================================================================================
class GP:
#==================================================================================================

    def __init__(self,data,theta0=None,thetaL_r=None,thetaU_r=None,**ka):
        self.data = data
        gp = GaussianProcess(theta0=theta0,thetaL=thetaL_r*theta0,thetaU=thetaU_r*theta0,**ka)
        gp.fit(data.x[:,newaxis],data.y)
        def q(x,regr=gp.get_params()['regr'],beta=gp.reduced_likelihood_function()[1]['beta']):
            return dot(regr(x),beta)
        self.q = q
        self.predict = gp.predict

#--------------------------------------------------------------------------------------------------
    def display(self,ax,clevel=None,nsample=None,logscale=None,gpmean=None,SQRT2=sqrt(2.),animate=None):
#--------------------------------------------------------------------------------------------------
        xmin = amin(self.data.x)
        xmax = amax(self.data.y)
        d = (xmax-xmin)/50.
        x = linspace(xmin-d,xmax+d,nsample)
        y,err = self.predict(x[:,newaxis],eval_MSE=True)
        sigma = sqrt(err)
        coef = SQRT2*erfinv(clevel)
        yl, yh = y-coef*sigma,y+coef*sigma
        ax.fill_between(x,yl,yh,alpha=.3,facecolor='b',edgecolor='b',label='{:.1%}% confidence bounds'.format(clevel))
        ax.plot(x,y,c='b',label='prediction')
        ax.scatter(self.data.x,self.data.y,color='k',s=1,marker='.',label='data')
        if gpmean:
            ybounds = ax.get_ylim()
            ax.plot(x,self.q(x[:,newaxis]),c='r',label='GP mean')
            ax.set_ylim(ybounds)
        if logscale: ax.set_xscale('symlog')
        ax.set_xlim((xmin,xmax))
        ax.set_xlabel(self.data.xlabel)
        ax.set_ylabel(self.data.ylabel)
        ax.set_title('data: {}'.format(self.data.title))
        ax.legend()

#==================================================================================================
# Utilities
#==================================================================================================

Data = namedtuple('Data',('x','y','xlabel','ylabel','title'))

#--------------------------------------------------------------------------------------------------
class cachedcsvdatagenerator:
#--------------------------------------------------------------------------------------------------
    xlabel = 'xlabel'
    ylabel = 'ylabel'
    title = 'title'
    name = 'the data'
    signature = None

    def transform(self,ur): raise NotImplementedError

    def configurator(self):
        from myutil import quickui
        Q = quickui.cbook
        return (
            'data',dict(),Q.Ctransf(self.__call__),
            ('datafile',dict(tooltip='File containing {} (a pickle cache will be created)'.format(self.name)),Q.Cbase.Filename(caption='Select file',filter='Data (*.csv *.pck)'))
            )

    def __call__(datafile):
        import csv, pickle, os
        fn,ext = os.path.splitext(datafile)
        assert ext=='.csv' or ext=='.pck'
        a = None
        if os.path.exists(fn+'.pck') and os.stat(fn+'.pck').st_mtime>=os.stat(fn+'.csv').st_mtime:
            with open(fn+'.pck','rb') as u:
                if pickle.load(u) == self.signature: a = pickle.load(u)
        if a is None:
            with open(fn+'.csv') as u:
                ur = csv.reader(u)
                tr,trall = self.transform(ur)
                a = trall(map(tr,ur))
            with open(fn+'.pck','wb') as v:
                pickle.dump(self.signature,v)
                pickle.dump(a,v)
        return Data(a[:,0],a[:,1],self.xlabel,self.ylabel,self.title)

#--------------------------------------------------------------------------------------------------
class testdatagenerator:
#--------------------------------------------------------------------------------------------------

    Funcs = ('x*sin(100.*x)','sum(log(abs(x-arange(0.,1.,.05))),axis=1,keepdims=True)')

    def configurator(self):
        from myutil import quickui
        Q = quickui.cbook
        return (
            'data',dict(),Q.Ctransf(self.__call__),
            ('func',dict(tooltip='True function to draw data from'),Q.Cbase.Set(options=self.Funcs)),
            ('nsample',dict(tooltip='Number of training samples to generate'),Q.Cbase.LinScalar(vtype=int,vmin=50,vmax=5000)),
            )

    def __call__(self,func=None,nsample=None):
        from numpy.random import uniform
        from numpy import sin, sum, log, exp, abs, arange, newaxis
        d = {}
        d.update(locals())
        funcbody = self.Funcs[func]
        exec('def f(x):\n\treturn '+funcbody,d)
        x = uniform(0.,1.,(nsample,))
        y = d['f'](x)
        return Data(x,y,'x','y','y={}'.format(funcbody))

datagen = testdatagenerator()

#--------------------------------------------------------------------------------------------------
def run():
#--------------------------------------------------------------------------------------------------
    from animate import launch
    syst = GP(
        datagen(
            func=0,
            nsample=1000,
            ),
        regr='quadratic',
        corr='squared_exponential',
        theta0=1.,
        thetaL_r=.001,
        thetaU_r=100.,
        nugget=.01,
        random_start=3
        )
    launch(syst,animate={},
        logscale=False,
        nsample=100,
        clevel=.95,
        gpmean=False,
        )

#--------------------------------------------------------------------------------------------------
def runui():
#--------------------------------------------------------------------------------------------------
    from animate import launchui
    from myutil import set_qtbinding; set_qtbinding('pyqt4')  
    datasection = testdatagenerator().configurator
    def config(*animate):
        from myutil import quickui
        GPRegrTypes = (
            ('constant',None),
            ('linear',None),
            ('quadratic',None),
            )
        GPKernels = (
            ('absolute_exponential','exp(-theta|d|)'),
            ('squared_exponential','exp(-theta|d|^2)'),
            #'generalized_exponential',
            ('cubic',None),
            ('linear',None),
            )
        def GPconv(regr=None,corr=None,**ka):
            return GP(regr=GPRegrTypes[regr][0],corr=GPKernels[corr][0],**ka)
        Q = quickui.cbook
        return quickui.configuration(
            Q.Ctransf(partial(dict,logscale=False,nsample=100,clevel=.95,gpmean=False)),
            ('syst',dict(anchor=True),Q.Ctransf(partial(GPconv,regr=2,corr=1,theta0=.1,thetaL_r=.01,thetaU_r=100.,nugget=.01,random_start=1)),
             datasection(),
             ('regr',dict(accept=Q.i,tooltip='Regression type'),Q.Cbase.Set(options=GPRegrTypes)),
             ('corr',dict(accept=Q.i,tooltip='GP kernel'),Q.Cbase.Set(options=GPKernels)),
             ('nugget',dict(accept=Q.i,tooltip='Tychonov regularisation'),Q.Cbase.LogScalar(vmin=.0001,vmax=1000)),
             ('theta0',dict(accept=Q.i,tooltip='Larger means faster decrease of prior correlation when distance increases'),Q.Cbase.LogScalar(vmin=.001,vmax=10.)),
             ('thetaL_r',dict(accept=Q.i,tooltip='Ratio to theta0 of lower bound of parameter space exploration'),Q.Cbase.LogScalar(vmin=.001,vmax=1.)),
             ('thetaU_r',dict(accept=Q.i,tooltip='Ratio to theta0 of upper bound of parameter space exploration'),Q.Cbase.LogScalar(vmin=1.,vmax=1000.)),
             ('random_start',dict(accept=Q.i,tooltip='number of MLE repeats from randomly chosen initialisation'),Q.Cbase.LinScalar(vtype=int,vmin=1,vmax=10)),
             ),
            ('clevel',dict(accept=Q.i,tooltip='Confidence level'),Q.Cbase.LinScalar(vtype=float,vmin=.9,vmax=.9999)),
            ('gpmean',dict(accept=Q.i,sel=None,tooltip='Check to show GP mean (linear combination of basis functions)'),Q.Cbase.Boolean(),),
            ('nsample',dict(accept=Q.i,tooltip='Number of test samples'),Q.Cbase.LinScalar(vtype=int,vmin=50,vmax=500)),
            ('logscale',dict(accept=Q.i,sel=None,tooltip='Check for x-axis in logscale'),Q.Cbase.Boolean(),),
            )
    launchui(config,width=800)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    runui()

