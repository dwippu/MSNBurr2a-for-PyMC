import numpy as np
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable
import pymc as pm

from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none

from typing import Optional, Tuple
    
def quantile(p: int, mu, sigma, alpha)->int:
    if sigma.all() <= 0:
        raise ValueError("sigma must more than 0")
    if alpha.all() <= 0:
        raise ValueError("alpha must more than 0")
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    q = mu + sigma/omega*(np.log(alpha)+np.log((1-p)**(-1/alpha)-1))
    return q

def random(mu, sigma, alpha, rng = np.random.default_rng(), size: Optional[Tuple[int]]=None):
    if sigma.all() <= 0:
        raise ValueError("sigma must more than 0")
    if alpha.all() <= 0:
        raise ValueError("alpha must more than 0")
    u = rng.uniform(low=0, high=1, size=size)
    random = quantile(u, mu, sigma, alpha)
    return np.asarray(random)

def moment(rv, size, mu, sigma, alpha):
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    moment= mu + sigma/omega*(pt.digamma(1)-pt.digamma(alpha)+pt.log(alpha))
    if not rv_size_is_none(size):
        moment = pt.full(size, moment)
    return moment
    
def logp(y: TensorVariable, mu: TensorVariable, sigma: TensorVariable, alpha: TensorVariable, **kwargs):
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    epart = omega/sigma*(y-mu) - pt.log(alpha)
    logpdf = pt.switch(
        pt.eq(y, -np.inf),
        -np.inf,  # logpdf should be -inf if y is -inf
        pt.log(omega)-pt.log(sigma)+(omega/sigma*(y-mu))-((alpha+1)*pt.log1pexp(epart))
    )

    return check_parameters(
        logpdf,
        alpha > 0,
        sigma > 0,
        msg=f"alpha must more than 0, sigma must more than 0",
    )

def logcdf(y: TensorVariable, mu: TensorVariable, sigma: TensorVariable, alpha: TensorVariable, **kwargs):
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    logcdf = pt.log(1-(1+pt.exp((omega*(y-mu))/sigma)/alpha)**-alpha)
    return check_parameters(
        logcdf,
        alpha > 0,
        sigma > 0,
        msg=f"alpha must more than 0, sigma must more than 0",
    )


class Msnburr_iia:

    def __new__(self, name:str, mu, sigma, alpha, observed=None, **kwargs):
        if int(pm.__version__[0])==5 and int(pm.__version__[2:4]) >= 11:
            return pm.CustomDist(
                name,
                mu, sigma, alpha,
                logp=logp,
                logcdf=logcdf,
                random=random,
                support_point=moment,
                observed=observed,
                **kwargs
            )
        else:
            return pm.CustomDist(
                name,
                mu, sigma, alpha,
                logp=logp,
                logcdf=logcdf,
                random=random,
                moment=moment,
                observed=observed,
                **kwargs
            )
    
    @classmethod
    def dist(cls, mu, sigma, alpha, **kwargs):
        if int(pm.__version__[0])==5 and int(pm.__version__[2:4]) >= 11:
            return pm.CustomDist.dist(
                mu, sigma, alpha,
                class_name="WeightedNormal",
                logp=logp,
                logcdf=logcdf,
                random=random,
                support_point=moment,
                **kwargs
            )
        else:
            return pm.CustomDist.dist(
                mu, sigma, alpha,
                class_name="WeightedNormal",
                logp=logp,
                logcdf=logcdf,
                random=random,
                moment=moment,
                **kwargs
            )

    quantile = quantile
    logp = logp
    logcdf = logcdf
    random = random