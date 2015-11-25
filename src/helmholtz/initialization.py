
from __future__ import division

import numpy

from blocks.initialization import NdarrayInitialization, Uniform


class ShapeDependentInitialization(NdarrayInitialization):
    """Initialize 

    Parameters
    ----------
    weights_init : :class:`NdarrayInitialization` instance
        The unscaled initialization scheme to initialize the weights with.
    """
    def __init__(self, weights_init):
        super(ShapeDependentInitialization, self).__init__()
        self.weights_init = weights_init

    def generate(self, rng, shape):
        weights = self.weights_init.generate(rng, shape)
        scale = self.scale_func(*shape)
        return scale*weights

    # TODO: Abstract
    def scale_func(self, *shape):
        pass


class TanhInitialization(ShapeDependentInitialization):
    """Normalized initialization for tanh MLPs. 

    This class initializes parameters by drawing from the uniform 
    distribution   with the interval 

        [- sqrt(6)/sqrt(dim_in+dim_out)  .. sqrt(6)/sqrt(dim_in+dim_out)]
    """
    def __init__(self):
        super(TanhInitialization, self).__init__(Uniform(mean=0., width=2.))

    def scale_func(self, dim_in, dim_out):
        return numpy.sqrt(6)/numpy.sqrt(dim_in+dim_out)


class RWSInitialization(ShapeDependentInitialization):
    def __init__(self, factor=1.):
        super(RWSInitialization, self).__init__(Uniform(mean=0., width=2.))
        self.factor = factor

    def scale_func(self, dim_in, dim_out):
        return self.factor * numpy.sqrt(6)/numpy.sqrt(dim_in+dim_out)/dim_in
