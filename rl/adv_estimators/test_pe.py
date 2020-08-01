# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from rl.adv_estimators.performance_estimate import PerformanceEstimate as PE


gamma=1.0
delta=0.9
lambd=1
pe = PE(gamma, lambd, delta)
c = np.arange(10)
V = np.arange(10)*2

#td
adv = pe.adv(c, V, lambd=0, done=True)
assert np.isclose(adv[0],  c[0]+delta*V[1]-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])

#monte carlo
adv = pe.adv(c, V, lambd=1., done=True)
q = np.sum(delta**np.arange(10)*c)
assert np.isclose(adv[0],  q-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])


#test weights
w = np.random.random(c.shape)
w = w[:-1]
adv = pe.adv(c, V, w=w, lambd=0, done=True)
assert np.isclose(adv[0],  c[0]+delta*V[1]-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])



from rl.adv_estimators.performance_estimate import SimplePerformanceEstimate as SPE
spe = SPE(gamma, lambd, delta)
c = np.arange(10)
V = np.arange(10)*2

#td
adv = spe.adv(c, V, lambd=0, done=True)
assert np.isclose(adv[0],  c[0]+delta*V[1]-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])

#monte carlo
adv = spe.adv(c, V, lambd=1., done=True)
q = np.sum(delta**np.arange(10)*c)
assert np.isclose(adv[0],  q-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])


#test weights
w = np.random.random(c.shape)
w = w[:-1]
adv = spe.adv(c, V, w=w, lambd=0, done=True)
assert np.isclose(adv[0],  c[0]+delta*V[1]-V[0])
assert np.isclose(adv[-1], c[-2]+delta*c[-1]-V[-2])

# test consistency
c = np.arange(1000)
V = np.arange(1000)*2

gamma=0.412
delta=0.8
lambd=0.3
pe = PE(gamma, lambd, delta)
spe = SPE(gamma, lambd, delta)
#td
adv1= pe.adv(c, V,  done=True)
adv2 = spe.adv(c, V, done=True)
assert np.all(np.isclose(adv1, adv2))


