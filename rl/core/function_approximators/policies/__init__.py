from .policy import Policy

import tensorflow as tf
if tf.__version__[0]=='2':
    from .tf2_policies import tfPolicy, tfGaussianPolicy, RobustKerasMLPGassian

