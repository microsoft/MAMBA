from .oracle import Oracle
from .oracles import LikelihoodRatioOracle
from .meta_oracles import MetaOracle, AdversarialOracle, MvAvgOracle

import tensorflow as tf
if tf.__version__[0]=='2':
    from .tf2_oracles import tfOracle, tfLikelihoodRatioOracle
