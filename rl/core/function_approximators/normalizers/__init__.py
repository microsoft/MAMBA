from .normalizer import Normalizer, NormalizerMax, NormalizerStd, NormalizerClip

import tensorflow as tf
if tf.__version__[0]=='2':
    from .tf2_normalizer import tfNormalizerMax, tfNormalizerStd, tfNormalizerClip

