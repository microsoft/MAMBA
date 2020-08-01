from .supervised_learner import SupervisedLearner
import tensorflow as tf
if tf.__version__[0]=='2':
    from .tf2_supervised_learners import SuperRobustKerasFuncApp, SuperRobustKerasMLP, SuperKerasMLP
#from .tf_supervised_learners import tfSupervisedLearner, tfMLPSupervisedLearner
