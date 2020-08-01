# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from abc import abstractmethod, ABC
from rl.core.online_learners import OnlineLearner


class Algorithm(ABC):
    """ An abtract interface required by Experimenter. """

    # For update
    @abstractmethod
    def pretrain(self, gen_ro):
        """ Pretrain the policy.

            `gen_ro` takes an Agent and returns rollouts as a Dataset and the
            Agent that collects it.
        """
    @abstractmethod
    def update(self, ro, agent):
        """ Update the policy based on rollouts. """

    # Outcome of an algorithm
    @abstractmethod
    def get_policy(self):
        """ Return a Policy object which is the outcome of the algorithm. """

    # For data collection
    @abstractmethod
    def agent(self, mode):
        """ Return a picklable Agent for data collection.

            `mode` is either 'behavior' or 'target'.
        """

class Agent(ABC):

    @abstractmethod
    def pi(self, ob, t, done):
        """ Policy used in online querying. """

    @abstractmethod
    def logp(self, obs, acs):
        """ Log probability of the behavior policy.

            Need to support batch querying. """

    @abstractmethod
    def callback(self, ro):
        """ A method called at the end of each rollout. """


class PolicyAgent(Agent):
    """ A trivial example based on Policy. """

    def __init__(self, policy):
        self.policy = policy

    def pi(self, ob, t, done):
        return self.policy(ob)

    def logp(self, obs, acs):
        return self.policy.logp(obs, acs)

    def callback(self, ro):
        pass
