from gym.spaces import Discrete
import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class SACMATFCriticModel(TFModelV2):
    """Extension of standard TFModel for SAC critic model.

    Data flow:
        obs -> forward() -> model_out
        model_out, actions -> get_q_values() -> Q(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 critic_hidden_activation="relu",
                 critic_hiddens=(256, 256),):
        """Initialize variables of this model.

        Extra model kwargs:
            critic_hidden_activation (str): activation for critic network
            critic_hiddens (list): hidden layers sizes for critic network

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of SACModel.
        """
        super(SACMATFCriticModel, self).__init__(obs_space, action_space, num_outputs,
                                                 model_config, name)
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
            q_outs = self.action_dim
        else:
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            q_outs = 1

        self.model_out = tf.keras.layers.Input(
            shape=(self.num_outputs, ), name="model_out")

        self.actions_input = None
        if not self.discrete:
            self.actions_input = tf.keras.layers.Input(
                shape=(self.action_dim, ), name="actions")

        self.q_net = tf.keras.Sequential(([
                     tf.keras.layers.Concatenate(axis=1),
                 ] if not self.discrete else []) + [
                    tf.keras.layers.Dense(
                        units=units,
                        activation=getattr(tf.nn, critic_hidden_activation, None),
                        name="{}_hidden_{}".format(name, i))
                    for i, units in enumerate(critic_hiddens)
                ] + [
                    tf.keras.layers.Dense(
                        units=q_outs, activation=None, name="{}_out".format(name))
                ])

        # Build and register model
        if self.discrete: # via build
            self.q_net.build(self.model_out.shape)
        else:  # via call
            self.q_net = tf.keras.Model([self.model_out, self.actions_input],
                                   self.q_net([self.model_out, self.actions_input]))
        self.register_variables(self.q_net.variables)

    def get_q_values(self, model_out, actions=None):
        """Return the Q estimates for the most recent forward pass.

        This implements Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim]. If None (discrete action
                case), return Q-values for all actions.

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        if actions is not None:
            return self.q_net([model_out, actions])
        else:
            return self.q_net(model_out)
