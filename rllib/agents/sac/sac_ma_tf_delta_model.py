from gym.spaces import Discrete
import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class SACMATFDeltaModel(TFModelV2):
    """Extension of standard TFModel for SAC.

    Data flow:
        obs -> forward() -> model_out
        model_out, actions -> get_d_values() -> d(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 critic_hidden_activation="relu",
                 critic_hiddens=(256, 256),
                 divergence_type="none",):
        """Initialize variables of this model.

        Extra model kwargs:
            critic_hidden_activation (str): activation for delta network
            critic_hiddens (list): hidden layers sizes for delta network

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of SACModel.
        """
        super(SACMATFDeltaModel, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
            action_outs = self.action_dim
        else:
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            action_outs = 2 * self.action_dim

        self.divergence_type = divergence_type
        self.model_out = tf.keras.layers.Input(
            shape=(self.num_outputs, ), name="model_out")

        self.actions_input = None
        if not self.discrete:
            self.actions_input = tf.keras.layers.Input(
                shape=(self.action_dim, ), name="actions")

        if divergence_type in ["state", "state_action"]:
            # todo: implement continuous action support
            self.d_outs = 2 * action_outs if divergence_type == 'state_action' else 2
            self.d_concat_input = False
            d_net = tf.keras.Sequential(([
                         tf.keras.layers.Concatenate(axis=1),
                     ] if self.d_concat_input else []) +
                    [tf.keras.layers.Dense(
                        units=units,
                        activation=getattr(tf.nn, critic_hidden_activation, None),
                        name="d_hidden_{}".format(i))
                        for i, units in enumerate(critic_hiddens)
                    ] + [
                        tf.keras.layers.Dense(
                            units=self.d_outs, activation=None, name="d_out")
                    ])

            # build the model via call
            if self.d_concat_input:
                self.d_net = tf.keras.Model([self.model_out, self.actions_input],
                                            d_net([self.model_out, self.actions_input]))
            else:
                self.d_net = tf.keras.Model(self.model_out, d_net(self.model_out))

            self.register_variables(self.d_net.variables)

    def get_d_values(self, model_out, actions):
        if self.d_concat_input:
            d_out = self.d_net([model_out, actions])
        else:
            d_out = self.d_net(model_out)
        return tf.reshape(d_out, [-1, 2, int(self.d_outs/2)])
