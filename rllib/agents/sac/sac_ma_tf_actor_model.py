from gym.spaces import Discrete
import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class SACMATFActorModel(TFModelV2):
    """Extension of standard TFModel for SAC actor model.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 actor_hidden_activation="relu",
                 actor_hiddens=(256, 256),
                 initial_alpha=1.0,
                 alpha=None,
                 target_entropy=None,
                 entropy_scale=1,
                 initial_beta=1.0,
                 beta=None,
                 target_div=None,):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation (str): activation for actor network
            actor_hiddens (list): hidden layers sizes for actor network
            initial_alpha (float): The initial value for the to-be-optimized
                alpha parameter (default: 1.0).
            initial_beta (float): The initial value for the to-be-optimized
                beta parameter (default: 1.0).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of SACModel.
        """
        super(SACMATFActorModel, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        if isinstance(action_space, Discrete):
            action_outs = action_space.n
            self.discrete = True
        else:
            action_outs = 2 * action_space.n
            self.discrete = False

        self.model_out = tf.keras.layers.Input(
            shape=(self.num_outputs, ), name="model_out")
        self.action_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=hidden,
                activation=getattr(tf.nn, actor_hidden_activation, None),
                name="action_{}".format(i + 1))
            for i, hidden in enumerate(actor_hiddens)
        ] + [
            tf.keras.layers.Dense(
                units=action_outs, activation=None, name="action_out")
        ])
        # build the model and register variables
        self.action_model.build(self.model_out.shape)
        self.register_variables(self.action_model.variables)

        self.policy_id = tf.Variable(-1, dtype=tf.int32, name="policy_id", trainable=False)
        self.register_variables([self.policy_id])
        self.updated_policy_id = False
        ######################
        # ALPHA
        if alpha is not None:
            initial_alpha = alpha
            print(f":::setting a constant alpha value! ({alpha}):::")

        self.log_alpha = tf.Variable(np.log(initial_alpha), dtype=tf.float32, name="log_alpha")
        self.alpha = tf.exp(self.log_alpha)

        # Auto-calculate the target entropy.
        if target_entropy is None or target_entropy == "auto":
            # See hyperparams in [2] (README.md).
            if self.discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / action_space.n), dtype=np.float32)
            # See [1] (README.md).
            else:
                target_entropy = -np.prod(action_space.shape)
        self.target_entropy = entropy_scale * target_entropy
        self.register_variables([self.log_alpha])
        ######################
        # BETA
        if beta is not None:
            initial_beta = beta
            print(f":::setting a constant beta value! ({beta}):::")
        self.log_beta = tf.Variable(np.log(initial_beta), dtype=tf.float32, name="log_beta",
                                    constraint=lambda x: tf.clip_by_value(x, np.log(1e-10), np.log(100)))
        self.beta = tf.exp(self.log_beta)
        self.target_div = tf.Variable(target_div, dtype=tf.float32, name="target_div")
        self.register_variables([self.log_beta, self.target_div])
        self.updated_target_div = False

    def update_target_div(self, x, session):
        if not self.updated_target_div:
            print(f"Updating target divergence value: {x}")
            if session is None:
                self.target_div.assign(x)
            else:
                session.run(self.target_div.assign(x))
            self.updated_target_div = True

    def update_policy_id(self, x, session):
        if not self.updated_policy_id:
            print(f"Updating policy id: {x}")
            if session is None:
                self.policy_id.assign(x)
            else:
                session.run(self.policy_id.assign(x))
            self.updated_policy_id = True

    def get_policy_output(self, model_out):
        """Return the action output for the most recent forward pass.

        This outputs the support for pi(s). For continuous action spaces, this
        is the action directly. For discrete, is is the mean / std dev.

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_out_size]
        """
        return self.action_model(model_out)
