from gym.spaces import MultiDiscrete
import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf = try_import_tf()


class SACEnsembleTFModel(TFModelV2):
    """Extension of standard TFModel for SAC.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

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
                 critic_hidden_activation="relu",
                 critic_hiddens=(256, 256),
                 twin_q=False,
                 initial_alpha=1.0,
                 target_entropy=None,
                 ensemble_size=1,
                 shared_body=False,
                 constant_alpha=False):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation (str): activation for actor network
            actor_hiddens (list): hidden layers sizes for actor network
            critic_hidden_activation (str): activation for critic network
            critic_hiddens (list): hidden layers sizes for critic network
            twin_q (bool): build twin Q networks.
            initial_alpha (float): The initial value for the to-be-optimized
                alpha parameter (default: 1.0).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of SACModel.
        """
        super(SACEnsembleTFModel, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        if isinstance(action_space, MultiDiscrete):
            ensembe_action_dims = action_space.nvec
            assert all(x == ensembe_action_dims[0] for x in ensembe_action_dims)
            self.action_dim = ensembe_action_dims[0]
            self.discrete = True
            action_outs = q_outs = self.action_dim
        else:
            self.action_dim = np.product(action_space.shape[1:])
            self.discrete = False
            action_outs = 2 * self.action_dim
            q_outs = 1

        self.model_out = tf.keras.layers.Input(
            shape=(self.num_outputs,), name="model_out")

        self.twin_q = twin_q
        self.ensemble_size = ensemble_size
        self.shared_body = shared_body
        self.constant_alpha = constant_alpha
        self.action_model = [None] * ensemble_size
        self.shift_and_log_scale_diag = [None] * ensemble_size

        if self.shared_body:
            x = None
            for i, hidden in enumerate(actor_hiddens):
                if x is None:
                    x = self.model_out
                x = tf.keras.layers.Dense(
                    units=hidden,
                    activation=getattr(tf.nn, actor_hidden_activation, None),
                    name="action_{}".format(i + 1))(x)

            for eidx in range(ensemble_size):
                a_out = tf.keras.layers.Dense(units=action_outs, activation=None, name="action_out_{}".format(eidx))(x)
                self.action_model[eidx] = tf.keras.Model(self.model_out, a_out)

                self.shift_and_log_scale_diag[eidx] = self.action_model[eidx](self.model_out)

                self.register_variables(self.action_model[eidx].variables)
        else:
            for eidx in range(ensemble_size):
                self.action_model[eidx] = tf.keras.Sequential([
                    tf.keras.layers.Dense(
                        units=hidden,
                        activation=getattr(tf.nn, actor_hidden_activation, None),
                        name="action_{}_{}".format(eidx, i + 1))
                    for i, hidden in enumerate(actor_hiddens)
                ] + [
                    tf.keras.layers.Dense(
                        units=action_outs, activation=None, name="action_out_{}".format(eidx))
                ])
            self.shift_and_log_scale_diag[eidx] = self.action_model[eidx](self.model_out)

            self.register_variables(self.action_model[eidx].variables)

        self.actions_input = None
        if not self.discrete:
            self.actions_input = tf.keras.layers.Input(
                shape=(self.action_dim, ), name="actions")

        def build_q_net(name, observations, actions, eidx):
            # For continuous actions: Feed obs and actions (concatenated)
            # through the NN. For discrete actions, only obs.
            q_net = tf.keras.Sequential(([
                tf.keras.layers.Concatenate(axis=1),
            ] if not self.discrete else []) + [
                tf.keras.layers.Dense(
                    units=units,
                    activation=getattr(tf.nn, critic_hidden_activation, None),
                    name="{}_hidden_{}_{}".format(name, i, eidx))
                for i, units in enumerate(critic_hiddens)
            ] + [
                tf.keras.layers.Dense(
                    units=q_outs, activation=None, name="{}_out".format(name))
            ])

            # TODO(hartikainen): Remove the unnecessary Model calls here
            if self.discrete:
                q_net = tf.keras.Model(observations, q_net(observations))
            else:
                q_net = tf.keras.Model([observations, actions],
                                       q_net([observations, actions]))
            return q_net

        self.q_net = [None] * ensemble_size
        self.twin_q_net = [None] * ensemble_size
        self.log_alpha = [None] * ensemble_size
        self.alpha = [None] * ensemble_size

        for eidx in range(ensemble_size):
            self.q_net[eidx] = build_q_net("q", self.model_out, self.actions_input, eidx)
            self.register_variables(self.q_net[eidx].variables)

            if twin_q:
                self.twin_q_net[eidx] = build_q_net("twin_q", self.model_out,
                                              self.actions_input, eidx)
                self.register_variables(self.twin_q_net[eidx].variables)
            else:
                self.twin_q_net[eidx] = None

        # Auto-calculate the target entropy.
        if target_entropy is None or target_entropy == "auto":
            # See hyperparams in [2] (README.md).
            if self.discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / self.action_dim), dtype=np.float32)
                # TODO: find the correct entropy value for the ensemble
            # See [1] (README.md).
            else:
                # TODO: find the correct entropy value for the ensemble
                target_entropy = -np.prod(action_space.shape[1:])
        self.target_entropy = target_entropy

        if constant_alpha:
            initial_alpha = 2 / self.ensemble_size
            print("=================CONSTANT ALPHA====================")

        print(f"target entropy: {self.target_entropy}, initial alpha: {initial_alpha}")

        log_alpha_vec = [np.log(initial_alpha)] * ensemble_size
        log_alpha_vec = np.expand_dims(log_alpha_vec, axis=1)
        self.log_alpha = tf.Variable(log_alpha_vec, dtype=tf.float32, name="log_alpha")
        self.alpha = tf.exp(self.log_alpha)
        if not constant_alpha:
            self.register_variables([self.log_alpha])

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
        # TODO: consider remove casting after debug
        model_out = tf.cast(model_out, tf.float32)
        if actions is not None:
            actions = tf.unstack(actions, axis=1)
            q_value_list = [qnet([model_out, act]) for qnet, act in zip(self.q_net, actions)]
        else:
            q_value_list = [qnet(model_out) for qnet in self.q_net]
        return tf.stack(q_value_list, axis=1)

    def get_twin_q_values(self, model_out, actions=None):
        """Same as get_q_values but using the twin Q net.

        This implements the twin Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim]. If None (discrete action
                case), return Q-values for all actions.

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        # TODO: consider remove casting after debug
        model_out = tf.cast(model_out, tf.float32)
        if actions is not None:
            actions = tf.unstack(actions, axis=1)
            twin_q_value_list = [twin_qnet([model_out, act]) for twin_qnet, act in zip(self.twin_q_net, actions)]
        else:
            twin_q_value_list = [twin_qnet(model_out) for twin_qnet in self.twin_q_net]
        return tf.stack(twin_q_value_list, axis=1)

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

        policy_output_list = [self.action_model[eidx](model_out) for eidx in range(self.ensemble_size)]
        return tf.stack(policy_output_list, axis=1)

    def policy_variables(self):
        """Return the list of variables for the policy net."""

        vars = []
        for eidx in range(self.ensemble_size):
            vars += self.action_model[eidx].variables
        return vars

    def q_variables(self):
        """Return the list of variables for Q / twin Q nets."""

        vars = []
        # We assume that the list is ordered as [Q vars, twin Q vars]
        # 1. First list Q variables
        for eidx in range(self.ensemble_size):
            vars += self.q_net[eidx].variables
        # 2. Second list twin Q variables
        if self.twin_q:
            for eidx in range(self.ensemble_size):
                vars += self.twin_q_net[eidx].variables
        return vars
