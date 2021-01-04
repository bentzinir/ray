from ray.rllib.models.tf.layers import NoisyLayer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
import numpy as np

tf1, tf, tfv = try_import_tf()


class DistributionalQDeltaTFModel(TFModelV2):
    """Extension of standard TFModel to provide distributional Q values.

    It also supports options for noisy nets and parameter space noise.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_q_value_distributions() -> Q(s, a) atoms
        model_out -> get_state_value() -> V(s)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            q_hiddens=(256, ),
            dueling=False,
            num_atoms=1,
            use_noisy=False,
            v_min=-10.0,
            v_max=10.0,
            sigma0=0.5,
            # TODO(sven): Move `add_layer_norm` into ModelCatalog as
            #  generic option, then error if we use ParameterNoise as
            #  Exploration type and do not have any LayerNorm layers in
            #  the net.
            add_layer_norm=False,
            divergence_type='none',
            initial_beta=None,
            beta=None,
            target_div=None,
            shared_base=None,
            ensemble_size=None,
            multi_binary=False
            ):
        """Initialize variables of this model.

        Extra model kwargs:
            q_hiddens (List[int]): List of layer-sizes after(!) the
                Advantages(A)/Value(V)-split. Hence, each of the A- and V-
                branches will have this structure of Dense layers. To define
                the NN before this A/V-split, use - as always -
                config["model"]["fcnet_hiddens"].
            dueling (bool): Whether to build the advantage(A)/value(V) heads
                for DDQN. If True, Q-values are calculated as:
                Q = (A - mean[A]) + V. If False, raw NN output is interpreted
                as Q-values.
            num_atoms (int): If >1, enables distributional DQN.
            use_noisy (bool): Use noisy nets.
            v_min (float): Min value support for distributional DQN.
            v_max (float): Max value support for distributional DQN.
            sigma0 (float): Initial value of noisy layers.
            add_layer_norm (bool): Enable layer norm (for param noise).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the Q head. Those layers for forward()
        should be defined in subclasses of DistributionalQModel.
        """

        super(DistributionalQDeltaTFModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        if shared_base is not None and \
            model_config["custom_model_config"].get("shared_base_model", False):
            self.base_model = shared_base
            print(" !!! reusing base model !!! ")

        # setup the Q head output (i.e., model for get_q_values)
        self.model_out = tf.keras.layers.Input(
            shape=(num_outputs, ), name="model_out")

        def build_action_value(prefix, model_out, nunits):
            if q_hiddens:
                action_out = model_out
                for i in range(len(q_hiddens)):
                    if use_noisy:
                        action_out = NoisyLayer(
                            "{}hidden_{}".format(prefix, i),
                            q_hiddens[i],
                            sigma0)(action_out)
                    elif add_layer_norm:
                        action_out = tf.keras.layers.Dense(
                            units=q_hiddens[i],
                            activation=tf.nn.relu)(action_out)
                        action_out = \
                            tf.keras.layers.LayerNormalization()(
                                action_out)
                    else:
                        action_out = tf.keras.layers.Dense(
                            units=q_hiddens[i],
                            activation=tf.nn.relu,
                            name="hidden_%d" % i)(action_out)
            else:
                # Avoid postprocessing the outputs. This enables custom models
                # to be used for parametric action DQN.
                action_out = model_out

            if use_noisy:
                action_scores = NoisyLayer(
                    "{}output".format(prefix),
                    self.action_space.n * num_atoms,
                    sigma0,
                    activation=None)(action_out)
            elif q_hiddens:
                action_scores = tf.keras.layers.Dense(
                    units=nunits * num_atoms,
                    activation=None)(action_out)
            else:
                action_scores = model_out

            if num_atoms > 1:
                # Distributional Q-learning uses a discrete support z
                # to represent the action value distribution
                z = tf.range(num_atoms, dtype=tf.float32)
                z = v_min + z * (v_max - v_min) / float(num_atoms - 1)

                def _layer(x):
                    support_logits_per_action = tf.reshape(
                        tensor=x, shape=(-1, self.action_space.n, num_atoms))
                    support_prob_per_action = tf.nn.softmax(
                        logits=support_logits_per_action)
                    x = tf.reduce_sum(
                        input_tensor=z * support_prob_per_action, axis=-1)
                    logits = support_logits_per_action
                    dist = support_prob_per_action
                    return [x, z, support_logits_per_action, logits, dist]

                return tf.keras.layers.Lambda(_layer)(action_scores)
            else:
                logits = tf.expand_dims(tf.ones_like(action_scores), -1)
                dist = tf.expand_dims(tf.ones_like(action_scores), -1)
                return [action_scores, logits, dist]

        def build_state_score(prefix, model_out):
            state_out = model_out
            for i in range(len(q_hiddens)):
                if use_noisy:
                    state_out = NoisyLayer(
                        "{}dueling_hidden_{}".format(prefix, i),
                        q_hiddens[i],
                        sigma0)(state_out)
                else:
                    state_out = tf.keras.layers.Dense(
                        units=q_hiddens[i], activation=tf.nn.relu)(state_out)
                    if add_layer_norm:
                        state_out = tf.keras.layers.LayerNormalization()(
                            state_out)
            if use_noisy:
                state_score = NoisyLayer(
                    "{}dueling_output".format(prefix),
                    num_atoms,
                    sigma0,
                    activation=None)(state_out)
            else:
                state_score = tf.keras.layers.Dense(
                    units=num_atoms, activation=None)(state_out)
            return state_score

        q_out = build_action_value(name + "/action_value/", self.model_out, self.action_space.n)
        self.q_value_head = tf.keras.Model(self.model_out, q_out)
        self.register_variables(self.q_value_head.variables)

        assert divergence_type != 'state_action', "Not supported"
        if divergence_type == 'state':
            self.n_dunits = 2 * ensemble_size if multi_binary else 2
            d_out, _, _ = build_action_value(name + "/delta/", tf.stop_gradient(self.model_out), self.n_dunits)
            self.delta = tf.keras.Model(self.model_out, d_out)
            self.register_variables(self.delta.variables)

        if dueling:
            state_out = build_state_score(
                name + "/state_value/", self.model_out)
            self.state_value_head = tf.keras.Model(self.model_out, state_out)
            self.register_variables(self.state_value_head.variables)

        ######################
        # BETA
        self.train_beta = True
        if beta is not None:
            initial_beta = beta
            self.train_beta = False
            print(f":::setting a constant beta value! ({beta}):::")
        self.opponent_size = ensemble_size if multi_binary else 1
        initial_beta = np.log([initial_beta for _ in range(self.opponent_size)])
        self.log_beta = tf.Variable(initial_beta, dtype=tf.float32, name="log_beta",
                                    constraint=lambda x: tf.clip_by_value(x, np.log(1e-20), np.log(100)))
        self.beta = tf.exp(self.log_beta)
        self.register_variables([self.log_beta])
        self.updated_beta = False
        self.target_div = target_div

        ###########################
        # Policy id
        self.policy_id = tf.Variable(-1, dtype=tf.int32, name="policy_id", trainable=False)
        self.register_variables([self.policy_id])
        self.updated_policy_id = False

    def update_beta(self, x, policy):
        if not self.updated_beta:
            print(f"Updating log beta value: {x}")
            session = policy.get_session()
            if session is None:
                self.log_beta.assign([x] * self.opponent_size)
            else:
                session.run(self.log_beta.assign([x] * self.opponent_size))
            self.updated_beta = True

    def update_policy_id(self, x, policy):
        if not self.updated_policy_id:
            print(f"Updating policy id: {x}")
            session = policy.get_session()
            if session is None:
                self.policy_id.assign(x)
            else:
                session.run(self.policy_id.assign(x))
            self.updated_policy_id = True

    def get_q_value_distributions(self, model_out):
        """Returns distributional values for Q(s, a) given a state embedding.

        Override this in your custom model to customize the Q output head.

        Arguments:
            model_out (Tensor): embedding from the model layers

        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """

        return self.q_value_head(model_out)

    def get_state_value(self, model_out):
        """Returns the state value prediction for the given state embedding."""

        return self.state_value_head(model_out)

    def get_delta_values(self, model_out):
        if hasattr(self, "delta"):
            d_out = self.delta(model_out)
            return tf.reshape(d_out, [-1, int(self.n_dunits/2), 2])
        else:
            return None
