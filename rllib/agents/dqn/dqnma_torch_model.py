from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import numpy as np

torch, nn = try_import_torch()


class DQNMATorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 to provide dueling-Q functionality.
    """

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            *,
            q_hiddens=(256, ),
            dueling=False,
            dueling_activation="relu",
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
            initial_beta=1.0,
            beta=None,
            target_div=None,
            shared_base=None,
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
            dueling_activation (str): The activation to use for all dueling
                layers (A- and V-branch). One of "relu", "tanh", "linear".
            num_atoms (int): If >1, enables distributional DQN.
            use_noisy (bool): Use noisy layers.
            v_min (float): Min value support for distributional DQN.
            v_max (float): Max value support for distributional DQN.
            sigma0 (float): Initial value of noisy layers.
            add_layer_norm (bool): Enable layer norm (for param noise).
        """
        nn.Module.__init__(self)
        super(DQNMATorchModel, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        if shared_base is not None and \
            model_config["custom_model_config"].get("shared_base_model", False):
            self._base_model = shared_base
            print(" !!! reusing base model !!! ")

        self.dueling = dueling
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.sigma0 = sigma0
        self.q_params = get_params(self._base_model)
        ins = num_outputs

        advantage_module = nn.Sequential()
        value_module = nn.Sequential()
        delta_module = nn.Sequential()

        # Dueling case: Build the shared (advantages and value) fc-network.
        for i, n in enumerate(q_hiddens):
            if use_noisy:
                advantage_module.add_module(
                    "dueling_A_{}".format(i),
                    NoisyLayer(
                        ins, n, sigma0=self.sigma0,
                        activation=dueling_activation))
                value_module.add_module(
                    "dueling_V_{}".format(i),
                    NoisyLayer(
                        ins, n, sigma0=self.sigma0,
                        activation=dueling_activation))
                delta_module.add_module(
                    "dueling_D_{}".format(i),
                    NoisyLayer(
                        ins, n, sigma0=self.sigma0,
                        activation=dueling_activation))
            else:
                advantage_module.add_module(
                    "dueling_A_{}".format(i),
                    SlimFC(ins, n, activation_fn=dueling_activation))
                value_module.add_module(
                    "dueling_V_{}".format(i),
                    SlimFC(ins, n, activation_fn=dueling_activation))
                delta_module.add_module(
                    "dueling_D_{}".format(i),
                    SlimFC(ins, n, activation_fn=dueling_activation))
                # Add LayerNorm after each Dense.
                if add_layer_norm:
                    advantage_module.add_module(
                        "LayerNorm_A_{}".format(i), nn.LayerNorm(n))
                    value_module.add_module(
                        "LayerNorm_V_{}".format(i), nn.LayerNorm(n))
                    delta_module.add_module(
                        "LayerNorm_D_{}".format(i), nn.LayerNorm(n))
            ins = n

        # Actual Advantages layer (nodes=num-actions).
        if use_noisy:
            advantage_module.add_module("A", NoisyLayer(
                ins,
                self.action_space.n * self.num_atoms,
                sigma0,
                activation=None))
        elif q_hiddens:
            advantage_module.add_module(
                "A",
                SlimFC(
                    ins, action_space.n * self.num_atoms,
                    activation_fn=None))

        self.advantage_module = advantage_module
        self.q_params.extend(get_params(self.advantage_module))

        # Value layer (nodes=1).
        if self.dueling:
            value_module.add_module("V", SlimFC(ins, 1, activation_fn=None))
            self.value_module = value_module
            self.q_params.extend(get_params(self.value_module))

        # Delta layer
        if divergence_type in ["state", "state_action"]:
            self.n_dunits = 2 * self.action_space.n if divergence_type == 'state_action' else 2
            delta_module.add_module("D", SlimFC(ins, self.n_dunits * self.num_atoms, activation_fn=None))
            self.delta_module = delta_module
            self.delta_params = get_params(self.delta_module)

        ######################
        # BETA
        self.train_beta = True
        if beta is not None:
            initial_beta = beta
            self.train_beta = False
            print(f":::setting a constant beta value! ({beta}):::")
        self.log_beta = torch.tensor(data=[np.log(initial_beta + 1e-16)], dtype=torch.float32, requires_grad=True)
        self.updated_beta = False
        self.target_div = torch.tensor(data=[target_div], dtype=torch.float32, requires_grad=False)

        ###########################
        # Policy id
        self.policy_id = torch.tensor(data=[-1], dtype=torch.int32, requires_grad=False)
        self.updated_policy_id = False

    def update_beta(self, x, **kwargs):
        if not self.updated_beta:
            print(f"Updating log beta value: {x}")
            self.log_beta[0] = x
            self.updated_beta = True

    def update_policy_id(self, x, **kwargs):
        if not self.updated_policy_id:
            print(f"Updating policy id: {x}")
            self.policy_id[0] = x
            self.updated_policy_id = True

    def get_q_value_distributions(self, model_out):
        """Returns distributional values for Q(s, a) given a state embedding.

        Override this in your custom model to customize the Q output head.

        Args:
            model_out (Tensor): Embedding from the model layers.

        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """
        action_scores = self.advantage_module(model_out)

        if self.num_atoms > 1:
            # Distributional Q-learning uses a discrete support z
            # to represent the action value distribution
            z = torch.range(0.0, self.num_atoms - 1, dtype=torch.float32)
            z = self.v_min + \
                z * (self.v_max - self.v_min) / float(self.num_atoms - 1)

            support_logits_per_action = torch.reshape(
                action_scores, shape=(-1, self.action_space.n, self.num_atoms))
            support_prob_per_action = nn.functional.softmax(
                support_logits_per_action)
            action_scores = torch.sum(z * support_prob_per_action, dim=-1)
            logits = support_logits_per_action
            probs = support_prob_per_action
            return action_scores, z, support_logits_per_action, logits, probs
        else:
            logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
            return action_scores, logits, logits

    def get_state_value(self, model_out):
        """Returns the state value prediction for the given state embedding."""

        return self.value_module(model_out)

    def get_delta_values(self, model_out):
        if hasattr(self, "delta_module"):
            d_out = self.delta_module(model_out)
            return d_out.view(-1, 2, int(self.n_dunits/2))
        else:
            return


def get_params(module):
    return [val for key, val in module.state_dict(keep_vars=True).items()]