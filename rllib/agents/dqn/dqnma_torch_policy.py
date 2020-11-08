from gym.spaces import Discrete

import ray
from ray.rllib.agents.dqn.dqnma_tf_policy import postprocess_nstep_and_prio, \
    PRIO_WEIGHTS, Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.dqn.dqnma_torch_model import DQNMATorchModel
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import huber_loss, reduce_mean_ignore_inf, \
    softmax_cross_entropy_with_logits
from ray.rllib.agents.dqn.dqnma_tf_policy import AGENT_LABEL, OPPONENT_LABEL

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional


class QLoss:
    def __init__(self,
                 q_t_selected,
                 q_logits_t_selected,
                 q_tp1_best,
                 q_probs_tp1_best,
                 importance_weights,
                 rewards,
                 done_mask,
                 gamma=0.99,
                 n_step=1,
                 num_atoms=1,
                 v_min=-10.0,
                 v_max=10.0):

        if num_atoms > 1:
            # Distributional Q-learning which corresponds to an entropy loss
            z = torch.range(0.0, num_atoms - 1, dtype=torch.float32)
            z = v_min + z * (v_max - v_min) / float(num_atoms - 1)

            # (batch_size, 1) * (1, num_atoms) = (batch_size, num_atoms)
            r_tau = torch.unsqueeze(
                rewards, -1) + gamma**n_step * torch.unsqueeze(
                    1.0 - done_mask, -1) * torch.unsqueeze(z, 0)
            r_tau = torch.clamp(r_tau, v_min, v_max)
            b = (r_tau - v_min) / ((v_max - v_min) / float(num_atoms - 1))
            lb = torch.floor(b)
            ub = torch.ceil(b)

            # Indispensable judgement which is missed in most implementations
            # when b happens to be an integer, lb == ub, so pr_j(s', a*) will
            # be discarded because (ub-b) == (b-lb) == 0.
            floor_equal_ceil = (ub - lb < 0.5).float()

            # (batch_size, num_atoms, num_atoms)
            l_project = F.one_hot(lb.long(), num_atoms)
            # (batch_size, num_atoms, num_atoms)
            u_project = F.one_hot(ub.long(), num_atoms)
            ml_delta = q_probs_tp1_best * (ub - b + floor_equal_ceil)
            mu_delta = q_probs_tp1_best * (b - lb)
            ml_delta = torch.sum(
                l_project * torch.unsqueeze(ml_delta, -1), dim=1)
            mu_delta = torch.sum(
                u_project * torch.unsqueeze(mu_delta, -1), dim=1)
            m = ml_delta + mu_delta

            # Rainbow paper claims that using this cross entropy loss for
            # priority is robust and insensitive to `prioritized_replay_alpha`
            self.td_error = softmax_cross_entropy_with_logits(
                logits=q_logits_t_selected, labels=m)
            self.loss = torch.mean(self.td_error * importance_weights)
            self.stats = {
                # TODO: better Q stats for dist dqn
                "mean_td_error": torch.mean(self.td_error),
            }
        else:
            q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best

            # compute RHS of bellman equation
            q_t_selected_target = rewards + gamma**n_step * q_tp1_best_masked

            # compute the error (potentially clipped)
            self.td_error = q_t_selected - q_t_selected_target.detach()
            self.loss = torch.mean(
                importance_weights.float() * huber_loss(self.td_error))
            self.stats = {
                "mean_q": torch.mean(q_t_selected),
                "min_q": torch.min(q_t_selected),
                "max_q": torch.max(q_t_selected),
                "mean_td_error": torch.mean(self.td_error),
            }


class ComputeTDErrorMixin:
    def __init__(self):
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t})
            input_dict[SampleBatch.ACTIONS] = act_t
            input_dict[SampleBatch.REWARDS] = rew_t
            input_dict[SampleBatch.NEXT_OBS] = obs_tp1
            input_dict[SampleBatch.DONES] = done_mask
            input_dict[PRIO_WEIGHTS] = importance_weights

            # Do forward pass on loss to update td error attribute
            build_q_losses(self, self.model, None, input_dict)

            return self.q_loss.td_error

        self.compute_td_error = compute_td_error


def build_q_model_and_distribution(policy, obs_space, action_space, config):
    import ray.rllib.examples.dqn_multiagent_train as base_script
    if not hasattr(base_script, "BASE_MODEL"):
        base_script.base_model_init()

    if not isinstance(action_space, Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))

    if config["hiddens"]:
        # try to infer the last layer size, otherwise fall back to 256
        num_outputs = ([256] + config["model"]["fcnet_hiddens"])[-1]
        config["model"]["no_final_linear"] = True
    else:
        num_outputs = action_space.n

    # TODO(sven): Move option to add LayerNorm after each Dense
    #  generically into ModelCatalog.
    add_layer_norm = (
        isinstance(getattr(policy, "exploration", None), ParameterNoise)
        or config["exploration_config"]["type"] == "ParameterNoise")

    policy.q_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch",
        model_interface=DQNMATorchModel,
        name=Q_SCOPE,
        q_hiddens=config["hiddens"],
        dueling=config["dueling"],
        num_atoms=config["num_atoms"],
        use_noisy=config["noisy"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
        divergence_type=config["div_type"],
        initial_beta=config["initial_beta"],
        beta=config["beta"],
        target_div=config["target_div"],
        shared_base=base_script.BASE_MODEL["main"]
    )

    policy.target_q_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch",
        model_interface=DQNMATorchModel,
        name=Q_TARGET_SCOPE,
        q_hiddens=config["hiddens"],
        dueling=config["dueling"],
        num_atoms=config["num_atoms"],
        use_noisy=config["noisy"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
        divergence_type=config["div_type"],
        initial_beta=config["initial_beta"],
        beta=config["beta"],
        target_div=config["target_div"],
        shared_base=base_script.BASE_MODEL["target"])

    policy.q_func_vars = policy.q_model.q_params
    policy.target_q_func_vars = policy.target_q_model.q_params

    if base_script.BASE_MODEL["main"] is None:
        base_script.BASE_MODEL = {"main": policy.q_model._base_model,
                                  "target": policy.target_q_model._base_model}

    return policy.q_model, TorchCategorical


def get_distribution_inputs_and_class(policy,
                                      model,
                                      obs_batch,
                                      *,
                                      explore=True,
                                      is_training=False,
                                      **kwargs):
    q_vals, _ = compute_q_values(policy, model, obs_batch, explore, is_training)
    q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals

    policy.q_values = q_vals
    return policy.q_values, TorchCategorical, []  # state-out


def build_q_losses(policy, model, _, train_batch):
    config = policy.config
    # Q-network evaluation.
    (q_t, q_logits_t, q_probs_t), delta_t = compute_q_values(
        policy,
        policy.q_model,
        train_batch[SampleBatch.CUR_OBS],
        explore=False,
        is_training=True)

    # Target Q-network evaluation.
    (q_tp1, q_logits_tp1, q_probs_tp1), delta_tp1 = compute_q_values(
        policy,
        policy.target_q_model,
        train_batch[SampleBatch.NEXT_OBS],
        explore=False,
        is_training=True)

    # placeholders
    l_policy = (train_batch["data_id"] < model.policy_id).float()
    leq_policy = (train_batch["data_id"] <= model.policy_id).float()
    eq_policy = (train_batch["data_id"] == model.policy_id).float()
    disc_label = (OPPONENT_LABEL * l_policy + AGENT_LABEL * eq_policy).long()
    opp_action_dist = train_batch["action_dist_inputs"]
    l_count = 1e-8 + torch.sum(l_policy)
    leq_count = 1e-8 + torch.sum(leq_policy)

    # Q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(
        train_batch[SampleBatch.ACTIONS], policy.action_space.n)
    q_t_selected = torch.sum(
        torch.where(q_t > -float("inf"), q_t, torch.tensor(0.0, device=policy.device)) *
        one_hot_selection, 1)
    q_logits_t_selected = torch.sum(
        q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1)

    # compute estimate of best possible value starting from state at t + 1
    if config["double_q"]:
        (q_tp1_using_online_net, q_logits_tp1_using_online_net, \
            q_dist_tp1_using_online_net), _ = compute_q_values(
                policy,
                policy.q_model,
                train_batch[SampleBatch.NEXT_OBS],
                explore=False,
                is_training=True)
        q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = F.one_hot(
            q_tp1_best_using_online_net, policy.action_space.n)
        q_tp1_best = torch.sum(
            torch.where(q_tp1 > -float("inf"), q_tp1, torch.tensor(0.0, device=policy.device)) *
            q_tp1_best_one_hot_selection, 1)
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1)
    else:
        q_tp1_best_one_hot_selection = F.one_hot(
            torch.argmax(q_tp1, 1), policy.action_space.n)
        q_tp1_best = torch.sum(
            torch.where(q_tp1 > -float("inf"), q_tp1, torch.tensor(0.0, device=policy.device)) *
            q_tp1_best_one_hot_selection, 1)
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1)

    # Apply ensemble diversity regularization
    if policy.config["div_type"] == 'action':
        delta_t = torch.nn.LogSoftmax(dim=1)(opp_action_dist)
        log_delta = torch.sum(one_hot_selection * delta_t, dim=1)
    elif policy.config["div_type"] == 'state':
        log_delta = torch.nn.LogSoftmax(dim=1)(delta_tp1)
        # we choose positive agent-based regularization in oppose to negative opponent-based pne
        log_delta = -log_delta[:, AGENT_LABEL, :]
        # log_delta = log_delta[:, OPPONENT_LABEL, :]
    elif policy.config["div_type"] == 'state_action':
        one_hot_3d = tf.tile(tf.expand_dims(one_hot_selection, axis=1), multiples=[1, 2, 1])
        delta_selected = tf.reduce_sum(one_hot_3d * delta_t, axis=2, keepdims=True)
        log_delta = tf.nn.log_softmax(delta_selected, axis=1)
        log_delta = -tf.slice(log_delta, begin=[0, AGENT_LABEL, 0], size=[-1, 1, -1])
    else:
        raise ValueError

    beta = torch.exp(model.log_beta)
    q_tp1_best -= beta * torch.squeeze(log_delta)

    policy.q_loss = QLoss(
        q_t_selected, q_logits_t_selected, q_tp1_best, q_probs_tp1_best,
        train_batch[PRIO_WEIGHTS], train_batch[SampleBatch.REWARDS],
        train_batch[SampleBatch.DONES].float(), config["gamma"],
        config["n_step"], config["num_atoms"],
        config["v_min"], config["v_max"])

    # Train diversity model
    if policy.config["div_type"] == 'action':
        delta_loss = torch.tensor(data=0.0, dtype=torch.float32, requires_grad=True)  # non-parametric diversity regularization mode
        div_vec = torch.argmax(q_t, 1) != torch.argmax(train_batch["action_dist_inputs"], 1)
        div_vec = div_vec * l_policy
        div_rate = torch.sum(div_vec) / l_count
    else:
        if policy.config["div_type"] == 'state_action':
            d_t = torch.sum(one_hot_3d * delta_t, dim=2)
        elif policy.config["div_type"] == 'state':
            d_t = torch.squeeze(delta_t, dim=2)
        else:
            raise ValueError
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        delta_loss_vec = ce_loss(d_t, disc_label)
        delta_loss = torch.sum(leq_policy * delta_loss_vec) / leq_count
        div_vec = disc_label == torch.argmax(d_t, 1)
        div_vec = div_vec * leq_policy
        div_rate = torch.sum(div_vec) / leq_count

    # Auto adjust divergence coefficient
    beta_backup = (model.target_div - div_rate).detach()
    beta_loss = - torch.mean(model.log_beta * beta_backup)

    # mask beta/delta loss for policy 0:
    if not model.train_beta or model.policy_id == 0:
        beta_loss = torch.tensor(data=0.0, dtype=torch.float32, requires_grad=True)
    if model.policy_id == 0:
        delta_loss = torch.tensor(data=0.0, dtype=torch.float32, requires_grad=True)

    policy.delta_loss = delta_loss
    policy.beta_loss = beta_loss
    policy.beta_value = beta
    policy.target_div = model.target_div
    policy.div_rate = div_rate
    policy.delta_penalty = log_delta

    if not hasattr(policy.model, "delta_module"):
        return tuple([beta_loss] + [policy.q_loss.loss])
    else:
        return tuple([beta_loss] + [delta_loss] + [policy.q_loss.loss])


def adam_optimizer(policy, config):
    policy.q_optim = torch.optim.Adam(policy.q_func_vars, lr=policy.cur_lr, eps=config["adam_epsilon"])

    policy.beta_optim = torch.optim.Adam(
        params=[policy.model.log_beta],
        lr=config["entropy_lr"],
        eps=config["adam_epsilon"])
    if not hasattr(policy.model, "delta_module"):
        return tuple([policy.beta_optim] + [policy.q_optim])
    else:
        policy.delta_optim = torch.optim.Adam(
            params=policy.q_model.delta_params,
            lr=policy.cur_lr,
            eps=config["adam_epsilon"])
        return tuple([policy.beta_optim] + [policy.delta_optim] + [policy.q_optim])


def build_q_stats(policy, batch):
    return dict({
        "cur_lr": policy.cur_lr,
        # nirbz: ensemble learning add ons
        "delta_loss": policy.delta_loss,
        "beta_loss": torch.mean(policy.beta_loss),
        "beta_value": torch.mean(policy.beta_value),
        "target_div": torch.mean(policy.target_div),
        "div_rate": policy.div_rate,
        "delta_penalty": policy.delta_penalty,
        "policy_id": policy.model.policy_id,
    }, **policy.q_loss.stats)


def setup_early_mixins(policy, obs_space, action_space, config):
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def after_init(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)
    # Move target net to device (this is done autoatically for the
    # policy.model, but not for any other models the policy has).
    policy.target_q_model = policy.target_q_model.to(policy.device)
    policy.model.log_beta = policy.model.log_beta.to(policy.device)
    policy.model.target_div = policy.model.target_div.to(policy.device)
    policy.model.policy_id = policy.model.policy_id.to(policy.device)


def compute_q_values(policy, model, obs, explore, is_training=False):
    config = policy.config

    model_out, state = model({
        SampleBatch.CUR_OBS: obs,
        "is_training": is_training,
    }, [], None)

    if config["num_atoms"] > 1:
        (action_scores, z, support_logits_per_action, logits,
         probs_or_logits) = model.get_q_value_distributions(model_out)
    else:
        (action_scores, logits,
         probs_or_logits) = model.get_q_value_distributions(model_out)
    delta = model.get_delta_values(model_out)

    if config["dueling"]:
        state_score = model.get_state_value(model_out)
        if policy.config["num_atoms"] > 1:
            support_logits_per_action_mean = torch.mean(
                support_logits_per_action, dim=1)
            support_logits_per_action_centered = (
                support_logits_per_action - torch.unsqueeze(
                    support_logits_per_action_mean, dim=1))
            support_logits_per_action = torch.unsqueeze(
                state_score, dim=1) + support_logits_per_action_centered
            support_prob_per_action = nn.functional.softmax(
                support_logits_per_action)
            value = torch.sum(z * support_prob_per_action, dim=-1)
            logits = support_logits_per_action
            probs_or_logits = support_prob_per_action
        else:
            advantages_mean = reduce_mean_ignore_inf(action_scores, 1)
            advantages_centered = action_scores - torch.unsqueeze(
                advantages_mean, 1)
            value = state_score + advantages_centered
    else:
        value = action_scores

    return (value, logits, probs_or_logits), delta


def grad_process_and_td_error_fn(policy, optimizer, loss):
    # Clip grads if configured.
    return apply_grad_clipping(policy, optimizer, loss)


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    return {"q_values": policy.q_values}


DQNMATorchPolicy = build_torch_policy(
    name="DQNMATorchPolicy",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.agents.dqn.dqnma.DEFAULT_CONFIG,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.q_loss.td_error},
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    after_init=after_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])