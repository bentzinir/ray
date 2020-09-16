from gym.spaces import Box, Discrete
import logging
import numpy as np
import random
import ray
import ray.experimental.tf_utils
from ray.rllib.agents.ddpg.ddpg_tf_policy import ComputeTDErrorMixin
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.agents.sac.sac_ma_tf_actor_model import SACMATFActorModel
from ray.rllib.agents.sac.sac_ma_tf_critic_model import SACMATFCriticModel
from ray.rllib.agents.sac.sac_ma_tf_delta_model import SACMATFDeltaModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Beta, Categorical, \
    DiagGaussian, SquashedGaussian
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_tfp
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils.annotations import override
from ray.rllib.policy.tf_policy import TFPolicy


tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)

AGENT_LABEL = 0
OPPONENT_LABEL = 1


def build_sac_model(policy, obs_space, action_space, config):
    # 2 cases:
    # 1) with separate state-preprocessor (before obs+action concat).
    # 2) no separate state-preprocessor: concat obs+actions right away.
    if config["use_state_preprocessor"]:
        num_outputs = 256  # Flatten last Conv2D to this many nodes.
    else:
        num_outputs = 0
        # No state preprocessor: fcnet_hiddens should be empty.
        if config["model"]["fcnet_hiddens"]:
            logger.warning(
                "When not using a state-preprocessor with SAC, `fcnet_hiddens`"
                " will be set to an empty list! Any hidden layer sizes are "
                "defined via `policy_model.fcnet_hiddens` and "
                "`Q_model.fcnet_hiddens`.")
            config["model"]["fcnet_hiddens"] = []

    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's "Q_model" and "policy_model"
    # settings.

    policy.actor = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACMATFActorModel,
        name="actor_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        alpha=config["alpha"],
        entropy_scale=config["entropy_scale"],
        initial_beta=config["initial_beta"],
        beta=config["beta"],
        target_div=config["target_div"],)

    policy.critic = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACMATFCriticModel,
        name="critic_model",
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],)

    policy.target_critic = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACMATFCriticModel,
        name="target_critic_model",
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"], )

    if config["twin_q"]:
        policy.twin_critic = ModelCatalog.get_model_v2(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=config["model"],
            framework=config["framework"],
            model_interface=SACMATFCriticModel,
            name="twin_critic_model",
            critic_hidden_activation=config["Q_model"]["fcnet_activation"],
            critic_hiddens=config["Q_model"]["fcnet_hiddens"], )

        policy.target_twin_critic = ModelCatalog.get_model_v2(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=config["model"],
            framework=config["framework"],
            model_interface=SACMATFCriticModel,
            name="target_twin_critic_model",
            critic_hidden_activation=config["Q_model"]["fcnet_activation"],
            critic_hiddens=config["Q_model"]["fcnet_hiddens"], )

    if config["divergence_type"] in ["state", "state_action"]:
        policy.delta = ModelCatalog.get_model_v2(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=config["model"],
            framework=config["framework"],
            model_interface=SACMATFDeltaModel,
            name="delta_model",
            critic_hidden_activation=config["Q_model"]["fcnet_activation"],
            critic_hiddens=config["Q_model"]["fcnet_hiddens"],
            divergence_type=config["divergence_type"],)

    return policy.actor


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    if other_agent_batches is not None:
        opponent_list = list(other_agent_batches.keys())
        # overload only one opponent member at a time
        if len(opponent_list) > 0:
            opponent_list = [random.choice(opponent_list)]
            for opponent_id in opponent_list:
                opponent_batch = other_agent_batches[opponent_id][1]
                for key in opponent_batch.keys():
                    sample_batch[key] = np.append(sample_batch[key], opponent_batch[key], axis=0)

    sample_batch["disc_label"] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.int32)
    sample_batch["l_agent"] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    sample_batch["leq_agent"] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    sample_batch["eq_agent"] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    sample_batch["opponent_logp"] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    # todo: change logic below to support continuous actions
    sample_batch["opponent_action_dist"] = np.ones((len(sample_batch[SampleBatch.REWARDS]),
                                                   policy.model.action_space.n), dtype=np.float32)

    if 'agent_id' in sample_batch:
        agent_id = sample_batch["agent_id"][0]
        for i, member_id in enumerate(sample_batch['agent_id']):
            if member_id < agent_id:
                sample_batch["l_agent"][i] = 1.
                sample_batch["leq_agent"][i] = 1.
                sample_batch["disc_label"][i] = OPPONENT_LABEL
                sample_batch["opponent_logp"][i] = sample_batch["action_logp"][i]
                sample_batch["opponent_action_dist"][i] = sample_batch["action_dist_inputs"][i]
            elif member_id == agent_id:
                sample_batch["leq_agent"][i] = 1.
                sample_batch["eq_agent"][i] = 1.
                sample_batch["disc_label"][i] = AGENT_LABEL
            if 'infos' in sample_batch:
                sample_batch[SampleBatch.DONES][i] = sample_batch[SampleBatch.DONES][i] or \
                                                     sample_batch['infos'][i]['internal_done']
            else:
                raise AssertionError
        if policy.config["shuffle_data"]:
            j = random.choice(range(len(sample_batch['t'])))
            sample_batch = sample_batch.slice(j, j+1)
    return postprocess_nstep_and_prio(policy, sample_batch)


def get_dist_class(config, action_space):
    if isinstance(action_space, Discrete):
        return Categorical
    else:
        if config["normalize_actions"] or "env_config" in config and config["env_config"]["normalize_actions"]:
            return SquashedGaussian if \
                not config["_use_beta_distribution"] else Beta
        else:
            return DiagGaussian


def get_distribution_inputs_and_class(policy,
                                      model,
                                      obs_batch,
                                      *,
                                      explore=True,
                                      **kwargs):
    # Get base-model output.
    model_out, state_out = model({
        "obs": obs_batch,
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)
    # Get action model output from base-model output.
    distribution_inputs = model.get_policy_output(model_out)
    action_dist_class = get_dist_class(policy.config, policy.action_space)
    return distribution_inputs, action_dist_class, state_out


def sac_actor_critic_loss(policy, model, _, train_batch):
    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    cur_obs = train_batch[SampleBatch.CUR_OBS]
    next_obs = train_batch[SampleBatch.NEXT_OBS]

    if policy.config["env_config"]["normalize_obs"]:
        cur_obs = tf.cast(cur_obs, tf.float32) / 255.
        next_obs = tf.cast(next_obs, tf.float32) / 255.
    else:
        print("====== No Obs Normalization ======")

    # actor models
    actor_t, _ = policy.actor({
        "obs": cur_obs,
        "is_training": policy._get_is_training_placeholder()}, [], None)

    actor_tp1, _ = policy.actor({
        "obs": next_obs,
        "is_training": policy._get_is_training_placeholder()}, [], None)

    # critic models
    critic_t, _ = policy.critic({
        "obs": cur_obs,
        "is_training": policy._get_is_training_placeholder()}, [], None)

    target_critic_tp1, _ = policy.target_critic({
        "obs": next_obs,
        "is_training": policy._get_is_training_placeholder()}, [], None)

    # twin critic models
    if policy.config["twin_q"]:
        twin_critic_t, _ = policy.twin_critic({
            "obs": cur_obs,
            "is_training": policy._get_is_training_placeholder()}, [], None)

        target_twin_critic_tp1, _ = policy.target_twin_critic({
            "obs": next_obs,
            "is_training": policy._get_is_training_placeholder()}, [], None)

    # delta models
    if policy.config["divergence_type"] in ['state', 'state_action']:
        delta_t, _ = policy.delta({
            "obs": cur_obs,
            "is_training": policy._get_is_training_placeholder()}, [], None)

        delta_tp1, _ = policy.delta({
            "obs": next_obs,
            "is_training": policy._get_is_training_placeholder()}, [], None)

    # Discrete case.
    if policy.actor.discrete:
        # Get all action probs directly from pi and form their logp.
        log_pis_t = tf.nn.log_softmax(policy.actor.get_policy_output(actor_t), -1)
        policy_t = tf.math.exp(log_pis_t)
        log_pis_tp1 = tf.nn.log_softmax(
            policy.actor.get_policy_output(actor_tp1), -1)
        policy_tp1 = tf.math.exp(log_pis_tp1)
        # Q-values.
        q_t = policy.critic.get_q_values(critic_t)
        # Target Q-values.
        q_tp1 = policy.target_critic.get_q_values(target_critic_tp1)
        if policy.config["twin_q"]:
            twin_q_t = policy.twin_critic.get_q_values(twin_critic_t)
            twin_q_tp1 = policy.target_twin_critic.get_q_values(target_twin_critic_tp1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_tp1 -= model.alpha * log_pis_tp1

        # Apply ensemble diversity regularization
        if policy.config["divergence_type"] == 'action':
            # todo: clarify possible bug. penalty based on s_t instead of s_tp1
            logd_tp1 = tf.expand_dims(train_batch["opponent_logp"], axis=1)
            logd_tp1 = tf.tile(logd_tp1, multiples=[1, q_t.shape.as_list()[-1]])
        elif policy.config["divergence_type"] in ['state', 'state_action']:
            d_tp1 = policy.delta.get_d_values(delta_tp1, actions=None)
            logd_tp1 = tf.nn.log_softmax(d_tp1, axis=1)
            # we choose positive agent-based regularization in oppose to negative opponent-based pne
            logd_tp1 = -tf.slice(logd_tp1, begin=[0, AGENT_LABEL, 0], size=[-1, 1, -1])
            # logd_tp1 = tf.slice(logd_tp1, begin=[0, OPPONENT_LABEL, 0], size=[-1, 1, -1])
            if policy.config["divergence_type"] == 'state':
                logd_tp1 = tf.tile(logd_tp1, multiples=[1, 1, q_t.shape.as_list()[-1]])
            logd_tp1 = tf.squeeze(logd_tp1, axis=1)
        else:
            raise ValueError
        q_tp1 -= policy.actor.beta * logd_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = tf.one_hot(
            train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1])
        q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
        q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * q_tp1_best

    # Continuous actions case.
    else:
        assert False, 'not re-implemented yet'
        # Sample single actions from distribution.
        action_dist_class = get_dist_class(policy.config, policy.action_space)
        action_dist_t = action_dist_class(
            model.get_policy_output(model_out_t), policy.model)
        policy_t = action_dist_t.sample() if not deterministic else \
            action_dist_t.deterministic_sample()
        log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else \
            action_dist_tp1.deterministic_sample()
        log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS])

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(
                model_out_t, policy_t)
            q_t_det_policy = tf.reduce_min(
                (q_t_det_policy, twin_q_t_det_policy), axis=0)

        # target q network evaluation
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
                                                 policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1)
            # Take min over both twin-NNs.
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)

        q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        q_tp1 -= model.alpha * log_pis_tp1

        q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES],
                                           tf.float32)) * q_tp1_best

    assert policy.config["n_step"] == 1, "TODO(hartikainen) n_step > 1"

    # compute RHS of bellman equation
    q_t_selected_target = tf.stop_gradient(
        train_batch[SampleBatch.REWARDS] +
        policy.config["gamma"] ** policy.config["n_step"] * q_tp1_best_masked)

    # Compute the TD-error (potentially clipped).
    base_td_error = tf.math.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = tf.math.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [
        0.5 * tf.keras.losses.MSE(
            y_true=q_t_selected_target, y_pred=q_t_selected)
    ]
    if policy.config["twin_q"]:
        critic_loss.append(
            0.5 * tf.keras.losses.MSE(
                y_true=q_t_selected_target,
                y_pred=twin_q_t_selected))

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    l_count = 1e-8 + tf.reduce_sum(train_batch["l_agent"])
    leq_count = 1e-8 + tf.reduce_sum(train_batch["leq_agent"])
    eq_count = 1e-8 + tf.reduce_sum(train_batch["eq_agent"])
    if model.discrete:
        min_q_t = tf.stop_gradient(tf.reduce_min((q_t, twin_q_t), axis=0))
        actor_loss = tf.reduce_mean(
            tf.reduce_sum(
                # todo: experiment with the newly derived actor loss
                # tf.multiply(policy_t, tf.stop_gradient(model.alpha) * log_pis_t - tf.nn.log_softmax(min_q_t, axis=1)),
                tf.multiply(policy_t, tf.stop_gradient(model.alpha) * log_pis_t - min_q_t),
                axis=-1))
        entropy_vec = -tf.reduce_sum(policy_t * log_pis_t, axis=-1)
        entropy = tf.reduce_sum(train_batch["eq_agent"] * entropy_vec) / eq_count
        # take log alpha instead of alpha
        alpha_loss = - model.log_alpha * tf.stop_gradient(model.target_entropy - entropy)
    else:
        alpha_loss = -tf.reduce_mean(model.log_alpha * tf.stop_gradient(log_pis_t + model.target_entropy))
        actor_loss = tf.reduce_mean(model.alpha * log_pis_t - q_t_det_policy)

    # Train diversity model
    if policy.config["divergence_type"] == 'action':
        delta_loss = 0  # non-parametric diversity regularization mode
        divergence_vec = tf.math.not_equal(
            tf.argmax(policy_t, axis=1, output_type=tf.int32),
            tf.argmax(train_batch["opponent_action_dist"], axis=1, output_type=tf.int32))
        divergence_vec = tf.cast(divergence_vec, tf.float32)
        divergence_vec = divergence_vec * train_batch["l_agent"]
        div_rate = tf.reduce_sum(divergence_vec) / l_count
    elif policy.config["divergence_type"] in ['state_action', 'state']:
        d_t = policy.delta.get_d_values(delta_t, actions=None)
        if policy.delta.divergence_type == 'state_action':
            one_hot_3d = tf.tile(tf.expand_dims(one_hot, axis=1), multiples=[1, 2, 1])
            d_t = tf.reduce_sum(one_hot_3d * d_t, axis=2)
        else:  # state-only discrimination mode
            d_t = tf.squeeze(d_t, axis=2)
        delta_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(train_batch["disc_label"], d_t)

        delta_loss = tf.reduce_sum(train_batch["leq_agent"] * delta_loss_vec) / leq_count
        divergence_vec = tf.math.equal(train_batch["disc_label"],
                                       tf.argmax(d_t, axis=1, output_type=tf.int32))
        divergence_vec = tf.cast(divergence_vec, tf.float32)
        divergence_vec = divergence_vec * train_batch["leq_agent"]
        div_rate = tf.reduce_sum(divergence_vec) / leq_count
    else:
        raise ValueError

    # Auto adjust divergence coefficient
    beta_backup = tf.stop_gradient(model.target_div - div_rate)
    beta_loss = - tf.reduce_mean(model.log_beta * beta_backup)

    # save for stats function
    policy.policy_t = policy_t
    policy.q_t = q_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.alpha_value = policy.actor.alpha
    policy.entropy = entropy
    policy.target_entropy = policy.actor.target_entropy
    policy.delta_loss = delta_loss
    policy.beta_loss = beta_loss
    policy.beta_value = policy.actor.beta
    policy.target_div = policy.actor.target_div
    policy.div_rate = div_rate
    policy.delta_penalty = logd_tp1
    policy.entropy_penalty = log_pis_tp1

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    return actor_loss + tf.math.add_n(critic_loss) + alpha_loss + beta_loss + delta_loss


def gradients_fn(policy, optimizer, loss):
    # Eager: Use GradientTape.

    # Bug fix: gradients are defined over non-filtered variable list
    if policy.config["framework"] in ["tf2", "tfe"]:
        tape = optimizer.tape
        actor_grads_and_vars = list(zip(tape.gradient(
            policy.actor_loss, policy.actor.variables()), policy.actor.variables()))
        critic_grads_and_vars = list(zip(tape.gradient(
            policy.critic_loss[0], policy.critic.variables()), policy.critic.variables()))
        if policy.config["twin_q"]:
            twin_critic_grads_and_vars = list(zip(tape.gradient(
                policy.critic_loss[1], policy.twin_critic.variables()), policy.twin_critic.variables()))

        alpha_grads_and_vars = list(zip(tape.gradient(
            policy.alpha_loss, [policy.actor.log_alpha]), [policy.actor.log_alpha]))
        beta_grads_and_vars = list(zip(tape.gradient(
            policy.beta_loss, [policy.actor.log_beta]), [policy.actor.log_beta]))
        if hasattr(policy, "delta"):
            delta_grads_and_vars = list(zip(tape.gradient(
                policy.delta_loss, policy.delta.variables()), policy.delta.variables()))

    # Tf1.x: Use optimizer.compute_gradients()
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.actor.variables())

        critic_grads_and_vars = policy._critic_optimizer.compute_gradients(
            policy.critic_loss[0], var_list=policy.critic.variables())
        if policy.config["twin_q"]:
            twin_critic_grads_and_vars = policy._twin_critic_optimizer.compute_gradients(
                policy.critic_loss[1], var_list=policy.twin_critic.variables())
        else:
            twin_critic_grads_and_vars = []

        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
            policy.alpha_loss, var_list=[policy.actor.log_alpha])
        beta_grads_and_vars = policy._beta_optimizer.compute_gradients(
            policy.beta_loss, var_list=[policy.actor.log_beta])
        if hasattr(policy, "delta"):
            delta_grads_and_vars = policy._delta_optimizer.compute_gradients(
                policy.delta_loss, var_list=policy.delta.variables())

    # Clip if necessary.
    if policy.config["grad_clip"]:
        clip_func = tf.clip_by_norm
    else:
        clip_func = tf.identity

    # Save grads and vars for later use in `build_apply_op`.
    policy._actor_grads_and_vars = [
        (clip_func(g), v) for (g, v) in actor_grads_and_vars if g is not None]
    policy._critic_grads_and_vars = [
        (clip_func(g), v) for (g, v) in critic_grads_and_vars if g is not None]
    if policy.config["twin_q"]:
        policy._twin_critic_grads_and_vars = [
            (clip_func(g), v) for (g, v) in twin_critic_grads_and_vars if g is not None]
    else:
        policy._twin_critic_grads_and_vars = []
    policy._alpha_grads_and_vars = [
        (clip_func(g), v) for (g, v) in alpha_grads_and_vars if g is not None]
    if hasattr(policy.model, "d_net"):
        policy._delta_grads_and_vars = [
            (clip_func(g), v) for (g, v) in delta_grads_and_vars if g is not None]
    else:
        policy._delta_grads_and_vars = []
    policy._beta_grads_and_vars = [
        (clip_func(g), v) for (g, v) in beta_grads_and_vars if g is not None]

    grads_and_vars = (
        policy._actor_grads_and_vars + policy._critic_grads_and_vars + policy._twin_critic_grads_and_vars +
        policy._alpha_grads_and_vars + policy._beta_grads_and_vars + policy._delta_grads_and_vars)
    return grads_and_vars


def apply_gradients(policy, optimizer, grads_and_vars):
    actor_apply_ops = policy._actor_optimizer.apply_gradients(
        policy._actor_grads_and_vars)

    critic_apply_ops = policy._critic_optimizer.apply_gradients(
        policy._critic_grads_and_vars)

    if policy.config["twin_q"]:
        twin_critic_apply_ops = [policy._twin_critic_optimizer.apply_gradients(
            policy._twin_critic_grads_and_vars)]
    else:
        twin_critic_apply_ops = []

    if hasattr(policy.model, "d_net"):
        delta_apply_ops = [policy._delta_optimizer.apply_gradients(
            policy._delta_grads_and_vars)]
    else:
        delta_apply_ops = []
    if policy.config["framework"] in ["tf2", "tfe"]:
        if policy.config["alpha"] is None:
            policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars)
        if policy.config["beta"] is None:
            policy._beta_optimizer.apply_gradients(policy._beta_grads_and_vars)
        return
    else:
        apply_ops = [actor_apply_ops] + [critic_apply_ops] + twin_critic_apply_ops + delta_apply_ops
        if policy.config["alpha"] is None:
            alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
                policy._alpha_grads_and_vars,
                global_step=tf1.train.get_or_create_global_step())
            apply_ops += [alpha_apply_ops]
        if policy.config["beta"] is None:
            beta_apply_ops = policy._beta_optimizer.apply_gradients(
                policy._beta_grads_and_vars,
                global_step=tf1.train.get_or_create_global_step())
            apply_ops += [beta_apply_ops]
        return tf.group(apply_ops)


def stats(policy, train_batch):
    return {
        "policy_t": policy.policy_t,
        "td_error": policy.td_error,
        "mean_td_error": tf.reduce_mean(policy.td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss),
        "alpha_loss": tf.reduce_mean(policy.alpha_loss),
        "alpha_value": tf.reduce_mean(policy.alpha_value),
        "beta_loss": tf.reduce_mean(policy.beta_loss),
        "delta_loss": tf.reduce_mean(policy.delta_loss),
        "beta_value": tf.reduce_mean(policy.beta_value),
        "target_entropy": tf.constant(policy.target_entropy),
        "target_div": tf.constant(policy.target_div),
        "mean_q": tf.reduce_mean(policy.q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "min_q": tf.reduce_min(policy.q_t),
        "div_rate": tf.reduce_mean(policy.div_rate),
        "entropy": tf.reduce_mean(policy.entropy),
        "entropy_penalty": tf.reduce_mean(policy.entropy_penalty),
        "delta_penalty": tf.reduce_mean(policy.delta_penalty),
    }


class TargetNetworkMixin:
    def __init__(self, config):
        @make_tf_callable(self.get_session())
        def update_target_fn(tau):
            tau = tf.convert_to_tensor(tau, dtype=tf.float32)
            update_target_expr = []
            model_vars = self.critic.variables()
            target_model_vars = self.target_critic.variables()
            if config["twin_q"]:
                model_vars += self.twin_critic.variables()
                target_model_vars += self.target_twin_critic.variables()
            assert len(model_vars) == len(target_model_vars), \
                (model_vars, target_model_vars)
            for var, var_target in zip(model_vars, target_model_vars):
                update_target_expr.append(
                    var_target.assign(tau * var + (1.0 - tau) * var_target))
                logger.debug("Update target op {}".format(var_target))
            return tf.group(*update_target_expr)

        # Hard initial update.
        self._do_update = update_target_fn
        self.update_target(tau=1.0)

    # Support both hard and soft sync.
    def update_target(self, tau=None):
        self._do_update(np.float32(tau or self.config.get("tau")))

    # @override(TFPolicy)
    # def variables(self):
    #     assert False, "do we need to be here?"
    #     return self.model.variables() + self.target_model.variables()


class ActorCriticOptimizerMixin:
    def __init__(self, config):
        # - Create global step for counting the number of update operations.
        # - Use separate optimizers for actor & critic.
        if config["framework"] in ["tf2", "tfe"]:
            self.global_step = get_variable(0, tf_name="global_step")
            self._actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["actor_learning_rate"])
            self._critic_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config["optimization"]["critic_learning_rate"])
            if config["twin_q"]:
                self._twin_critic_optimizer = tf.keras.optimizers.Adam(
                        learning_rate=config["optimization"]["critic_learning_rate"])
            self._alpha_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["entropy_learning_rate"])
            self._beta_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["beta_learning_rate"])
            if config["divergence_type"] in ["state_action", "state"]:
                self._delta_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config["optimization"]["critic_learning_rate"])
        else:
            self.global_step = tf1.train.get_or_create_global_step()
            self._actor_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["actor_learning_rate"])
            self._critic_optimizer = tf1.train.AdamOptimizer(
                    learning_rate=config["optimization"]["critic_learning_rate"])
            if config["twin_q"]:
                self._twin_critic_optimizer = tf1.train.AdamOptimizer(
                        learning_rate=config["optimization"]["critic_learning_rate"])
            self._alpha_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["entropy_learning_rate"])
            self._beta_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["beta_learning_rate"])
            if config["divergence_type"] in ["state_action", "state"]:
                self._delta_optimizer = tf1.train.AdamOptimizer(
                    learning_rate=config["optimization"]["critic_learning_rate"])


def setup_early_mixins(policy, obs_space, action_space, config):
    ActorCriticOptimizerMixin.__init__(policy, config)


def setup_mid_mixins(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy, sac_actor_critic_loss)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)


def validate_spaces(pid, observation_space, action_space, config):
    if not isinstance(action_space, (Box, Discrete)):
        raise UnsupportedSpaceException(
            "Action space ({}) of {} is not supported for "
            "SAC.".format(action_space, pid))
    if isinstance(action_space, Box) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space ({}) of {} has multiple dimensions "
            "{}. ".format(action_space, pid, action_space.shape) +
            "Consider reshaping this into a single dimension, "
            "using a Tuple action space, or the multi-agent API.")


SACMATFPolicy = build_tf_policy(
    name="SACMATFPolicy",
    get_default_config=lambda: ray.rllib.agents.sac.sac.DEFAULT_CONFIG,
    make_model=build_sac_model,
    postprocess_fn=postprocess_trajectory,
    action_distribution_fn=get_distribution_inputs_and_class,
    loss_fn=sac_actor_critic_loss,
    stats_fn=stats,
    gradients_fn=gradients_fn,
    apply_gradients_fn=apply_gradients,
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error},
    mixins=[
        TargetNetworkMixin, ActorCriticOptimizerMixin, ComputeTDErrorMixin
    ],
    validate_spaces=validate_spaces,
    before_init=setup_early_mixins,
    before_loss_init=setup_mid_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False)
