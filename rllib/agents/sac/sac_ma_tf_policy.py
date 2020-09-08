from gym.spaces import Box, Discrete
import logging
import numpy as np
import random
import ray
import ray.experimental.tf_utils
from ray.rllib.agents.ddpg.ddpg_tf_policy import ComputeTDErrorMixin, \
    TargetNetworkMixin
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.agents.sac.sac_ma_tf_model import SACMATFModel
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Beta, Categorical, \
    DiagGaussian, SquashedGaussian
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf, \
    try_import_tfp

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)

AGENT_LABEL = 0
OPPONENT_LABEL = 1
LEARN_IT = .1


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
    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACTorchModel
        if config["framework"] == "torch" else SACMATFModel,
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        alpha=config["alpha"],
        beta=config["beta"],
        entropy_scale=config["entropy_scale"],
        target_acc=config["target_acc"],
        divergence_type=config["divergence_type"],)

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACTorchModel
        if config["framework"] == "torch" else SACMATFModel,
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        alpha=config["alpha"],
        beta=config["beta"],
        entropy_scale=config["entropy_scale"],
        target_acc=config["target_acc"],
        divergence_type=config["divergence_type"],)

    return policy.model


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    if other_agent_batches is not None:
        for opponent_id in other_agent_batches.keys():
            opponent_batch = other_agent_batches[opponent_id][1]
            for key in opponent_batch.keys():
                sample_batch[key] = np.append(sample_batch[key], opponent_batch[key], axis=0)
    sample_batch["disc_valid"] = np.ones_like(sample_batch[SampleBatch.REWARDS])
    sample_batch["disc_label"] = AGENT_LABEL * np.ones_like(sample_batch[SampleBatch.REWARDS], dtype=np.int32)
    sample_batch["own_experience"] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    if 'agent_id' in sample_batch:
        agent_id = sample_batch["agent_id"][0]
        for i, member_id in enumerate(sample_batch['agent_id']):
            if member_id < agent_id:
                sample_batch["disc_label"][i] = OPPONENT_LABEL
                sample_batch["own_experience"][i] = 0.
            elif member_id == agent_id:
                sample_batch["disc_label"][i] = AGENT_LABEL
                sample_batch["own_experience"][i] = 1.
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

    model_out_t, _ = model({
        "obs": cur_obs,
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    model_out_tp1, _ = model({
        "obs": next_obs,
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    target_model_out_tp1, _ = policy.target_model({
        "obs": next_obs,
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        log_pis_t = tf.nn.log_softmax(model.get_policy_output(model_out_t), -1)
        policy_t = tf.math.exp(log_pis_t)
        log_pis_tp1 = tf.nn.log_softmax(
            model.get_policy_output(model_out_tp1), -1)
        policy_tp1 = tf.math.exp(log_pis_tp1)
        # Q-values.
        q_t = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(model_out_t)
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_tp1 -= model.alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = tf.one_hot(
            train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1])
        q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
        # Diversity reward
        if model.divergence_type == 'state_action':
            # calculated based on s_t, a_t. First extract discrimination for all actions in time t
            at_time = 'tp1'
            if at_time == 't':
                discrimination = model.get_d_values(model_out_t, actions=None)
                # Slice actually selected action a_t
                one_hot_3d = tf.tile(tf.expand_dims(one_hot, axis=1), multiples=[1, 2, 1])
                discrimination = tf.reduce_sum(discrimination * one_hot_3d, axis=2)
            elif at_time == 'tp1':
                discrimination = model.get_d_values(model_out_tp1, actions=None)
                action_dist_class = get_dist_class(policy.config, policy.action_space)
                action_dist_tp1 = action_dist_class(
                    model.get_policy_output(model_out_tp1), policy.model)
                sampled_a_tp1 = action_dist_tp1.sample()
                # Slice sampled action a_tp1
                one_hot_tp1 = tf.one_hot(sampled_a_tp1, depth=q_t.shape.as_list()[-1])
                one_hot_3d_tp1 = tf.tile(tf.expand_dims(one_hot_tp1, axis=1), multiples=[1, 2, 1])
                discrimination = tf.reduce_sum(discrimination * one_hot_3d_tp1, axis=2)
            else:
                raise ValueError
        elif model.divergence_type == 'state':
            # calculated based on s_tp1
            discrimination = model.get_d_values(model_out_tp1, actions=None)
            discrimination = tf.squeeze(discrimination, axis=2)
        else:
            print(f"Unsupported divergence type: {model.divergence_type}")
            raise ValueError
        log_discrimination = tf.nn.log_softmax(discrimination, axis=1)
        agent_discrimination_logit = tf.split(log_discrimination, 2, axis=1)[AGENT_LABEL]
        q_tp1_best += model.beta * tf.squeeze(agent_discrimination_logit, axis=1)
        q_tp1_best_masked = \
            (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
            q_tp1_best
    # Continuous actions case.
    else:
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
        # Diversity reward
        if model.divergence_type == 'state_action':
            at_time = "tp1"
            if at_time == 't':
                # calculate based on s_t, a_t
                discrimination = model.get_d_values(model_out_t, train_batch[SampleBatch.ACTIONS])
            elif at_time == 'tp1':
                # calculate based on s_tp1, a_tp1
                discrimination = model.get_d_values(model_out_tp1, policy_tp1)
            else:
                raise ValueError
        elif model.divergence_type == 'state':
            # calculate based on s_tp1
            discrimination = model.get_d_values(model_out_tp1, actions=None)
        else:
            print(f"Unsupported divergence type: {model.divergence_type}")
            raise ValueError
        discrimination = tf.squeeze(discrimination, axis=2)
        log_discrimination = tf.nn.log_softmax(discrimination, axis=1)
        agent_discrimination_logit = tf.split(log_discrimination, 2, axis=1)[AGENT_LABEL]
        q_tp1_best += model.beta * tf.squeeze(agent_discrimination_logit, axis=1)
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
    own_exp_sum = 1e-4 + tf.reduce_sum(train_batch["own_experience"])
    if model.discrete:
        alpha_loss = tf.reduce_sum(
            train_batch["own_experience"] * tf.reduce_sum(
                tf.multiply(
                    tf.stop_gradient(policy_t), -model.log_alpha *
                    tf.stop_gradient(log_pis_t + model.target_entropy)),
                axis=-1)) / own_exp_sum
        entropy = -tf.reduce_sum(
            train_batch["own_experience"] * tf.reduce_sum(tf.multiply(policy_t, log_pis_t), axis=-1)) / own_exp_sum
        # action_dist_class = get_dist_class(policy.config, policy.action_space)
        # action_dist_t = action_dist_class(model.get_policy_output(model_out_t), policy.model)
        # entropy = action_dist_t.entropy()
        actor_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    model.alpha * log_pis_t - tf.stop_gradient(q_t)),
                axis=-1))
        # actor_loss = tf.reduce_mean(tf.reduce_sum(one_hot * (model.alpha * log_pis_t - q_t), axis=1))
    else:
        alpha_loss = -tf.reduce_sum(
            train_batch["own_experience"] * model.log_alpha *
            tf.stop_gradient(log_pis_t + model.target_entropy)) / own_exp_sum
        entropy = -tf.reduce_sum(train_batch["own_experience"] * log_pis_t) / own_exp_sum
        actor_loss = tf.reduce_mean(model.alpha * log_pis_t - q_t_det_policy)

    # discrimination
    d_t = model.get_d_values(model_out_t, train_batch[SampleBatch.ACTIONS])
    if d_t.shape.as_list()[2] > 1:
        assert model.discrete
        one_hot_3d = tf.tile(tf.expand_dims(one_hot, axis=1), multiples=[1, 2, 1])
        d_t = tf.reduce_sum(d_t * one_hot_3d, axis=2)
    else:
        d_t = tf.squeeze(d_t, axis=2)
    valid_count = 1e-4 + tf.reduce_sum(train_batch["disc_valid"])
    disc_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(train_batch["disc_label"], d_t)
    disc_loss = tf.reduce_sum(train_batch["disc_valid"] * disc_loss_vec) / valid_count
    disc_accuracy_vec = tf.math.equal(train_batch["disc_label"], tf.argmax(d_t, axis=1, output_type=tf.int32))
    disc_accuracy_vec = tf.cast(disc_accuracy_vec, tf.float32) * train_batch["disc_valid"]
    disc_accuracy = tf.reduce_sum(disc_accuracy_vec) / valid_count
    beta_loss = model.beta * tf.stop_gradient(disc_accuracy - model.target_acc)

    # save for stats function
    policy.policy_t = policy_t
    policy.q_t = q_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.alpha_value = model.alpha
    policy.beta_loss = beta_loss
    policy.beta_value = model.beta
    policy.target_entropy = model.target_entropy
    policy.disc_loss = disc_loss
    policy.disc_acc = disc_accuracy
    policy.entropy = entropy
    policy.agent_discrimination_logit = agent_discrimination_logit

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    return actor_loss + tf.math.add_n(critic_loss) + alpha_loss + disc_loss + beta_loss


def gradients_fn(policy, optimizer, loss):
    # Eager: Use GradientTape.
    if policy.config["framework"] in ["tf2", "tfe"]:
        tape = optimizer.tape
        pol_weights = policy.model.policy_variables()
        actor_grads_and_vars = list(zip(tape.gradient(
            policy.actor_loss, pol_weights), pol_weights))
        q_weights = policy.model.q_variables()
        if policy.config["twin_q"]:
            half_cutoff = len(q_weights) // 2
            grads_1 = tape.gradient(
                policy.critic_loss[0], q_weights[:half_cutoff])
            grads_2 = tape.gradient(
                policy.critic_loss[1], q_weights[half_cutoff:])
            critic_grads_and_vars = \
                list(zip(grads_1, q_weights[:half_cutoff])) + \
                list(zip(grads_2, q_weights[half_cutoff:]))
        else:
            critic_grads_and_vars = list(zip(tape.gradient(
                policy.critic_loss[0], q_weights), q_weights))

        alpha_vars = [policy.model.log_alpha]
        alpha_grads_and_vars = list(zip(tape.gradient(
            policy.alpha_loss, alpha_vars), alpha_vars))
        d_weights = policy.model.d_variables()
        disc_grads_and_vars = list(zip(tape.gradient(
            policy.disc_loss, d_weights), d_weights))
        beta_vars = [policy.model.log_beta]
        beta_grads_and_vars = list(zip(tape.gradient(
            policy.beta_loss, beta_vars), beta_vars))
    # Tf1.x: Use optimizer.compute_gradients()
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.model.policy_variables())

        q_weights = policy.model.q_variables()
        if policy.config["twin_q"]:
            half_cutoff = len(q_weights) // 2
            base_q_optimizer, twin_q_optimizer = policy._critic_optimizer
            critic_grads_and_vars = base_q_optimizer.compute_gradients(
                policy.critic_loss[0], var_list=q_weights[:half_cutoff]
            ) + twin_q_optimizer.compute_gradients(
                policy.critic_loss[1], var_list=q_weights[half_cutoff:])
        else:
            critic_grads_and_vars = policy._critic_optimizer[
                0].compute_gradients(
                    policy.critic_loss[0], var_list=q_weights)
        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
            policy.alpha_loss, var_list=[policy.model.log_alpha])
        disc_grads_and_vars = policy._disc_optimizer.compute_gradients(
            policy.disc_loss, var_list=policy.model.d_variables())
        beta_grads_and_vars = policy._beta_optimizer.compute_gradients(
            policy.beta_loss, var_list=[policy.model.log_beta])

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
    policy._alpha_grads_and_vars = [
        (clip_func(g), v) for (g, v) in alpha_grads_and_vars if g is not None]
    policy._disc_grads_and_vars = [
        (clip_func(g), v) for (g, v) in disc_grads_and_vars if g is not None]
    policy._beta_grads_and_vars = [
        (clip_func(g), v) for (g, v) in beta_grads_and_vars if g is not None]

    grads_and_vars = (
        policy._actor_grads_and_vars + policy._critic_grads_and_vars +
        policy._alpha_grads_and_vars + policy._disc_grads_and_vars + policy._beta_grads_and_vars)
    return grads_and_vars


def apply_gradients(policy, optimizer, grads_and_vars):
    actor_apply_ops = policy._actor_optimizer.apply_gradients(
        policy._actor_grads_and_vars)

    cgrads = policy._critic_grads_and_vars
    half_cutoff = len(cgrads) // 2
    if policy.config["twin_q"]:
        critic_apply_ops = [
            policy._critic_optimizer[0].apply_gradients(cgrads[:half_cutoff]),
            policy._critic_optimizer[1].apply_gradients(cgrads[half_cutoff:])
        ]
    else:
        critic_apply_ops = [
            policy._critic_optimizer[0].apply_gradients(cgrads)
        ]
    disc_apply_ops = policy._disc_optimizer.apply_gradients(
        policy._disc_grads_and_vars)
    if policy.config["framework"] in ["tf2", "tfe"]:
        if policy.config["alpha"] is None:
            policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars)
        if policy.config["beta"] is None:
            policy._beta_optimizer.apply_gradients(policy._beta_grads_and_vars)
        return
    else:
        apply_ops = [actor_apply_ops] + critic_apply_ops + [disc_apply_ops]
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
        # "policy_t": policy.policy_t,
        # "td_error": policy.td_error,
        "mean_td_error": tf.reduce_mean(policy.td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss),
        "alpha_loss": tf.reduce_mean(policy.alpha_loss),
        "alpha_value": tf.reduce_mean(policy.alpha_value),
        "beta_loss": tf.reduce_mean(policy.beta_loss),
        "beta_value": tf.reduce_mean(policy.beta_value),
        "target_entropy": tf.constant(policy.target_entropy),
        "mean_q": tf.reduce_mean(policy.q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "min_q": tf.reduce_min(policy.q_t),
        "disc_loss": tf.reduce_mean(policy.disc_loss),
        "disc_acc": tf.reduce_mean(policy.disc_acc),
        "entropy": tf.reduce_mean(policy.entropy),
        "agent_discrimination_logit": tf.reduce_mean(policy.agent_discrimination_logit),
    }


class ActorCriticOptimizerMixin:
    def __init__(self, config):
        # - Create global step for counting the number of update operations.
        # - Use separate optimizers for actor & critic.
        if config["framework"] in ["tf2", "tfe"]:
            self.global_step = get_variable(0, tf_name="global_step")
            self._actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["actor_learning_rate"])
            self._critic_optimizer = [
                tf.keras.optimizers.Adam(
                    learning_rate=config["optimization"][
                        "critic_learning_rate"])
            ]
            if config["twin_q"]:
                self._critic_optimizer.append(
                    tf.keras.optimizers.Adam(
                        learning_rate=config["optimization"][
                            "critic_learning_rate"]))
            self._alpha_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["entropy_learning_rate"])
            self._disc_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["critic_learning_rate"])
            self._beta_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["beta_learning_rate"])
        else:
            self.global_step = tf1.train.get_or_create_global_step()
            self._actor_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["actor_learning_rate"])
            self._critic_optimizer = [
                tf1.train.AdamOptimizer(
                    learning_rate=config["optimization"][
                        "critic_learning_rate"])
            ]
            if config["twin_q"]:
                self._critic_optimizer.append(
                    tf1.train.AdamOptimizer(
                        learning_rate=config["optimization"][
                            "critic_learning_rate"]))
            self._alpha_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["entropy_learning_rate"])
            self._disc_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["critic_learning_rate"])
            self._beta_optimizer = tf1.train.AdamOptimizer(
                learning_rate=config["optimization"]["beta_learning_rate"])


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