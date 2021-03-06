from gym.spaces import Box, MultiDiscrete
import logging
import numpy as np
import ray
import ray.experimental.tf_utils
from ray.rllib.agents.ddpg.ddpg_tf_policy import ComputeTDErrorMixin, \
    TargetNetworkMixin
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.agents.sac.sac_ensemble_tf_model import SACEnsembleTFModel
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Beta, MultiCategorical, \
    DiagGaussian, MultiSquashedGaussian
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_ensemble_policy_template import build_tf_ensemble_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf, \
    try_import_tfp

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)


def build_sac_ensemble_model(policy, obs_space, action_space, config):
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
        if config["framework"] == "torch" else SACEnsembleTFModel,
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        alpha=config["alpha"],
        target_entropy=config["target_entropy"],
        ensemble_size=config["partial_ensemble_size"],
        timescale=config["timescale"],
        shared_actor=config["shared_actor"],)

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACTorchModel
        if config["framework"] == "torch" else SACEnsembleTFModel,
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        alpha=config["alpha"],
        target_entropy=config["target_entropy"],
        ensemble_size=config["partial_ensemble_size"],
        timescale=config["timescale"],
        shared_actor=config["shared_actor"],)

    return policy.model


def slice_loss(x, idx, mode='slice'):
    xshape = x.shape.as_list()
    if mode == 'slice':
        begin = [0] * len(xshape)
        size = [-1] * len(xshape)
        begin[1] = idx
        size[1] = 1
        return tf.reduce_mean(tf.slice(x, begin, size))
    elif mode == 'mask':
        onehot_vec = tf.expand_dims(tf.one_hot(idx, depth=E), 0)
        if len(xshape) == 3:
            onehot_vec = tf.expand_dims(onehot_vec, -1)
        masked_x = tf.multiply(x, onehot_vec)
        return tf.reduce_mean(tf.reduce_sum(masked_x, axis=1))
    else:
        raise ValueError


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    if 'infos' not in sample_batch:
        sample_batch['members'] = np.ones_like(sample_batch[SampleBatch.REWARDS]).astype(np.int32)
        print("infos field not in sample_batch !!!")
    else:
        sample_batch['members'] = np.array([info['active_member'] for info in sample_batch['infos']], dtype=np.int32)
    return postprocess_nstep_and_prio(policy, sample_batch)


def get_dist_class(config, action_space):
    if isinstance(action_space, MultiDiscrete):
        return MultiCategorical
    else:
        if config["normalize_actions"]:
            return MultiSquashedGaussian if \
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

    model_out_t, _ = model({
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    # Broadcast the action of active ensemble member to all other ensemble members,
    # because this action is the one responsible for the transition.
    E = policy.config['partial_ensemble_size']
    dones = tf.tile(tf.expand_dims(train_batch[SampleBatch.DONES], 1), [1, E])
    rewards = tf.tile(tf.expand_dims(train_batch[SampleBatch.REWARDS], 1), [1, E])
    member_mat = tf.one_hot(train_batch['members'], depth=E)

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
        ######################### CROSS ENTROPY #########################
        # old:
        # q_tp1 -= model.alpha * log_pis_tp1
        # new:
        if policy.config["asymmetric"]:
            print(f"============ Asymmetric Ensemble===========")
            cum_log_pis_tp1 = tf.math.cumsum(log_pis_tp1, axis=1)
            arange = tf.range(start=1, limit=E + 1, delta=1, dtype=tf.float32, name='range')
            inv_arange = tf.math.divide(1., arange)
            w = tf.tile(tf.expand_dims(inv_arange, 1), [1, q_t.shape.as_list()[-1]])
            ens_log_pis_tp1 = w * cum_log_pis_tp1
            q_tp1 -= model.alpha * ens_log_pis_tp1
        else:
            beta = 1 / E * tf.ones((E, E), dtype=tf.float32)
            q_tp1 -= model.alpha * tf.matmul(beta, log_pis_tp1)
        #################################################################
        # Actually selected Q-values (from the actions batch).
        actions_mat = tf.cast(member_mat, train_batch[SampleBatch.ACTIONS].dtype) * train_batch[SampleBatch.ACTIONS]
        actions = tf.reduce_sum(actions_mat, axis=1)
        bcast_actions = tf.tile(tf.expand_dims(actions, 1), [1, E])
        one_hot = tf.one_hot(bcast_actions, depth=q_t.shape.as_list()[-1])
        q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
        q_tp1_best_masked = \
            (1.0 - tf.cast(dones, tf.float32)) * \
            q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = get_dist_class(policy.config, policy.action_space)
        action_dist_t = action_dist_class(
            model.get_policy_output(model_out_t), policy.model)
        policy_t = action_dist_t.sample() if not deterministic else \
            action_dist_t.deterministic_sample()
        log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t, reduce=False), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else \
            action_dist_tp1.deterministic_sample()
        log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1, reduce=False), -1)

        # Q-values for the actually selected actions.
        ex_member_mat = tf.tile(tf.expand_dims(member_mat, 2), [1, 1, policy_t.shape.as_list()[-1]])
        active_actions = tf.reduce_sum(ex_member_mat * train_batch[SampleBatch.ACTIONS], axis=1, keepdims=True)
        active_action_mat = tf.tile(active_actions, [1, E, 1])
        q_t = model.get_q_values(model_out_t, active_action_mat)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, active_action_mat)

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
        ######################### CROSS ENTROPY #########################
        # old:
        # q_tp1 -= model.alpha * log_pis_tp1
        # new:
        if policy.config["asymmetric"]:
            print(f"============ Asymmetric Ensemble===========")
            arange = tf.range(start=1, limit=E + 1, delta=1, dtype=tf.float32, name='range')
            inv_arange = tf.math.divide(1., arange)
            w = tf.tile(tf.expand_dims(inv_arange, 1), [1, q_t.shape.as_list()[-1]])
            cum_log_pis_tp1 = tf.math.cumsum(log_pis_tp1, axis=1)
            ens_log_pis_tp1 = w * cum_log_pis_tp1
            q_tp1 -= model.alpha * ens_log_pis_tp1
        else:
            beta = 1 / E * tf.ones((E, E), dtype=tf.float32)
            q_tp1 -= model.alpha * tf.matmul(beta, log_pis_tp1)
        #################################################################
        q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (1.0 - tf.cast(dones, tf.float32)) * q_tp1_best

    assert policy.config["n_step"] == 1, "TODO(hartikainen) n_step > 1"

    # compute RHS of bellman equation
    q_t_selected_target = tf.stop_gradient(
        rewards + policy.config["gamma"]**policy.config["n_step"] * q_tp1_best_masked)

    # Compute the TD-error (potentially clipped).
    base_td_error = tf.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = tf.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    crnt_trnng_idx = tf.cast(policy.model.flrd_cntr, tf.int32)

    critic_ens_loss = 0.5 * tf.square(q_t_selected_target - q_t_selected)

    slice_mode = 'slice'
    critic_loss = [slice_loss(critic_ens_loss, crnt_trnng_idx, mode=slice_mode)]

    if policy.config["twin_q"]:
        twin_c_ens_loss = 0.5 * tf.square(q_t_selected_target - twin_q_t_selected)
        critic_loss.append(slice_loss(twin_c_ens_loss, crnt_trnng_idx, mode=slice_mode))

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        # ens_pis_t = tf.reduce_mean(policy_t, axis=1)
        # ens_log_pis_t = tf.log(ens_pis_t)
        # alpha_loss = tf.reduce_mean(
        #     mask *
        #     tf.reduce_sum(
        #         tf.multiply(
        #             tf.stop_gradient(ens_pis_t), -model.log_alpha *
        #             tf.stop_gradient(ens_log_pis_t + model.target_entropy)),
        #         axis=-1))
        actor_ens_loss = tf.reduce_sum(tf.multiply(policy_t, model.alpha * log_pis_t - tf.stop_gradient(q_t)), axis=-1)
        actor_loss = slice_loss(actor_ens_loss, crnt_trnng_idx, mode=slice_mode)
    else:
        # alpha_loss = -tf.reduce_mean(
        #     model.log_alpha *
        #     tf.stop_gradient(log_pis_t + model.target_entropy))
        actor_ens_loss = model.alpha * log_pis_t - q_t_det_policy
        actor_loss = slice_loss(actor_ens_loss, crnt_trnng_idx, slice_mode)

    # save for stats function
    policy.policy_t = policy_t
    policy.q_t = q_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    # policy.alpha_loss = alpha_loss
    policy.alpha_value = model.alpha
    policy.target_entropy = model.target_entropy

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    return actor_loss + tf.add_n(critic_loss)  # + alpha_loss


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

        # alpha_vars = [policy.model.log_alpha]
        # alpha_grads_and_vars = list(zip(tape.gradient(
        #     policy.alpha_loss, alpha_vars), alpha_vars))
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
        # alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
        #     policy.alpha_loss, var_list=[policy.model.log_alpha])

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
    # policy._alpha_grads_and_vars = [
    #     (clip_func(g), v) for (g, v) in alpha_grads_and_vars if g is not None]

    grads_and_vars = (
        policy._actor_grads_and_vars \
        + policy._critic_grads_and_vars \
        # + policy._alpha_grads_and_vars
    )
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

    if policy.config["framework"] in ["tf2", "tfe"]:
        # policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars)
        assert False, 'implement counter apply op'
        return
    else:
        # alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
        #     policy._alpha_grads_and_vars,
        #     global_step=tf1.train.get_or_create_global_step())
        return tf.group([actor_apply_ops, policy.model.cntr_inc_op] + critic_apply_ops)
    
    # # alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
    # #     policy._alpha_grads_and_vars,
    # #     global_step=tf.train.get_or_create_global_step())
    # apply_ops = [actor_apply_ops] + critic_apply_ops
    # apply_ops += [policy.model.cntr_inc_op]
    # 
    # # if policy.config["alpha"] is None:
    # #     apply_ops += [alpha_apply_ops]
    # return tf.group(apply_ops)


def stats(policy, train_batch):
    return {
        # "policy_t": policy.policy_t,
        # "td_error": policy.td_error,
        "mean_td_error": tf.reduce_mean(policy.td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss),
        # "alpha_loss": tf.reduce_mean(policy.alpha_loss),
        "alpha_value": tf.reduce_mean(policy.alpha_value),
        "target_entropy": tf.constant(policy.target_entropy),
        "mean_q": tf.reduce_mean(policy.q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "min_q": tf.reduce_min(policy.q_t),
        "counter": policy.model.cntr,
        "floored_counter": policy.model.flrd_cntr,
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


def setup_early_mixins(policy, obs_space, action_space, config):
    ActorCriticOptimizerMixin.__init__(policy, config)


def setup_mid_mixins(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy, sac_actor_critic_loss)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)


def validate_spaces(pid, observation_space, action_space, config):
    if not isinstance(action_space, (Box, MultiDiscrete)):
        raise UnsupportedSpaceException(
            "Action space ({}) of {} is not supported for "
            "SAC.".format(action_space, pid))
    if isinstance(action_space, Box) and len(action_space.shape) != 2:
        raise UnsupportedSpaceException(
            "Action space ({}) of {} has multiple dimensions "
            "{}. ".format(action_space, pid, action_space.shape) +
            "Consider reshaping this into a single dimension, "
            "using a Tuple action space, or the multi-agent API.")


SACEnsembleTFPolicy = build_tf_ensemble_policy(
    name="SACTFPolicy",
    get_default_config=lambda: ray.rllib.agents.sac.sac_ensemble.DEFAULT_CONFIG,
    make_model=build_sac_ensemble_model,
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
