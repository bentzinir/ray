from gym.spaces import Box, Discrete, MultiDiscrete
import logging

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.ddpg.ddpg_tf_policy import ComputeTDErrorMixin, \
    TargetNetworkMixin
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.agents.sac.sac_ensemble_tf_model_unstack import SACEnsembleTFModel
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Beta, MultiCategorical, Categorical, \
    DiagGaussian, MultiSquashedGaussian
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_ensemble_policy_template import build_tf_ensemble_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.tf_ops import minimize_and_clip

tf = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)


def build_sac_ensemble_model(policy, obs_space, action_space, config):
    if config["model"].get("custom_model"):
        logger.warning(
            "Setting use_state_preprocessor=True since a custom model "
            "was specified.")
        config["use_state_preprocessor"] = True
    # if not isinstance(action_space, (Box, Discrete)):
    #     raise UnsupportedSpaceException(
    #         "Action space {} is not supported for SAC.".format(action_space))
    # if isinstance(action_space, Box) and len(action_space.shape) > 1:
    #     raise UnsupportedSpaceException(
    #         "Action space has multiple dimensions "
    #         "{}. ".format(action_space.shape) +
    #         "Consider reshaping this into a single dimension, "
    #         "using a Tuple action space, or the multi-agent API.")

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
                "defined via `policy_model.hidden_layer_sizes` and "
                "`Q_model.hidden_layer_sizes`.")
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
        model_interface=SACTorchModel if config["framework"] == "torch" else SACEnsembleTFModel,
        # model_interface=SACTorchModel if config["framework"] == "torch" else SACTFModel,
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        ensemble_size=config["partial_ensemble_size"],
        shared_actor_body=config["shared_actor_body"],
        constant_alpha=config["constant_alpha"],
        shared_entropy=config["shared_entropy"])

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=SACTorchModel if config["framework"] == "torch" else SACEnsembleTFModel,
        # model_interface=SACTorchModel if config["framework"] == "torch" else SACTFModel,
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        ensemble_size=config["partial_ensemble_size"],
        shared_actor_body=config["shared_actor_body"],
        constant_alpha=config["constant_alpha"],
        shared_entropy=config["shared_entropy"])

    return policy.model


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    return postprocess_nstep_and_prio(policy, sample_batch)


def get_dist_class(config, action_space):
    # if isinstance(action_space, Discrete):
    #     return Categorical
    if isinstance(action_space, MultiDiscrete):
        return MultiCategorical
    else:
        if config["normalize_actions"]:
            # return SquashedGaussian if not config["_use_beta_distribution"] else Beta
            return MultiSquashedGaussian if not config["_use_beta_distribution"] else Beta
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

    def build_member_loss(midx):

        mask = tf.cast(tf.math.equal(train_batch['members'], midx), dtype=tf.float32)

        if policy.config["shared_entropy"]:
            alpha_m = model.alpha
            log_alpha_m = model.log_alpha
        else:
            alpha_m = tf.unstack(model.alpha, axis=0)[midx]
            log_alpha_m = tf.unstack(model.log_alpha, axis=0)[midx]

        if not policy.config["experience_masking"]:
            mask = tf.ones_like(mask)

        # Discrete case.
        if model.discrete:
            # Get all action probs directly from pi and form their logp.
            log_pis_t = tf.nn.log_softmax(model.get_policy_output(model_out_t, midx=midx), -1)
            policy_t = tf.exp(log_pis_t)
            log_pis_tp1 = tf.nn.log_softmax(
                model.get_policy_output(model_out_tp1, midx=midx), -1)
            policy_tp1 = tf.exp(log_pis_tp1)
            # Q-values.
            q_t = model.get_q_values(model_out_t, midx=midx)
            # Target Q-values.
            q_tp1 = policy.target_model.get_q_values(target_model_out_tp1, midx=midx)
            if policy.config["twin_q"]:
                twin_q_t = model.get_twin_q_values(model_out_t, midx=midx)
                twin_q_tp1 = policy.target_model.get_twin_q_values(
                    target_model_out_tp1, midx=midx)
                q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)

            ###########################
            # old
            # q_tp1 -= tf.unstack(model.alpha, axis=0)[midx] * log_pis_tp1
            # new
            ens_log_pis_tp1 = tf.nn.log_softmax(model.get_policy_output(model_out_tp1, midx=None), -1)
            q_tp1 -= alpha_m * tf.reduce_mean(ens_log_pis_tp1, axis=1)
            ###########################

            # Actually selected Q-values (from the actions batch).
            member_mat = tf.one_hot(train_batch['members'], depth=policy.config['partial_ensemble_size'])
            actions_mat = tf.cast(member_mat, tf.int64) * train_batch[SampleBatch.ACTIONS]
            actions = tf.reduce_sum(actions_mat, axis=1)
            one_hot = tf.one_hot(actions, depth=q_t.shape.as_list()[-1])
            q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
            if policy.config["twin_q"]:
                twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
            # Discrete case: "Best" means weighted by the policy (prob) outputs.
            q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
            q_tp1_best_masked = \
                (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
                q_tp1_best
        # Continuous actions case.
        else:
            # Sample single actions from distribution.
            action_dist_class = get_dist_class(policy.config, policy.action_space)
            ens_policy_t = model.get_policy_output(model_out_t)
            action_dist_t = action_dist_class(ens_policy_t, policy.model)
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
        base_td_error = tf.abs(q_t_selected - q_t_selected_target)
        if policy.config["twin_q"]:
            twin_td_error = tf.abs(twin_q_t_selected - q_t_selected_target)
            td_error = 0.5 * (base_td_error + twin_td_error)
        else:
            td_error = base_td_error

        # c_loss = tf.losses.mean_squared_error(labels=q_t_selected_target, predictions=q_t_selected, weights=0.5)
        c_loss = 0.5 * tf.square(q_t_selected_target - q_t_selected)
        critic_loss = [c_loss * mask]

        if policy.config["twin_q"]:
            # twin_c_loss = tf.losses.mean_squared_error(labels=q_t_selected_target, predictions=twin_q_t_selected,
            #                                            weights=0.5)
            twin_c_loss = 0.5 * tf.square(q_t_selected_target - twin_q_t_selected)
            critic_loss.append(twin_c_loss * mask)

        # Alpha- and actor losses.
        # Note: In the papers, alpha is used directly, here we take the log.
        # Discrete case: Multiply the action probs as weights with the original
        # loss terms (no expectations needed).
        if model.discrete:
            y = tf.reduce_sum(
                    tf.multiply(
                        tf.stop_gradient(policy_t), -log_alpha_m *
                                                    tf.stop_gradient(log_pis_t + model.target_entropy)),
                    axis=-1)
            alpha_loss = tf.reduce_mean(y * mask)

            z = tf.reduce_sum(
                    tf.multiply(
                        # NOTE: No stop_grad around policy output here
                        # (compare with q_t_det_policy for continuous case).
                        policy_t,
                        alpha_m * log_pis_t - tf.stop_gradient(q_t)),
                    axis=-1)
            actor_loss = tf.reduce_mean(z * mask)
        else:
            assert False, 'un debugged code'
            alpha_loss = -tf.reduce_mean(
                tf.unstack(model.log_alpha, axis=0)[midx] *
                tf.stop_gradient(log_pis_t + model.target_entropy))
            actor_loss = tf.reduce_mean(tf.unstack(model.alpha, axis=0)[midx] * log_pis_t - q_t_det_policy)

        return actor_loss, critic_loss, alpha_loss, policy_t, td_error, q_t

    actors_loss = []
    critics_loss = []
    alphas_loss = []
    policies_t = []
    td_errs = []
    q_ts = []
    for midx in range(policy.config['partial_ensemble_size']):
        actor_loss_m, critic_loss_m, alpha_loss_m, policy_t_m, td_error_m, q_t_m = build_member_loss(midx)
        actors_loss.append(actor_loss_m)
        critics_loss.append(critic_loss_m)
        alphas_loss.append(alpha_loss_m)
        policies_t.append(policy_t_m)
        td_errs.append(td_error_m)
        q_ts.append(q_t_m)

    critic0 = tf.reduce_mean([critics_loss[eidx][0] for eidx in range(policy.config['partial_ensemble_size'])])
    critic1 = tf.reduce_mean([critics_loss[eidx][1] for eidx in range(policy.config['partial_ensemble_size'])])

    alpha_loss = tf.reduce_mean(alphas_loss)

    if policy.config["shared_entropy"]:
        if model.discrete:
            ens_log_pis_t = tf.nn.log_softmax(model.get_policy_output(model_out_t, midx=None), -1)
            mean_ens_log_pis_t = tf.reduce_mean(ens_log_pis_t, axis=1)
            ens_policy_t = tf.exp(mean_ens_log_pis_t)
            y = tf.reduce_sum(
                tf.multiply(
                    tf.stop_gradient(ens_policy_t), -model.log_alpha *
                                                tf.stop_gradient(mean_ens_log_pis_t + model.target_entropy)),
                axis=-1)
            alpha_loss = tf.reduce_mean(y)
        else:
            assert False, 'un debugged code'

    # entropy = -tf.reduce_mean(tf.multiply(ens_policy_t, mean_ens_log_pis_t))

    # save for stats function
    policy.policy_t = tf.reduce_mean(policies_t, axis=0)
    policy.q_t = tf.reduce_mean(q_ts, axis=0)
    policy.td_error = tf.reduce_mean(td_errs)
    policy.actor_loss = tf.reduce_mean(actors_loss)
    policy.critic_loss = [critic0, critic1]
    policy.alpha_value = model.alpha
    policy.target_entropy = model.target_entropy
    # policy.current_entropy = entropy
    policy.mask_0 = tf.reduce_mean(tf.cast(tf.math.equal(train_batch['members'], 0), dtype=tf.float32))
    policy.mask_1 = tf.reduce_mean(tf.cast(tf.math.equal(train_batch['members'], 1), dtype=tf.float32))
    if not model.constant_alpha:
        policy.alpha_loss = alpha_loss

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    loss = tf.reduce_mean(actors_loss) + tf.add_n([critic0, critic1])
    if not model.constant_alpha:
        loss += alpha_loss
    return loss


def gradients(policy, optimizer, loss):
    if policy.config["grad_clip"]:
        actor_grads_and_vars = minimize_and_clip(
            optimizer,  # isn't optimizer not well defined here (which one)?
            policy.actor_loss,
            var_list=policy.model.policy_variables(),
            clip_val=policy.config["grad_clip"])
        if policy.config["twin_q"]:
            q_variables = policy.model.q_variables()
            half_cutoff = len(q_variables) // 2
            critic_grads_and_vars = []
            critic_grads_and_vars += minimize_and_clip(
                optimizer,
                policy.critic_loss[0],
                var_list=q_variables[:half_cutoff],
                clip_val=policy.config["grad_clip"])
            critic_grads_and_vars += minimize_and_clip(
                optimizer,
                policy.critic_loss[1],
                var_list=q_variables[half_cutoff:],
                clip_val=policy.config["grad_clip"])
        else:
            critic_grads_and_vars = minimize_and_clip(
                optimizer,
                policy.critic_loss[0],
                var_list=policy.model.q_variables(),
                clip_val=policy.config["grad_clip"])
        if not policy.model.constant_alpha:
            alpha_grads_and_vars = minimize_and_clip(
                optimizer,
                policy.alpha_loss,
                var_list=[policy.model.log_alpha],
                clip_val=policy.config["grad_clip"])
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.model.policy_variables())
        if policy.config["twin_q"]:
            q_variables = policy.model.q_variables()
            half_cutoff = len(q_variables) // 2
            base_q_optimizer, twin_q_optimizer = policy._critic_optimizer
            critic_grads_and_vars = base_q_optimizer.compute_gradients(
                policy.critic_loss[0], var_list=q_variables[:half_cutoff]
            ) + twin_q_optimizer.compute_gradients(
                policy.critic_loss[1], var_list=q_variables[half_cutoff:])
        else:
            critic_grads_and_vars = policy._critic_optimizer[
                0].compute_gradients(
                    policy.critic_loss[0], var_list=policy.model.q_variables())
        if not policy.model.constant_alpha:
            alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
                policy.alpha_loss, var_list=[policy.model.log_alpha])

    # save these for later use in build_apply_op
    policy._actor_grads_and_vars = [(g, v) for (g, v) in actor_grads_and_vars
                                    if g is not None]
    policy._critic_grads_and_vars = [(g, v) for (g, v) in critic_grads_and_vars
                                     if g is not None]
    if not policy.model.constant_alpha:
        policy._alpha_grads_and_vars = [(g, v) for (g, v) in alpha_grads_and_vars
                                        if g is not None]
    grads_and_vars = (policy._actor_grads_and_vars + policy._critic_grads_and_vars)
    if not policy.model.constant_alpha:
        grads_and_vars += policy._alpha_grads_and_vars
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
    if not policy.model.constant_alpha:
        alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
            policy._alpha_grads_and_vars,
            global_step=tf.train.get_or_create_global_step())
    actor_ops = [actor_apply_ops]
    if not policy.model.constant_alpha:
        actor_ops += [alpha_apply_ops]
    apply_ops = tf.group(actor_ops + critic_apply_ops)
    apply_ops = tf.group(apply_ops)
    return apply_ops


def stats(policy, train_batch):
    stat_dict = {
        # "policy_t": policy.policy_t,
        # "td_error": policy.td_error,
        "mean_td_error": tf.reduce_mean(policy.td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss),
        "alpha_loss": tf.constant(0) if policy.model.constant_alpha else tf.reduce_mean(policy.alpha_loss),
        "alpha_value": tf.reduce_mean(policy.alpha_value),
        "target_entropy": tf.constant(policy.target_entropy),
        "mean_q": tf.reduce_mean(policy.q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "min_q": tf.reduce_min(policy.q_t),
        # "current_entropy": policy.current_entropy,
        "mask_0": policy.mask_0,
        "mask_1": policy.mask_1,
    }

    # for i in range(policy.model.ensemble_size):
    #     stat_dict[f"alpha_{i}_value"] = tf.squeeze(tf.slice(policy.model.alpha, begin=[i, 0], size=[1, 1]))

    return stat_dict


class ActorCriticOptimizerMixin:
    def __init__(self, config):
        # create global step for counting the number of update operations
        self.global_step = tf.train.get_or_create_global_step()

        # use separate optimizers for actor & critic
        self._actor_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["actor_learning_rate"])
        self._critic_optimizer = [
            tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["critic_learning_rate"])
        ]
        if config["twin_q"]:
            self._critic_optimizer.append(
                tf.train.AdamOptimizer(learning_rate=config["optimization"][
                    "critic_learning_rate"]))
        self._alpha_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["entropy_learning_rate"])


def setup_early_mixins(policy, obs_space, action_space, config):
    ActorCriticOptimizerMixin.__init__(policy, config)


def setup_mid_mixins(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy, sac_actor_critic_loss)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)


SACEnsembleTFPolicy = build_tf_ensemble_policy(
    name="SACEnsembleTFPolicy",
    get_default_config=lambda: ray.rllib.agents.sac.sac_ensemble.DEFAULT_CONFIG,
    make_model=build_sac_ensemble_model,
    # make_model=build_sac_model,
    postprocess_fn=postprocess_trajectory,
    action_distribution_fn=get_distribution_inputs_and_class,
    loss_fn=sac_actor_critic_loss,
    stats_fn=stats,
    gradients_fn=gradients,
    apply_gradients_fn=apply_gradients,
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error},
    mixins=[
        TargetNetworkMixin, ActorCriticOptimizerMixin, ComputeTDErrorMixin
    ],
    before_init=setup_early_mixins,
    before_loss_init=setup_mid_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False)
