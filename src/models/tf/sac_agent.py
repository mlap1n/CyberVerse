import tensorflow as tf

from tf_agents.train.utils import spec_utils
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.train.utils import spec_utils


def make_networks(env,
                  strategy,
                  actor_net_layer,
                  critic_net_layer,
                  ):
    with strategy.scope():
        obs_spec, act_spec, ts_spec = spec_utils.get_tensor_specs(env)
        critic_net = critic_network.CriticNetwork(
                (obs_spec, act_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=critic_net_layer,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')

        actor_net = actor_distribution_network.ActorDistributionNetwork(
        obs_spec,
        act_spec,
        fc_layer_params=actor_net_layer,
        continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))

    return actor_net, critic_net

def make_agent(env,
               strategy,
               actor_net,
               critic_net,
               train_step,
               ):
    # Now create the agent using the actor and critic networks
    with strategy.scope():
        obs_spec, act_spec, ts_spec = spec_utils.get_tensor_specs(env)

        critic_learning_rate = 3e-4
        actor_learning_rate = 3e-4
        alpha_learning_rate = 3e-4
        target_update_tau = 0.005
        target_update_period = 1
        gamma = 0.99
        reward_scale_factor = 1.0

        actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        critic_opt = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        alpha_opt = tf.keras.optimizers.Adam(learning_rate=alpha_learning_rate)

        agent = sac_agent.SacAgent(
                ts_spec,
                act_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=actor_opt,
                critic_optimizer=critic_opt,
                alpha_optimizer=alpha_opt,
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                train_step_counter=train_step)
    return agent
