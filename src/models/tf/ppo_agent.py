import tempfile

import tensorflow as tf

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.train import actor
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.agents import PPOAgent
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.agents import PPOClipAgent


class ImageLayer(tf.keras.layers.Layer):
    def __init__(self, input=(84, 84, 3), **kwargs):
        super().__init__(**kwargs)
        self.reshape = tf.keras.layers.Reshape(input)
        self.rescale = tf.keras.layers.Rescaling(scale=1/255.0)
        self.cv_model = tf.keras.applications.ResNet50(include_top=False,
                                                       weights="imagenet")
        self.cv_model.trainable = False
        self.linear = tf.keras.layers.Dense(64)

    def call(self, x):
        x = self.reshape(x)
        x = self.rescale(x)
        x = self.cv_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.linear(x)
        return x


def make_networks(env,
                  strategy,
                  actor_net_layer,
                  value_net_layer,
                  lstm_size,
                  dropout_layer_params,
                  use_cnn):
    obs_spec, act_spec, ts_spec = spec_utils.get_tensor_specs(env)

    if use_cnn:
        #cv_model = tf.keras.applications.ResNet50(include_top=False,
        #                                               weights="imagenet")
        #cv_model.trainable = False

        with strategy.scope():
            preprocessing_layers = {
                 'com_velocity': tf.keras.models.Sequential([
                     tf.keras.layers.Flatten(),]),
                 'velocity': tf.keras.models.Sequential([
                     tf.keras.layers.Flatten(),]),

                 'torso_vertical': tf.keras.models.Sequential([
                     tf.keras.layers.Flatten(),]),
                 'extremities': tf.keras.models.Sequential([
                     tf.keras.layers.Flatten(),]),
                 'head_height': tf.keras.models.Sequential([
                     tf.keras.layers.Flatten(),]),
                 'joint_angles': tf.keras.models.Sequential([
                     tf.keras.layers.Flatten(),]),
            }

            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

            actor_net = ActorDistributionRnnNetwork(obs_spec,
                                                    act_spec,
                                                    preprocessing_layers=preprocessing_layers,
                                                    preprocessing_combiner=preprocessing_combiner,
                                                    input_fc_layer_params=actor_net_layer,
                                                    lstm_size=lstm_size,
                                                    output_fc_layer_params=(128,),
            )
            value_net = ValueRnnNetwork(obs_spec,
                                        preprocessing_layers=preprocessing_layers,
                                        preprocessing_combiner=preprocessing_combiner,
                                        input_fc_layer_params=value_net_layer,
                                        lstm_size=lstm_size,
                                        output_fc_layer_params=(128,),
            )

            #actor_net = ActorDistributionNetwork(obs_spec,
            #                                    act_spec,
            #                                    preprocessing_layers=preprocessing_layers,
            #                                    preprocessing_combiner=preprocessing_combiner,
            #                                    fc_layer_params=actor_net_layer,
            #)
            #value_net = ValueNetwork(obs_spec,
            #                        preprocessing_layers=preprocessing_layers,
            #                        preprocessing_combiner=preprocessing_combiner,
            #                        fc_layer_params=value_net_layer,
            #)
    return actor_net, value_net

def make_agent(env,
               strategy,
               actor_net,
               critic_net,
               lr=4e-4,
               dropout_layer_params=(0.15,0.15),
               use_cnn=True):
    # Now create the agent using the actor and critic networks
    obs_spec, act_spec, ts_spec = spec_utils.get_tensor_specs(env)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    with strategy.scope():
        global_step = tf.compat.v1.train.get_or_create_global_step()

        agent = PPOClipAgent(ts_spec,
                            act_spec,
                            optimizer=optimizer,
                            actor_net=actor_net,
                            value_net=critic_net,
                            train_step_counter=global_step,
        )
    return agent
