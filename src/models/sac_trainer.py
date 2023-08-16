import reverb

from tf_agents.utils import common
from tf_agents.replay_buffers import reverb_replay_buffer
import os

import numpy as np

from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.metrics import py_metrics
from tf_agents.train import triggers
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.train.utils import strategy_utils
from tf_agents.replay_buffers import reverb_utils
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train.utils import train_utils

from src.models.sac_agent import make_agent, make_networks
from src.utils.post_processing import save_video
from src.utils.logger import Logger


def get_eval_metrics():
    eval_metrics = [
        py_metrics.AverageReturnMetric(),
        py_metrics.AverageEpisodeLengthMetric(),
    ]
    return eval_metrics


def get_step_metrics():
    step_metrics = [
        py_metrics.NumberOfEpisodes(),
        py_metrics.EnvironmentSteps(),
    ]
    return step_metrics


def get_train_metrics(batch_size):
    train_metrics = [
        py_metrics.AverageReturnMetric(batch_size=batch_size),
        py_metrics.AverageEpisodeLengthMetric(batch_size=batch_size),
    ]
    return train_metrics


class SACTrainer:
    """
    Trainer class for SAC to update policies.
    """
    def __init__(self, env, eval_env, args=None, cfg=None):
        self.args = args
        self.name_task = "site"
        self.logger = Logger()

        self.strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)
        with self.strategy.scope():
            self.train_step = train_utils.create_train_step()

            self.actor_net, self.critic_net = make_networks(
                env=env,
                strategy=self.strategy,
                actor_net_layer=self.args.actor_net_layer,
                critic_net_layer=self.args.critic_net_layer,
            )
            self.agent = make_agent(
                actor_net=self.actor_net,
                critic_net=self.critic_net,
                env=env,
                strategy=self.strategy,
                train_step=self.train_step,
            )
            self.agent.initialize()

        self.env = env
        self.eval_env = eval_env

        self.policy = py_tf_eager_policy.PyTFEagerPolicy(
            self.agent.policy,
            use_tf_function=True,
        )
        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            self.agent.collect_policy,
            use_tf_function=True,
        )
        self.random_policy = random_py_policy.RandomPyPolicy(
            self.env.time_step_spec(),
            self.env.action_spec(),
        )

        if self.args.verbose:
            self.logger.info("Observation spec: {} \n".format(self.env.observation_spec()))
            self.logger.info("Action spec: {} \n".format(self.env.action_spec()))
            self.logger.info("policy: {} \n".format(self.agent.policy))

        rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
            samples_per_insert=3.0,
            min_size_to_sample=3,
            error_buffer=3.0,
        )

        table_name = 'uniform_table'
        table = reverb.Table(
            table_name,
            max_size=self.args.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )

        self.reverb_server = reverb.Server([table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            sequence_length=2,
            table_name=table_name,
            local_server=self.reverb_server,
        )

        dataset = self.replay_buffer.as_dataset(
            sample_batch_size=self.args.batch_size, num_steps=2).prefetch(50)
        self.dataset_fn = lambda: dataset

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            self.replay_buffer.py_client,
            table_name,
            sequence_length=2,
            stride_length=1,
        )

        self.eval_metrics = get_eval_metrics()
        self.step_metrics = get_step_metrics()
        self.train_metrics = get_train_metrics(self.env.batch_size)
        self.all_train_metrics = self.step_metrics + self.train_metrics

        self.agent.train = common.function(self.agent.train, autograph=True)
        self.agent.update_normalizers_in_train = False

        self.initial_collect_actor = actor.Actor(
            self.env,
            self.random_policy,
            self.train_step,
            steps_per_run=self.args.num_warmup_iter,
            observers=[self.rb_observer]+self.step_metrics,
        )

        self.collect_actor = actor.Actor(
            self.env,
            self.collect_policy,
            self.train_step,
            steps_per_run=1,
            metrics=actor.collect_metrics(10),
            summary_dir=os.path.join(self.args.tb_dir, learner.TRAIN_DIR),
            observers=[self.rb_observer]+self.all_train_metrics,
        )

        self.eval_actor = actor.Actor(
            self.eval_env,
            self.policy,
            self.train_step,
            episodes_per_run=self.args.num_eval_episodes,
            metrics=actor.eval_metrics(self.args.num_eval_episodes),
            summary_dir=os.path.join(self.args.tb_dir, 'eval'),
            observers=[self.rb_observer]+self.eval_metrics,
        )

        saved_model_dir = os.path.join(self.args.checkpoints_dir, learner.POLICY_SAVED_MODEL_DIR)

        learning_triggers = [
            triggers.PolicySavedModelTrigger(
                saved_model_dir,
                self.agent,
                self.train_step,
                interval=self.args.save_interval,
            ),
            triggers.StepPerSecondLogTrigger(self.train_step, interval=1000),
        ]

        self.agent_learner = learner.Learner(
            self.args.tb_dir,
            self.train_step,
            self.agent,
            self.dataset_fn,
            triggers=learning_triggers,
            strategy=self.strategy,
        )

    def eval(self, step):
        self.logger.info("starting the evaluation stage...")
        self.eval_actor.run()
        results = {}
        for metric in self.eval_actor.metrics:
            results[metric.name] = metric.result()

        eval_results = (', ').join('{} = {:.6f}'.format(name, result) for name, result in results.items())
        self.logger.info('step = {0}: {1}'.format(step, eval_results))

    def make_video(self):
        self.logger.info("video recording...")
        time_step = self.eval_env.reset()
        state = self.policy.get_initial_state(self.eval_env.batch_size)
        frames = []
        while not time_step.is_last():
            policy_step = self.policy.action(time_step, state)
            state = policy_step.state
            time_step = self.eval_env.step(policy_step.action)
            frames.append([self.eval_env.render()])
        all_frames = np.concatenate(frames, axis=0)
        name = self.name_task + "_" + str(self.agent_learner.train_step_numpy) + ".gif"
        save_video(all_frames, 30, name)
        self.logger.info(f"file saved: results/{name}")
        self.logger.info("done!")

    def train(self):
        self.agent.train_step_counter.assign(0)
        self.logger.info("start of agent training...")

        self.make_video()
        self.eval(0)
        self.initial_collect_actor.run()

        for _ in range(self.args.num_train_iter):
            self.collect_actor.run()
            loss_info = self.agent_learner.run(iterations=1)

            step = self.agent_learner.train_step_numpy

            if self.args.eval_interval and step % self.args.eval_interval == 0:
                self.eval(step)
                self.make_video()

            if self.args.train_log_interval and step % self.args.train_log_interval == 0:
                self.logger.info('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

        self.rb_observer.close()
        self.reverb_server.stop()

        self.logger.info("================")
        self.logger.info("training is done")
        self.logger.info("================")
