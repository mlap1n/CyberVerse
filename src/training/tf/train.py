from typing import Callable, Optional
import argparse
import warnings

from tf_agents.environments import tf_py_environment
import tensorflow as tf
from tf_agents.environments import suite_pybullet

import src.utils.options as options
from src.models.tf.sac_trainer import SACTrainer

def main(modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None):
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.autograph.set_verbosity(0)

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = options.get_train_parser()
    args = options.parse_args(parser, modify_parser=modify_parser)
    env_name = "MinitaurBulletEnv-v0"
    env = suite_pybullet.load(env_name)

    env.reset()

    print('Observation Spec:')
    print(env.time_step_spec().observation)
    print('Action Spec:')
    print(env.action_spec())

    train_env = suite_pybullet.load(env_name)
    eval_env = suite_pybullet.load(env_name)

    # tf_train_env = tf_py_environment.TFPyEnvironment(train_env)
    # tf_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

    trainer = SACTrainer(
        args=args,
        env=train_env,
        eval_env=eval_env,
    )

    trainer.train()

if __name__ == "__main__":
    main()
