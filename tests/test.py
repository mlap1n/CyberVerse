"""Tests for `dm_control.manipulation_suite`."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import manipulation
import numpy as np
import cv2


flags.DEFINE_boolean(
    'fix_seed', True,
    'Whether to fix the seed for the environment\'s random number generator. '
    'This the default since it prevents non-deterministic failures, but it may '
    'be useful to allow the seed to vary in some cases, for example when '
    'repeating a test many times in order to detect rare failure events.')

FLAGS = flags.FLAGS

_FIX_SEED = None
_NUM_EPISODES = 5
_NUM_STEPS_PER_EPISODE = 10


class Test(parameterized.TestCase):
    """Tests run on all the tasks registered."""
    def _validate_observation(self, observation, observation_spec):
        self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
        for name, array_spec in observation_spec.items():
            array_spec.validate(observation[name])

    def _validate_reward_range(self, reward):
        self.assertIsInstance(reward, float)
        self.assertBetween(reward, 0, 1)

    def _validate_discount(self, discount):
        self.assertIsInstance(discount, float)
        self.assertBetween(discount, 0, 1)

    @parameterized.parameters(*manipulation.ALL)
    def test_task_runs(self, task_name):
        """Tests that the environment runs and is coherent with its specs."""
        seed = 99
        bob = Robot("mjcf_models/mjmodel.xml")
        task = BaseTask([bob])
        env = composer.Environment(task, random_state=np.random.RandomState(self.seed))

        prop = props.Duplo(
            observable_options=observations.make_options(
                obs_settings, observations.FREEPROP_OBSERVABLES))
        cradle = SphereCradle()

        env = Place(arena=arena, arm=arm, hand=hand, prop=prop,
                    obs_settings=obs_settings,
                    workspace=_WORKSPACE,
                    control_timestep=constants.CONTROL_TIMESTEP,
                    cradle=cradle)

        random_state = np.random.RandomState(seed)

        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        time_step = env.reset()
        render = env.physics.render(height=480, width=640)
        cv2.imwrite(f"{task_name}.png", render)


if __name__ == '__main__':
  absltest.main()
