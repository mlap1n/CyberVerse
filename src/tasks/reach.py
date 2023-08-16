from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import workspaces
from dm_control.utils import rewards

import numpy as np

from config import obs
from envs.base_task import BaseTask


class Reach(BaseTask):
    """Bring the hand close to a target prop or site."""

    def __init__(self,
                 arena: composer.Entity = None,
                 robot: composer.Entity = None,
                 prop: composer.Entity = None,
                 obs_settings: obs.ObservationSettings = None,
                 workspace: tuple = None,
                 control_timestep: float = None,
                 target_radius: float = 0.05
        ):
        """
        Initializes a new `Reach` task.

        :param arena: `composer.Entity`:
        :param prop: `composer.Entity`:
        :param obs_settings: `observations.ObservationSettings`:
        :param workspace: A `_PlaceWorkspace`:
        :param control_timestep: `float`: specifying the control timestep in seconds.
        """
        super().__init__(arena=arena,
                         robot=robot,
                         obs_settings=obs_settings,
                         workspace=workspace,
                         control_timestep=control_timestep)
        target_pos_distribution = distributions.Uniform(*workspace.target_bbox)
        self._prop = prop
        self._target_radius = target_radius
        if prop:
            self._make_target_site(parent_entity=prop, visible=False)
            self._target = self._arena.add_free_entity(prop)
            self._prop_placer = initializers.PropPlacer(
                props=[prop],
                position=target_pos_distribution,
                quaternion=workspaces.uniform_z_rotation,
                settle_physics=True)
        else:
            self._target = self._make_target_site(parent_entity=self._arena, visible=True)
            self._target_placer = target_pos_distribution

            obs = observable.MJCFFeature('pos', self._target)
            obs.configure(**obs_settings.prop_pose._asdict())
            self._task_observables['target_position'] = obs

            # Add sites for visualizing the prop and target bounding boxes.
            workspaces.add_bbox_site(
                body=self.root_entity.mjcf_model.worldbody,
                lower=workspace.target_bbox.lower, upper=workspace.target_bbox.upper,
                rgba=constants.BLUE, name='target_spawn_area')

    def _make_target_site(self, parent_entity, visible):
        return workspaces.add_target_site(
            body=parent_entity.mjcf_model.worldbody,
            radius=self._target_radius, visible=visible,
            rgba=constants.RED, name='target_site')

    @property
    def task_observables(self):
        return self._task_observables

    def get_reward(self, physics):
        hand_pos = physics.bind(self._robot._hand_center_point).xpos
        target_pos = physics.bind(self._target).xpos
        distance = np.linalg.norm(hand_pos - target_pos)
        return rewards.tolerance(distance,
                                 bounds=(0, self._target_radius),
                                 margin=self._target_radius,
                                 sigmoid="long_tail")

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        if self._prop:
            self._prop_placer(physics, random_state)
        else:
            physics.bind(self._target).pos = (
                self._target_placer(random_state=random_state))
