from dm_control import mjcf
from dm_control import composer

from dm_control.manipulation.shared import constants
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from config import obs
from dm_control.manipulation.shared import workspaces
from dm_control.composer import define
from dm_control.composer.observation import observable
from dm_control.utils import rewards

import numpy as np

from envs.base_task import BaseTask


class Prop(composer.Entity):
    """A primitive MuJoCo geom prop."""

    def _build(self, path, is_free=False, name=None, is_site=False):
        """
        Initializes this prop.

        Args:
        name: (optional) A string, the name of this prop.
        """
        self._mjcf_root = mjcf.from_path(path)
        self._name = name if name is not None else str(self.mjcf_model.worldbody.body[0].name)
        self._is_free = is_free

        self._linear_velocity = self._mjcf_root.sensor.add(
            'framelinvel', name='linear_velocity', objtype='body',
            objname=self._name)

        self._angular_velocity = self._mjcf_root.sensor.add(
            'frameangvel', name='angular_velocity', objtype='body',
            objname=self._name)

        if is_site:
            self._make_target_site(self.frame_obj, True)
            self._target_site = self.mjcf_model.find('site', 'target_site')

    def _build_observables(self):
        return PropObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def frame_obj(self):
        return self.mjcf_model.worldbody.body[0]

    @property
    def name(self):
        return self._name

    @property
    def is_free(self):
        return self._is_free

    @property
    def linear_velocity(self):
        """Ground truth velocity sensor."""
        return self._linear_velocity

    @property
    def angular_velocity(self):
        """Ground truth angular velocity sensor."""
        return self._angular_velocity

    @property
    def target_site(self):
        return self._target_site

    def _make_target_site(self, parent_entity, visible):
        return workspaces.add_target_site(
            body=parent_entity,
            radius=0.1, visible=visible,
            rgba=constants.RED, name='target_site')


class PropObservables(composer.Observables):
    """Observables for the `Pedestal` prop."""


class MoveProp(BaseTask):
    """
    Place the prop on top of another fixed prop held up by a pedestal.
    """
    def __init__(self,
                 arena: composer.Entity = None,
                 robot: composer.Entity = None,
                 items: composer.Entity=None,
                 props: composer.Entity = None,
                 target_prop: composer.Entity = None,
                 target_site: bool = None,
                 obs_settings: obs.ObservationSettings = None,
                 workspace: tuple = None,
                 control_timestep: float = None):
        """
        Initializes a new `MoveProp` task.

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
        self._items = items
        self._props = props
        self._target_prop = target_prop
        self._target_radius = 0.1

        if self._items:
            self._items_frame = [self._arena.add_entity(i) for i in self._items]

        if self._target_prop:
            self._target_prop_frame = [self._arena.add_entity(i) for i in self._target_prop]

        if self._props:
            self._props_frame = [self._arena.add_entity(i) for i in self._props]

            self._prop_placer = initializers.PropPlacer(
                props=self._props,
                position=distributions.Uniform(*workspace.prop_bbox),
                quaternion=workspaces.uniform_z_rotation,
                settle_physics=True,
                max_attempts_per_prop=50)

            # Add sites for visual debugging.
            workspaces.add_bbox_site(
                body=self.root_entity.mjcf_model.worldbody,
                lower=workspace.prop_bbox.lower,
                upper=workspace.prop_bbox.upper,
                rgba=constants.BLUE, name='prop_spawn_area')

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        if self._props:
            self._prop_placer(physics, random_state)

    def get_reward(self, physics):
        target = physics.bind(self._target_prop[0].target_site).xpos
        obj = physics.bind(self._props_frame[0]).xpos
        hand_pos = physics.bind(self._robot._hand_center_point).xpos

        hand_to_obj_dist = np.linalg.norm(obj - hand_pos)
        grasp = rewards.tolerance(hand_to_obj_dist,
                                  bounds=(0, self._target_radius),
                                  margin=self._target_radius,
                                  sigmoid='long_tail')

        obj_to_tgt_dist = np.linalg.norm(obj - target)
        in_place = rewards.tolerance(obj_to_tgt_dist,
                                     bounds=(0, self._target_radius),
                                     margin=self._target_radius,
                                     sigmoid='long_tail')

        tcp_to_tgt_dist = np.linalg.norm(hand_pos - target)
        hand_away = rewards.tolerance(tcp_to_tgt_dist,
                                      bounds=(4*self._target_radius, np.inf),
                                      margin=3*self._target_radius,
                                      sigmoid='long_tail')
        in_place_weight = 10.
        grasp_or_hand_away = grasp * (1 - in_place) + hand_away * in_place
        return (grasp_or_hand_away + in_place_weight * in_place) / (1 + in_place_weight)
