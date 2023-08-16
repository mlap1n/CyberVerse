import itertools

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import workspaces
from dm_control.utils import rewards

import numpy as np

from config import obs
from envs.base_task import BaseTask


class _VertexSitesMixin:
  """Mixin class that adds sites corresponding to the vertices of a box."""

  def _add_vertex_sites(self, box_geom_or_site):
    """Add sites corresponding to the vertices of a box geom or site."""
    offsets = (
        (-half_length, half_length) for half_length in box_geom_or_site.size)
    site_positions = np.vstack(list(itertools.product(*offsets)))
    if box_geom_or_site.pos is not None:
      site_positions += box_geom_or_site.pos
    self._vertices = []
    for i, pos in enumerate(site_positions):
      site = box_geom_or_site.parent.add(
          'site', name='vertex_' + str(i), pos=pos, type='sphere', size=[0.002],
          rgba=constants.RED, group=constants.TASK_SITE_GROUP)
      self._vertices.append(site)

  @property
  def vertices(self):
    return self._vertices


class _BoxWithVertexSites(props.Primitive, _VertexSitesMixin):
  """Subclass of `Box` with sites marking the vertices of the box geom."""

  def _build(self, *args, **kwargs):
    super()._build(*args, geom_type='box', **kwargs)
    self._add_vertex_sites(self.geom)


class _DuploWithVertexSites(props.Duplo, _VertexSitesMixin):
  """Subclass of `Duplo` with sites marking the vertices of its sensor site."""

  def _build(self, *args, **kwargs):
    super()._build(*args, **kwargs)
    self._add_vertex_sites(self.mjcf_model.find('site', 'bounding_box'))


class Lift(BaseTask):
    """A task where the goal is to elevate a prop."""
    def __init__(self,
                 arena: composer.Entity = None,
                 robot: composer.Entity = None,
                 prop: composer.Entity = None,
                 obs_settings: obs.ObservationSettings = None,
                 workspace: tuple = None,
                 control_timestep: float = None,
                 distance_to_lift: float = 0.3,
        ):
        """
        Initializes a new `Lift` task.

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
        self._prop = prop
        self._distance_to_lift = distance_to_lift
        self._arena.add_free_entity(prop)
        self._prop_placer = initializers.PropPlacer(
            props=[prop],
            position=distributions.Uniform(*workspace.prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=True,
            settle_physics=True)

        # Add sites for visualizing bounding boxes and target height.
        self._target_height_site = workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=(-1, -1, 0),
            upper=(1, 1, 0),
            rgba=constants.RED,
            name='target_height')
        workspaces.add_bbox_site(body=self.root_entity.mjcf_model.worldbody,
                                 lower=workspace.prop_bbox.lower,
                                 upper=workspace.prop_bbox.upper,
                                 rgba=constants.BLUE,
                                 name='prop_spawn_area')

    # @property
    # def task_observables(self):
    #     return self._task_observables

    def _get_height_of_lowest_vertex(self, physics):
        return min(physics.bind(self._prop.vertices).xpos[:, 2])

    def get_reward(self, physics):
        prop_height = self._get_height_of_lowest_vertex(physics)
        return rewards.tolerance(prop_height,
                                 bounds=(self._target_height, np.inf),
                                 margin=self._distance_to_lift,
                                 value_at_margin=0,
                                 sigmoid='long_tail')

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._prop_placer(physics, random_state)

        initial_prop_height = self._get_height_of_lowest_vertex(physics)
        self._target_height = self._distance_to_lift + initial_prop_height
        physics.bind(self._target_height_site).pos[2] = self._target_height
