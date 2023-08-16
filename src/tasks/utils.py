from dm_control.manipulation.shared import constants
from dm_control import composer
from dm_control.entities import props
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import tags

from config import obs
from agents.robot import Robot
from envs.arenas import CustomArena
from tasks.move_prop import MoveProp, Prop
from tasks.reach import Reach
from tasks.lift import Lift, _DuploWithVertexSites, _BoxWithVertexSites
import tasks.task_workspace as tws


def move_prop(obs_settings):
    """
    Configure and instantiate a `MoveProp` task.

    :param obs_settings: `observations.ObservationSettings`:

    Returns:
        An instance of `move_prop.MoveProp`.
    """
    item_names = [
        # {'name': '', 'dir': 'mjcf_models/playroom/',
        #  'free': True, 'path': 'obj3.xml'},
        # {'name': '', 'dir': 'mjcf_models/playroom/',
        #  'free': True, 'path': 'obj1.xml'},
        # {'name': '', 'dir': 'mjcf_models/playroom/',
        #  'free': True, 'path': 'BlockBin.xml'},
        {'name': '', 'dir': 'mjcf_models/playroom/',
         'free': True, 'path': 'obj2.xml'},
    ]
    target_names = [
        {'name': '', 'dir': 'mjcf_models/playroom/',
         'free': False, 'path': 'BlockBin.xml'},
    ]
    prop_names = [
        {'name': '', 'dir': 'mjcf_models/playroom/',
         'free': True, 'path': 'obj1.xml'},
    ]

    robot_obs_settings = obs.make_options(obs_settings, obs.ROBOT_OBSERVABLES)
    bob = Robot("mjcf_models/mjmodel.xml")
    kwargs = {
        "xml_path": "mjcf_models/playroom/playroom.xml"
    }

    items = [Prop(f"{i['dir']}{i['path']}", is_free=i['free'])
             for i in item_names]
    targets = [Prop(f"{i['dir']}{i['path']}", is_free=i['free'], is_site=True)
               for i in target_names]
    props = [Prop(f"{i['dir']}{i['path']}", is_free=i['free'])
             for i in prop_names]

    task = MoveProp(arena=CustomArena(**kwargs),
                    robot=bob,
                    items=items,
                    target_prop=targets,
                    props=props,
                    obs_settings=obs_settings,
                    workspace=tws.move_prop_workspace,
                    control_timestep=constants.CONTROL_TIMESTEP)
    return task


def reach(obs_settings, use_site=True):
    """
    Configure and instantiate a `Reach` task.

    :param obs_settings: `observations.ObservationSettings`:
    :param use_site: `bool`: if True then the target will be a fixed site, otherwise
    it will be a moveable brick.

    Returns:
        An instance of `reach.Reach`.
    """
    robot_obs_settings = obs.make_options(obs_settings, obs.ROBOT_OBSERVABLES)
    bob = Robot("mjcf_models/mjmodel.xml")
    kwargs = {
        "xml_path": "mjcf_models/playroom/playroom.xml"
    }
    if use_site:
        workspace = tws.reach_site_workspace
        prop = None
    else:
        workspace = tws.brick_workspace
        prop = props.Brick(observable_options=obs.make_options(
           obs_settings, obs.FREEPROP_OBSERVABLES))

    task = Reach(arena=CustomArena(**kwargs),
                 robot=bob,
                 prop=prop,
                 obs_settings=obs_settings,
                 workspace=workspace,
                 control_timestep=constants.CONTROL_TIMESTEP)
    return task


def lift(obs_settings, prop_name):
    """
    Configure and instantiate a `Lift` task.

    :param obs_settings: `observations.ObservationSettings`:
    :param prop_name: `str`: the name of the prop to be lifted

    Returns:
        An instance of `lift.Lift`.

    Raises:
        ValueError: If `prop_name` is neither 'brick' nor 'box'.
    """
    robot_obs_settings = obs.make_options(obs_settings, obs.ROBOT_OBSERVABLES)
    bob = Robot("mjcf_models/mjmodel.xml")
    kwargs = {
        "xml_path": "mjcf_models/playroom/playroom.xml"
    }

    if prop_name == 'brick':
        workspace = tws._duplo_workspace
        prop = _DuploWithVertexSites(
            observable_options=obs.make_options(
                obs_settings, obs.FREEPROP_OBSERVABLES))
    elif prop_name == 'box':
        workspace = tws.lift_box_workspace
        prop = _BoxWithVertexSites(
            size=[tws._box_size] * 3,
            observable_options=obs.make_options(
                obs_settings, obs.FREEPROP_OBSERVABLES))
        prop.geom.mass = tws._box_mass
    else:
        raise ValueError('`prop_name` must be either \'brick\' or \'box\'.')

    task = Lift(arena=CustomArena(**kwargs),
                robot=bob,
                prop=prop,
                obs_settings=obs_settings,
                workspace=workspace,
                control_timestep=constants.CONTROL_TIMESTEP)
    return task


@registry.add(tags.VISION, tags.EASY)
def reach_site_vision():
  return reach(obs_settings=obs.VISION, use_site=True)

@registry.add(tags.VISION)
def lift_brick_vision():
  return lift(obs_settings=obs.VISION, prop_name='brick')

@registry.add(tags.VISION)
def lift_large_box_vision():
  return lift(obs_settings=obs.VISION, prop_name='box')

@registry.add(tags.VISION)
def place_brick_vision():
  return move_prop(obs_settings=obs.VISION)
