"""Tools for adding custom cameras to the arena."""

import collections

from dm_control.composer.observation import observable


CameraSpec = collections.namedtuple('CameraSpec', ['name', 'pos', 'xyaxes'])

# Custom cameras that may be added to the arena for particular tasks.
front_close = CameraSpec(
    name='front_close',
    pos=(2.0, -0.5, 1.),
    xyaxes=(0., 1., 0., 0., 0, 0.75)
)
front_far = CameraSpec(
    name='front_far',
    pos=(4.0, -0.5, 1.),
    xyaxes=(0., 1., 0., 0., 0, 0.75)
)


left_close = CameraSpec(
    name='left_close',
    pos=(-2, -1, 1.85),
    xyaxes=(0, -1, 0, 1, 0, 2)
)
left_far = CameraSpec(
    name='left_far',
    pos=(-2, -1.5, 1.85),
    xyaxes=(0, -1, 0, 1, 0, 2)
)


right_close = CameraSpec(
    name='right_close',
    pos=(-1, -3, 1.85),
    xyaxes=(0., 1., 0., -0.7, 0., 0.75)
)
right_far = CameraSpec(
    name='right_far',
    pos=(-3, -5, 1.85),
    xyaxes=(0., 1., 0., -0.7, 0., 0.75)
)


def add_camera_observables(entity, obs_settings, *camera_specs):
    """Adds cameras to an entity's worldbody and configures observables for them.

    Args:
        entity: A `composer.Entity`.
        obs_settings: An `observations.ObservationSettings` instance.
        *camera_specs: Instances of `CameraSpec`.

    Returns:
        A `collections.OrderedDict` keyed on camera names, containing pre-configured
        `observable.MJCFCamera` instances.
    """
    obs_dict = collections.OrderedDict()
    for spec in camera_specs:
        camera = entity.mjcf_model.worldbody.add('camera', **spec._asdict())
        obs = observable.MJCFCamera(camera)
        obs.configure(**obs_settings.camera._asdict())
        obs_dict[spec.name] = obs
    return obs_dict
