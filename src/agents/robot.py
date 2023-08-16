import collections

import numpy as np

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.mujoco import wrapper as mj_wrapper
from dm_control.mujoco.wrapper import mjbindings

from config import obs


ROBOT_OFFSET = (0.0, 0.0, 0.12)
_INVALID_JOINTS_ERROR = (
    'All non-hinge joints must have limits. Model contains the following '
    'non-hinge joints which are unbounded:\n{invalid_str}')


def make_robot(obs_settings):
    """
    :param obs_settings: `observations.ObservationSettings` instance.
    """
    return Robot(observable_options=obs.make_options(obs_settings,
                                                     obs.ROBOT_OBSERVABLES))


class AgentObservables(composer.Observables):
    @composer.observable
    def joints_torque(self):
        def get_torques(physics):
            torques = physics.bind(self._entity.joint_torque_sensors).sensordata
            joint_axes = physics.bind(self._entity.joints).axis
            return np.einsum('ij,ij->i', torques.reshape(-1, 3), joint_axes)
        return observable.Generic(get_torques)

    @composer.observable
    def joints_vel(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)

    @composer.observable
    def sensors_touch_fingertips(self):
        touch_sensors = self._entity.mjcf_model.sensor.touch
        touch_sensors = [i for i in touch_sensors if "fingertip" in i.name]
        return observable.MJCFFeature('sensordata', touch_sensors)

    @composer.observable
    def sensors_touch_fingerpads(self):
        touch_sensors = self._entity.mjcf_model.sensor.touch
        touch_sensors = [i for i in touch_sensors if "fingerpad" in i.name]
        return observable.MJCFFeature('sensordata', touch_sensors)

    @composer.observable
    def sensors_gyro(self):
        return observable.MJCFFeature('sensordata',
                                      self._entity.mjcf_model.sensor.gyro)

    @composer.observable
    def sensors_accelerometer(self):
        return observable.MJCFFeature('sensordata',
                                      self._entity.mjcf_model.sensor.accelerometer)

    @composer.observable
    def egocentric_camera(self):
        options = mj_wrapper.MjvOption()
        width, height = 8, 8
        return observable.MJCFCamera(self._entity.egocentric_camera,
                                     width=width, height=height, scene_option=options)

    @composer.observable
    def actuator_activation(self):
        return observable.MJCFFeature('act',
                                    self._entity.mjcf_model.find_all('actuator'))

    @property
    def proprioception(self):
        return [
            self.joints_vel,
            self.actuator_activation,
        ] + self._collect_from_attachments('proprioception')


_GRIP_SITE = "r_gripper_palm_site"

class Robot(composer.Robot):
    def _build(self, path=None, obs_settings=None):
        if path:
            assert path is not None, "scene_path can't be None if default_scene is False"
            self._model = mjcf.from_path(path)
        else:
            self._model = mjcf.RootElement()

        self._joint_torque_sensors = [self._add_torque_sensor(joint) for joint in self.joints]
        self.mjcf_model.actuator.motor.clear()

        self._grip_site = self.mjcf_model.find('site', _GRIP_SITE)


    def _build_observables(self):
        return AgentObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def joints(self):
        """List of joint elements"""
        return self._model.find_all('joint')

    @property
    def actuators(self):
        return tuple(self._model.find_all('actuator'))

    @property
    def joint_torque_sensors(self):
        return self._joint_torque_sensors

    @property
    def _hand_center_point(self):
        return self._grip_site

    def _get_joint_pos_sampling_bounds(self, physics):
        """Returns lower and upper bounds for sampling arm joint positions.
        Args:
        physics: An `mjcf.Physics` instance.
        Returns:
        A (2, num_joints) numpy array containing (lower, upper) position bounds.
        For hinge joints without limits the bounds are defined as [0, 2pi].
        Raises:
        RuntimeError: If the model contains unlimited joints that are not hinges.
        """
        bound_joints = physics.bind(self.joints)
        limits = np.array(bound_joints.range, copy=True)
        is_hinge = bound_joints.type == mjbindings.enums.mjtJoint.mjJNT_HINGE
        is_limited = bound_joints.limited.astype(bool)
        invalid = ~is_hinge & ~is_limited
        if any(invalid):
            invalid_str = '\n'.join(str(self.joints[i]) for i in np.where(invalid)[0])
            raise RuntimeError(_INVALID_JOINTS_ERROR.format(invalid_str=invalid_str))
        # For unlimited hinges we sample positions between 0 and 2pi.
        limits[is_hinge & ~is_limited] = 0., 2*np.pi
        return limits.T

    def randomize_arm_joints(self, physics, random_state):
        """Randomizes the qpos of all arm joints.
        The ranges of qpos values is determined from the MJCF model.
        Args:
        physics: A `mujoco.Physics` instance.
        random_state: An `np.random.RandomState` instance.
        """
        lower, upper = self._get_joint_pos_sampling_bounds(physics)
        physics.bind(self.joints).qpos = random_state.uniform(lower, upper)

    def rsi(self, physics, close_factors):
        if not isinstance(close_factors, collections.abc.Iterable):
            close_factors = (close_factors,) * len(self.joints)

        for joint, finger_factor in zip(self.joints, close_factors):
            joint_mj = physics.bind(joint)
            min_value, max_value = joint_mj.range
            joint_mj.qpos = min_value + (max_value - min_value) * finger_factor
        physics.after_reset()

        # Set target joint torque to zero.
        physics.bind(self.actuators).ctrl = 0

    def _add_torque_sensor(self, joint):
        site = joint.parent.add(
            'site', size=[1e-3], group=composer.SENSOR_SITES_GROUP,
            name=joint.name+'_site')
        return joint.root.sensor.add('torque', site=site, name=joint.name+'_torque')

    @composer.cached_property
    def egocentric_camera(self):
        return self._model.find('camera', 'egocentric_camera')
