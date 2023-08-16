import unittest
import matplotlib.animation as animation
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/max/projects/behavioral-cloning-hands/")

import numpy as np
from dm_control import composer
from dm_control.entities import props
from dm_control.manipulation.shared import constants
from config import obs

from envs.base_task import BaseTask
from agents.robot import Robot
from tasks.task_workspace import move_prop_workspace
from tasks.utils import lift, reach, move_prop
from envs.base_task import BaseTask
from envs.arenas import CustomArena
from tasks.move_prop import MoveProp, Prop
from dm_control import viewer

import cv2


def display_video(frames, framerate=30, name_file="video.gif"):
    print(frames[0].shape)
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    f = f"results/{name_file}"
    writergif = animation.PillowWriter(fps=framerate)
    anim.save(f, writer=writergif)


class CheckAllEnvs(unittest.TestCase):
    def setUp(self):
        #self.envs_cfg = json.loads("envs_cfg.json")
        self.seed = 50

    def t_check(self):
        bob = Robot("mjcf_models/mjmodel.xml")
        task = BaseTask(bob)
        env = composer.Environment(task, random_state=np.random.RandomState(self.seed))
        action_spec = env.action_spec()
        obs_spec = env.observation_spec()

        print("action:")
        print(action_spec, sep="\n")

        print("obs:")
        print(obs_spec, sep="\n")

        def sample_random_action():
            return env.random_state.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
            ).astype(action_spec.dtype, copy=False)

        frames = []
        timestep = env.reset()
        for _ in range(5000):
            timestep = env.step(sample_random_action())
            #print(timestep.observation)
            frames.append([env.physics.render(height=480, width=640)])
        all_frames = np.concatenate(frames, axis=0)
        display_video(all_frames, 30)

    def test_t(self):
        prop_names = [#{'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_Upper_3.xml'},
                      #{'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_SeeSaw_2.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_2.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_Upper_2.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Rug_01.xml'},
                      {'name': '', 'dir': 'mjcf_models/playroom/', 'free': True, 'path': 'obj3.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'StudyTable_Drawer.xml'},
                      {'name': '', 'dir': 'mjcf_models/playroom/', 'free': True, 'path': 'obj1.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Rug_02.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Painting_01.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_1.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Painting_02.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_3.xml'},
                      #{'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_SeeSaw_3.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Lamp_01.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Lamp_02.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'StudyTable_Slide.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'StudyTable.xml'},
                      #{'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_SeeSaw_1.xml'},
                      {'name': '', 'dir': 'mjcf_models/playroom/', 'free': True, 'path': 'BlockBin.xml'},
                      {'name': '', 'dir': 'mjcf_models/playroom/', 'free': True, 'path': 'obj2.xml'},
                    #   {'name': '', 'dir': 'mjcf_models/playroom/', 'free': False, 'path': 'Button_Upper_1.xml'}
        ]
        obs_settings = obs.VISION
        task = reach(obs_settings)
        env = composer.Environment(task, random_state=np.random.RandomState(self.seed))

        random_state = np.random.RandomState(self.seed)

        def sample_random_action():
            return env.random_state.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
            ).astype(action_spec.dtype, copy=False)

        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        time_step = env.reset()
        # frames = []
        # for _ in range(60):
        #     timestep = env.step(sample_random_action())
        #     frames.append([env.physics.render(height=480,
        #                                       width=640,
        #                                       camera_id="front_close")])
        # all_frames = np.concatenate(frames, axis=0)
        # display_video(all_frames, 30)
        viewer.launch(env)
        #cv2.imwrite(f"test.png", render)

if __name__ == "__main__":
    unittest.main()
