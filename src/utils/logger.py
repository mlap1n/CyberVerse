import os
import logging
from datetime import datetime
from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import tensorflow as tf


class Logger:
    def __init__(self,
                 log_interval: int = 200,
                 log_name: str = "logfile.log",
                 tb_name: str = "",
                 log_dir: str = "logs",
                 tb_dir: str = "tensorboard",
                 use_tb: bool = False,
                 use_wandb: bool = False,
                 ):
        """
        :param log_interval (int): log progress every N steps
        :param log_name (str): file to save log data
        :param tb_name (str): file to save tensorboard data
        :param log_dir (str): path to log_name
        :param tb_dir (str): path to tb_name
        :param use_tb (bool): use tensorboard or not
        """
        self.time_ext = datetime.now().strftime("%Y%m%d-%H%M")
        self.log_interval = log_interval
        self.use_tb = use_tb

        # create logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # create file handler
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, log_name))
        fh.setLevel(logging.DEBUG)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add ch and fh to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        if self.use_tb:
            tb_dir = PurePath(__file__).parts[0] if tb_dir is None else dir
            self.tb_dir = Path(tb_dir) / "tensorboard" / tb_name.format(self.time_ext)
            Path(self.tb_dir).mkdir(parents=True, exist_ok=True)
            self.tb_writer = tf.summary.create_file_writer(self.tb_dir)

    def set_lvl(self):
        pass

    def tb_write(self, val, step, title="Average Training Reward"):
        with self.tb_writer.as_default():
            tf.summary.scalar(title, val, step=step)

    def info(self, info):
        self.logger.info(info)

    def _end_train(self, step):
        print("Reached the end of training with {} training steps".format(step))

    def plot_eval(self):
        """ Plots average evaluation returns for an agent over time.

        Creates a matplotlib plot for the average evaluation returns for each
        agent over time.  This file is saved to the 'policy_save_dir', as
        specified in the constructor.
        """
        if self.use_separate_agents:  # Make graphs for N separate agents
            for car_id in range(self.num_agents):
                # TODO(rms): How to plot for multiple agents?
                xs = [i * self.eval_interval for
                      i in range(len(self.eval_returns[car_id]))]
                plt.plot(xs, self.eval_returns[car_id])
                plt.xlabel("Training epochs")
                plt.ylabel("Average Return")
                plt.title("Average Returns as a Function "
                          "of Training (Agent {})".format(car_id))
                save_path = os.path.join(self.policy_save_dir,
                                         "eval_returns_agent_{}"
                                         ".png".format(car_id))
                plt.savefig(save_path)
                print("Created plot of returns for agent {}...".format(car_id))

        else:
            xs = [i * self.eval_interval for i in range(len(self.eval_returns))]
            plt.plot(xs, self.eval_returns)
            plt.xlabel("Training epochs")
            plt.ylabel("Average Return")
            plt.title("Average Returns as a Function of Training")
            save_path = os.path.join(self.policy_save_dir, "eval_returns.png")
            plt.savefig(save_path)
            print("CREATED PLOT OF RETURNS")
