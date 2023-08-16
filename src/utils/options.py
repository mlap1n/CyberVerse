import argparse


def get_parser(desc, default_task=""):
    pass


def get_preprocessing_parser(default_task):
    parser = get_parser("Preprocessing", default_task)
    return parser


def get_eval_parser(parser):
    #parser = get_parser("Validation", default_task)
    parser.add_argument("--eval_interval",
                        default=10_000,
                        type=int,
                        help="Evaluate every time epoch % eval_interval = 0.",
    )
    parser.add_argument("--num_eval_episodes",
                        default=20,
                        type=int,
                        help="Evaluate over eval_episodes evaluation episodes.",
    )
    return parser


def get_test_parser(default_task):
    parser = get_parser("Test", default_task)
    return parser


def get_marl_parser(parser):
    parser.add_argument("--self_play",
                        required=False,
                        action="store_true",
                        help="to use a single master PPO agent.",
    )
    parser.add_argument("--separate_agents",
                        required=False,
                        action="store_true",
                        help="to use a N PPO agents.",
    )
    parser.add_argument("--num_agents",
                        default=1,
                        type=int,
                        help="Number of agents in the env.",
    )
    return parser


def get_logging_parser(parser):
    parser.add_argument("--save_interval",
                        default=10_000,
                        type=int,
                        help="Save policies every time epoch % eval_interval = 0.",
    )
    parser.add_argument("--train_log_interval",
                        default=2000,
                        type=int,
                        help="Log results every time epoch % eval_interval = 0.",
    )
    parser.add_argument("--tb",
                        required=False,
                        action="store_true",
                        help="Log with tensorboard as well.",
    )
    parser.add_argument("--wandb",
                        required=False,
                        action="store_true",
                        help="Log with wandb as well.",
    )
    parser.add_argument("--logs",
                        required=False,
                        action="store_true",
                        help="text log",
    )
    parser.add_argument('--tb_dir',
                        type=str,
                        help='',
                        default='./tb_logs',
    )
    parser.add_argument('--logs_dir',
                        type=str,
                        help='',
                        default='./logs',
    )
    parser.add_argument("--verbose",
                        required=False,
                        action="store_true",
                        help="",
    )
    return parser


def get_checkpoint_parser(parser):
    parser.add_argument('--checkpoints_dir',
                        type=str,
                        help='',
                        default='./checkpoints',
    )
    parser.add_argument('--max_checkpoints',
                        type=int,
                        help='',
                        default=5,
    )
    return parser


def get_imitation_learning_parser(parser):
    parser.add_argument('--goal_weight',
                        type=float,
                        help='',
                        default=0.7,
    )
    parser.add_argument('--imitation_weight',
                        type=float,
                        help='',
                        default=0.3,
    )
    return parser


def get_optimization_parser(parser):
    parser.add_argument("-lr",
                        "--learning_rate",
                        default=5e-8,
                        type=float,
                        help="Learning rate",
    )
    parser.add_argument("--init_lr",
                        default=5e-3,
                        type=float,
                        help="initial learning rate",
    )
    parser.add_argument('--decay',
                        type=str,
                        default="exponential",
                        help='',
    )
    parser.add_argument('--decay_steps',
                        type=int,
                        default=1000,
                        help='',
    )
    parser.add_argument('--decay_rate',
                        type=float,
                        default=5e-8,
                        help='',
    )
    return parser


def get_distributed_training_parser(parser):
    pass


def get_dataset_parser(parser):
    parser.add_argument('--data_path',
                        type=str,
                        default="",
                        help='',
    )
    parser.add_argument('--seq_length',
                        type=int,
                        default=10,
                        help='',
    )
    return parser


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser = get_marl_parser(parser)
    parser = get_logging_parser(parser)
    parser = get_optimization_parser(parser)
    parser = get_dataset_parser(parser)
    parser = get_eval_parser(parser)
    parser = get_checkpoint_parser(parser)

    parser.add_argument("--num_train_iter",
                        default=100_000,
                        type=int,
                        help="Number of epochs to train agent over.",
    )
    parser.add_argument("--collect_steps_per_iter",
                        default=1,
                        type=int,
                        help="Number of steps to take per collection episode.",
    )
    parser.add_argument("--eps",
                        type=float,
                        default=0.0,
                        help="Probability of training on the greedy policy for a given episode",
    )
    parser.add_argument("--num_warmup_iter",
                        default=10_000,
                        type=int,
                        help="",
    )
    parser.add_argument("--batch_size",
                        default=512,
                        type=int,
                        help="",
    )
    parser.add_argument('--replay_buffer_max_length',
                        type=int,
                        help='replay_buffer_max_length',
                        default=10_000,
    )
    parser.add_argument('--replay_buffer_capacity',
                        type=int,
                        help='replay_buffer_capacity',
                        default=10_000,
    )
    parser.add_argument('--min_reward',
                        type=float,
                        help='',
                        default=-1.,
    )
    parser.add_argument('--max_reward',
                        type=float,
                        help='',
                        default=1.,
    )
    parser.add_argument("--norm",
                        action="store_true",
                        help="",
    )
    parser.add_argument("--actor_net_layer",
                        default=(256, 256),
                        type=tuple,
                        help="",
    )
    parser.add_argument("--critic_net_layer",
                        default=(256, 256),
                        type=tuple,
                        help="",
    )
    parser.add_argument("--dropout",
                        default=(0.1, 0.1),
                        type=tuple,
                        help="",
    )
    return parser


def parse_args(parser, modify_parser):
    return parser.parse_args()
