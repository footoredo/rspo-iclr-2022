import argparse

import torch


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args(add_extra_args_fn=None, config=None):
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=64)
    parser.add_argument(
        '--activation',
        type=str,
        default='tanh')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=False)
    parser.add_argument(
        '--action-activation',
        type=str)
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--parallel-limit',
        type=int,
        default=10,
        help='parallel limit')
    parser.add_argument(
        '--load-dvd-weights-dir',
        type=str
    )
    parser.add_argument(
        '--train-in-turn',
        dest='train_in_turn',
        action="store_true",
        default=True,
        help='train in turn')
    parser.add_argument(
        '--no-train-in-turn',
        dest='train_in_turn',
        action="store_false",
        help='don\'t train in turn')
    parser.add_argument(
        '--train',
        dest='train',
        action="store_true",
        default=True,
        help='train')
    parser.add_argument(
        '--no-train',
        dest='train',
        action="store_false",
        help='don\'t train')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--num-refs',
        type=int,
        nargs="*",
        help='number of reference agents (for loading)')
    parser.add_argument(
        '--collect-trajectories',
        action='store_true',
        default=False,
        help='collect trajectories for DIPG')
    parser.add_argument(
        '--use-rnd',
        action='store_true',
        default=False,
        help='use rnd')
    parser.add_argument(
        '--rnd-alpha',
        default=1.0,
        type=float)
    parser.add_argument(
        '--dipg',
        action='store_true',
        default=False,
        help='use DIPG')
    parser.add_argument(
        '--dipg-k',
        default=16,
        type=int,
        help='DIPG k for g')
    parser.add_argument(
        '--dipg-num-samples',
        default=32,
        type=int,
        help='DIPG num samples')
    parser.add_argument(
        '--dipg-alpha',
        default=1.0,
        type=float,
        help='DIPG alpha')
    parser.add_argument(
        '--use-reference',
        action='store_true',
        default=False,
        help='use reference agent')
    parser.add_argument(
        '--ppo-use-reference',
        action='store_true',
        default=False,
        help='ppo use reference agent')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--dice-lambda',
        type=float,
        default=0.95,
        help='Loaded DiCE lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--no-grad-norm-clip',
        action='store_true',
        default=False,
        help='don\'t clip norm of gradients (default: False)')
    parser.add_argument(
        '--reward-normalization',
        action='store_true',
        default=False,
        help='do reward normalization (default: False)')
    parser.add_argument(
        '--obs-normalization',
        action='store_true',
        default=False,
        help='do obs normalization (default: False)')
    parser.add_argument(
        '--use-attention',
        action='store_true',
        default=False,
        help='use attention mechanism in policy (default: False)')
    parser.add_argument(
        '--use-linear',
        action='store_true',
        default=False,
        help='use linear in policy (default: False)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-agents',
        type=int,
        default=2,
        help='number of agents in the environment (default: 2)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--dice-epoch',
        type=int,
        default=4,
        help='number of dice epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--likelihood-threshold',
        type=float,
        nargs="*",
        default=200.0,
        help='likelihood threshold')
    parser.add_argument(
        '--likelihood-cap',
        type=float,
        help='likelihood cap')
    parser.add_argument(
        '--threshold-annealing-schedule',
        type=str,
        help='likelihood threshold annealing schedule')
    parser.add_argument(
        '--exploration-reward-annealing-schedule',
        type=str,
        help='exploration reward annealing schedule')
    parser.add_argument(
        '--no-load-refs',
        action="store_true",
        default=False)
    parser.add_argument(
        '--use-likelihood-reward-cap',
        action="store_true",
        default=False)
    parser.add_argument(
        '--use-dynamic-prediction-alpha',
        action="store_false",
        default=True)
    parser.add_argument(
        '--use-timestep-mask',
        action="store_true",
        default=False)
    parser.add_argument(
        '--use-reward-predictor',
        action="store_true",
        default=False)
    parser.add_argument(
        '--auto-threshold',
        type=float)
    parser.add_argument(
        '--likelihood-alpha',
        type=float,
        default=0.0,
        help='likelihood alpha')
    parser.add_argument(
        '--reward-multiplier',
        type=float,
        default=1.0,
        help='reward multiplier')
    parser.add_argument(
        '--prediction-reward-alpha',
        type=float,
        default=1.0,
        help='prediction reward alpha')
    parser.add_argument(
        '--exploration-reward-alpha',
        type=float,
        default=1.0,
        help='exploration reward alpha')
    parser.add_argument(
        '--reward-prediction-loss-coef',
        type=float,
        default=1.0)
    parser.add_argument(
        '--reward-prediction-multiplier',
        type=float,
        default=10.,
        help='reward prediction multiplier')
    parser.add_argument(
        '--use-symmetry-for-reference',
        action="store_true",
        default=False)
    parser.add_argument(
        '--wandb-project')
    parser.add_argument(
        '--wandb-group')
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        default=False)
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--num-games-after-training',
        type=int,
        default=100,
        help='number of games played after training')
    parser.add_argument(
        '--episode-steps',
        type=int,
        help='number of steps per episode')
    parser.add_argument(
        '--test-episode-steps',
        type=int,
        help='number of steps per episode during test')
    parser.add_argument(
        '--env-name',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save data (default: ./trained_models/)')
    parser.add_argument(
        '--config',
        help='config file')
    parser.add_argument(
        '--env-config',
        help='environment config file')
    parser.add_argument(
        '--ref-config',
        help='reference config file')
    parser.add_argument(
        '--reject-sampling',
        action="store_true",
        default=False)
    parser.add_argument(
        '--interpolate-rewards',
        action="store_true",
        default=False)
    parser.add_argument(
        '--no-exploration-rewards',
        action="store_true",
        default=False)
    parser.add_argument(
        '--full-intrinsic',
        action="store_true",
        default=False)
    parser.add_argument(
        '--plot-joint-plot',
        action="store_true",
        default=False)
    parser.add_argument(
        '--fine-tune-start-iter',
        type=int)
    parser.add_argument(
        '--likelihood-gamma',
        type=float,
        default=0.995)
    parser.add_argument(
        '--exploration-threshold',
        type=float)
    parser.add_argument(
        '--log-level',
        default='info')
    parser.add_argument(
        '--play',
        dest='play',
        action="store_true",
        default=True,
        help='play after training')
    parser.add_argument(
        '--no-play',
        dest='play',
        action="store_false",
        help='don\'t play after training')
    parser.add_argument(
        '--gif',
        action="store_true",
        help='make gif')
    parser.add_argument(
        '--load',
        action="store_true",
        help='whether to load model')
    parser.add_argument(
        '--load-dir',
        help='directory to load model')
    parser.add_argument(
        '--load-step',
        type=int,
        help='step to load model')
    parser.add_argument(
        '--ref-load-dir',
        nargs="*",
        help='directory to load reference model')
    parser.add_argument(
        '--ref-load-step',
        nargs="*",
        type=int,
        help='step to load reference model')
    parser.add_argument(
        '--ref-num-ref',
        nargs="*",
        type=int,
        help='reference model used reference')
    parser.add_argument(
        '--reseed-step',
        type=int,
        help='step to reseed environment')
    parser.add_argument(
        '--render',
        action='store_true',
        help='if render after training')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='if plot after training')
    parser.add_argument(
        '--reseed-z',
        type=int,
        default=1,
        help='z to reseed environment')
    parser.add_argument(
        '--direction',
        type=int,
        help='direction')
    parser.add_argument(
        '--guided-updates',
        type=int,
        help='direction')
    parser.add_argument(
        '--dice-task',
        help='task for dice')
    parser.add_argument(
        '--test-branching-name',
        default="test-branching",
        help='test-branching-name')
    parser.add_argument(
        '--task',
        help='task')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    if add_extra_args_fn is not None:
        parser = add_extra_args_fn(parser)

    config_args = parser.parse_args()
    pre_args = argparse.Namespace()

    if config is not None:
        pre_args.__dict__.update(config)

    if config_args.config is not None:
        import json
        pre_args.__dict__.update(json.load(open(config_args.config)))

    args = parser.parse_args(namespace=pre_args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr', 'loaded-dice', 'hessian']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo', 'loaded-dice'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
