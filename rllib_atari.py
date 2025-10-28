# These tags allow extracting portions of this script on Anyscale.
# ws-template-imports-start
import gymnasium as gym

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.rllib.utils.test_utils import add_rllib_example_script_args

from pprint import pprint

# ws-template-imports-end
ENV_PREFIX = "ale_py:ALE"
ENV_ID = "Breakout-v5"

parser = add_rllib_example_script_args(
    default_reward=float("inf"),
    default_timesteps=3000000,
    default_iters=100000000000,
)
parser.set_defaults(
    env=f"{ENV_PREFIX}/{ENV_ID}"
)
# Use `parser` to add your own custom command line options to this script
# and (if needed) use their values to set up `config` below.
args = parser.parse_args()

NUM_LEARNERS = args.num_learners or 1
ENV = args.env


# These tags allow extracting portions of this script on Anyscale.
# ws-template-code-start
def _make_env_to_module_connector(env, spaces, device):
    return FrameStackingEnvToModule(num_frames=4)


def _make_learner_connector(input_observation_space, input_action_space):
    return FrameStackingLearner(num_frames=4)


# Create a custom Atari setup (w/o the usual RLlib-hard-coded framestacking in it).
# We would like our frame stacking connector to do this job.
def _env_creator(cfg):
    return wrap_atari_for_new_api_stack(
        gym.make(ENV, **cfg, render_mode="rgb_array"),
        # Perform frame-stacking through ConnectorV2 API.
        framestack=None,
    )


tune.register_env("env", _env_creator)

config = (
    PPOConfig()
    .environment(
        "env",
        env_config={
            "frameskip": 4,
            "full_action_space": False,
            "repeat_action_probability": 0.0,
        },
        clip_rewards=True,
    )
    .learners(
        # num_learner=0; 로컬learner 사용
        num_learners=0,
        num_gpus_per_learner=1
    )
    .env_runners(
        num_env_runners=8,
        num_envs_per_env_runner=4,
        num_cpus_per_env_runner=1,
        env_to_module_connector=_make_env_to_module_connector,
    )
    .training(
        learner_connector=_make_learner_connector,
        train_batch_size_per_learner=4000,
        minibatch_size=128,
        lambda_=0.95,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        num_epochs=10,
        lr=0.00015 * NUM_LEARNERS,
        grad_clip=100.0,
        grad_clip_by="global_norm",
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_filters=[[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
            conv_activation="relu",
            head_fcnet_hiddens=[256],
            vf_share_layers=True,
        ),
    )
)

algo = config.build()
for i in range(5):
    res = algo.train()
    try:
        # iteration 내에서 episode가 안 끝났을 때 KeyError 발생하게 됨.
        # num_env_runners * num_envs_per_env_runner 이 커지면 더 자주 발생.
        # 발생시키고 싶지 않으면 train_batch_size_per_learner 값을 키우면 됨
        ep_rew_mean = res["env_runners"]["episode_return_mean"]
        print(f"Iteration {i}: episode_return_mean = {ep_rew_mean}")
    except KeyError:
        pprint(res)

algo.stop()
