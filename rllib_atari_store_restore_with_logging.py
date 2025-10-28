# These tags allow extracting portions of this script on Anyscale.
# ws-template-imports-start
import gymnasium as gym
import logging
from pathlib import Path

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.rllib.utils.test_utils import add_rllib_example_script_args

from pprint import pprint

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

logger.info("="*100)
logger.info("RLlib Atari Training with Checkpoint Save/Restore Logging")
logger.info("="*100)
logger.info(f"Environment: {ENV}")
logger.info(f"Number of Learners: {NUM_LEARNERS}")


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
logger.info("✓ Environment registered with tune")

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
        # num_learners=0; 로컬learner 사용
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
    # 저장 시, 환경 설정이 freeze 되기에, 추후 평가할 때 수정이 어려움.
    # 학습 때 설정해두는 것이 좋음.
    .evaluation(
        evaluation_interval=None,               # 필요 시 변경 가능
        evaluation_num_env_runners=2,           # 평가에 사용할 EnvRunner 수
        evaluation_duration=10,                 # 평가 길이
        evaluation_duration_unit="episodes",    # 'episodes' 또는 'timesteps'
        evaluation_config={
            "explore": False,                    # 평가에서만 탐험 비활성화
            "env": "env",
            "env_config": {
                "frameskip": 4,
                "full_action_space": False,
                "repeat_action_probability": 0.0,
            },
        },
    )
)

logger.info("-"*100)
logger.info("Configuration Summary:")
logger.info(f"  num_env_runners: 8")
logger.info(f"  num_envs_per_env_runner: 4")
logger.info(f"  train_batch_size_per_learner: 4000")
logger.info(f"  evaluation_num_env_runners: 2")
logger.info(f"  evaluation_duration: 10 episodes")
logger.info("-"*100)

logger.info("Building Algorithm...")
algo = config.build()
logger.info("✓ Algorithm built successfully")

logger.info("\n" + "="*100)
logger.info("Starting Training Loop (5 iterations)")
logger.info("="*100)

for i in range(5):
    logger.info(f"\n{'='*100}")
    logger.info(f"Training Iteration {i+1}/5")
    logger.info(f"{'='*100}")
    
    res = algo.train()
    
    try:
        # iteration 내에서 episode가 안 끝났을 때 KeyError 발생하게 됨.
        # num_env_runners * num_envs_per_env_runner 이 커지면 더 자주 발생.
        # 발생시키고 싶지 않으면 train_batch_size_per_learner 값을 키우면 됨
        ep_rew_mean = res["env_runners"]["episode_return_mean"]
        ep_len_mean = res["env_runners"].get("episode_len_mean", "N/A")
        num_episodes = res["env_runners"].get("num_episodes", "N/A")
        
        logger.info(f"✓ Iteration {i+1} completed successfully")
        logger.info(f"  Episode Return Mean: {ep_rew_mean:.2f}")
        logger.info(f"  Episode Length Mean: {ep_len_mean}")
        logger.info(f"  Number of Episodes: {num_episodes}")
        
        if "num_env_steps_sampled_lifetime" in res:
            logger.info(f"  Total Env Steps (Lifetime): {res['num_env_steps_sampled_lifetime']}")
            
    except KeyError as e:
        logger.warning(f"⚠ Episode metrics not available yet (KeyError: {e})")
        logger.info("Full result structure:")
        pprint(res)

logger.info("\n" + "="*100)
logger.info("CHECKPOINT SAVE OPERATION")
logger.info("="*100)

# 체크포인트 저장
# 체크포인트 저장 경로를 PyArrow URI(예: file:///..., s3://...)로 해석하도록 되어 있어
# 로컬 경로라도 file:// 스킴을 붙이거나 절대경로의 파일 URI로 넘겨야 됨
target_dir = (Path.cwd() / "checkpoints" / ENV_ID).resolve()
target_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Checkpoint target directory: {target_dir}")
logger.info("Calling algo.save_to_path()...")

ckpt_path = algo.save_to_path(str(target_dir))

logger.info(f"✓ Checkpoint saved successfully")
logger.info(f"  Checkpoint path: {ckpt_path}")

# 체크포인트 구조 확인
ckpt_path_obj = Path(ckpt_path)
if ckpt_path_obj.exists():
    total_size = sum(f.stat().st_size for f in ckpt_path_obj.rglob('*') if f.is_file())
    logger.info(f"  Total checkpoint size: {total_size / (1024*1024):.2f} MB")
    
    logger.info("  Checkpoint directory structure:")
    items = sorted(ckpt_path_obj.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    for item in items[:15]:
        if item.is_dir():
            logger.info(f"    📁 {item.name}/")
        else:
            size_kb = item.stat().st_size / 1024
            logger.info(f"    📄 {item.name} ({size_kb:.1f} KB)")
    if len(list(ckpt_path_obj.iterdir())) > 15:
        logger.info(f"    ... ({len(list(ckpt_path_obj.iterdir())) - 15} more items)")

logger.info("\nStopping the original algorithm...")
algo.stop()
logger.info("✓ Original algorithm stopped")

logger.info("\n" + "="*100)
logger.info("CHECKPOINT RESTORE OPERATION")
logger.info("="*100)

# 체크포인트로부터 복원
# 중요: 복원 시에도 동일한 env 등록/creator, 커넥터 import 가 살아 있어야 함
logger.info(f"Restoring algorithm from checkpoint: {ckpt_path}")
logger.info("Calling Algorithm.from_checkpoint()...")

# from_checkpoint 는 체크포인트에 저장된 config를 그대로 로드함
algo_eval = Algorithm.from_checkpoint(ckpt_path)

logger.info("✓ Algorithm restored successfully from checkpoint")
logger.info(f"  Restored algorithm type: {type(algo_eval).__name__}")

logger.info("\n" + "="*100)
logger.info("EVALUATION OPERATION")
logger.info("="*100)

# 평가
logger.info("Starting evaluation with restored algorithm...")
eval_results = algo_eval.evaluate()
logger.info("✓ Evaluation completed")

# 결과 집계
try:
    eval_data = eval_results.get("evaluation", {})
    if not eval_data:
        logger.warning("⚠ No evaluation data found in results")
    else:
        eval_ep_ret_mean = eval_data.get("episode_return_mean", "N/A")
        eval_ep_len_mean = eval_data.get("episode_len_mean", "N/A")
        eval_num_episodes = eval_data.get("num_episodes", "N/A")
        
        logger.info("Evaluation Results:")
        logger.info(f"  Episode Return Mean: {eval_ep_ret_mean}")
        logger.info(f"  Episode Length Mean: {eval_ep_len_mean}")
        logger.info(f"  Number of Episodes: {eval_num_episodes}")
        
except KeyError as e:
    logger.warning(f"⚠ Could not extract evaluation metrics (KeyError: {e})")
    logger.info("Full evaluation result structure:")
    pprint(eval_results)

# 평가용 알고리즘 정리
logger.info("\nStopping evaluation algorithm...")
algo_eval.stop()
logger.info("✓ Evaluation algorithm stopped")

logger.info("\n" + "="*100)
logger.info("SCRIPT COMPLETED SUCCESSFULLY")
logger.info("="*100)
logger.info("Summary:")
logger.info(f"  - Trained for 5 iterations")
logger.info(f"  - Saved checkpoint to: {ckpt_path}")
logger.info(f"  - Restored algorithm from checkpoint")
logger.info(f"  - Performed evaluation")
logger.info("="*100)
