# RLlib Atari Breakout PPO 예제

이 프로젝트는 Ray RLlib를 사용해 Atari Breakout(`ale_py:ALE/Breakout-v5`) 환경에서 PPO 알고리즘을 학습·저장·복원·평가하는 최소 예제 스크립트 모음입니다. RLlib의 ConnectorV2를 활용하여 프레임 스태킹을 환경 외부가 아닌 커넥터에서 수행하도록 구성했습니다.

## 구성 파일

- `reinforece3/rllib_atari.py`
  - Breakout 환경에서 PPO 학습을 수행하는 기본 스크립트
  - 프레임 스태킹: `FrameStackingEnvToModule`, `FrameStackingLearner` (각 4프레임)
  - Atari 전처리: `wrap_atari_for_new_api_stack` 사용, `clip_rewards=True`
  - 5회 반복 학습 후 학습 통계(가능 시 `episode_return_mean`) 출력
- `reinforece3/rllib_atari_store_restore.py`
  - 위와 동일한 학습 구성에 더해, 평가(`.evaluation(...)`) 및 체크포인트 저장/복원/평가까지 포함
  - 체크포인트는 작업 디렉터리의 `checkpoints/Breakout-v5` 하위에 저장
  - `Algorithm.from_checkpoint(...)`로 복원 후 `evaluate()` 실행

## 주요 아이디어

- 프레임 스태킹을 환경 래퍼가 아닌 RLlib ConnectorV2(`EnvToModule`, `Learner`)로 처리하여 구성 유연성을 높임
- Atari 환경 설정(`frameskip=4`, `full_action_space=False`, `repeat_action_probability=0.0`)과 보상 클리핑 활성화
- 학습은 로컬 Learner를 사용하도록 고정(`num_learners=0`), GPU 1장 할당(`num_gpus_per_learner=1`)
- Env Runner 8개 × Runner당 4환경 = 총 32 환경 병렬 실행

## 하이퍼파라미터 요약

- PPO 학습(`.training(...)`)
  - `train_batch_size_per_learner=4000`, `minibatch_size=128`, `num_epochs=10`
  - `lambda_=0.95`, `kl_coeff=0.5`, `clip_param=0.1`, `vf_clip_param=10.0`, `entropy_coeff=0.01`
  - `lr=0.00015 * NUM_LEARNERS` (스크립트 인자 `--num-learners`에 비례)
  - `grad_clip=100.0`, `grad_clip_by="global_norm"`
- RL Module(`.rl_module(...)`)
  - 합성곱 필터: `[[16,4,2],[32,4,2],[64,4,2],[128,4,2]]` (out, kernel, stride)
  - 활성함수: ReLU, 헤드 MLP: `[256]`, 가치함수 공유: `vf_share_layers=True`
- 환경/런너(`.environment(...)`, `.env_runners(...)`)
  - Atari 전처리 래퍼 + 보상 클리핑
  - Runner 수: 8, Runner당 env 수: 4, Runner당 CPU: 1

## 필요 환경 및 설치

- Python 3.8+
- Ray RLlib, Gymnasium, ALE-Py(Atari)
- CUDA GPU 사용 시 CUDA 지원 PyTorch 설치 권장

예시(환경에 맞게 버전/옵션 조정):

```bash
pip install "ray[rllib]" gymnasium[atari,accept-rom-license] ale-py
# (GPU 사용 시) pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 실행 방법

- 기본 학습만 수행:

```bash
python reinforece3/rllib_atari.py \
  --env ale_py:ALE/Breakout-v5 \
  --num-learners 1
```

- 학습 + 체크포인트 저장/복원/평가까지 수행:

```bash
python reinforece3/rllib_atari_store_restore.py \
  --env ale_py:ALE/Breakout-v5 \
  --num-learners 1
```

참고:
- 스크립트는 내부에서 Learner 수를 `num_learners=0`으로 고정(로컬 Learner 사용)합니다. `--num-learners` 인자는 현재 LR 스케일링(`lr=0.00015 * NUM_LEARNERS`)에만 반영됩니다.
- 출력 통계에서 `env_runners.episode_return_mean`이 바로 없을 수 있습니다(에피소드 미완료 시). 이 경우 전체 결과 딕셔너리를 출력하도록 되어 있습니다.

## 체크포인트

- 저장 위치: 작업 디렉터리 기준 `checkpoints/Breakout-v5/`
- 저장 API: `algo.save_to_path(path)` 반환값은 저장된 체크포인트 경로
- 복원 API: `Algorithm.from_checkpoint(ckpt_path)`

## 자주 겪는 이슈와 팁

- `KeyError: 'episode_return_mean'`
  - 많은 병렬 환경에서 학습 반복 내에 에피소드가 끝나지 않으면 통계 키가 비어 있을 수 있습니다.
  - 해결: `train_batch_size_per_learner`를 늘리거나 학습 반복 수를 늘려 에피소드가 충분히 끝나도록 합니다.
- Atari ROM/라이선스
  - `gymnasium[atari,accept-rom-license]`로 설치 시 ROM 라이선스 동의가 포함됩니다.
- GPU 사용
  - CUDA가 설치된 환경에서 CUDA 지원 PyTorch를 설치해야 GPU를 활용할 수 있습니다.

## 폴더 구조

```
reinforece3/
├─ rllib_atari.py                  # 기본 학습
└─ rllib_atari_store_restore.py    # 학습 + 저장/복원/평가
```

## 참고 코드 위치

- 학습/환경 설정: `reinforece3/rllib_atari.py:1`
- 저장/복원/평가: `reinforece3/rllib_atari_store_restore.py:1`

---

질문이나 실행 환경 특화 설정(예: GPU 개수, EnvRunner 수, 배치 크기) 조정이 필요하시면 알려주세요. 원하시면 멀티 Learner 분산 구성으로 확장하는 설정 예시도 추가해드릴 수 있습니다.

