# Checkpoint Save & Restore의 역할 설명

## 1. Checkpoint Save (체크포인트 저장)의 역할

### 1.1 핵심 역할

**체크포인트 저장은 학습 중인 강화학습 에이전트의 "현재 상태 스냅샷"을 디스크에 저장하는 작업입니다.**

### 1.2 구체적인 저장 대상

```
checkpoint/
├── 신경망 가중치 (RLModule)
│   - Actor 네트워크 파라미터
│   - Critic 네트워크 파라미터
│   - 모든 레이어의 weights & biases
│
├── 학습 진행 상태
│   - 현재 iteration 번호
│   - 총 수집한 timesteps
│   - 총 학습한 timesteps
│
├── Optimizer 상태
│   - Adam/SGD의 momentum
│   - Learning rate schedule 상태
│   - Gradient 통계
│
├── 환경 상호작용 상태 (EnvRunner)
│   - Frame stacking buffer
│   - Observation normalization 통계 (mean, std)
│   - Action masking 상태
│
├── Connector 상태
│   - 전처리 파이프라인의 stateful 변환
│   - Running mean/std normalization
│   - Frame buffer
│
└── 메타데이터
    - 알고리즘 타입 (PPO, DQN 등)
    - 체크포인트 버전
    - 환경 설정 정보
```

### 1.3 왜 저장해야 하는가?

#### (1) **학습 중단 후 재개**

```python
# 시나리오: 학습 중 컴퓨터가 꺼짐
# 저장 없이:
# ❌ 처음부터 다시 학습 (수 시간~수 일 손실)

# 저장 있음:
algo.save_to_path("checkpoint/")  # 정기적으로 저장
# 재시작 후
algo = Algorithm.from_checkpoint("checkpoint/")  # 이어서 학습
algo.train()  # 중단된 시점부터 계속
```

#### (2) **최고 성능 모델 보존**

```python
best_reward = -float('inf')

for i in range(100):
    results = algo.train()
    current_reward = results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
    
    if current_reward > best_reward:
        best_reward = current_reward
        # 최고 성능 달성 시점의 모델 저장
        algo.save_to_path(f"best_checkpoint_reward_{current_reward:.2f}/")
```

#### (3) **모델 배포 (Deployment)**

```python
# 학습 완료 후
algo.save_to_path("production_model/")

# 다른 서버에서 추론
algo = Algorithm.from_checkpoint("production_model/")
action = algo.compute_single_action(observation)  # 실시간 추론
```

#### (4) **실험 재현성 (Reproducibility)**

```python
# 논문 작성 시: "우리 모델은 X 성능 달성"
algo.save_to_path("paper_model_v1/")

# 6개월 후 리뷰어가 재현 요청
# ✓ 정확히 같은 모델로 검증 가능
algo = Algorithm.from_checkpoint("paper_model_v1/")
```

#### (5) **A/B 테스트**

```python
# 버전 1 저장
config_v1.build().save_to_path("model_v1/")

# 버전 2 저장
config_v2.build().save_to_path("model_v2/")

# 실전 환경에서 비교
algo_v1 = Algorithm.from_checkpoint("model_v1/")
algo_v2 = Algorithm.from_checkpoint("model_v2/")
# 어느 버전이 더 나은지 측정
```

---

## 2. Checkpoint Restore (체크포인트 복원)의 역할

### 2.1 핵심 역할

**체크포인트 복원은 저장된 스냅샷으로부터 에이전트를 완전히 재구성하는 작업입니다.**

### 2.2 복원 프로세스

```python
# 1단계: 객체 구조 복원
algo = Algorithm.from_checkpoint("checkpoint/")
    ↓
# - Algorithm 클래스 인스턴스 생성
# - LearnerGroup 생성
# - EnvRunner 생성 (로컬 + 원격)
# - 모든 서브컴포넌트 재구성

# 2단계: 상태 복원
    ↓
# - 신경망 가중치 로드 → RLModule
# - Optimizer 상태 로드 → Learner
# - Normalization 통계 로드 → Connector
# - Iteration 정보 로드 → Algorithm

# 3단계: 분산 동기화 (중요!)
    ↓
# - 로컬 EnvRunner에 상태 적용
# - 모든 원격 EnvRunner에 가중치 동기화
# - Connector 상태를 모든 워커에 전파

# ✓ 완전히 복원된 에이전트 준비 완료
```

### 2.3 복원 후 가능한 작업들

#### (1) **학습 재개**

```python
algo = Algorithm.from_checkpoint("checkpoint_iter_500/")

# iteration 500부터 이어서 학습
for i in range(500, 1000):
    results = algo.train()
    print(f"Iteration {i}: {results[EPISODE_RETURN_MEAN]}")
```

#### (2) **추론 (Inference)**

```python
algo = Algorithm.from_checkpoint("trained_model/")

# 실시간 의사결정
env = gym.make("Breakout-v5")
obs, info = env.reset()

while True:
    action = algo.compute_single_action(obs)  # 학습된 정책으로 행동 선택
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

#### (3) **평가 (Evaluation)**

```python
algo = Algorithm.from_checkpoint("checkpoint/")

# 100 에피소드 평가
evaluation_results = algo.evaluate()
print(f"평균 리워드: {evaluation_results[EPISODE_RETURN_MEAN]}")
print(f"평균 에피소드 길이: {evaluation_results[EPISODE_LEN_MEAN]}")
```

#### (4) **Fine-tuning (전이 학습)**

```python
# Breakout에서 학습된 모델 로드
algo = Algorithm.from_checkpoint("breakout_checkpoint/")

# Pong 환경으로 fine-tuning
config.environment("Pong-v5")
algo.config = config

for i in range(100):
    algo.train()  # 적은 데이터로 빠르게 적응
```

#### (5) **모델 분석**

```python
algo = Algorithm.from_checkpoint("checkpoint/")

# 가중치 확인
weights = algo.get_module().get_state()
print(f"Actor 네트워크 첫 번째 레이어: {weights['pi.0.weight']}")

# 특정 상황에서 행동 분석
action, state, info = algo.compute_single_action(
    critical_observation,
    explore=False  # Deterministic policy
)
print(f"Action probabilities: {info['action_dist_inputs']}")
```

---

## 3. Save와 Restore의 관계

### 3.1 완벽한 대칭성

```python
# 저장
state_before = algo.get_state()
algo.save_to_path("checkpoint/")

# 복원
algo2 = Algorithm.from_checkpoint("checkpoint/")
state_after = algo2.get_state()

# ✓ state_before == state_after (비트 단위로 동일)
# ✓ 동일한 observation에 대해 동일한 action 출력
# ✓ 동일한 학습 곡선 재현
```

### 3.2 저장 시점 vs 복원 시점

```python
# 저장 시점 (Training 중)
algo.train()  # Iteration 100
algo.save_to_path("checkpoint/")
# 상태: 
# - timesteps_total = 400,000
# - learning_rate = 0.0001
# - epsilon = 0.05 (exploration)

# --- 시간 경과 (컴퓨터 재부팅 등) ---

# 복원 시점 (다음 날)
algo = Algorithm.from_checkpoint("checkpoint/")
# 상태: 동일하게 복원됨
# - timesteps_total = 400,000
# - learning_rate = 0.0001
# - epsilon = 0.05

algo.train()  # Iteration 101부터 정확히 이어짐
```

---

## 4. 내부 동작 원리

### 4.1 Save 내부 동작

```python
def save_to_path(self, path):
    # 1. 현재 상태 수집
    state = {
        "neural_network": self.module.get_state(),      # 가중치 추출
        "optimizer": self.optimizer.state_dict(),       # Optimizer 상태
        "iteration": self.training_iteration,           # 진행 상태
        "connectors": self.get_connectors_state(),      # 전처리 상태
    }
    
    # 2. 직렬화 (Serialization)
    # Python 객체 → 바이트 스트림
    serialized = pickle.dumps(state)  # 또는 msgpack
    
    # 3. 디스크에 쓰기
    with open(f"{path}/state.pkl", "wb") as f:
        f.write(serialized)
    
    # 4. 메타데이터 저장
    metadata = {
        "version": "2.1",
        "algorithm": "PPO",
        "timestamp": "2025-10-28 10:30:00"
    }
    json.dump(metadata, open(f"{path}/metadata.json", "w"))
```

### 4.2 Restore 내부 동작

```python
def from_checkpoint(cls, path):
    # 1. 메타데이터 로드
    metadata = json.load(open(f"{path}/metadata.json"))
    
    # 2. 역직렬화 (Deserialization)
    # 바이트 스트림 → Python 객체
    with open(f"{path}/state.pkl", "rb") as f:
        state = pickle.load(f)
    
    # 3. 객체 재구성
    algo = cls(config)  # 빈 Algorithm 생성
    
    # 4. 상태 적용
    algo.module.set_state(state["neural_network"])      # 가중치 로드
    algo.optimizer.load_state_dict(state["optimizer"])  # Optimizer 복원
    algo.training_iteration = state["iteration"]        # 진행 상태 복원
    algo.set_connectors_state(state["connectors"])      # 전처리 복원
    
    # 5. 분산 동기화
    algo.sync_weights_to_workers()  # 원격 워커에 전파
    
    return algo
```

---

## 5. 실전 예제

### 5.1 주기적 저장 (Checkpointing Strategy)

```python
algo = config.build()

for i in range(1000):
    results = algo.train()
    
    # 전략 1: 매 N iteration마다 저장
    if i % 10 == 0:
        algo.save_to_path(f"checkpoint_iter_{i}/")
    
    # 전략 2: 성능 개선 시 저장
    if results[EPISODE_RETURN_MEAN] > best_reward:
        algo.save_to_path("best_model/")
    
    # 전략 3: 시간 기반 저장 (1시간마다)
    if time.time() - last_save_time > 3600:
        algo.save_to_path(f"checkpoint_{datetime.now()}/")
```

### 5.2 장애 복구 (Fault Tolerance)

```python
try:
    # 마지막 체크포인트 찾기
    checkpoints = sorted(glob.glob("checkpoint_iter_*"))
    if checkpoints:
        latest = checkpoints[-1]
        algo = Algorithm.from_checkpoint(latest)
        print(f"✓ 복원 성공: {latest}")
    else:
        algo = config.build()
        print("새로운 학습 시작")
    
    # 학습 계속
    for i in range(1000):
        results = algo.train()
        
except KeyboardInterrupt:
    print("학습 중단됨")
    algo.save_to_path("interrupted_checkpoint/")
    print("✓ 체크포인트 저장 완료")
```

### 5.3 모델 버전 관리

```python
# 실험 추적
experiment_name = "PPO_Breakout_v1"

for trial in range(5):
    algo = config.build()
    
    for i in range(100):
        results = algo.train()
    
    # 각 trial 결과 저장
    final_reward = results[EPISODE_RETURN_MEAN]
    algo.save_to_path(f"{experiment_name}/trial_{trial}_reward_{final_reward:.2f}/")

# 나중에 최고 성능 모델 찾기
best_trial = max(glob.glob(f"{experiment_name}/*"), 
                 key=lambda x: float(x.split("_")[-1].rstrip("/")))
best_algo = Algorithm.from_checkpoint(best_trial)
```

---

## 6. 요약

| 작업 | Save (저장) | Restore (복원) |
|------|------------|---------------|
| **목적** | 현재 상태 스냅샷 생성 | 저장된 상태 재구성 |
| **타이밍** | 학습 중 주기적 | 학습 재개/배포 시 |
| **저장 대상** | 가중치, Optimizer, 통계 등 | - |
| **복원 대상** | - | 저장된 모든 상태 |
| **결과** | 디스크에 파일들 생성 | 동작 가능한 Algorithm 객체 |
| **사용 사례** | 백업, 최고 모델 보존 | 학습 재개, 추론, 평가 |

**핵심 원칙:**
- **Save = 시간 여행의 체크포인트 생성**
- **Restore = 특정 시점으로 시간 여행**
- 완벽한 복원 → 동일한 observation에 동일한 action → 재현 가능한 강화학습

---

## 7. 체크포인트 파일 구조 상세

### 7.1 실제 저장되는 파일들

```
my_checkpoint/
│
├── metadata.json                    # 알고리즘 정보, 버전, 타임스탬프
├── class_and_ctor_args.pkl         # Algorithm 클래스 및 생성자 인자
├── state.pkl                        # Algorithm의 메인 상태
│
├── learner_group/                   # 학습 컴포넌트
│   ├── metadata.json
│   ├── class_and_ctor_args.pkl
│   ├── state.pkl
│   │
│   └── learner_0000/               # 개별 Learner
│       ├── metadata.json
│       ├── state.pkl               # Optimizer 상태 포함
│       │
│       └── module/                 # 신경망 모델 (핵심!)
│           ├── metadata.json
│           ├── state.pkl           # 모든 가중치 저장
│           └── component_*/        # 하위 네트워크들
│
├── env_runner/                     # 환경 상호작용 컴포넌트
│   ├── metadata.json
│   ├── state.pkl
│   │
│   ├── env_to_module/             # 전처리 Connector
│   │   ├── metadata.json
│   │   └── state.pkl
│   │
│   └── module_to_env/             # 후처리 Connector
│       ├── metadata.json
│       └── state.pkl
│
└── metrics_logger/                 # 메트릭 히스토리
    ├── metadata.json
    └── state.pkl
```

### 7.2 각 파일의 역할

#### `metadata.json`
```json
{
  "checkpoint_version": "2.1",
  "class_path": "ray.rllib.algorithms.ppo.PPOAlgorithm",
  "timestamp": "2025-10-28T10:30:00",
  "ray_version": "2.9.0",
  "rllib_version": "2.9.0"
}
```

#### `class_and_ctor_args.pkl`
```python
{
  "class": <class 'PPOAlgorithm'>,
  "ctor_args_and_kwargs": (
    [],  # positional args
    {    # keyword args
      "config": <AlgorithmConfig>,
      "env": "Breakout-v5",
      ...
    }
  )
}
```

#### `state.pkl` (Algorithm)
```python
{
  "training_iteration": 100,
  "timesteps_total": 400000,
  "episodes_total": 1234,
  "env_runner": {...},         # EnvRunner 상태 참조
  "learner_group": {...},      # LearnerGroup 상태 참조
  "metrics_logger": {...}      # MetricsLogger 상태 참조
}
```

#### `state.pkl` (RLModule - 신경망)
```python
{
  "module_state": {
    # PyTorch의 경우
    "pi_network.conv1.weight": tensor([[...]]),     # Actor 네트워크
    "pi_network.conv1.bias": tensor([...]),
    "pi_network.fc.weight": tensor([[...]]),
    
    "vf_network.conv1.weight": tensor([[...]]),     # Critic 네트워크
    "vf_network.conv1.bias": tensor([...]),
    "vf_network.fc.weight": tensor([[...]]),
    
    # 총 수백만~수천만 개의 파라미터
  }
}
```

#### `state.pkl` (Learner - Optimizer)
```python
{
  "optimizer_state": {
    "state": {
      0: {  # 첫 번째 파라미터 그룹
        "momentum_buffer": tensor([...]),
        "exp_avg": tensor([...]),        # Adam의 1차 모멘트
        "exp_avg_sq": tensor([...]),     # Adam의 2차 모멘트
        "step": 10000
      },
      # ... 모든 파라미터에 대한 optimizer 상태
    },
    "param_groups": [{
      "lr": 0.0001,
      "betas": (0.9, 0.999),
      "eps": 1e-08,
      "weight_decay": 0
    }]
  }
}
```

---

## 8. 체크포인트 크기 분석

### 8.1 전형적인 크기

```
PPO on Atari (Breakout):
├── 신경망 가중치 (RLModule)        ~5-10 MB
├── Optimizer 상태                  ~10-20 MB (Adam의 경우)
├── Connector 상태                  ~1 MB
├── 메타데이터 및 설정              ~1 MB
└── 총 크기                        ~20-30 MB
```

### 8.2 크기에 영향을 주는 요소

1. **네트워크 구조**
   - 더 깊은/넓은 네트워크 → 더 큰 체크포인트
   - CNN vs MLP: CNN이 일반적으로 작음

2. **Optimizer 종류**
   - SGD: 추가 상태 없음 (작음)
   - Adam: 2개의 모멘트 버퍼 (큼)

3. **Frame Stacking**
   - 더 많은 frame → Connector 상태 증가

4. **분산 학습**
   - 여러 Learner → 각각의 체크포인트

---

## 9. 체크포인트 최적화 전략

### 9.1 저장 빈도 조절

```python
# ❌ 너무 자주 저장 (비효율적)
for i in range(1000):
    results = algo.train()
    algo.save_to_path(f"checkpoint_{i}/")  # 매 iteration

# ✓ 적절한 빈도
for i in range(1000):
    results = algo.train()
    if i % 50 == 0:  # 50 iteration마다
        algo.save_to_path(f"checkpoint_{i}/")
```

### 9.2 오래된 체크포인트 정리

```python
import glob
import os

# 최근 N개만 유지
MAX_CHECKPOINTS = 5

checkpoints = sorted(glob.glob("checkpoint_*"))
if len(checkpoints) > MAX_CHECKPOINTS:
    for old_cp in checkpoints[:-MAX_CHECKPOINTS]:
        shutil.rmtree(old_cp)
        print(f"Deleted old checkpoint: {old_cp}")
```

### 9.3 선택적 저장

```python
# 중요한 것만 저장
algo.save_to_path(
    "checkpoint/",
    # 특정 컴포넌트만 저장
    components=[COMPONENT_LEARNER_GROUP]  # 신경망만
)
```

### 9.4 압축 사용

```python
# Msgpack 사용 (Pickle보다 작음)
algo.save_to_path("checkpoint/", use_msgpack=True)

# 또는 외부 압축
import tarfile

with tarfile.open("checkpoint.tar.gz", "w:gz") as tar:
    tar.add("checkpoint/")
```

---

## 10. 체크포인트 관련 흔한 문제와 해결

### 10.1 문제: 복원 시 환경을 찾을 수 없음

```python
# ❌ 에러 발생
algo = Algorithm.from_checkpoint("checkpoint/")
# EnvError: Environment 'MyEnv-v0' not found

# ✓ 해결책
from ray import tune
tune.register_env("MyEnv-v0", lambda config: MyEnv(config))
algo = Algorithm.from_checkpoint("checkpoint/")
```

### 10.2 문제: Python 버전 불일치

```python
# Python 3.8에서 저장
algo.save_to_path("checkpoint/")

# Python 3.10에서 복원 시 에러 가능
# ✓ 해결책: Msgpack 사용 (버전 독립적)
algo.save_to_path("checkpoint/", use_msgpack=True)
```

### 10.3 문제: 클래스 정의 변경

```python
# 저장 시
class MyModule(RLModule):
    def __init__(self, config):
        self.layer1 = nn.Linear(10, 20)

algo.save_to_path("checkpoint/")

# 복원 시 (클래스가 변경됨)
class MyModule(RLModule):
    def __init__(self, config):
        self.layer1 = nn.Linear(10, 30)  # ❌ 크기 변경!
        self.layer2 = nn.Linear(30, 20)  # ❌ 새 레이어 추가!

algo = Algorithm.from_checkpoint("checkpoint/")  # 에러!

# ✓ 해결책: 버전 관리
# v1/my_module.py에 이전 정의 유지
# v2/my_module.py에 새 정의 사용
```

### 10.4 문제: 디스크 공간 부족

```python
# 모니터링 추가
import shutil

disk_usage = shutil.disk_usage("/")
free_gb = disk_usage.free / (1024**3)

if free_gb < 10:  # 10GB 미만
    print("⚠️ 디스크 공간 부족, 체크포인트 저장 스킵")
else:
    algo.save_to_path("checkpoint/")
```

---

## 11. 체크포인트를 활용한 고급 기법

### 11.1 모델 앙상블

```python
# 여러 체크포인트에서 복원
models = []
for cp_path in ["checkpoint_100/", "checkpoint_200/", "checkpoint_300/"]:
    models.append(Algorithm.from_checkpoint(cp_path))

# 앙상블 추론
def ensemble_action(obs):
    actions = [model.compute_single_action(obs) for model in models]
    # 다수결 또는 평균
    return most_common(actions)
```

### 11.2 Curriculum Learning

```python
# 단계 1: 쉬운 환경에서 학습
config.environment("EasyEnv-v0")
algo = config.build()
for i in range(100):
    algo.train()
algo.save_to_path("stage1_checkpoint/")

# 단계 2: 중간 환경으로 전이
config.environment("MediumEnv-v0")
algo = Algorithm.from_checkpoint("stage1_checkpoint/")
algo.config = config
for i in range(100):
    algo.train()
algo.save_to_path("stage2_checkpoint/")

# 단계 3: 어려운 환경으로 전이
config.environment("HardEnv-v0")
algo = Algorithm.from_checkpoint("stage2_checkpoint/")
algo.config = config
for i in range(100):
    algo.train()
```

### 11.3 하이퍼파라미터 탐색 후 Fine-tuning

```python
# 여러 설정으로 짧게 학습
configs = [
    {"lr": 0.0001, "gamma": 0.99},
    {"lr": 0.0003, "gamma": 0.995},
    {"lr": 0.001, "gamma": 0.999},
]

results = []
for i, hyperparams in enumerate(configs):
    config = PPOConfig().update_from_dict(hyperparams)
    algo = config.build()
    
    for _ in range(10):  # 짧게 학습
        result = algo.train()
    
    algo.save_to_path(f"trial_{i}/")
    results.append((i, result[EPISODE_RETURN_MEAN]))

# 최고 성능 설정 찾기
best_trial = max(results, key=lambda x: x[1])[0]

# 최고 설정으로 본격 학습
algo = Algorithm.from_checkpoint(f"trial_{best_trial}/")
for i in range(1000):  # 오래 학습
    algo.train()
```

---

## 12. 실무 체크포인트 관리 예제

### 12.1 완전한 학습 파이프라인

```python
import time
import json
from datetime import datetime

class CheckpointManager:
    def __init__(self, base_dir, max_checkpoints=5):
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        
    def save(self, algo, metrics):
        """체크포인트 저장 및 메타데이터 기록"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cp_name = f"checkpoint_{timestamp}"
        cp_path = self.base_dir / cp_name
        
        # 체크포인트 저장
        algo.save_to_path(str(cp_path))
        
        # 메타데이터 저장
        metadata = {
            "timestamp": timestamp,
            "iteration": algo.training_iteration,
            "reward_mean": metrics.get("episode_return_mean", 0),
            "reward_max": metrics.get("episode_return_max", 0),
        }
        
        with open(cp_path / "training_metrics.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.checkpoints.append((cp_path, metadata))
        
        # 오래된 체크포인트 정리
        self._cleanup()
        
        return cp_path
    
    def _cleanup(self):
        """오래된 체크포인트 삭제 (최고 성능은 유지)"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # 성능 순 정렬
        sorted_cps = sorted(
            self.checkpoints,
            key=lambda x: x[1]["reward_mean"],
            reverse=True
        )
        
        # 상위 N개 + 최신 것 유지
        to_keep = set(
            [cp[0] for cp in sorted_cps[:self.max_checkpoints-1]] +
            [self.checkpoints[-1][0]]  # 최신 것
        )
        
        # 나머지 삭제
        for cp_path, _ in self.checkpoints:
            if cp_path not in to_keep:
                shutil.rmtree(cp_path)
                print(f"Deleted checkpoint: {cp_path}")
        
        self.checkpoints = [
            cp for cp in self.checkpoints if cp[0] in to_keep
        ]
    
    def get_best(self):
        """최고 성능 체크포인트 반환"""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x[1]["reward_mean"])[0]


# 사용 예제
checkpoint_manager = CheckpointManager("experiments/ppo_breakout")

algo = config.build()
best_reward = -float('inf')

for i in range(1000):
    results = algo.train()
    current_reward = results[EPISODE_RETURN_MEAN]
    
    print(f"Iteration {i}: Reward = {current_reward:.2f}")
    
    # 주기적 저장 (50 iteration마다)
    if i % 50 == 0:
        checkpoint_manager.save(algo, results)
    
    # 최고 성능 갱신 시 저장
    if current_reward > best_reward:
        best_reward = current_reward
        best_path = checkpoint_manager.save(algo, results)
        print(f"✓ New best model saved: {best_path}")

# 최고 모델로 평가
print("\n평가 시작...")
best_algo = Algorithm.from_checkpoint(str(checkpoint_manager.get_best()))
eval_results = best_algo.evaluate()
print(f"Best model evaluation: {eval_results[EPISODE_RETURN_MEAN]:.2f}")
```

---

## 참고 자료

- **RLlib 공식 문서**: https://docs.ray.io/en/latest/rllib/
- **Checkpointable API**: `/home/com/ray/rllib/utils/checkpoints.py`
- **Algorithm 클래스**: `/home/com/ray/rllib/algorithms/algorithm.py`
- **실습 예제**: `/home/com/reinforce-project3/`