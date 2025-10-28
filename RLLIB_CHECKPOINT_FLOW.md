# RLlib 체크포인트 저장/복원 흐름 분석

## 개요

이 문서는 RLlib의 학습 흐름과 체크포인트 저장/복원 메커니즘을 상세히 설명합니다.

---

## 1. RLlib 전체 학습 흐름

### 1.1 초기화 단계

```
Algorithm 생성 (config.build())
    ↓
Config 검증 및 설정
    ↓
EnvRunner 생성 (로컬 + 원격)
    ↓
LearnerGroup 생성
    ↓
RLModule 생성 및 초기화
    ↓
Connector 파이프라인 구성
```

**주요 컴포넌트:**
- **Algorithm**: 전체 학습 과정을 총괄하는 최상위 클래스
- **EnvRunner**: 환경과 상호작용하며 데이터 수집
- **LearnerGroup**: 신경망 모델 학습 담당
- **RLModule**: 실제 신경망 모델
- **Connector**: 데이터 전처리 파이프라인

### 1.2 학습 반복 (Training Iteration)

```
algo.train() 호출
    ↓
1. EnvRunner들이 환경에서 데이터 수집 (샘플링)
   - 병렬로 여러 환경에서 동시 실행
   - num_env_runners × num_envs_per_env_runner 만큼의 환경
    ↓
2. 수집된 데이터를 Connector를 통해 전처리
   - Env-to-Module Connector: 환경 데이터 → 모델 입력 형식
   - Frame stacking, normalization 등
    ↓
3. LearnerGroup으로 데이터 전달
    ↓
4. Learner에서 실제 학습 수행
   - Forward pass
   - Loss 계산 (PPO loss, value loss, entropy loss)
   - Backward pass (gradient 계산)
   - Optimizer step (가중치 업데이트)
    ↓
5. 업데이트된 가중치를 EnvRunner들에게 동기화
    ↓
6. 메트릭 수집 및 로깅
   - Episode return, length
   - Learning rate, loss values
   - Sampling/training 시간 통계
    ↓
7. 결과 딕셔너리 반환
```

### 1.3 주요 데이터 흐름

```
Environment Observation
    ↓
EnvRunner (data collection)
    ↓
SampleBatch (raw experiences)
    ↓
Env-to-Module Connector (preprocessing)
    ↓
RLModule (forward pass for inference)
    ↓
Action selection
    ↓
Environment step
    ↓
[Training phase]
    ↓
LearnerGroup receives batches
    ↓
Learner Connector (training preprocessing)
    ↓
RLModule (forward pass for training)
    ↓
Loss computation
    ↓
Backpropagation
    ↓
Weight update
    ↓
Sync weights back to EnvRunners
```

---

## 2. 체크포인트 저장 메커니즘

### 2.1 저장 호출 흐름

```python
algo.save_to_path(path)
    ↓
Algorithm.save_to_path() (in algorithm.py)
    ↓
Checkpointable.save_to_path() (in checkpoints.py)
```

### 2.2 `Checkpointable.save_to_path()` 내부 동작

**위치:** `/home/com/ray/rllib/utils/checkpoints.py:94`

```python
def save_to_path(self, path, state=None, filesystem=None, use_msgpack=False):
    # 1. 경로 생성 및 검증
    logger.info(f"[CHECKPOINT SAVE] Starting checkpoint save operation for {type(self).__name__}")
    
    # 2. Filesystem 설정 (로컬 또는 클라우드)
    if path and not filesystem:
        filesystem, path = pyarrow.fs.FileSystem.from_uri(path)
    
    # 3. 메타데이터 저장 (JSON 파일)
    metadata = self.get_metadata()
    # checkpoint_version, algorithm_type 등 저장
    
    # 4. 클래스 및 생성자 정보 저장 (Pickle 파일)
    # 복원 시 동일한 객체를 생성하기 위함
    pickle.dump({
        "class": type(self),
        "ctor_args_and_kwargs": self.get_ctor_args_and_kwargs()
    }, f)
    
    # 5. 상태 정보 수집
    state = state or self.get_state(
        not_components=[c[0] for c in self.get_checkpointable_components()]
    )
    
    # 6. 하위 컴포넌트들을 재귀적으로 저장
    for comp_name, comp in self.get_checkpointable_components():
        comp_path = path / comp_name
        comp.save_to_path(comp_path, filesystem=filesystem, ...)
    
    # 7. 메인 상태 저장 (Pickle 또는 Msgpack)
    pickle.dump(state, f)
    
    logger.info(f"[CHECKPOINT SAVE] Checkpoint saved successfully to: {str(path)}")
    return str(path)
```

### 2.3 저장되는 체크포인트 구조

```
checkpoint_directory/
│
├── metadata.json                    # 메타데이터 (버전, 타입 등)
├── class_and_ctor_args.pkl         # 클래스 정보 및 생성자 인자
├── state.pkl (or state.msgpack)    # Algorithm의 메인 상태
│
├── learner_group/                   # LearnerGroup 체크포인트
│   ├── metadata.json
│   ├── class_and_ctor_args.pkl
│   ├── state.pkl
│   │
│   └── learner/                     # 개별 Learner 체크포인트
│       ├── metadata.json
│       ├── state.pkl
│       │
│       └── module/                  # RLModule (신경망) 체크포인트
│           ├── metadata.json
│           ├── state.pkl
│           └── [model weights files]
│
├── env_runner/                      # EnvRunner 체크포인트
│   ├── metadata.json
│   ├── state.pkl
│   │
│   ├── env_to_module_connector/    # Connector 상태
│   │   └── state.pkl
│   │
│   └── module_to_env_connector/    # Connector 상태
│       └── state.pkl
│
└── metrics_logger/                  # 메트릭 로거 상태
    └── state.pkl
```

### 2.4 Algorithm의 get_state() 메서드

**위치:** `/home/com/ray/rllib/algorithms/algorithm.py:2843`

```python
def get_state(self, components=None, not_components=None, **kwargs):
    state = {}
    
    # 1. EnvRunner 상태 수집 (RLModule 제외)
    if self.env_runner:
        state[COMPONENT_ENV_RUNNER] = self.env_runner.get_state(
            not_components=[COMPONENT_RL_MODULE]  # RLModule은 Learner에서 관리
        )
    
    # 2. Evaluation EnvRunner 상태
    if self.eval_env_runner:
        state[COMPONENT_EVAL_ENV_RUNNER] = self.eval_env_runner.get_state(...)
    
    # 3. LearnerGroup 상태 (RLModule 포함)
    state[COMPONENT_LEARNER_GROUP] = self.learner_group.get_state(...)
    
    # 4. MetricsLogger 상태
    state[COMPONENT_METRICS_LOGGER] = self.metrics.get_state()
    
    # 5. 학습 진행 상태
    state[TRAINING_ITERATION] = self.training_iteration
    
    return state
```

**중요 포인트:**
- RLModule (신경망 가중치)는 LearnerGroup을 통해서만 저장됨
- EnvRunner의 RLModule은 inference-only이므로 저장하지 않음
- 학습 중단 후 재개를 위해 iteration 정보도 저장

---

## 3. 체크포인트 복원 메커니즘

### 3.1 복원 호출 흐름 (from_checkpoint)

```python
algo = Algorithm.from_checkpoint(path)
    ↓
Algorithm.from_checkpoint() (in algorithm.py:296)
    ↓
Checkpointable.from_checkpoint() (in checkpoints.py:417)
```

### 3.2 `Checkpointable.from_checkpoint()` 내부 동작

**위치:** `/home/com/ray/rllib/utils/checkpoints.py:417`

```python
@classmethod
def from_checkpoint(cls, path, filesystem=None, **kwargs):
    logger.info(f"[CHECKPOINT FROM_CHECKPOINT] Creating new {cls.__name__} instance from checkpoint: {path}")
    
    # 1. Filesystem 설정
    if path and not filesystem:
        filesystem, path = pyarrow.fs.FileSystem.from_uri(path)
    
    # 2. 클래스 정보 및 생성자 인자 로드
    with filesystem.open_input_stream(...) as f:
        ctor_info = pickle.load(f)
    
    ctor = ctor_info["class"]          # 저장된 클래스
    ctor_args = ctor_info["ctor_args_and_kwargs"][0]
    ctor_kwargs = ctor_info["ctor_args_and_kwargs"][1]
    
    # 3. 빈 객체 생성 (상태는 아직 복원 안 됨)
    obj = ctor(*ctor_args, **ctor_kwargs)
    logger.info(f"[CHECKPOINT FROM_CHECKPOINT] Created {ctor.__name__} instance, now restoring state...")
    
    # 4. 상태 복원
    obj.restore_from_path(path, filesystem=filesystem, **kwargs)
    logger.info(f"[CHECKPOINT FROM_CHECKPOINT] Successfully created and restored {ctor.__name__} from checkpoint")
    
    return obj
```

### 3.3 `restore_from_path()` 내부 동작

**위치:** `/home/com/ray/rllib/utils/checkpoints.py:326`

```python
def restore_from_path(self, path, component=None, filesystem=None, **kwargs):
    logger.info(f"[CHECKPOINT RESTORE] Starting checkpoint restore for {type(self).__name__} from: {path}")
    
    # 1. 경로 검증
    if not _exists_at_fs_path(filesystem, path):
        raise FileNotFoundError(f"`path` ({path}) not found!")
    
    # 2. 하위 컴포넌트들을 재귀적으로 복원
    self._restore_all_subcomponents_from_path(path, filesystem, component=component, **kwargs)
    
    # 3. 메인 상태 로드
    if component is None:
        filename = path / self.STATE_FILE_NAME
        if filename.with_suffix(".msgpack").is_file():
            state = msgpack.load(f, strict_map_key=False)
        else:
            state = pickle.load(f)
        
        # 4. 상태 적용
        self.set_state(state)
    
    logger.info(f"[CHECKPOINT RESTORE] Checkpoint restored successfully from: {path}")
```

### 3.4 Algorithm의 set_state() 메서드

**위치:** `/home/com/ray/rllib/algorithms/algorithm.py:2925`

```python
def set_state(self, state: StateDict):
    # 1. EnvRunner 상태 복원
    if COMPONENT_ENV_RUNNER in state:
        if self.env_runner:
            self.env_runner.set_state(state[COMPONENT_ENV_RUNNER])
        
        # 2. 모든 원격 EnvRunner에 상태 동기화
        self.env_runner_group.sync_env_runner_states(
            config=self.config,
            from_worker=self.env_runner,
            env_to_module=self.env_to_module_connector,
            module_to_env=self.module_to_env_connector,
        )
    
    # 3. LearnerGroup 상태 복원
    if COMPONENT_LEARNER_GROUP in state:
        self.learner_group.set_state(state[COMPONENT_LEARNER_GROUP])
    
    # 4. MetricsLogger 상태 복원
    if COMPONENT_METRICS_LOGGER in state:
        self.metrics.set_state(state[COMPONENT_METRICS_LOGGER])
    
    # 5. 학습 진행 상태 복원
    if TRAINING_ITERATION in state:
        self.training_iteration = state[TRAINING_ITERATION]
```

### 3.5 Algorithm의 restore_from_path() 추가 처리

**위치:** `/home/com/ray/rllib/algorithms/algorithm.py:3011`

```python
def restore_from_path(self, path, *args, **kwargs):
    # 1. 부모 클래스의 복원 로직 실행
    super().restore_from_path(path, *args, **kwargs)
    
    # 2. LearnerGroup이 복원되었다면, 가중치를 EnvRunner들에 동기화
    path = pathlib.Path(path)
    if (path / COMPONENT_LEARNER_GROUP).is_dir():
        # 학습된 모델 가중치를 추론용 EnvRunner들에 전파
        self.env_runner_group.sync_weights(
            from_worker_or_learner_group=self.learner_group,
            inference_only=True,  # 추론 전용 가중치만
        )
    
    # 3. Connector 상태 복원 및 동기화
    if self.env_runner_group.num_remote_env_runners() > 0:
        if (path / COMPONENT_ENV_TO_MODULE_CONNECTOR).is_dir():
            self.env_to_module_connector.restore_from_path(...)
        
        if (path / COMPONENT_MODULE_TO_ENV_CONNECTOR).is_dir():
            self.module_to_env_connector.restore_from_path(...)
        
        # 모든 원격 EnvRunner에 Connector 상태 동기화
        self.env_runner_group.sync_env_runner_states(...)
```

---

## 4. 로깅이 추가된 위치 요약

### 4.1 checkpoints.py에 추가된 로그

**파일:** `/home/com/ray/rllib/utils/checkpoints.py`

1. **save_to_path() 시작** (라인 147)
   ```python
   logger.info(f"[CHECKPOINT SAVE] Starting checkpoint save operation for {type(self).__name__}")
   ```

2. **save_to_path() 완료** (라인 326)
   ```python
   logger.info(f"[CHECKPOINT SAVE] Checkpoint saved successfully to: {str(path)}")
   ```

3. **restore_from_path() 시작** (라인 378)
   ```python
   logger.info(f"[CHECKPOINT RESTORE] Starting checkpoint restore for {type(self).__name__} from: {path}")
   ```

4. **restore_from_path() 완료** (라인 421)
   ```python
   logger.info(f"[CHECKPOINT RESTORE] Checkpoint restored successfully from: {path}")
   ```

5. **from_checkpoint() 시작** (라인 425)
   ```python
   logger.info(f"[CHECKPOINT FROM_CHECKPOINT] Creating new {cls.__name__} instance from checkpoint: {path}")
   ```

6. **from_checkpoint() 객체 생성 후** (라인 509)
   ```python
   logger.info(f"[CHECKPOINT FROM_CHECKPOINT] Created {ctor.__name__} instance, now restoring state...")
   ```

7. **from_checkpoint() 완료** (라인 512)
   ```python
   logger.info(f"[CHECKPOINT FROM_CHECKPOINT] Successfully created and restored {ctor.__name__} from checkpoint")
   ```

### 4.2 스크립트에 추가된 로그

**파일:** `rllib_atari_with_logging.py`, `rllib_atari_store_restore_with_logging.py`

- 환경 설정 정보
- 각 training iteration 시작/완료
- Episode return, length, steps 메트릭
- 체크포인트 저장 전/후
- 체크포인트 구조 및 크기 정보
- Algorithm 복원 전/후
- 평가 시작/완료 및 결과

---

## 5. 체크포인트 호출 체인 전체 흐름

### 5.1 저장 시 호출 체인

```
사용자 코드: algo.save_to_path("/path/to/checkpoint")
    ↓
Algorithm.save_to_path()              [LOG: "Starting checkpoint save for Algorithm"]
    ↓
    ├─> LearnerGroup.save_to_path()   [LOG: "Starting checkpoint save for LearnerGroup"]
    │       ↓
    │       └─> Learner.save_to_path() [LOG: "Starting checkpoint save for Learner"]
    │               ↓
    │               └─> RLModule.save_to_path() [LOG: "Starting checkpoint save for RLModule"]
    │                       ↓
    │                       [신경망 가중치 저장] [LOG: "Checkpoint saved successfully..."]
    │
    ├─> EnvRunner.save_to_path()      [LOG: "Starting checkpoint save for EnvRunner"]
    │       ↓
    │       ├─> EnvToModuleConnector.save_to_path()
    │       └─> ModuleToEnvConnector.save_to_path()
    │
    └─> [Algorithm 메인 상태 저장]    [LOG: "Checkpoint saved successfully to: /path/to/checkpoint"]
```

### 5.2 복원 시 호출 체인

```
사용자 코드: algo = Algorithm.from_checkpoint("/path/to/checkpoint")
    ↓
Algorithm.from_checkpoint()           [LOG: "Creating new Algorithm instance from checkpoint"]
    ↓
    [빈 Algorithm 객체 생성]          [LOG: "Created Algorithm instance, now restoring state..."]
    ↓
Algorithm.restore_from_path()         [LOG: "Starting checkpoint restore for Algorithm"]
    ↓
    ├─> LearnerGroup.restore_from_path() [LOG: "Starting checkpoint restore for LearnerGroup"]
    │       ↓
    │       └─> Learner.restore_from_path()
    │               ↓
    │               └─> RLModule.restore_from_path()
    │                       ↓
    │                       [신경망 가중치 로드]
    │
    ├─> EnvRunner.restore_from_path()
    │       ↓
    │       ├─> EnvToModuleConnector.restore_from_path()
    │       └─> ModuleToEnvConnector.restore_from_path()
    │
    ├─> [Algorithm 메인 상태 로드 및 적용]
    │
    └─> [가중치를 원격 EnvRunner들에 동기화] [LOG: "Checkpoint restored successfully"]
    
    [LOG: "Successfully created and restored Algorithm from checkpoint"]
```

---

## 6. 주요 컴포넌트별 역할

### 6.1 Algorithm (algorithm.py)
- 전체 학습 프로세스 조율
- EnvRunner와 LearnerGroup 관리
- 체크포인트 최상위 로직 구현
- 학습 iteration 관리

### 6.2 LearnerGroup
- 여러 Learner를 관리 (분산 학습 지원)
- 학습 데이터 분배
- 가중치 집계 및 동기화

### 6.3 Learner
- 실제 신경망 학습 수행
- Loss 계산 및 역전파
- Optimizer 관리

### 6.4 RLModule
- 신경망 모델 정의
- Forward pass 구현
- 모델 가중치 저장/로드

### 6.5 EnvRunner
- 환경과의 상호작용
- 경험 데이터 수집
- Inference용 RLModule 보유

### 6.6 Connector
- 데이터 전처리 파이프라인
- Frame stacking, normalization 등
- Stateful 변환 가능

### 6.7 Checkpointable (checkpoints.py)
- 체크포인트 저장/복원 인터페이스
- 재귀적 컴포넌트 저장/복원
- Filesystem 추상화 (로컬/클라우드)

---

## 7. 체크포인트 사용 시 주의사항

### 7.1 환경 설정 Freeze
- 체크포인트 저장 시 환경 설정이 함께 저장됨
- 복원 시 동일한 환경이 필요
- 평가 설정은 학습 시 미리 정의하는 것이 좋음

### 7.2 Connector 및 Environment 등록
```python
# 복원 전에 반드시 필요!
tune.register_env("env", _env_creator)
# Connector 함수들도 import되어 있어야 함
```

### 7.3 Python 버전 호환성
- Pickle 파일은 Python 버전에 따라 호환성 문제 가능
- Msgpack 사용 권장 (언어 독립적)
- 클래스 정의가 변경되면 복원 실패 가능

### 7.4 분산 환경
- 원격 노드의 EnvRunner 상태도 동기화됨
- 네트워크를 통한 파일 전송 발생 가능
- PyArrow FileSystem을 통한 클라우드 저장 지원

---

## 8. 실행 예제

### 8.1 기본 학습 및 저장
```bash
cd /home/com/reinforce-project3
python rllib_atari_with_logging.py
```

**예상 로그 출력:**
```
2025-10-28 10:00:00 | INFO     | __main__                       | ============================================
2025-10-28 10:00:00 | INFO     | __main__                       | RLlib Atari Training with Checkpoint Logging
...
2025-10-28 10:05:00 | INFO     | __main__                       | ✓ Iteration 1 completed successfully
2025-10-28 10:05:00 | INFO     | __main__                       |   Episode Return Mean: 15.23
...
```

### 8.2 체크포인트 저장 및 복원
```bash
cd /home/com/reinforce-project3
python rllib_atari_store_restore_with_logging.py
```

**예상 로그 출력:**
```
...
2025-10-28 10:10:00 | INFO     | __main__                       | CHECKPOINT SAVE OPERATION
2025-10-28 10:10:00 | INFO     | ray.rllib.utils.checkpoints    | [CHECKPOINT SAVE] Starting checkpoint save operation for Algorithm
2025-10-28 10:10:00 | INFO     | ray.rllib.utils.checkpoints    | [CHECKPOINT SAVE] Starting checkpoint save operation for LearnerGroup
2025-10-28 10:10:01 | INFO     | ray.rllib.utils.checkpoints    | [CHECKPOINT SAVE] Checkpoint saved successfully to: /home/com/reinforce-project3/checkpoints/Breakout-v5
...
2025-10-28 10:10:05 | INFO     | __main__                       | CHECKPOINT RESTORE OPERATION
2025-10-28 10:10:05 | INFO     | ray.rllib.utils.checkpoints    | [CHECKPOINT FROM_CHECKPOINT] Creating new Algorithm instance from checkpoint: ...
2025-10-28 10:10:06 | INFO     | ray.rllib.utils.checkpoints    | [CHECKPOINT RESTORE] Starting checkpoint restore for Algorithm
2025-10-28 10:10:07 | INFO     | ray.rllib.utils.checkpoints    | [CHECKPOINT RESTORE] Checkpoint restored successfully from: ...
...
```

---

## 9. 디버깅 팁

### 9.1 체크포인트 저장 실패 시
- 로그에서 어떤 컴포넌트에서 실패했는지 확인
- 디스크 용량 확인
- 파일 시스템 권한 확인
- `[CHECKPOINT SAVE]` 태그가 붙은 로그 추적

### 9.2 체크포인트 복원 실패 시
- 환경 등록 확인: `tune.register_env()`
- Connector 함수 import 확인
- Python 버전 일치 확인
- `[CHECKPOINT RESTORE]` 태그가 붙은 로그 추적

### 9.3 로그 레벨 조정
```python
import logging
logging.getLogger("ray.rllib").setLevel(logging.DEBUG)  # 더 자세한 로그
```

---

## 10. 참고 파일 위치

### 10.1 핵심 파일
- **Algorithm**: `/home/com/ray/rllib/algorithms/algorithm.py`
- **Checkpointable**: `/home/com/ray/rllib/utils/checkpoints.py`
- **EnvRunner**: `/home/com/ray/rllib/env/env_runner.py`
- **LearnerGroup**: `/home/com/ray/rllib/core/learner/learner_group.py`
- **RLModule**: `/home/com/ray/rllib/core/rl_module/rl_module.py`

### 10.2 예제 스크립트
- **기본 학습**: `/home/com/reinforce-project3/rllib_atari.py`
- **저장/복원**: `/home/com/reinforce-project3/rllib_atari_store_restore.py`
- **로깅 포함 학습**: `/home/com/reinforce-project3/rllib_atari_with_logging.py`
- **로깅 포함 저장/복원**: `/home/com/reinforce-project3/rllib_atari_store_restore_with_logging.py`

---

## 요약

RLlib의 체크포인트 메커니즘은 **계층적 저장/복원 구조**를 사용합니다:

1. **Algorithm** → LearnerGroup, EnvRunner 등의 최상위 컴포넌트 관리
2. 각 컴포넌트는 **Checkpointable 인터페이스** 구현
3. **재귀적으로** 하위 컴포넌트 저장/복원
4. **가중치 동기화**를 통해 분산 환경에서도 일관성 유지
5. **로깅**을 통해 각 단계를 추적 가능

이를 통해 학습 중단 후 재개, 모델 배포, 평가 등을 안전하게 수행할 수 있습니다.
