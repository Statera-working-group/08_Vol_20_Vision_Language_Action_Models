**Volume 20. Vision Language Action (VLA) Models**

# Chapter 4. Action Models

## 4.1 Action Model Architecture Overview

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

액션 모델(Action Model)의 정책 헤드(Policy Head)는 현대 비전-언어-행동(VLA, Vision-Language-Action) 아키텍처에서 가장 중요한 구성 요소 중 하나이다. 비전 인코더(Vision Encoder)가 주변 환경을 이해하고 대규모 언어 모델(LLM, Large Language Model)이 사용자의 의도와 작업 목표를 해석한다면, 정책 헤드는 이러한 고수준 의미 정보를 실제 로봇이 실행할 수 있는 행동(Action)으로 변환하는 역할을 담당한다. 즉, 인공지능의 인지(Cognition)와 실제 물리적 제어(Control)를 연결하는 마지막 단계이다.

기존 로봇 시스템은 인식(Perception), 계획(Planning), 제어(Control)를 서로 독립된 모듈로 구현하였다. 카메라는 물체를 인식하고, 계획기는 작업 순서를 생성하며, 제어기는 모터를 제어하였다. 이러한 구조는 안정적이지만 모듈 간 정보 손실과 지연 시간이 발생하고, 환경 변화에 유연하게 대응하기 어렵다. 액션 모델은 이러한 한계를 해결하기 위해 인식부터 행동까지를 하나의 학습 모델로 연결하며, 정책 헤드는 그 최종 출력 계층(Output Layer)으로 동작한다.

VLA 전체 구조에서 센서 정보는 RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 촉각 센서(Tactile Sensor), 힘 센서(Force Sensor), IMU, 휠 엔코더(Wheel Encoder), GNSS 등의 다양한 장치로부터 수집된다. 비전 인코더는 이를 의미 특징(Semantic Feature)으로 변환하고, LLM은 자연어 명령과 작업 목표를 이해한다. 이후 멀티모달 융합(Multimodal Fusion)이 모든 정보를 하나의 잠재 표현(Latent Representation)으로 통합하며, 정책 헤드는 이를 기반으로 실제 행동을 생성한다.

정책 헤드의 출력은 일반적인 언어 생성과 다르다. 텍스트를 생성하는 대신 관절 위치(Joint Position), 관절 속도(Joint Velocity), 토크(Torque), 휠 속도(Wheel Velocity), 조향각(Steering Angle), 엔드이펙터(End-Effector) 자세, 이동 경로(Waypoint), 행동 트리(Behavior Tree), 조작 명령(Manipulation Command) 등 실제 로봇 제어에 필요한 정보를 생성한다. 따라서 의미적 정확성과 물리적 실행 가능성을 동시에 만족해야 한다.

정책 헤드는 기존의 PID 제어기(PID Controller)나 모델 기반 제어(Model-Based Control)와는 다르다. 기존 제어기는 미리 정의된 수학적 모델을 이용하여 목표 위치를 추종하지만, 정책 헤드는 대규모 데이터로부터 환경과 행동 사이의 관계를 학습한다. 다만 정책 헤드가 기존 제어기를 완전히 대체하는 것은 아니며, 일반적으로 정책 헤드는 목표 행동을 생성하고 저수준 제어기가 이를 안정적으로 실행한다.

정책 헤드는 연속 행동 공간(Continuous Action Space)과 이산 행동 공간(Discrete Action Space)을 모두 사용할 수 있다. 연속 정책은 관절 각도, 속도, 토크, 이동 속도 등을 직접 생성하여 부드러운 움직임을 만든다. 반면 이산 정책은 이동(Navigation), 집기(Grasping), 충전(Charging), 검사(Inspection), 문 열기(Open Door)와 같은 기술(Skill)을 선택한다. 실제 로봇은 두 방식을 함께 사용하는 경우가 많다.

최근에는 계층형 행동 생성(Hierarchical Action Generation)이 널리 사용된다. 최상위에서는 작업 목표(Mission Goal)를 결정하고, 중간 계층에서는 사용할 기술(Skill)을 선택하며, 최하위에서는 실제 모터 제어 명령을 생성한다. 정책 헤드는 주로 중간 계층에서 기술을 선택하거나 저수준 행동을 생성하는 역할을 수행한다.

시간 모델링(Temporal Modeling)은 정책 헤드에서 매우 중요한 요소이다. 현재 행동은 현재 상태뿐 아니라 이전 행동, 과거 환경 변화, 앞으로 수행할 작업에 영향을 받는다. 따라서 정책 헤드는 시간 정보를 유지하기 위해 트랜스포머(Transformer), 순환 신경망(RNN, Recurrent Neural Network), 시퀀스 모델(Sequence Model), 확산 모델(Diffusion Model) 등을 활용한다.

자동회귀 정책(Auto-Regressive Policy)은 이전 행동을 기반으로 다음 행동을 순차적으로 생성한다. 이는 언어 모델과 유사한 방식으로 긴 작업(Long-Horizon Task)을 자연스럽게 생성할 수 있다는 장점이 있지만, 시간이 지날수록 작은 오류가 누적될 수 있다는 단점도 존재한다.

최근에는 확산 정책(Diffusion Policy)이 매우 주목받고 있다. 확산 모델은 무작위 노이즈(Noisy Action)에서 시작하여 반복적으로 수정하면서 최종 행동을 생성한다. 동일한 환경에서도 여러 가지 가능한 행동을 자연스럽게 생성할 수 있으며, 조작(Manipulation)이나 접촉(Contact) 작업에서 높은 성능을 보인다.

트랜스포머 기반 정책 헤드(Transformer Policy Head)는 현재 가장 널리 사용되는 구조이다. 셀프 어텐션(Self-Attention)은 영상, 언어, 작업 이력, 환경 정보를 동시에 분석하고, 크로스 어텐션(Cross-Attention)은 특정 작업과 관련된 중요한 정보를 선택적으로 활용한다. 이를 통해 복잡한 환경에서도 높은 추론 성능을 유지할 수 있다.

정책 헤드는 외부 환경뿐 아니라 내부 상태(Proprioception)도 함께 사용한다. 관절 위치, 속도, 모터 전류(Motor Current), 온도, 배터리 상태(Battery Status), 그리퍼 상태(Gripper Status), 로봇 자세 등을 함께 입력받아 현재 로봇의 실제 상태를 정확하게 반영한다.

로봇의 형태(Embodiment)에 따라 정책 헤드의 구조도 달라진다. 휴머노이드(Humanoid)는 수십 개의 관절을 동시에 제어해야 하며, AMR은 자율주행과 장애물 회피가 중심이고, 산업용 매니퓰레이터(Manipulator)는 엔드이펙터의 위치와 힘 제어가 중요하다. 따라서 동일한 정책 헤드 구조라도 로봇의 특성에 맞게 최적화된다.

최근에는 행동 토큰(Action Token) 개념도 활발히 연구되고 있다. 언어 모델이 단어(Token)를 생성하는 것처럼 로봇도 이동, 회전, 집기 등의 기본 동작을 토큰으로 표현하고, 정책 헤드는 이러한 행동 토큰을 생성한 후 실제 제어 명령으로 변환한다.

기술 조건 정책(Skill-Conditioned Policy)은 선택된 기술에 따라 행동을 생성한다. 예를 들어 이동 기술이 선택되면 자율주행 명령을 생성하고, 집기 기술이 선택되면 그리퍼 제어 명령을 생성한다. 이를 통해 복잡한 작업도 재사용 가능한 기술의 조합으로 수행할 수 있다.

목표 조건 정책(Goal-Conditioned Policy)은 작업 목표를 명시적으로 입력받는다. 목표는 자연어, 목표 위치, 물체 이름, 공간 좌표, 작업 명령 등 다양한 형태로 표현될 수 있으며, 정책 헤드는 이를 기반으로 행동을 생성한다. 따라서 새로운 작업에도 높은 일반화 성능을 가진다.

세계 모델(World Model)은 정책 헤드와 함께 사용되는 경우가 많다. 현재 상태만 고려하는 것이 아니라 앞으로 환경이 어떻게 변할지를 예측하여 행동을 생성하므로 장기 계획(Long-Horizon Planning)의 성공률이 크게 향상된다.

어포던스 추론(Affordance Reasoning)도 정책 생성 과정에 포함된다. 의미적으로 적절한 행동이라도 실제로 수행할 수 없는 경우에는 선택되지 않는다. 예를 들어 접근이 불가능하거나 배터리가 부족하거나 충돌 위험이 있는 행동은 제외되고, 현재 환경에서 실행 가능한 행동만 생성된다.

안전(Safety)은 정책 헤드보다 상위의 독립적인 시스템에서 관리된다. 정책 헤드는 행동을 생성하지만, 충돌 예측(Collision Prediction), 기하학적 검증(Geometric Validation), 운동학 제약(Kinematic Constraint), 동역학 제한(Dynamic Limit), 비상 정지(Emergency Stop)는 별도의 안전 계층(Safety Layer)이 담당한다.

최근 정책 헤드는 불확실성(Uncertainty)도 함께 예측한다. 행동과 함께 신뢰도(Confidence)를 계산하여 확신이 낮은 경우에는 추가 센서 정보를 요청하거나 사람에게 질문하거나 더 안전한 행동을 선택한다. 이러한 확률 기반 정책은 실제 환경에서 매우 중요한 역할을 한다.

다중 작업 학습(Multi-Task Learning)은 하나의 정책 헤드가 자율주행, 조작, 검사, 사람과의 대화, 보고서 생성 등을 동시에 학습하는 방식이다. 여러 작업이 공통적인 특징을 공유하기 때문에 학습 효율이 향상되고 새로운 작업에도 빠르게 적응할 수 있다.

정책 헤드는 대부분 모방 학습(Imitation Learning)으로 학습된다. 원격 조작(Teleoperation), 전문가 시연(Expert Demonstration), 산업용 작업 데이터, 시뮬레이션 결과 등을 이용하여 관측(Observation)과 행동(Action)의 관계를 학습한다. 이를 통해 사람과 유사한 작업 수행 능력을 획득한다.

행동 복제(Behavior Cloning)는 가장 기본적인 모방 학습 방법이다. 전문가의 행동을 그대로 따라 학습하지만, 장시간 작업에서는 작은 오차가 누적될 수 있다. 이를 해결하기 위해 강화학습(Reinforcement Learning), 온라인 수정(Online Correction), 지속적 학습(Continual Learning)이 함께 사용된다.

강화학습은 정책 헤드를 더욱 향상시킨다. 성공적인 작업 수행, 충돌 회피, 에너지 절감, 부드러운 움직임, 작업 완료 등을 보상 함수(Reward Function)로 정의하여 반복 학습함으로써 사람보다 더 효율적인 행동을 발견할 수 있다.

시뮬레이션(Simulation)은 정책 헤드 개발에서 매우 중요한 역할을 한다. 디지털 트윈(Digital Twin), 아이작 심(Isaac Sim), 가제보(Gazebo) 등을 이용하여 수백만 개의 작업 데이터를 생성하고 다양한 환경을 학습할 수 있다. 실제 로봇보다 훨씬 저렴하고 안전하게 데이터를 확보할 수 있다.

Sim-to-Real 기술은 시뮬레이션에서 학습한 정책을 실제 로봇에 적용하기 위한 기술이다. 센서 잡음(Sensor Noise), 물리 엔진 오차(Physics Variation), 조명 변화(Lighting Variation), 통신 지연(Latency) 등을 시뮬레이션에 추가하여 실제 환경에서도 안정적으로 동작하도록 만든다.

파운데이션 정책(Foundation Policy)은 여러 종류의 로봇에서 공통적으로 학습된 정책 모델이다. 이후 특정 플랫폼에 맞는 미세조정(Fine-Tuning)을 수행하여 관절 구조, 센서 구성, 작업 환경, 장비 특성에 맞게 빠르게 적응시킨다.

파라미터 효율적 미세조정(PEFT, Parameter-Efficient Fine-Tuning)은 전체 모델을 다시 학습하지 않고 로라(LoRA, Low-Rank Adaptation) 등의 기법을 이용하여 일부 파라미터만 수정한다. 이를 통해 공장 자동화, 물류, 의료, 농업, 시설 점검 등 다양한 환경에 빠르게 적용할 수 있다.

메모리(Memory)는 정책 생성에도 활용된다. 에피소드 메모리(Episodic Memory)는 이전 작업 경험을 저장하고, 의미 메모리(Semantic Memory)는 환경 정보를 저장하며, 절차 메모리(Procedural Memory)는 기술 수행 방법을 저장한다. 정책 헤드는 이러한 정보를 검색하여 더욱 적절한 행동을 생성한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 정책 생성에도 활용된다. 디지털 트윈, 시설 지도(Map), 유지보수 문서, 과거 작업 기록, 기업 데이터베이스를 검색하여 현재 작업에 필요한 정보를 정책 생성 과정에 반영한다.

다중 로봇(Multi-Robot) 환경에서는 각 로봇이 개별 정책 헤드를 가지면서도 전체 플릿(Fleet)의 목표를 함께 고려한다. 작업 분배(Task Allocation), 자원 공유(Resource Sharing), 충돌 방지(Collision Avoidance), 협업 조작(Cooperative Manipulation)을 위해 서로 협력하여 행동을 생성한다.

정책 헤드는 반드시 실시간 추론(Real-Time Inference)을 만족해야 한다. 양자화(Quantization), 가지치기(Pruning), 지식 증류(Knowledge Distillation), 효율적인 어텐션(Efficient Attention), GPU 최적화, 혼합 정밀도(Mixed Precision), 엣지-클라우드 협업(Edge-Cloud Collaboration) 등을 이용하여 낮은 지연 시간(Low Latency)을 유지한다.

산업용 제조에서는 정책 헤드가 생산 목표를 이해하고, 부품을 조립하며, 품질 검사를 수행하고, 사람과 협업하며, 생산 일정 변경에도 유연하게 대응한다. 하나의 통합 정책 모델이 다양한 작업을 수행할 수 있기 때문에 시스템 구성이 단순해진다.

물류 창고에서는 자율주행, 팔레트 운반, 충전, 재고 관리, 사람과의 협업을 하나의 정책 헤드가 통합적으로 수행한다. 의료 로봇에서는 환자의 안전을 최우선으로 고려하면서 이동, 물품 전달, 대화, 작업 수행을 동시에 처리한다.

향후 정책 헤드는 트랜스포머 기반 멀티모달 추론(Multimodal Reasoning), 확산 정책(Diffusion Policy), 세계 모델(World Model), 계층형 계획(Hierarchical Planning), 외부 메모리(External Memory), 검색 증강 생성(RAG), 지속적 학습(Continual Learning), 강화학습(Reinforcement Learning), 예측 시뮬레이션(Predictive Simulation)을 하나의 통합 구조로 결합하게 될 것으로 예상된다.

결국 정책 헤드는 인공지능의 인지 능력을 실제 물리적 행동으로 연결하는 핵심 구성 요소이다. 멀티모달 융합(Multimodal Fusion), 시간 모델링(Temporal Modeling), 계층형 기술 생성(Hierarchical Skill Generation), 세계 모델(World Model), 어포던스 추론(Affordance Reasoning), 불확실성 추정(Uncertainty Estimation), 메모리(Memory), 모방 학습(Imitation Learning), 강화학습(Reinforcement Learning), 실시간 추론 최적화(Real-Time Inference Optimization), 안전 검증(Safety Verification)을 통합함으로써 차세대 물리 AI(Physical AI)의 핵심 실행 엔진(Execution Engine)으로 발전하고 있다.

## 4.2 Deterministic Action Heads (with Code)

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

결정론적 액션 헤드(Deterministic Action Head)는 현대 액션 모델(Action Model)에서 가장 기본적인 정책(Policy) 구조 중 하나이다. 확률 기반 정책(Probabilistic Policy)이나 생성형 정책(Generative Policy)이 여러 가능한 행동(Action)을 생성하는 것과 달리, 결정론적 정책은 동일한 입력이 주어지면 항상 동일한 행동을 출력한다. 즉, 동일한 환경(Environment), 동일한 작업(Task), 동일한 로봇 상태(Robot State), 동일한 언어 명령(Language Instruction)이 입력되면 언제나 같은 제어 명령(Control Command)이 생성된다. 이러한 특성은 반복성과 예측 가능성이 중요한 산업용 로봇에서 매우 큰 장점이 된다.

비전-언어-행동(VLA, Vision-Language-Action) 아키텍처에서 결정론적 액션 헤드는 최종 행동 생성 계층(Final Action Generation Layer)에 위치한다. 비전 인코더(Vision Encoder)는 RGB 영상(RGB Image), 깊이 영상(Depth Image), 라이다(LiDAR), 촉각 센서(Tactile Sensor), 내부 상태(Proprioception) 등을 의미 특징(Semantic Feature)으로 변환하고, 대규모 언어 모델(LLM, Large Language Model)은 자연어 명령과 작업 목표(Task Goal)를 이해한다. 이후 멀티모달 융합(Multimodal Fusion)을 통해 하나의 잠재 표현(Latent Representation)이 생성되며, 결정론적 액션 헤드는 이를 실제 제어 명령으로 변환한다.

회귀 기반 정책(Regression-Based Policy)은 행동 생성을 함수 근사(Function Approximation) 문제로 정의한다. 즉, 상태(State), 관측(Observation), 목표(Goal)를 입력으로 받아 연속적인 행동(Action)을 출력하는 함수 f를 학습한다. 학습이 완료된 이후에는 동일한 입력에 대해 항상 동일한 결과를 출력하기 때문에 매우 안정적이고 예측 가능한 동작을 수행할 수 있다.

로봇은 대부분 연속 행동 공간(Continuous Action Space)에서 동작한다. 산업용 로봇은 관절 각도(Joint Position), 속도(Joint Velocity), 토크(Torque)를 생성해야 하며, 자율주행 로봇은 휠 속도(Wheel Velocity), 조향각(Steering Angle), 가속도(Acceleration)를 생성한다. 보행 로봇(Legged Robot)은 각 다리의 궤적(Trajectory)을 연속적으로 계산해야 하므로 회귀 기반 정책이 매우 적합하다.

결정론적 정책은 언어 모델처럼 단어(Token)를 생성하는 것이 아니라 연속적인 수치 값을 직접 출력한다. 따라서 생성 결과는 실제 모터 제어기(Motion Controller)가 바로 사용할 수 있는 제어 명령 형태로 제공된다. 이러한 특성은 실시간 제어(Real-Time Control)에 매우 적합하다.

학습 과정에서는 예측된 행동과 전문가 행동(Expert Demonstration)의 차이를 최소화하도록 회귀 손실 함수(Regression Loss)를 사용한다. 평균제곱오차(MSE, Mean Squared Error), 평균절대오차(MAE, Mean Absolute Error), 허버 손실(Huber Loss), Smooth L1 Loss 등이 대표적으로 사용된다. 이러한 손실 함수는 실제 행동과 최대한 유사한 제어 명령을 생성하도록 정책을 학습시킨다.

평균제곱오차(MSE)는 가장 널리 사용되는 손실 함수이다. 큰 오차에 더 큰 패널티를 부여하므로 높은 정밀도를 요구하는 로봇 작업에 적합하다. 그러나 실제 로봇에서는 궤적 부드러움(Trajectory Smoothness), 에너지 소비(Energy Consumption), 충돌 회피(Collision Avoidance), 자세 안정성(Posture Stability) 등을 고려한 다양한 손실 함수를 함께 사용하는 경우가 많다.

결정론적 액션 헤드는 일반적으로 여러 개의 연속 출력(Continuous Output)을 동시에 생성한다. 예를 들어 6축 산업용 로봇은 6개의 관절 위치와 그리퍼(Gripper) 상태를 함께 출력하고, 자율주행 로봇은 이동 속도, 회전 속도, 조향각을 동시에 생성한다. 휴머노이드(Humanoid)는 수십 개의 관절을 동시에 예측해야 한다.

시간적 일관성(Temporal Consistency)은 매우 중요한 요소이다. 연속된 행동이 갑자기 크게 변하면 로봇은 불안정한 움직임을 보이게 된다. 따라서 정책은 현재 행동뿐 아니라 이전 행동과의 연속성을 유지하도록 학습된다. 이를 위해 시간 정규화(Temporal Regularization)나 시퀀스 학습(Sequence Learning)이 활용된다.

최근에는 트랜스포머(Transformer) 기반 정책 헤드가 널리 사용된다. 셀프 어텐션(Self-Attention)은 과거의 행동, 현재 환경, 미래 작업 목표를 동시에 고려하여 보다 안정적인 행동을 생성한다. 긴 시간에 걸친 작업(Long-Horizon Task)에서도 높은 성능을 유지할 수 있다는 장점이 있다.

순환 신경망(RNN, Recurrent Neural Network) 역시 시간 정보를 처리하는 데 사용된다. 트랜스포머보다 계산량이 적기 때문에 임베디드(Embedded) 로봇에서는 여전히 활용되는 경우가 있으며, 제한된 하드웨어에서 효율적인 시간 모델링을 제공한다.

결정론적 정책은 대부분 교사 강요(Teacher Forcing) 방식으로 학습된다. 사람의 원격 조작(Teleoperation), 전문가 시연(Expert Demonstration), 산업 자동화 데이터, 시뮬레이션 데이터를 이용하여 입력과 행동의 관계를 지도학습(Supervised Learning)으로 학습한다.

행동 복제(Behavior Cloning)는 가장 대표적인 학습 방법이다. 사람이 수행한 작업을 그대로 학습하여 동일한 행동을 생성한다. 학습 속도가 빠르고 구현이 간단하지만, 장시간 작업에서는 작은 오차가 누적되어 성능이 저하될 수 있다는 한계도 존재한다.

이러한 문제를 분포 변화(Distribution Shift)라고 한다. 실제 환경에서 작은 예측 오차가 발생하면 로봇은 학습 데이터에 없던 새로운 상태에 진입하게 되고, 이후 예측 오차가 점점 커질 수 있다. 이를 해결하기 위해 데이터셋 집계(Dataset Aggregation), 온라인 수정(Online Correction), 강화학습(Reinforcement Learning)이 함께 사용된다.

결정론적 정책은 특히 산업용 제조 환경에서 매우 뛰어난 성능을 보인다. 생산 라인(Assembly Line), 반도체 제조(Semiconductor Manufacturing), 자동 검사(Inspection), 실험실 자동화(Laboratory Automation), 물류 자동화(Logistics Automation)처럼 반복성이 높은 작업에서는 항상 동일한 품질의 행동을 생성하는 것이 매우 중요하기 때문이다.

또 다른 큰 장점은 낮은 추론 지연(Low Inference Latency)이다. 결정론적 정책은 단 한 번의 순전파(Forward Pass)만 수행하면 즉시 행동을 생성할 수 있다. 확산 정책(Diffusion Policy)처럼 반복 계산이 필요하지 않으므로 실시간 제어에 매우 적합하다.

고주파 제어(High-Frequency Control)에서도 결정론적 정책은 강력한 성능을 보인다. 산업용 로봇은 수십에서 수백 Hz로 제어 명령을 갱신하며, 자율주행 로봇은 센서 정보를 실시간으로 반영하여 이동 속도를 수정해야 한다. 보행 로봇 역시 매 보행 주기마다 새로운 제어 명령을 생성해야 하므로 빠른 추론이 필수적이다.

결정론적 정책은 하드웨어 효율성(Hardware Efficiency)도 우수하다. 확률 분포 계산이나 샘플링(Sampling), 반복 최적화가 필요 없기 때문에 GPU, NPU(Neural Processing Unit), FPGA(Field Programmable Gate Array), 엣지 AI 컴퓨터(Edge AI Computer)에서 매우 효율적으로 실행된다.

정책 네트워크는 일반적으로 다층 퍼셉트론(MLP, Multi-Layer Perceptron)이나 트랜스포머 구조를 사용한다. 멀티모달 특징을 입력받아 여러 개의 은닉층(Hidden Layer)을 통과한 후 연속적인 행동 벡터(Action Vector)를 생성한다. 잔차 연결(Residual Connection), 정규화(Normalization), 활성화 함수(Activation Function)가 함께 사용되어 안정적인 학습을 지원한다.

목표 조건 정책(Goal-Conditioned Policy)은 일반화 성능을 크게 향상시킨다. 목표를 자연어(Natural Language), 물체 이름(Object Name), 위치(Coordinate), 작업(Task Description) 등으로 입력받아 동일한 환경에서도 서로 다른 행동을 생성할 수 있다.

로봇의 형태(Embodiment)에 따라 출력 구조도 달라진다. 차동 구동(Differential Drive)은 선속도와 각속도를 출력하고, 애커만 조향(Ackermann Steering)은 조향각과 가속도를 생성한다. 산업용 매니퓰레이터(Manipulator)는 각 관절의 위치를 출력하며, 휴머노이드는 전신 관절을 동시에 제어한다.

다중 작업 학습(Multi-Task Learning)은 하나의 정책이 자율주행, 조작, 충전, 검사, 사람과의 상호작용 등을 동시에 수행하도록 한다. 여러 작업이 동일한 환경 정보를 공유하기 때문에 학습 효율이 높고 새로운 작업에도 빠르게 적응할 수 있다.

계층형 제어(Hierarchical Control)는 작업을 여러 단계로 나누어 처리한다. 상위 계층은 작업 목표를 결정하고, 중간 계층인 정책 헤드는 연속적인 행동 목표를 생성하며, 하위 계층은 PID 제어기(PID Controller), 모델 예측 제어(MPC, Model Predictive Control), 역기구학(Inverse Kinematics) 등을 이용하여 실제 모터를 제어한다.

결정론적 정책은 기존 ROS 2(Robot Operating System 2), MoveIt, 행동 트리(Behavior Tree), 경로 계획기(Path Planner)와도 쉽게 연동된다. 기존 로봇 소프트웨어를 크게 변경하지 않고도 최신 AI 정책을 적용할 수 있다는 장점이 있다.

안전(Safety)은 정책과 독립적으로 관리된다. 정책 헤드는 행동을 생성하지만, 충돌 예측(Collision Prediction), 작업 공간 검증(Workspace Validation), 속도 제한(Velocity Limit), 힘 제한(Force Limit), 비상 정지(Emergency Stop)는 독립적인 안전 계층(Safety Layer)이 담당한다.

결정론적 정책의 가장 큰 단점은 다중 해답(Multi-Modal Action)을 표현하기 어렵다는 점이다. 하나의 물체를 여러 방향에서 잡을 수 있는 경우 평균적인 행동을 생성하여 오히려 좋지 않은 결과를 만들 수 있다. 이러한 이유로 최근에는 확산 정책(Diffusion Policy)이나 확률 기반 정책도 함께 연구되고 있다.

데이터 품질(Data Quality)은 정책 성능에 매우 큰 영향을 미친다. 다양한 물체 위치, 조명 변화, 센서 잡음, 환경 구조를 포함하는 대규모 데이터셋은 정책의 일반화 성능을 크게 향상시킨다. 실제로 모델 크기보다 데이터 품질이 성능에 더 큰 영향을 주는 경우도 많다.

시뮬레이션(Simulation)은 결정론적 정책 학습에서 매우 중요한 역할을 한다. 디지털 트윈(Digital Twin)을 이용하여 수백만 개의 입력과 행동 데이터를 생성할 수 있으며, 물리 엔진(Physics Engine), 조명 변화, 센서 잡음, 통신 지연 등을 함께 학습하여 실제 환경에 강한 정책을 만들 수 있다.

Sim-to-Real 기술은 시뮬레이션에서 학습한 정책을 실제 로봇으로 이전하는 기술이다. 적은 양의 실제 데이터를 이용한 미세조정(Fine-Tuning)만으로도 실제 환경에서 높은 성능을 얻을 수 있다.

파라미터 효율적 미세조정(PEFT, Parameter-Efficient Fine-Tuning)은 전체 모델을 다시 학습하지 않고 일부 계층만 수정하여 새로운 로봇 플랫폼에 빠르게 적용할 수 있도록 한다. 로봇의 관절 구조, 센서 구성, 작업 환경에 맞추어 효율적으로 적응할 수 있다.

최근에는 결정론적 정책에도 불확실성 추정(Uncertainty Estimation)이 함께 사용된다. 행동과 함께 신뢰도를 예측하여 신뢰도가 낮은 경우에는 추가 센서 정보를 요청하거나 사람의 개입(Human Intervention)을 요청하거나 더 안전한 행동을 선택한다.

메모리(Memory)는 장시간 작업에서도 중요한 역할을 한다. 에피소드 메모리(Episodic Memory)는 이전 작업 경험을 저장하고, 의미 메모리(Semantic Memory)는 환경 정보를 저장하며, 절차 메모리(Procedural Memory)는 반복 가능한 기술(Skill)을 저장한다. 정책은 이러한 정보를 활용하여 더욱 안정적인 행동을 생성한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)도 정책 생성에 활용된다. 설비 매뉴얼, 지도(Map), 유지보수 문서, 작업 기록, 기업 데이터베이스를 검색하여 현재 상황에 가장 적합한 행동을 생성할 수 있다.

산업용 조립(Industrial Assembly)은 결정론적 정책이 가장 적합한 분야 중 하나이다. 조립, 용접(Welding), 나사 체결(Fastening), 접착(Adhesive Application), 검사(Inspection), 포장(Packaging)처럼 반복성과 정밀도가 중요한 작업에서는 항상 동일한 행동을 생성하는 것이 매우 중요하다.

물류 창고(Warehouse)에서도 결정론적 정책은 뛰어난 성능을 보인다. AMR은 반복적인 경로를 이동하고, 정해진 위치에 도킹(Docking)하며, 자동 충전(Auto Charging)을 수행하므로 예측 가능한 행동이 전체 시스템의 안정성을 높인다.

의료 로봇(Healthcare Robot) 역시 약품 배송(Medication Delivery), 검체 운반(Specimen Transport), 실험실 자동화(Laboratory Automation)와 같이 반복성과 신뢰성이 중요한 작업에서 결정론적 정책을 많이 활용한다.

농업 로봇(Agricultural Robot)은 작물 관리(Crop Monitoring), 정밀 방제(Precision Spraying), 수확(Harvesting), 반복적인 농장 이동(Field Navigation) 등에서 연속적인 제어를 수행하므로 회귀 기반 정책이 매우 적합하다.

추론 최적화(Inference Optimization)는 결정론적 정책의 장점을 더욱 강화한다. 양자화(Quantization), 가지치기(Pruning), 혼합 정밀도(Mixed Precision), 효율적인 어텐션(Efficient Attention), GPU 최적화, 엣지 AI(Edge AI)를 적용하면 매우 낮은 지연 시간으로 실시간 제어가 가능하다.

결정론적 정책의 성능 평가는 일반 언어 모델과 다르다. 회귀 정확도(Regression Accuracy), 엔드이펙터 오차(End-Effector Error), 궤적 부드러움(Trajectory Smoothness), 제어 안정성(Control Stability), 작업 성공률(Task Success Rate), 에너지 소비(Energy Consumption), 사이클 타임(Cycle Time), 반복 정밀도(Repeatability), 충돌 발생률(Collision Rate), 장시간 운용 안정성(Long-Term Reliability) 등을 종합적으로 평가한다.

향후 결정론적 액션 헤드는 트랜스포머 기반 시간 모델링(Temporal Modeling), 멀티모달 융합(Multimodal Fusion), 세계 모델(World Model), 불확실성 추정(Uncertainty Estimation), 지속적 학습(Continual Learning), 외부 메모리(External Memory), 검색 증강 생성(RAG), 계층형 제어(Hierarchical Control)를 하나의 통합 액션 모델(Action Model)로 발전시킬 것으로 예상된다.

결국 결정론적 회귀 기반 액션 헤드는 멀티모달 인식(Multimodal Perception), 연속 회귀(Continuous Regression), 시간 모델링(Temporal Modeling), 계층형 계획(Hierarchical Planning), 실시간 추론(Real-Time Inference), 모방 학습(Imitation Learning), 외부 지식(RAG), 미세조정(Fine-Tuning), 안전 검증(Safety Verification)을 통합하여 산업 자동화, 물류, 의료, 농업, 시설 점검 등 반복성과 신뢰성이 요구되는 다양한 물리 AI(Physical AI) 시스템의 핵심 실행 엔진(Execution Engine)으로 자리잡고 있다.

## 4.3 Diffusion-Based Action Policies (with Code)

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

확산 기반 액션 헤드(Diffusion-Based Action Head)는 최근 로봇 정책 학습(Robot Policy Learning)에서 가장 주목받는 기술 중 하나이다. 기존의 결정론적 정책(Deterministic Policy)은 하나의 입력에 대해 하나의 행동만을 예측하지만, 확산 정책(Diffusion Policy)은 하나의 상황에서 여러 개의 가능한 행동(Action Distribution)을 모두 학습한다. 실제 로봇 환경에서는 동일한 목표를 달성할 수 있는 여러 경로와 여러 조작 방법이 존재하므로, 이러한 확률 기반 행동 생성은 훨씬 높은 유연성과 일반화 성능을 제공한다.

결정론적 회귀 정책(Regression Policy)은 하나의 정답(Action)을 예측하는 것을 목표로 한다. 그러나 실제 환경에서는 여러 개의 정답이 존재하는 경우가 많다. 예를 들어 컵을 집을 때 왼쪽에서 접근하거나 오른쪽에서 접근하거나 모두 성공할 수 있다. 회귀 기반 정책은 이러한 여러 행동을 평균화(Averaging)하는 경향이 있어 실제 사람이 수행하지 않은 중간 경로를 생성할 수 있으며, 이는 조작 성능을 저하시킬 수 있다.

확산 기반 정책은 이러한 문제를 해결하기 위해 행동(Action)을 직접 예측하지 않는다. 대신 초기의 무작위 가우시안 노이즈(Gaussian Noise)에서 시작하여 여러 번의 노이즈 제거(Denoising)를 반복하면서 점진적으로 실제 행동 궤적(Action Trajectory)을 생성한다. 최종 행동은 여러 단계의 반복적인 정제(Refinement)를 거쳐 완성되므로 더욱 자연스럽고 안정적인 움직임을 생성할 수 있다.

이 개념은 원래 이미지 생성(Image Generation)을 위해 개발된 DDPM(Denoising Diffusion Probabilistic Model)에서 시작되었다. 학습 과정에서는 전문가 행동(Expert Demonstration)에 점진적으로 노이즈를 추가하여 완전히 무작위 상태까지 변형시키고, 신경망은 반대로 노이즈를 제거하는 방법을 학습한다. 추론 시에는 순수한 노이즈에서 시작하여 반복적으로 노이즈를 제거하면서 실제 로봇이 수행할 행동을 복원하게 된다.

자동회귀(Auto-Regressive) 방식은 한 번에 하나의 행동을 생성하지만, 확산 정책은 여러 시점(Time Step)의 행동을 동시에 최적화한다. 따라서 하나의 제어 명령만 생성하는 것이 아니라 미래의 전체 행동 궤적(Trajectory)을 함께 생성할 수 있으며, 시간적 일관성(Temporal Consistency)이 크게 향상된다.

비전-언어-행동(VLA, Vision-Language-Action) 아키텍처에서 RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 촉각 센서(Tactile Sensor), 힘 센서(Force Sensor), 로봇 내부 상태(Proprioception) 등이 먼저 환경을 인식한다. 대규모 언어 모델(LLM, Large Language Model)은 작업 목표와 사용자 명령을 이해하며, 멀티모달 융합(Multimodal Fusion)은 이 정보를 하나의 잠재 표현(Latent Representation)으로 통합한다. 확산 액션 헤드는 이 정보를 조건(Condition)으로 사용하여 행동을 생성한다.

조건부 확산(Conditional Diffusion)은 확산 정책의 핵심 기술이다. 노이즈를 제거하는 모든 단계에서 물체 위치(Object Position), 작업 목표(Task Goal), 로봇 상태(Robot State), 이전 행동(Previous Action), 자연어 명령(Language Instruction) 등이 함께 입력된다. 따라서 생성되는 행동은 단순히 무작위가 아니라 현재 작업 목적에 정확하게 맞는 행동이 된다.

순방향 확산(Forward Diffusion)은 학습 과정에서만 사용된다. 전문가 행동에 점차 노이즈를 추가하여 최종적으로는 완전한 랜덤 데이터(Random Noise)로 만든다. 이후 신경망은 각 단계에서 추가된 노이즈를 예측하도록 학습하며, 결과적으로 원래 행동을 복원하는 능력을 얻게 된다.

실제 추론(Inference)에서는 역확산(Reverse Diffusion)이 수행된다. 초기의 순수한 노이즈에서 시작하여 반복적으로 노이즈를 제거하면서 점점 더 실제 행동과 유사한 궤적으로 변환한다. 모든 반복이 끝나면 최종적으로 실행 가능한 행동(Action Sequence)이 생성된다.

확산 정책은 다중 행동(Multi-Modal Behavior)을 자연스럽게 표현할 수 있다. 동일한 작업이라도 초기 노이즈가 달라지면 서로 다른 행동이 생성될 수 있으며, 각각이 모두 성공적인 결과를 만들 수 있다. 이는 기존 결정론적 정책과 가장 큰 차이점이다.

미래 행동 예측(Action Horizon Prediction)도 확산 정책의 중요한 장점이다. 현재 행동만 생성하는 것이 아니라 수십 개의 미래 제어 명령을 동시에 생성할 수 있다. 이후 모델 예측 제어(MPC, Model Predictive Control)는 일부만 실행하고 다시 새로운 궤적을 생성하는 반복적 계획(Receding Horizon Planning)을 수행한다.

행동 예측 길이(Action Horizon)는 성능에 큰 영향을 준다. 짧은 예측은 빠른 반응성을 제공하며 장애물 회피에 적합하고, 긴 예측은 조립(Assembly), 장거리 이동(Long Navigation), 복잡한 조작(Complex Manipulation)과 같은 장기 작업(Long-Horizon Task)에 유리하다.

최근에는 트랜스포머 기반(Transformer-Based) 확산 정책이 가장 널리 사용된다. 셀프 어텐션(Self-Attention)은 여러 시점의 행동을 동시에 분석하며, 크로스 어텐션(Cross-Attention)은 언어, 영상, 센서 정보를 지속적으로 행동 생성 과정에 반영한다. 이를 통해 환경 변화에도 안정적인 행동을 생성할 수 있다.

UNet 구조도 확산 정책에서 많이 사용된다. 인코더(Encoder)는 전체적인 행동 구조를 이해하고, 디코더(Decoder)는 세부적인 움직임을 복원한다. 스킵 연결(Skip Connection)은 세밀한 행동 정보를 유지하면서도 전체적인 행동 계획을 함께 고려할 수 있도록 한다.

노이즈 스케줄링(Noise Scheduling)은 확산 정책의 중요한 요소이다. 노이즈를 얼마나 빠르게 추가하고 제거할 것인지에 따라 생성 품질과 학습 안정성이 달라진다. 선형 스케줄(Linear Schedule), 코사인 스케줄(Cosine Schedule), 학습 기반 스케줄(Learned Schedule) 등이 사용된다.

확산 정책은 행동 자체를 학습하는 것이 아니라 추가된 노이즈를 예측한다. 일반적으로 평균제곱오차(MSE, Mean Squared Error)를 이용하여 예측한 노이즈와 실제 노이즈의 차이를 최소화하도록 학습한다. 이러한 방식은 다양한 행동 분포를 안정적으로 학습할 수 있게 해준다.

확산 정책은 본질적으로 불확실성(Uncertainty)을 표현할 수 있다. 서로 다른 초기 노이즈는 서로 다른 행동을 생성하며, 여러 행동 모두 성공 가능성이 있다. 따라서 복잡한 환경에서도 다양한 해결책을 제시할 수 있다.

접촉 기반 조작(Contact-Rich Manipulation)은 확산 정책이 가장 뛰어난 성능을 보이는 분야이다. 부품 삽입(Insertion), 케이블 연결(Cable Routing), 천 조작(Cloth Manipulation), 식품 조리(Food Preparation), 의료 보조(Surgical Assistance)처럼 접촉(Contact)이 많은 작업에서는 기존 회귀 정책보다 훨씬 높은 성공률을 보인다.

장기 작업(Long-Horizon Task)도 확산 정책에 적합하다. 조립, 물류 피킹(Picking), 자동 점검(Inspection), 농업 수확(Harvesting), 협업 제조(Collaborative Manufacturing) 등은 여러 단계의 행동을 하나의 궤적으로 생성하기 때문에 전체 작업의 일관성이 크게 향상된다.

언어 조건 확산(Language-Conditioned Diffusion)은 자연어 명령을 행동 생성 과정에 직접 반영한다. 작업 목표(Task Goal), 안전 규칙(Safety Rule), 작업 대상(Object), 작업 순서(Task Sequence)가 모든 노이즈 제거 단계에서 함께 사용되므로 사람이 원하는 행동을 더욱 정확하게 생성할 수 있다.

멀티모달 조건(Multimodal Conditioning)은 RGB 영상, 깊이 정보, 의미 지도(Semantic Map), 촉각 정보, 위치 추정(Localization), 자연어를 동시에 활용한다. 확산 모델은 이러한 다양한 정보를 통합하여 환경 변화에 강한 행동을 생성한다.

로봇의 형태(Embodiment)에 따라 출력도 달라진다. 매니퓰레이터(Manipulator)는 관절 궤적(Joint Trajectory)과 그리퍼 제어를 생성하고, AMR은 이동 속도와 조향 명령을 생성하며, 보행 로봇은 발 위치(Foot Placement)와 균형(Balance)을 함께 생성한다. 휴머노이드는 전신 관절을 동시에 제어한다.

실행 단계에서는 일반적으로 반복적 계획(Receding Horizon Execution)을 사용한다. 전체 궤적을 생성하더라도 처음 일부만 실행하고, 새로운 센서 정보를 반영하여 다시 새로운 궤적을 생성한다. 이를 통해 장기 계획과 실시간 반응성을 동시에 확보할 수 있다.

시뮬레이션(Simulation)은 확산 정책 학습에서 매우 중요한 역할을 한다. 디지털 트윈(Digital Twin), 아이작 심(Isaac Sim) 등을 이용하여 수백만 개의 행동 데이터를 생성할 수 있으며, 다양한 물체 배치, 조명, 접촉 상황, 센서 오차 등을 포함하여 강인한 정책을 학습할 수 있다.

확산 정책 역시 행동 복제(Behavior Cloning)를 이용하여 학습한다. 원격 조작(Teleoperation), VR 조작, 산업 자동화 데이터, 강화학습 정책 등을 전문가 데이터로 사용하며, 하나의 행동이 아니라 전체 행동 분포를 학습한다는 점이 기존 회귀 정책과 다르다.

미세조정(Fine-Tuning)은 파운데이션 정책(Foundation Policy)을 특정 로봇 플랫폼이나 산업 환경에 맞게 최적화하는 과정이다. 소량의 실제 데이터만으로도 제조, 물류, 의료, 농업 등 다양한 분야에 빠르게 적용할 수 있다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 유지보수 문서, 시설 지도(Map), 디지털 트윈, 과거 작업 기록, 소프트웨어 문서 등을 검색하여 행동 생성에 활용한다. 외부 지식을 함께 사용함으로써 더욱 상황에 적합한 행동을 생성할 수 있다.

메모리(Memory)는 장시간 작업에서 매우 중요하다. 에피소드 메모리(Episodic Memory)는 이전 성공 사례를 저장하고, 의미 메모리(Semantic Memory)는 환경 정보를 저장하며, 절차 메모리(Procedural Memory)는 반복 가능한 기술(Skill)을 저장한다. 이러한 정보는 확산 과정의 조건으로 사용되어 행동 품질을 향상시킨다.

세계 모델(World Model)은 미래 환경 변화를 예측한다. 사람의 이동, 장애물 변화, 배터리 소비, 작업 진행 상황 등을 예측하여 미래 상태를 고려한 행동을 생성하므로 장기 계획 성능이 크게 향상된다.

어포던스 추론(Affordance Reasoning)은 실제 수행 가능한 행동만 생성하도록 제한한다. 접근 가능한 물체, 충돌이 없는 경로, 안정적인 파지(Grasp), 이동 가능한 공간 등을 미리 분석하여 비현실적인 행동이 생성되지 않도록 한다.

안전(Safety)은 확산 정책과 독립적으로 관리된다. 충돌 예측(Collision Prediction), 기하학적 검증(Geometric Verification), 운동학 제한(Kinematic Constraint), 비상 정지(Emergency Stop), 런타임 모니터(Runtime Monitor)는 생성된 행동을 별도로 검증한 후 실행을 허가한다.

확산 정책의 가장 큰 단점은 계산량(Computational Complexity)이 크다는 점이다. 일반적인 DDPM은 수십에서 수백 번의 노이즈 제거를 수행해야 하므로 회귀 정책보다 추론 시간이 길다. 이를 해결하기 위해 DDIM(Denoising Diffusion Implicit Model), 일관성 모델(Consistency Model), 증류(Distillation), 잠재 확산(Latent Diffusion), 적응형 샘플링(Adaptive Sampling) 등이 활발히 연구되고 있다.

실시간 로봇에서는 GPU, Tensor Core, AI 가속기(AI Accelerator), 혼합 정밀도(Mixed Precision), 양자화(Quantization), 가지치기(Pruning)를 이용하여 추론 시간을 크게 단축한다. 이를 통해 확산 정책도 실시간 제어에 적용할 수 있도록 발전하고 있다.

클라우드-엣지 협업(Cloud-Edge Collaboration)은 현실적인 해결책이다. 엣지 로봇은 압축된 확산 모델로 즉시 행동을 생성하고, 클라우드는 장기 계획(Long-Horizon Planning), 재학습(Retraining), 디지털 트윈 시뮬레이션을 수행한다. 네트워크가 끊겨도 기본적인 자율성은 유지된다.

산업 제조에서는 정밀 조립, 용접(Welding), 연마(Polishing), 케이블 조립(Cable Routing)과 같이 여러 가능한 경로가 존재하는 작업에서 확산 정책이 우수한 성능을 보인다. 물류 창고에서는 유연한 피킹(Picking), 팔레트 운반, 장애물 회피 등에 활용되며, 의료 분야에서는 수술 보조, 재활 로봇, 실험실 자동화 등에 적용되고 있다. 농업에서는 과일 수확, 가지치기, 작물 관리 등 다양한 자연환경 작업에 활용되고 있다.

확산 정책은 단순한 정확도만으로 평가하지 않는다. 궤적 부드러움(Trajectory Smoothness), 작업 성공률(Task Success Rate), 조작 성공률(Manipulation Success), 접촉 안정성(Contact Stability), 에너지 소비(Energy Consumption), 행동 다양성(Trajectory Diversity), 환경 변화에 대한 강인성(Robustness), 충돌 빈도(Collision Frequency), 추론 시간(Inference Latency), 장시간 운용 안정성(Long-Term Reliability)을 함께 평가한다.

향후 확산 기반 액션 헤드는 트랜스포머 추론(Transformer Reasoning), 멀티모달 인식(Multimodal Perception), 세계 모델(World Model), 외부 메모리(External Memory), 검색 증강 생성(RAG), 지속적 학습(Continual Learning), 강화학습(Reinforcement Learning), 계층형 계획(Hierarchical Planning)을 통합한 차세대 물리 AI(Physical AI) 구조로 발전할 것으로 예상된다. 또한 고속 확산(Fast Diffusion), 일관성 모델(Consistency Model), 잠재 확산(Latent Diffusion), 하드웨어 최적화(Hardware Optimization)가 결합되어 실시간 성능도 지속적으로 향상될 것이다.

결국 DDPM 기반 확산 액션 헤드는 확률적 행동 생성(Probabilistic Action Generation), 반복적 노이즈 제거(Iterative Denoising), 멀티모달 조건(Multimodal Conditioning), 트랜스포머 기반 시퀀스 학습(Transformer Sequence Learning), 세계 모델(World Model), 검색 증강 생성(RAG), 시뮬레이션 기반 학습(Simulation-Based Learning), 미세조정(Fine-Tuning), 하드웨어 가속(Hardware Acceleration), 안전 검증(Safety Verification)을 하나의 통합 액션 모델(Action Model)로 결합하여, 복잡한 실제 환경에서도 다양하고 안정적이며 실행 가능한 행동을 생성하는 차세대 물리 AI의 핵심 정책 기술로 자리잡고 있다.

## 4.4 Flow-Matching Action Models (with Code)

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

플로우 매칭(Flow Matching)은 최근 로봇 정책(Action Policy) 학습에서 가장 주목받는 생성형(Generative) 기술 중 하나이다. 기존의 확산 모델(DDPM, Denoising Diffusion Probabilistic Model)은 매우 뛰어난 행동 생성 능력을 보여주었지만, 추론(Inference) 시 수십에서 수백 번의 노이즈 제거(Denoising)를 반복해야 하므로 실시간 로봇에는 계산량이 큰 문제가 있었다. Flow Matching은 이러한 문제를 해결하기 위해 연속적인 벡터 필드(Vector Field)를 직접 학습하여 훨씬 적은 계산으로 동일한 수준의 행동 생성 성능을 제공한다. 최근 Pi0(Physical Intelligence 0)와 같은 차세대 로봇 파운데이션 모델(Foundation Model)에서 핵심 정책 구조로 채택되고 있다.

Flow Matching의 핵심 개념은 로봇 행동을 하나의 결과가 아니라 연속적인 변환(Continuous Transformation) 과정으로 보는 것이다. 기존 회귀 기반 정책(Regression Policy)은 하나의 행동을 직접 예측하고, 확산 정책(Diffusion Policy)은 노이즈를 반복적으로 제거하여 행동을 생성한다. 반면 Flow Matching은 초기의 단순한 확률 분포(Simple Distribution)를 목표 행동(Target Action Distribution)으로 이동시키는 연속적인 흐름(Flow)을 학습한다. 따라서 행동 생성이 하나의 연속적인 동역학(Dynamics) 문제가 된다.

Flow Matching은 행동(Action) 자체를 예측하지 않는다. 대신 현재 위치에서 목표 행동으로 이동하기 위한 순간 속도(Instantaneous Velocity)를 예측한다. 즉, 신경망은 현재 상태에서 어느 방향(Direction)으로 얼마나 이동해야 하는지를 나타내는 벡터 필드(Vector Field)를 학습한다. 이러한 속도장을 시간(Time)에 따라 적분(Integration)하면 최종 행동 궤적(Action Trajectory)을 얻을 수 있다.

비전-언어-행동(VLA, Vision-Language-Action) 아키텍처에서 RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 촉각 센서(Tactile Sensor), 내부 상태(Proprioception), 위치 추정(Localization) 등은 환경 정보를 의미 특징(Semantic Feature)으로 변환한다. 대규모 언어 모델(LLM, Large Language Model)은 자연어 명령과 작업 목표(Task Goal)를 이해하며, 멀티모달 융합(Multimodal Fusion)은 이를 하나의 잠재 표현(Latent Representation)으로 통합한다. Flow Matching 액션 헤드(Action Head)는 이 정보를 조건(Condition)으로 사용하여 행동을 생성한다.

Flow Matching은 DDPM과 근본적으로 다른 학습 방식을 가진다. DDPM은 먼저 전문가 행동에 노이즈를 추가하고 이를 제거하는 방법을 학습하지만, Flow Matching은 처음부터 목표 행동으로 이동하는 경로(Path)를 정의하고 그 경로를 따라 이동하는 속도(Velocity)를 직접 학습한다. 따라서 노이즈 제거 과정을 별도로 학습할 필요가 없으며 구조가 더욱 단순하다.

Flow Matching의 이론적 기반은 최적 수송(Optimal Transport)에 있다. 확률 분포(Probability Distribution)를 서로 독립된 데이터가 아니라 공간 안에서 이동하는 질량(Mass)으로 해석한다. 신경망은 이러한 확률 질량이 어떻게 이동해야 하는지를 나타내는 벡터 필드를 학습하며, 이는 실제 로봇의 연속적인 움직임과 매우 유사한 특성을 가진다.

학습 과정에서는 보간 경로(Interpolation Path)가 사용된다. 하나의 단순한 초기 분포(Source Distribution)와 전문가 행동(Target Distribution) 사이의 중간 상태를 생성하고, 각 지점에서 목표 방향으로 이동하기 위한 속도를 학습한다. 즉, 모든 중간 상태에서 올바른 이동 방향을 학습함으로써 전체 행동 생성 과정을 완성한다.

추론(Inference)에서는 확산 모델처럼 반복적인 노이즈 제거를 수행하지 않는다. 초기 잠재 변수(Latent Variable)에서 시작하여 학습된 벡터 필드를 따라 적분만 수행하면 최종 행동이 생성된다. 오일러 적분(Euler Integration), 룽게-쿠타(Runge-Kutta), 적응형 적분기(Adaptive Solver) 등이 일반적으로 사용된다.

Flow Matching은 결정론적(Deterministic) 특성도 가진다. 동일한 초기 상태와 동일한 입력이 주어지면 항상 동일한 행동을 생성하므로 디버깅(Debugging), 검증(Verification), 인증(Certification)에 유리하다. 반면 초기 잠재 변수만 변경하면 서로 다른 행동도 생성할 수 있어 행동 다양성(Diversity)도 유지할 수 있다.

추론 속도(Inference Speed)는 Flow Matching의 가장 큰 장점이다. DDPM은 일반적으로 50\~100회의 반복 계산이 필요하지만, Flow Matching은 몇 번의 적분만으로 동일한 수준의 결과를 생성할 수 있다. 따라서 실시간 자율주행, 조작, 산업용 로봇에서 매우 적합한 정책 구조로 평가받는다.

실시간 로봇에서는 수 밀리초(ms)의 지연도 매우 중요하다. 자율주행 로봇은 장애물을 즉시 회피해야 하고, 산업용 매니퓰레이터는 이동 중인 부품을 정확하게 잡아야 하며, 휴머노이드는 사람의 움직임에 즉시 반응해야 한다. Flow Matching의 낮은 지연 시간(Low Latency)은 이러한 응용 분야에서 큰 장점을 제공한다.

Flow Matching은 단순한 한 단계 행동이 아니라 전체 행동 궤적(Trajectory)을 생성한다. 연속적인 벡터 필드(Vector Field)를 따라 미래의 여러 행동이 동시에 생성되므로 시간적 일관성(Temporal Consistency)이 매우 뛰어나다. 결과적으로 급격한 방향 전환이나 불필요한 가속이 감소하여 더욱 부드럽고 안정적인 움직임을 만든다.

최근에는 트랜스포머 기반(Transformer-Based) Flow Matching이 가장 널리 연구되고 있다. 셀프 어텐션(Self-Attention)은 영상, 언어, 환경 정보, 과거 행동을 동시에 분석하며, 크로스 어텐션(Cross-Attention)은 멀티모달 정보를 지속적으로 벡터 필드 생성 과정에 반영한다. 이러한 구조는 대규모 파운데이션 모델 학습에 매우 적합하다.

Flow Matching은 연속 시간(Continuous Time)을 직접 모델링한다. 기존 시퀀스 모델처럼 고정된 시간 간격으로 행동을 생성하는 것이 아니라 임의의 시간(Time)에 대한 행동을 계산할 수 있다. 따라서 다양한 제어 주기(Control Frequency)에서도 동일한 정책을 사용할 수 있으며 매우 높은 유연성을 가진다.

로봇의 형태(Embodiment)에 따라 출력도 달라진다. 산업용 매니퓰레이터는 관절 궤적(Joint Trajectory), 엔드이펙터 자세(End-Effector Pose), 그리퍼 힘(Gripper Force)을 생성하고, AMR은 속도(Velocity), 조향(Steering), 이동 경로(Path)를 생성한다. 사족보행 로봇(Quadruped)은 발 위치(Foot Placement)와 균형(Balance)을 생성하며, 휴머노이드는 전신 관절을 동시에 제어한다.

멀티모달 조건(Multimodal Conditioning)은 Flow Matching에서도 매우 중요하다. 비전(Vision)은 물체의 위치와 형태를 제공하고, 의미 지도(Semantic Map)는 환경 구조를 제공하며, 촉각(Tactile)은 접촉 상태를 제공하고, 자연어(Language)는 작업 목표를 제공한다. 이러한 다양한 정보가 하나의 벡터 필드 생성 과정에 통합된다.

장기 작업(Long-Horizon Task)에서도 Flow Matching은 뛰어난 성능을 보인다. 조립(Assembly), 물류(Logistics), 시설 점검(Inspection), 협업 제조(Collaborative Manufacturing), 가정용 서비스(Service Robot), 농업(Agriculture)과 같이 여러 단계의 행동이 필요한 작업에서는 전체 궤적을 하나의 연속적인 흐름으로 생성하므로 일관성이 크게 향상된다.

세계 모델(World Model)은 Flow Matching의 성능을 더욱 향상시킨다. 현재 환경뿐 아니라 사람의 이동, 물체의 움직임, 장애물 변화, 배터리 상태 등을 미래까지 예측하여 벡터 필드 생성에 반영한다. 결과적으로 미래 환경까지 고려한 행동을 생성할 수 있다.

어포던스 추론(Affordance Reasoning)은 실제 수행 가능한 행동만 생성하도록 한다. 접근 가능한 물체, 충돌 없는 경로, 안정적인 파지(Grasp), 이동 가능한 공간 등을 분석하여 실행 불가능한 행동은 생성되지 않도록 제한한다.

안전(Safety)은 Flow Matching과 독립적으로 유지된다. 충돌 예측(Collision Prediction), 기하학적 검증(Geometric Verification), 운동학 제한(Kinematic Constraint), 동역학 제한(Dynamic Constraint), 비상 정지(Emergency Stop), 런타임 모니터(Runtime Monitor)는 생성된 행동을 항상 검증한 후 실행을 허가한다.

Flow Matching은 대부분 모방 학습(Imitation Learning)으로 학습된다. 원격 조작(Teleoperation), VR 조작, 전문가 시연(Expert Demonstration), 시뮬레이션(Simulation), 강화학습(Reinforcement Learning)의 결과를 이용하여 연속적인 행동 흐름을 학습한다. 단순한 행동이 아니라 행동이 변화하는 과정 전체를 학습한다는 점이 특징이다.

시뮬레이션은 매우 중요한 역할을 한다. 디지털 트윈(Digital Twin)은 수백만 개의 행동 궤적을 생성하며, 다양한 조명, 센서 잡음, 물체 위치, 날씨, 장비 상태를 포함한 데이터를 제공한다. 이러한 대규모 데이터는 실제 환경에서도 강인한 정책을 만드는 기반이 된다.

Sim-to-Real 기술은 시뮬레이션에서 학습한 Flow Matching 정책을 실제 로봇에 적용하기 위한 기술이다. 물리 오차, 센서 잡음, 통신 지연 등을 포함한 도메인 랜덤화(Domain Randomization)를 적용한 후 소량의 실제 데이터로 미세조정(Fine-Tuning)을 수행한다.

파라미터 효율적 적응(Parameter-Efficient Adaptation)은 하나의 파운데이션 정책(Foundation Policy)을 다양한 로봇 플랫폼에 적용할 수 있도록 한다. 전체 모델을 다시 학습하지 않고 일부 계층만 수정하여 산업용 로봇, 물류 AMR, 농업 로봇, 의료 로봇 등에 빠르게 적용할 수 있다.

메모리(Memory)는 Flow Matching에서도 중요한 역할을 한다. 에피소드 메모리(Episodic Memory)는 과거 성공 사례를 저장하고, 의미 메모리(Semantic Memory)는 환경 정보를 저장하며, 절차 메모리(Procedural Memory)는 기술(Skill)을 저장한다. 이러한 정보는 벡터 필드 생성 시 조건 정보로 활용된다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 기업 문서, 유지보수 매뉴얼, 디지털 트윈, 작업 기록, 시설 지도(Map) 등을 검색하여 행동 생성에 반영한다. 따라서 모델 내부의 지식뿐 아니라 최신 외부 정보를 활용할 수 있다.

계층형 제어(Hierarchical Control)는 상위 계획기(Planner)가 작업 순서를 결정하고, Flow Matching 액션 헤드는 연속적인 행동 궤적을 생성하며, 하위 제어기(PID, MPC, Impedance Control)는 실제 모터를 제어한다. 이를 통해 AI와 기존 제어 기술을 자연스럽게 결합할 수 있다.

Flow Matching은 DDPM보다 추론 과정이 더 안정적이다. 반복적인 확률 샘플링(Stochastic Sampling)이 없기 때문에 결과의 재현성(Reproducibility)이 높고 디버깅과 인증이 쉬워 산업 현장에 적합하다.

다만 적분 정확도(Numerical Integration Accuracy)는 중요한 과제이다. 적분 단계를 너무 줄이면 행동 품질이 낮아지고, 너무 많이 사용하면 계산량이 증가한다. 따라서 적응형 적분기(Adaptive Solver)를 사용하여 정확도와 속도의 균형을 맞춘다.

GPU, Tensor Core, NPU(Neural Processing Unit), 혼합 정밀도(Mixed Precision), 양자화(Quantization), 메모리 최적화(Memory Optimization)를 이용하면 Flow Matching의 추론 속도를 더욱 향상시킬 수 있다. 반복적인 노이즈 제거가 없기 때문에 하드웨어 효율도 매우 높다.

산업 제조에서는 정밀 조립, 삽입(Insertion), 용접(Welding), 연마(Polishing), 케이블 조립(Cable Routing) 등에 활용되며, 물류에서는 유연한 피킹(Picking), 팔레트 운반, 다중 AMR 협업에 적용된다. 의료에서는 재활, 수술 보조, 약품 배송, 실험실 자동화에 활용되며, 농업에서는 수확(Harvesting), 가지치기(Pruning), 작물 관리 등에 적용된다.

Flow Matching의 평가는 단순한 예측 정확도가 아니라 궤적 부드러움(Trajectory Smoothness), 작업 성공률(Task Success Rate), 시간적 일관성(Temporal Consistency), 에너지 효율(Energy Efficiency), 액추에이터 마모(Actuator Wear), 추론 지연(Inference Latency), 충돌 발생률(Collision Frequency), 환경 변화에 대한 강인성(Robustness), 장시간 운용 안정성(Long-Term Reliability)을 함께 평가한다.

최근 Pi0와 같은 파운데이션 로봇 모델은 Flow Matching이 범용 로봇 지능(General-Purpose Embodied Intelligence)으로 확장될 수 있음을 보여주고 있다. 영상, 언어, 행동, 환경 정보를 대규모로 학습하여 하나의 정책이 다양한 로봇과 다양한 작업을 수행할 수 있는 기반을 마련하고 있다.

향후에는 세계 모델(World Model), 계층형 계획(Hierarchical Planning), 멀티모달 추론(Multimodal Reasoning), 검색 증강 생성(RAG), 외부 메모리(External Memory), 지속적 학습(Continual Learning), 강화학습(Reinforcement Learning), 적응형 적분기(Adaptive Solver), 하드웨어 가속(Hardware Acceleration)이 모두 통합된 차세대 Flow Matching 구조가 등장할 것으로 예상된다.

결국 Pi0 스타일의 Flow Matching 액션 헤드는 연속 확률 수송(Continuous Probability Transport), 벡터 필드 학습(Vector Field Learning), 멀티모달 융합(Multimodal Fusion), 장기 행동 생성(Long-Horizon Trajectory Generation), 세계 모델(World Model), 검색 증강 생성(RAG), 시뮬레이션 기반 파운데이션 학습(Foundation Learning), 실시간 하드웨어 최적화(Hardware Optimization), 안전 검증(Safety Verification)을 하나의 통합 액션 모델(Action Model)로 결합하여, 차세대 물리 AI(Physical AI)에서 빠르고 부드러우며 확장성이 뛰어난 범용 로봇 정책의 핵심 기술로 자리잡고 있다.

## 4.5 Discrete Action Tokenization (with Code)

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

이산 행동 토큰화(Discrete Action Tokenization)는 현대 비전-언어-행동(VLA, Vision-Language-Action) 아키텍처에서 가장 중요한 기술 중 하나이다. 이 기술은 연속적인 로봇 제어(Continuous Robot Control)를 자연어 처리(NLP, Natural Language Processing)의 단어(Token) 생성과 유사한 토큰(Token) 생성 문제로 변환한다. 대규모 언어 모델(LLM, Large Language Model)이 단어를 생성하듯이, 로봇도 행동(Action)을 토큰 형태로 생성하도록 학습할 수 있다는 아이디어에서 출발하였다. 대표적인 사례가 RT-2(Robotics Transformer 2)이며, 이를 통해 로봇 파운데이션 모델(Robot Foundation Model)의 가능성이 크게 확대되었다.

기존 로봇은 관절 위치(Joint Position), 속도(Joint Velocity), 토크(Torque), 조향각(Steering Angle), 휠 속도(Wheel Velocity)와 같은 연속적인 제어 값을 직접 생성하였다. 이러한 방식은 높은 정밀도를 제공하지만, 행동 공간(Action Space)이 매우 크고 복잡하여 학습이 어렵다. 또한 작은 오차가 반복되면 장기 작업(Long-Horizon Task)에서 성능이 크게 저하될 수 있다.

이산 행동 토큰화는 이러한 문제를 해결하기 위해 연속적인 행동을 유한한 개수의 행동 토큰(Action Token)으로 변환한다. 정책 모델은 더 이상 연속적인 제어 값을 직접 생성하지 않고, 미리 정의된 행동 토큰을 순차적으로 생성한다. 이후 별도의 디코더(Decoder)가 토큰을 실제 로봇 제어 명령으로 변환하여 실행한다. 즉, 로봇 행동 생성이 자연어 문장 생성과 동일한 문제로 변환된다.

언어 모델과의 유사성은 매우 크다. 자연어는 단어(Token)가 모여 문장을 구성하고, 문장이 의미를 전달한다. 마찬가지로 로봇도 이동(Move), 회전(Rotate), 집기(Grasp), 놓기(Place), 검사(Inspect)와 같은 행동 토큰들이 순차적으로 연결되어 하나의 작업(Task)을 수행한다. 따라서 동일한 트랜스포머(Transformer) 구조를 그대로 로봇에도 적용할 수 있다.

비전-언어-행동(VLA) 시스템에서는 RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 촉각 센서(Tactile Sensor), 힘 센서(Force Sensor), 내부 상태(Proprioception), 위치 추정(Localization) 등이 먼저 환경을 인식한다. 대규모 언어 모델은 자연어 명령과 작업 목표(Task Goal)를 이해하며, 멀티모달 융합(Multimodal Fusion)은 이 정보를 하나의 잠재 표현(Latent Representation)으로 통합한다. 이후 액션 토큰 헤드(Action Token Head)가 다음 행동 토큰을 생성한다.

행동 토큰(Action Token)을 만들기 위해서는 먼저 행동 어휘(Action Vocabulary)를 구축해야 한다. 원격 조작(Teleoperation), 산업 자동화, 시뮬레이션(Simulation), VR 조작, 전문가 시연(Expert Demonstration) 등을 통해 수집한 연속적인 행동 데이터를 분석하고, 자주 나타나는 행동 패턴을 클러스터링(Clustering)하거나 벡터 양자화(Vector Quantization)하여 각각의 행동에 고유한 토큰을 부여한다.

행동 토큰은 다양한 수준(Level)을 가질 수 있다. 저수준 토큰(Low-Level Token)은 관절 이동, 휠 회전, 그리퍼 개폐와 같은 기본 동작을 표현하고, 고수준 토큰(High-Level Token)은 집기(Grasp), 도킹(Docking), 충전(Charging), 문 열기(Open Door), 버튼 누르기(Button Press)와 같은 복합 기술(Skill)을 표현한다. 실제 시스템은 두 가지 수준을 함께 사용하는 경우가 많다.

행동 토큰을 자동으로 학습하기 위해 가장 많이 사용되는 기술 중 하나가 VQ-VAE(Vector Quantized Variational Autoencoder)이다. 연속적인 행동 데이터를 잠재 공간(Latent Space)으로 변환한 후 이를 코드북(Codebook)의 이산 벡터로 양자화(Quantization)한다. 이후 디코더는 다시 연속적인 행동으로 복원하며, 학습을 반복하면서 의미 있는 행동 토큰이 자동으로 생성된다.

기존 회귀 기반 정책(Regression Policy)은 연속적인 수치를 예측하지만, 행동 토큰 정책(Action Token Policy)은 범주(Classification)를 예측한다. 즉, 가능한 행동 토큰들 중에서 가장 적절한 토큰을 선택하는 문제로 바뀌며, 선택된 토큰은 다시 실제 로봇 제어 명령으로 변환된다.

자동회귀(Auto-Regressive) 생성 방식도 그대로 적용된다. 현재까지 생성된 행동 토큰과 환경 정보, 언어 명령을 입력으로 받아 다음 행동 토큰을 하나씩 생성한다. 이러한 과정은 GPT와 같은 언어 모델이 단어를 하나씩 생성하는 방식과 거의 동일하다.

행동 토큰은 시간적 관계(Temporal Relationship)를 매우 효과적으로 학습한다. 트랜스포머의 셀프 어텐션(Self-Attention)은 이전 행동과 이후 행동의 관계를 동시에 고려하므로 긴 작업에서도 일관성 있는 행동을 생성할 수 있다. 별도의 시간 모델(Time Model)을 설계하지 않아도 장기 의존성(Long-Term Dependency)을 자연스럽게 학습할 수 있다.

행동 토큰은 계층적 행동(Hierarchical Behavior)도 쉽게 표현한다. 개별 토큰은 기본 동작을 구성하고, 여러 토큰이 하나의 기술(Skill)이 되며, 여러 기술이 하나의 작업(Task)이 되고, 여러 작업이 하나의 미션(Mission)을 구성한다. 이러한 구조는 문자(Character), 단어(Word), 문장(Sentence), 문서(Document)로 구성되는 자연어와 매우 유사하다.

언어 접지(Language Grounding)도 자연스럽게 이루어진다. "Pick", "Place", "Push", "Rotate", "Inspect", "Charge"와 같은 언어 토큰(Language Token)은 대응되는 행동 토큰(Action Token)과 함께 학습되므로 새로운 명령도 기존 의미를 조합하여 수행할 수 있다. 이것이 RT-2가 보인 뛰어난 일반화 성능의 핵심 원리 중 하나이다.

멀티모달 추론(Multimodal Reasoning)은 행동 토큰 구조에서 더욱 강력해진다. 비전 토큰(Vision Token)은 환경을 표현하고, 언어 토큰(Language Token)은 의미를 표현하며, 행동 토큰(Action Token)은 실제 물리적 행동을 표현한다. 트랜스포머는 세 가지 토큰을 하나의 시퀀스(Sequence)로 처리하여 인식과 행동을 동시에 학습한다.

행동 토큰은 일반화(Generalization) 성능도 향상시킨다. 연속 제어는 작은 수치 변화에도 민감하지만, 행동 토큰은 행동 자체를 상징(Symbolic Representation)으로 표현하므로 새로운 환경에서도 동일한 행동 패턴을 재사용할 수 있다. 따라서 학습하지 않은 작업(Unseen Task)에도 쉽게 적응할 수 있다.

대규모 사전학습(Large-Scale Pretraining)도 가능하다. 인터넷 비전-언어 데이터(Internet Vision-Language Data), 로봇 시연(Robot Demonstration), 시뮬레이션 데이터, 산업 자동화 로그를 모두 하나의 트랜스포머 모델로 학습할 수 있다. 이는 언어 모델에서 사용되는 스케일링 법칙(Scaling Law)을 로봇에도 적용할 수 있게 만든다.

RT-2는 이러한 개념을 실제로 입증하였다. 인터넷에서 학습한 비전-언어 지식을 로봇 행동 생성에 그대로 활용하여, 학습하지 않은 물체나 새로운 명령도 이해하고 수행할 수 있었다. 이는 언어 모델의 지식이 행동 생성으로 자연스럽게 전이(Transfer)될 수 있음을 보여준다.

로봇의 형태(Embodiment)에 따라 행동 토큰도 달라진다. 산업용 매니퓰레이터는 파지(Grasp), 삽입(Insertion), 힘 제어(Force Control)를 중심으로 구성되고, AMR은 이동(Navigation), 회피(Avoidance), 도킹(Docking), 충전을 중심으로 구성된다. 휴머노이드는 보행(Walking), 균형(Balance), 제스처(Gesture), 양팔 조작(Bimanual Manipulation)을 포함하며, 사족보행 로봇은 보행(Gait), 발 위치(Foot Placement), 자세 제어(Posture Control)를 포함한다.

토큰의 크기(Token Granularity)는 중요한 설계 요소이다. 작은 토큰은 세밀한 행동 표현이 가능하지만 시퀀스 길이가 길어지고 계산량이 증가한다. 큰 토큰은 계산 효율이 높지만 세부 동작 표현이 어려워진다. 실제 시스템은 작업 특성에 맞추어 적절한 토큰 크기를 선택한다.

자연어 처리의 서브워드(Subword) 개념도 행동 토큰에 적용된다. 하나의 기술을 하나의 토큰으로 표현하는 대신, 자주 사용되는 작은 동작 조각을 재사용 가능한 토큰으로 정의하면 다양한 복합 행동을 더욱 효율적으로 생성할 수 있다.

학습은 대부분 모방 학습(Imitation Learning) 기반으로 수행된다. 전문가 행동을 행동 토큰으로 변환한 후, 언어 모델과 동일하게 크로스 엔트로피(Cross Entropy) 손실 함수를 사용하여 다음 행동 토큰을 예측하도록 학습한다. 따라서 기존 LLM 학습 기술을 거의 그대로 사용할 수 있다.

행동 토큰은 장기 계획(Long-Horizon Planning)에도 유리하다. 하나의 행동만 생성하는 것이 아니라 미래의 행동 토큰 시퀀스를 생성하므로 전체 작업 절차를 계획할 수 있다. 이후 모델 예측 제어(MPC, Model Predictive Control)나 반복적 계획(Receding Horizon Planning)을 통해 일부를 실행하고 다시 계획을 갱신한다.

외부 메모리(External Memory)는 행동 토큰 정책의 성능을 향상시킨다. 에피소드 메모리(Episodic Memory)는 과거 성공 사례를 저장하고, 의미 메모리(Semantic Memory)는 환경 지식을 저장하며, 절차 메모리(Procedural Memory)는 작업 절차를 행동 토큰 시퀀스로 저장한다. 현재 작업은 이러한 메모리를 검색하여 더욱 효율적으로 수행된다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 행동 토큰 정책에서도 활용된다. 유지보수 문서, 디지털 트윈(Digital Twin), 시설 지도(Map), 기업 데이터베이스, 작업 기록 등을 검색하여 행동 토큰 생성에 반영한다. 외부 지식을 활용함으로써 모델을 다시 학습하지 않아도 최신 정보를 사용할 수 있다.

세계 모델(World Model)은 미래 환경을 예측하여 행동 토큰 선택에 활용된다. 사람의 이동, 물체 변화, 배터리 상태, 장애물 이동 등을 예측한 후 가장 적절한 행동 토큰을 선택하므로 장기 계획 성능이 크게 향상된다.

어포던스 추론(Affordance Reasoning)은 실행 가능한 행동만 선택하도록 한다. 행동 토큰이 존재하더라도 접근 불가능하거나 충돌 위험이 있는 경우에는 선택되지 않는다. 접근 가능성(Reachability), 충돌 여부(Collision), 안정적 파지(Stability) 등이 함께 고려된다.

안전(Safety)은 행동 토큰과 독립적으로 동작한다. 생성된 행동 토큰은 실제 실행 전에 충돌 검사(Collision Detection), 운동학 검증(Kinematic Verification), 속도 제한(Velocity Limit), 힘 제한(Force Limit), 런타임 모니터(Runtime Monitor), 비상 정지(Emergency Stop)를 통해 검증된다.

시뮬레이션(Simulation)은 행동 토큰 학습에서 매우 중요하다. 디지털 트윈을 이용하여 다양한 물체, 조명, 날씨, 센서 잡음, 작업 실패 사례를 포함한 수백만 개의 행동 데이터를 생성할 수 있으며, 이를 통해 행동 어휘(Action Vocabulary)가 더욱 풍부해진다.

Sim-to-Real 기술은 시뮬레이션에서 학습한 행동 토큰 정책을 실제 로봇으로 이전한다. 도메인 랜덤화(Domain Randomization), 물리 오차(Physics Variation), 센서 잡음 등을 적용한 후 소량의 실제 데이터로 미세조정(Fine-Tuning)을 수행하여 실제 환경에서도 높은 성능을 유지한다.

파라미터 효율적 적응(Parameter-Efficient Adaptation)은 하나의 대규모 행동 토큰 모델을 산업용 로봇, 물류 로봇, 농업 로봇, 의료 로봇 등 다양한 플랫폼에 빠르게 적용할 수 있도록 한다. 전체 모델을 다시 학습하지 않고 일부 계층만 수정하면 되므로 매우 효율적이다.

추론 효율(Inference Efficiency)은 확산 정책보다 높다. 확산 모델처럼 반복적인 노이즈 제거가 필요하지 않고, 트랜스포머가 행동 토큰을 순차적으로 생성하기 때문에 GPU, Tensor Core, NPU(Neural Processing Unit), 혼합 정밀도(Mixed Precision), 양자화(Quantization)를 이용하면 실시간 추론도 가능하다.

행동 토큰화의 가장 큰 과제는 이산화 오차(Discretization Error)이다. 연속적인 행동을 유한한 토큰으로 표현하는 과정에서 일부 정밀도가 손실될 수 있다. 이를 해결하기 위해 계층형 토큰(Hierarchical Tokenization), 적응형 코드북(Adaptive Codebook), 잔차 벡터 양자화(RVQ, Residual Vector Quantization), 연속 보정 계층(Continuous Refinement Layer)이 연구되고 있다.

또 다른 문제는 토큰 모호성(Token Ambiguity)이다. 비슷한 행동이 서로 다른 토큰으로 표현되거나, 동일한 토큰이 로봇에 따라 조금씩 다른 의미를 가질 수 있다. 따라서 최근에는 고수준은 행동 토큰으로 계획하고, 저수준은 연속 제어(Continuous Control)로 보정하는 하이브리드(Hybrid) 구조가 많이 사용된다.

산업 제조에서는 조립, 검사, 포장, 체결(Fastening), 팔레트 운반 등이 행동 토큰으로 표현된다. 물류에서는 이동, 도킹, 충전, 재고 관리가 토큰화되며, 의료에서는 약품 배송, 환자 지원, 재활, 검사, 농업에서는 수확, 가지치기, 방제, 작물 관리가 토큰 시퀀스로 표현된다.

행동 토큰 정책의 평가는 단순한 토큰 정확도가 아니라 작업 성공률(Task Success Rate), 조작 성공률(Manipulation Success), 시퀀스 일관성(Sequence Consistency), 언어와 행동의 의미 연결(Language Grounding), 일반화 성능(Generalization), 플랫폼 전이(Embodiment Transfer), 추론 속도(Inference Latency), 안전성(Safety Compliance), 장기 작업 안정성(Long-Horizon Stability)을 종합적으로 평가한다.

향후에는 행동 토큰화와 Flow Matching, 확산 정책(Diffusion Policy), 세계 모델(World Model), 검색 증강 생성(RAG), 외부 메모리(External Memory), 지속적 학습(Continual Learning), 강화학습(Reinforcement Learning), 적응형 코드북(Adaptive Codebook), 하이브리드 연속 제어(Hybrid Continuous Control)가 하나의 통합 액션 모델(Action Model)로 발전할 것으로 예상된다.

결국 RT-2 스타일의 이산 행동 토큰화는 로봇 제어를 언어 생성 문제로 변환함으로써 트랜스포머 기반 대규모 사전학습, 멀티모달 추론(Multimodal Reasoning), 파운데이션 모델(Foundation Model), 검색 증강 생성(RAG), 세계 모델(World Model), 안전 검증(Safety Verification)을 하나의 통합 프레임워크로 결합한다. 이를 통해 제조, 물류, 의료, 농업, 서비스 로봇, 시설 점검, 협업 로봇 등 다양한 물리 AI(Physical AI) 분야에서 범용 로봇 지능(General-Purpose Robotic Intelligence)을 실현하는 핵심 기술로 자리잡고 있다.

## 4.6 Hierarchical Action Models (with Code)

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

계층형 액션 모델(Hierarchical Action Model)은 현대 물리 AI(Physical AI)에서 가장 중요한 아키텍처 원리 중 하나이다. 하나의 거대한 신경망이 사람의 명령 이해, 장기 계획(Long-Horizon Planning), 경로 생성(Trajectory Generation), 모터 제어(Motor Control)를 모두 수행하는 대신, 역할을 여러 계층으로 분리하여 각각의 정책(Policy)이 자신의 역할만 담당하도록 구성한다. 이러한 구조는 확장성(Scalability), 해석 가능성(Interpretability), 계산 효율성(Computational Efficiency), 안전성(Safety), 일반화(Generalization)를 크게 향상시킨다.

계층형 제어(Hierarchical Control)의 핵심 아이디어는 인간의 사고 과정과 유사하다. 사람은 커피를 만들기 위해 근육 하나하나를 의식하지 않는다. 먼저 목표를 설정하고, 이를 여러 단계의 작업으로 분해한 후, 마지막으로 손과 팔을 움직인다. 로봇도 동일하게 상위 정책(High-Level Policy)은 목표를 결정하고, 중간 정책(Mid-Level Policy)은 작업을 계획하며, 하위 정책(Low-Level Policy)은 실제 액추에이터(Actuator)를 제어한다.

기존 End-to-End 정책은 센서 입력을 바로 모터 제어 명령으로 변환한다. 구조는 단순하지만 장기 계획, 언어 이해, 안전성, 실시간 제어를 하나의 모델이 동시에 수행해야 하므로 학습이 매우 어렵고 일반화 성능도 제한된다. 계층형 액션 모델은 이러한 기능을 여러 정책으로 분리하여 각 계층이 자신의 문제만 해결하도록 만든다.

최상위 계층은 전략적 추론(Strategic Reasoning)을 담당한다. 상위 정책(High-Level Policy)은 자연어 명령(Natural Language Instruction), 작업 목표(Task Goal), 환경 정보(Environment Context), 기업 규칙(Business Rule), 과거 대화(Dialog History) 등을 이해한다. 이 계층은 직접 모터를 제어하지 않고 작업 계획(Task Plan), 행동 순서(Task Sequence), 기술 선택(Skill Selection), 미션 계획(Mission Plan)과 같은 추상적인 목표를 생성한다.

대규모 언어 모델(LLM, Large Language Model)은 상위 정책으로 매우 적합하다. 예를 들어 "생산 라인의 모든 설비를 검사하고 이상 온도를 보고하라", "315호 병실까지 약품을 배송하라", "익은 딸기만 수확하고 손상된 과일은 피하라"와 같은 자연어 명령을 이해하고 이를 실행 가능한 작업 목표로 변환한다.

비전-언어 추론(Vision-Language Reasoning)은 상위 정책을 더욱 강화한다. 비전 인코더(Vision Encoder)는 물체(Object), 사람(Human), 설비(Facility), 위험 요소(Hazard)를 인식하고, 언어 모델은 이를 작업 목표와 결합하여 현재 환경에 적합한 전략을 수립한다. 따라서 고정된 작업 절차가 아니라 현재 상황에 맞는 계획을 생성할 수 있다.

작업 분해(Task Decomposition)는 상위 정책의 가장 중요한 역할이다. 하나의 복잡한 작업을 여러 개의 작은 작업으로 나누어 하위 정책이 실행할 수 있도록 한다. 예를 들어 창고 작업은 이동(Navigation), 위치 보정(Localization), 선반 탐색(Shelf Detection), 물체 인식(Object Recognition), 파지(Grasp), 운반(Transport), 도킹(Docking), 충전(Charging), 보고(Reporting) 등으로 분해된다.

상위 정책은 행동 트리(Behavior Tree), 계층형 작업 네트워크(HTN, Hierarchical Task Network), 상태 머신(State Machine), 계획 그래프(Planning Graph) 등을 생성할 수도 있다. 최근에는 LLM이 이러한 구조를 실시간으로 생성하거나 수정하여 더욱 유연한 작업 수행이 가능해지고 있다.

상위 정책과 하위 정책 사이에는 중간 정책(Mid-Level Policy)이 존재하는 경우가 많다. 중간 정책은 추상적인 목표를 실제 경로(Path), 조작 계획(Manipulation Plan), 이동 계획(Motion Planning), 기술 실행(Skill Execution)으로 변환한다. 의미 수준(Semantic Level)과 물리 제어(Physical Control)를 연결하는 중요한 역할을 수행한다.

하위 정책(Low-Level Policy)은 실제 로봇 제어를 담당한다. 수십에서 수백 Hz의 높은 주기로 관절 위치(Joint Position), 휠 속도(Wheel Velocity), 조향각(Steering Angle), 토크(Torque), 힘 제어(Force Control), 균형 유지(Balance Control), 궤적 추종(Trajectory Tracking)을 수행한다. 이 계층에서는 매우 빠른 응답성과 안정성이 요구된다.

하위 정책은 회귀 기반 정책(Regression Policy), 확산 정책(Diffusion Policy), Flow Matching, 모델 예측 제어(MPC, Model Predictive Control), 역기구학(Inverse Kinematics), PID 제어기(PID Controller), 강화학습(Reinforcement Learning) 등을 활용할 수 있다. 실제 산업에서는 AI와 기존 제어기를 함께 사용하는 하이브리드(Hybrid) 구조가 가장 많이 사용된다.

시간적 추상화(Temporal Abstraction)는 계층형 구조의 핵심이다. 상위 정책은 수분(Minute) 단위의 장기 계획을 담당하고, 중간 정책은 수초(Second) 단위의 기술 실행을 담당하며, 하위 정책은 밀리초(Millisecond) 단위의 실시간 제어를 수행한다. 각 계층이 서로 다른 시간 해상도를 담당함으로써 계산 효율성이 크게 향상된다.

공간적 추상화(Spatial Abstraction)도 계층별로 다르다. 상위 정책은 공장 전체, 창고 전체와 같은 의미 공간(Semantic Space)을 이해하고, 중간 정책은 작업 공간(Workspace)과 장애물(Obstacle)을 계획하며, 하위 정책은 관절(Joint), 모터(Motor), 엔코더(Encoder), 힘 센서(Force Sensor)를 직접 제어한다.

비전-언어-행동(VLA) 아키텍처에서는 RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 의미 분할(Semantic Segmentation), 촉각(Tactile), 힘 센서(Force Sensor), 위치 추정(Localization) 등이 모든 계층에서 공유된다. 단, 각 계층은 동일한 정보를 서로 다른 수준으로 해석한다.

언어 접지(Language Grounding)는 상위 정책뿐 아니라 하위 계층에도 영향을 준다. 상위 정책은 작업 목표를 이해하고, 중간 정책은 기술 선택을 수행하며, 하위 정책은 파지 방향(Grasp Orientation), 힘 조절(Force Control), 이동 방식(Motion Style)과 같은 세부 명령까지 반영할 수 있다.

계층형 메모리(Hierarchical Memory)는 장시간 작업에서 매우 중요하다. 에피소드 메모리(Episodic Memory)는 과거 작업 경험을 저장하고, 의미 메모리(Semantic Memory)는 환경 지식을 저장하며, 절차 메모리(Procedural Memory)는 기술(Skill)을 저장한다. 작업 메모리(Working Memory)는 현재 작업 상태를 유지하여 모든 계층이 필요한 정보를 공유하도록 한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 계층형 구조에서도 매우 효과적이다. 유지보수 문서, 디지털 트윈(Digital Twin), 생산 일정, 작업 기록, 시설 지도(Map) 등을 검색하여 상위 정책은 작업 계획을 수립하고, 하위 정책은 실행 시 필요한 세부 정보를 얻는다.

세계 모델(World Model)은 상위와 하위 정책을 연결하는 중요한 요소이다. 사람의 이동, 장애물 변화, 배터리 상태, 작업 진행 상황 등을 예측하여 미래 환경을 고려한 계획을 수립할 수 있다. 이를 통해 단순한 반응형 제어가 아니라 예측 기반 제어(Predictive Control)가 가능해진다.

어포던스 추론(Affordance Reasoning)은 계층마다 다른 역할을 수행한다. 상위 정책은 사용할 수 있는 물체와 작업을 선택하고, 중간 정책은 접근 가능성과 충돌 여부를 판단하며, 하위 정책은 실제 힘과 접촉(Contact)을 제어한다.

안전(Safety)은 모든 계층에서 동시에 관리된다. 상위 정책은 작업 승인과 규정 준수를 확인하고, 중간 정책은 위험 지역과 충돌 경로를 피하며, 하위 정책은 속도 제한(Velocity Limit), 힘 제한(Force Limit), 충돌 감지(Collision Detection), 비상 정지(Emergency Stop)를 수행한다. 다단계 안전 구조는 End-to-End 정책보다 훨씬 높은 신뢰성을 제공한다.

계층형 강화학습(Hierarchical Reinforcement Learning)은 상위 정책이 장기 보상(Long-Term Reward)을 학습하고, 하위 정책은 안정성(Stability), 에너지 효율(Energy Efficiency), 정밀도(Precision)를 학습한다. 이러한 역할 분리는 학습 효율을 크게 향상시킨다.

행동 복제(Behavior Cloning) 역시 계층형 구조에 적합하다. 사람의 시연에는 계획, 기술 선택, 실제 조작이 모두 포함되어 있으므로 이를 계층별로 분리하여 학습하면 적은 데이터로도 높은 일반화 성능을 얻을 수 있다.

시뮬레이션(Simulation)은 계층형 정책 개발에 매우 중요하다. 디지털 트윈은 장기 작업, 다양한 환경, 사람과의 상호작용, 센서 잡음, 장비 고장 등을 생성하여 상위 정책과 하위 정책을 동시에 학습할 수 있도록 지원한다.

Sim-to-Real 기술에서는 상위 정책은 비교적 환경 변화에 강하며, 하위 정책은 센서 오차, 마찰(Friction), 통신 지연(Latency), 물리 오차를 보정하도록 미세조정(Fine-Tuning)된다. 계층 구조 덕분에 실제 환경 적응이 훨씬 쉬워진다.

파라미터 효율적 미세조정(PEFT, Parameter-Efficient Fine-Tuning)은 상위 정책은 그대로 유지하고, 하위 정책만 새로운 로봇 플랫폼에 맞게 수정할 수 있도록 한다. 따라서 하나의 파운데이션 모델(Foundation Model)을 다양한 로봇에 효율적으로 적용할 수 있다.

로봇의 형태(Embodiment)는 주로 하위 정책에 영향을 준다. AMR, 산업용 매니퓰레이터, 휴머노이드(Humanoid), 사족보행 로봇(Quadruped), 농업 로봇 모두 상위 정책은 거의 동일하게 사용할 수 있지만, 하위 정책은 각각의 관절 구조와 센서 구성에 맞게 달라진다.

계층형 구조는 다중 로봇(Multi-Robot) 협업에도 매우 적합하다. 중앙 서버는 상위 정책을 통해 전체 미션을 계획하고, 개별 로봇은 자신의 하위 정책을 이용하여 자율적으로 이동과 작업을 수행한다. 중앙 집중식 계획과 분산 제어가 자연스럽게 결합된다.

클라우드-엣지 협업(Cloud-Edge Collaboration)도 계층 구조와 잘 맞는다. 클라우드는 LLM 추론, 세계 모델, 장기 계획, 다중 로봇 관리 등을 수행하고, 엣지는 실시간 인식, 제어, 안전 감시를 수행한다. 통신이 끊겨도 하위 정책은 계속 동작하므로 안전성이 유지된다.

계층형 정책은 비동기 실행(Asynchronous Execution)이 가능하다. 상위 정책은 필요할 때만 계획을 수정하고, 중간 정책은 환경 변화가 있을 때만 경로를 다시 생성하며, 하위 정책은 항상 일정한 주기로 동작한다. 계산 자원을 효율적으로 사용할 수 있는 구조이다.

설명 가능성(Explainability)도 매우 우수하다. 상위 정책은 사람이 이해할 수 있는 작업 계획을 생성하며, 중간 정책은 선택한 기술을 설명할 수 있고, 하위 정책은 실제 제어 데이터를 제공한다. 따라서 디버깅(Debugging), 인증(Certification), 유지보수가 쉬워진다.

계산 효율성도 향상된다. LLM과 같은 무거운 모델은 계획을 세울 때만 실행되고, 실시간 제어는 경량 정책(Lightweight Policy)이 수행하므로 전체 시스템의 연산량이 크게 감소한다.

평가(Evaluation)는 계층별로 수행된다. 상위 정책은 작업 성공률(Task Success Rate), 계획 품질(Planning Quality), 명령 이해(Instruction Following)를 평가하고, 중간 정책은 경로 품질(Path Quality), 기술 선택(Skill Selection)을 평가하며, 하위 정책은 제어 안정성(Control Stability), 추종 오차(Tracking Error), 에너지 효율(Energy Efficiency), 안전성(Safety)을 평가한다.

산업 제조에서는 조립(Assembly), 검사(Inspection), 물류(Logistics), 협업 로봇(Collaborative Robot)에 활용되며, 창고에서는 재고 관리와 자율주행을 결합한다. 의료에서는 환자 지원, 약품 배송, 실험실 자동화에 적용되고, 농업에서는 작물 관리, 수확, 자율주행을 통합하는 구조로 활용된다.

최근 로봇 파운데이션 모델은 계층형 액션 모델을 기본 구조로 채택하고 있다. 비전, 언어, 행동 데이터를 동시에 학습하면서 상위 추론과 하위 실행을 분리함으로써 다양한 작업과 다양한 로봇에 쉽게 적용할 수 있는 범용 정책을 구축하고 있다.

향후에는 Flow Matching, 확산 정책(Diffusion Policy), 행동 토큰(Action Token), 세계 모델(World Model), 검색 증강 생성(RAG), 지속적 학습(Continual Learning), 외부 메모리(External Memory), 강화학습(Reinforcement Learning), 다중 로봇 협업(Multi-Robot Coordination)이 계층형 구조 안에서 통합될 것으로 예상된다.

결국 계층형 액션 모델은 상위 정책(High-Level Policy)의 의미 기반 추론(Semantic Reasoning), 중간 정책(Mid-Level Policy)의 기술 계획(Skill Planning), 하위 정책(Low-Level Policy)의 실시간 제어(Real-Time Control)를 하나의 통합 구조로 결합한다. 여기에 멀티모달 인식(Multimodal Perception), LLM 기반 계획(Language Planning), 세계 모델(World Model), 검색 증강 생성(RAG), 파운데이션 모델(Foundation Model), 클라우드-엣지 협업(Cloud-Edge Collaboration), 안전 검증(Safety Verification)을 통합함으로써 제조, 물류, 의료, 농업, 시설 점검, 서비스 로봇 등 다양한 물리 AI(Physical AI) 시스템을 위한 가장 현실적이고 확장성 높은 차세대 액션 모델(Action Model) 아키텍처로 자리매김하고 있다.

## 4.7 Temporal Consistency and Action Chunking (with Code)

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

시간적 일관성(Temporal Consistency)은 현대 로봇 정책(Robot Policy) 학습에서 가장 중요한 요소 중 하나이다. 로봇은 단순히 올바른 개별 행동(Action)을 생성하는 것만으로는 충분하지 않으며, 여러 시간(Time)에 걸쳐 부드럽고 안정적인 움직임을 유지해야 한다. 기존 정책은 매 시점마다 하나의 제어 명령만 생성하기 때문에 궤적(Trajectory)이 불연속적이거나 진동(Oscillation)이 발생하고, 장기 작업(Long-Horizon Task)에서는 오차(Error)가 누적되는 문제가 있었다. ACT(Action Chunking Transformer)는 이러한 문제를 해결하기 위해 여러 개의 미래 행동을 하나의 덩어리(Action Chunk)로 동시에 생성한다.

ACT(Action Chunking Transformer)의 기본 아이디어는 인간의 운동 제어(Human Motor Control)에서 영감을 얻었다. 사람은 매 순간 개별 근육을 독립적으로 제어하지 않는다. 걷기(Walking)는 보행 주기(Gait Cycle) 단위로 계획되고, 팔을 뻗는 동작도 하나의 연속된 궤적으로 계획된다. 물체를 잡을 때 역시 손가락을 각각 제어하는 것이 아니라 하나의 파지(Grasp) 동작 전체를 계획한다. ACT는 이러한 생물학적 제어 방식을 로봇 정책에 적용하여 여러 개의 행동을 하나의 연속된 행동 조각(Action Chunk)으로 생성한다.

기존 반응형 정책(Reactive Policy)은 현재 상태(Current State)만 보고 다음 한 개의 행동만 생성한다. 이 방식은 구현이 간단하지만 작은 예측 오차가 다음 예측으로 계속 전달되면서 장기적으로 큰 오차가 발생한다. 또한 센서 잡음(Sensor Noise)이나 환경 변화(Environment Change)가 누적되면 로봇은 불필요한 진동, 반복적인 보정 움직임, 급격한 속도 변화 등을 보일 수 있다.

ACT는 이러한 문제를 해결하기 위해 미래의 여러 제어 명령(Control Command)을 동시에 생성한다. 하나의 행동 청크(Action Chunk)는 일정 시간 동안 실행될 여러 개의 연속 행동으로 구성된다. 로봇은 이 청크를 실행하면서 동시에 주변 환경을 계속 관찰하고, 일정 시간이 지나면 새로운 상태를 반영하여 다음 행동 청크를 다시 생성한다. 이러한 반복적 계획(Receding Horizon Planning)은 장기 계획과 실시간 적응성을 동시에 제공한다.

시간적 일관성(Temporal Consistency)이란 연속된 행동들이 물리적으로 자연스럽고 논리적으로 연결되는 특성을 의미한다. 로봇의 움직임은 모터의 관성(Inertia), 기계 구조(Mechanical Constraint), 접촉(Contact), 동역학(Dynamics)을 모두 만족해야 한다. ACT는 여러 시점의 행동을 동시에 최적화하기 때문에 독립적인 행동 생성보다 훨씬 부드럽고 안정적인 움직임을 제공한다.

비전-언어-행동(VLA, Vision-Language-Action) 아키텍처에서는 RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 촉각 센서(Tactile Sensor), 힘 센서(Force Sensor), 내부 상태(Proprioception), 위치 추정(Localization), 의미 지도(Semantic Map) 등이 먼저 환경을 인식한다. 비전 인코더(Vision Encoder)는 이를 잠재 표현(Latent Representation)으로 변환하며, 대규모 언어 모델(LLM, Large Language Model)은 자연어 명령과 작업 목표(Task Goal)를 이해한다. 멀티모달 융합(Multimodal Fusion)은 이러한 정보를 통합하여 ACT의 입력으로 제공한다.

기존 액션 헤드(Action Head)는 하나의 제어 명령만 출력하지만, ACT는 여러 개의 미래 행동으로 구성된 행동 청크(Action Chunk)를 출력한다. 이 청크는 관절 궤적(Joint Trajectory), 엔드이펙터 자세(End-Effector Pose), 휠 속도(Wheel Velocity), 조향각(Steering Angle), 그리퍼 상태(Gripper State), 힘 제어(Force Control) 등으로 구성될 수 있다. 여러 행동이 동시에 생성되므로 시간적 관계가 자연스럽게 유지된다.

트랜스포머(Transformer)는 ACT에 매우 적합한 구조이다. 셀프 어텐션(Self-Attention)은 여러 미래 시점(Time Step)의 행동을 동시에 분석하여 현재 행동이 미래 행동과 일관성을 유지하도록 한다. 기존 순환 신경망(RNN, Recurrent Neural Network)보다 장기 의존성(Long-Term Dependency)을 더욱 효과적으로 학습할 수 있다.

ACT는 행동을 단순한 제어 명령이 아니라 시퀀스(Sequence)로 이해한다. 자연어 문장이 단어(Token)의 연속으로 구성되듯이, 로봇 행동도 연속적인 행동 시퀀스로 구성된다. 따라서 행동 간의 시간적 관계를 학습하여 더욱 자연스럽고 안정적인 움직임을 생성할 수 있다.

행동 청크(Action Chunk)의 길이는 매우 중요한 설계 요소이다. 짧은 청크는 빠른 환경 변화에 대응하기 쉽지만 계획 능력이 제한되고, 긴 청크는 부드러운 움직임과 장기 계획에는 유리하지만 즉각적인 반응성이 감소할 수 있다. 실제 시스템은 작업(Task), 환경(Environment), 계산 성능(Computing Power)에 따라 적절한 길이를 선택한다.

ACT는 중첩 청크(Overlapping Chunk)도 사용할 수 있다. 연속된 두 행동 청크가 일부 행동을 공유하도록 하여 청크 경계에서 발생할 수 있는 불연속적인 움직임을 줄인다. 이러한 중첩 구조는 매우 자연스러운 행동 연결을 가능하게 한다.

슬라이딩 윈도우(Sliding Window) 실행 방식도 함께 사용된다. 하나의 행동 청크를 모두 실행하지 않고 앞부분만 실행한 후 새로운 센서 정보를 반영하여 다음 청크를 다시 생성한다. 이를 통해 장기 계획을 유지하면서도 실시간 환경 변화에 적응할 수 있다.

ACT는 시간적 평활화(Temporal Smoothing)를 자연스럽게 제공한다. 여러 행동을 동시에 최적화하기 때문에 가속도(Acceleration), 속도(Velocity), 방향(Direction)이 부드럽게 변화하며, 액추에이터(Actuator)의 진동이 줄어들고 에너지 소비(Energy Consumption)도 감소한다.

오차 누적(Error Accumulation)은 기존 정책의 가장 큰 문제였다. 하나의 행동 오차가 다음 행동으로 계속 전달되면서 장기적으로 큰 실패를 유발할 수 있었다. ACT는 여러 행동을 동시에 최적화하므로 오차가 개별 행동마다 증폭되지 않고 훨씬 안정적인 장기 수행(Long-Horizon Execution)이 가능하다.

ACT는 대부분 행동 복제(Behavior Cloning) 방식으로 학습된다. 원격 조작(Teleoperation), VR 조작, 산업 자동화 데이터, 강화학습(Reinforcement Learning), 시뮬레이션(Simulation) 등을 이용하여 전문가 행동을 수집하고, 개별 행동이 아니라 전체 행동 청크를 학습 대상으로 사용한다.

손실 함수(Loss Function)도 개별 행동이 아니라 전체 궤적(Trajectory)을 기준으로 계산된다. 평균제곱오차(MSE, Mean Squared Error), Smooth L1 Loss, 속도 정규화(Velocity Regularization), 가속도 패널티(Acceleration Penalty), Jerk 최소화(Jerk Minimization), 자세 일관성(Orientation Consistency) 등을 함께 사용하여 전체 움직임의 품질을 높인다.

ACT에서는 개별 행동의 정확도보다 전체 궤적의 품질이 더욱 중요하다. 일부 시점에서 작은 오차가 있더라도 전체 움직임이 자연스럽고 작업 성공률(Task Success Rate)이 높으면 더 좋은 정책으로 평가된다.

ACT는 연속 행동 공간(Continuous Action Space)을 그대로 지원한다. 행동 토큰(Action Token)처럼 이산화(Discretization)를 수행하지 않고 관절 위치(Joint Position), 휠 속도(Wheel Velocity), 토크(Torque), 힘 제어(Force Control) 등을 연속적인 값으로 예측한다.

계층형 액션 모델(Hierarchical Action Model)과도 매우 잘 결합된다. 상위 정책(High-Level Policy)은 작업 목표를 생성하고, 중간 정책(Mid-Level Policy)은 기술(Skill)과 경로(Path)를 계획하며, ACT는 이를 실행하기 위한 행동 청크를 생성한다. 하위 제어기(Low-Level Controller)는 PID, MPC(Model Predictive Control), 역기구학(Inverse Kinematics)을 이용하여 이를 실제 모터 제어로 변환한다.

세계 모델(World Model)은 ACT의 성능을 더욱 향상시킨다. 현재 환경뿐 아니라 미래의 물체 이동(Object Motion), 사람의 움직임(Human Activity), 장애물 변화, 배터리 상태 등을 예측하여 행동 청크 생성에 반영한다. 이를 통해 미래를 고려한 계획(Predictive Planning)이 가능해진다.

어포던스 추론(Affordance Reasoning)은 실행 가능한 행동만 생성하도록 한다. 접근 가능한 물체(Reachable Object), 충돌 없는 경로(Collision-Free Path), 안정적인 파지(Stable Grasp), 이동 가능한 공간(Navigable Area)을 고려하여 실제 수행 가능한 행동 청크만 생성한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 유지보수 문서, 디지털 트윈(Digital Twin), 작업 기록, 시설 지도(Map), 기업 데이터베이스를 검색하여 행동 청크 생성에 필요한 추가 정보를 제공한다. 외부 지식이 행동 계획에 직접 활용된다.

메모리(Memory)는 장시간 작업에서 매우 중요한 역할을 한다. 에피소드 메모리(Episodic Memory)는 이전 성공 사례를 저장하고, 의미 메모리(Semantic Memory)는 환경 지식을 저장하며, 절차 메모리(Procedural Memory)는 반복 가능한 기술(Skill)을 저장한다. 작업 메모리(Working Memory)는 현재 진행 상황을 유지하여 행동 청크 생성에 활용된다.

시뮬레이션(Simulation)은 ACT 학습에서 필수적이다. 디지털 트윈(Digital Twin)은 다양한 물체 배치, 센서 잡음, 날씨, 환경 변화, 장비 고장 등을 포함한 수백만 개의 행동 청크를 생성하여 정책의 일반화 성능을 크게 향상시킨다.

Sim-to-Real 기술은 시뮬레이션에서 학습한 ACT 정책을 실제 로봇에 적용한다. 도메인 랜덤화(Domain Randomization), 물리 오차(Physics Variation), 센서 오차(Sensor Noise), 통신 지연(Latency)을 포함하여 학습한 후 실제 데이터로 미세조정(Fine-Tuning)을 수행한다.

로봇의 형태(Embodiment)에 따라 행동 청크도 달라진다. 산업용 매니퓰레이터는 관절 궤적과 그리퍼 동작을 생성하고, AMR은 속도와 조향 궤적을 생성하며, 사족보행 로봇(Quadruped)은 보행 주기(Gait Cycle)를 생성하고, 휴머노이드는 보행과 조작을 포함한 전신 움직임을 생성한다.

클라우드-엣지 협업(Cloud-Edge Collaboration)에서도 ACT는 매우 적합하다. 클라우드는 LLM 추론, 세계 모델, 시뮬레이션, 장기 계획을 수행하고, 엣지 컴퓨터는 행동 청크 생성, 실시간 제어, 안전 감시를 담당한다. 통신이 끊겨도 행동 청크를 이용하여 일정 시간 동안 자율 동작을 계속 수행할 수 있다.

ACT는 추론 효율(Inference Efficiency)도 우수하다. 하나의 순전파(Forward Pass)로 여러 개의 미래 행동을 동시에 생성하므로 매 시점마다 정책을 반복 실행하는 것보다 계산량이 크게 감소한다. GPU, Tensor Core, NPU(Neural Processing Unit), 혼합 정밀도(Mixed Precision), 양자화(Quantization)를 적용하면 실시간 성능도 충분히 확보할 수 있다.

안전(Safety)은 ACT와 독립적으로 유지된다. 생성된 행동 청크는 실행 전에 충돌 감지(Collision Detection), 운동학 검증(Kinematic Verification), 동역학 제한(Dynamic Constraint), 작업 공간 제한(Workspace Limit), 비상 정지(Emergency Stop), 런타임 모니터(Runtime Monitor)를 통해 반드시 검증된다.

다중 로봇(Multi-Robot) 협업에서도 ACT는 효과적이다. 여러 로봇이 서로 시간적으로 일관된 행동 청크를 생성하여 협업 운반(Cooperative Transport), 조립(Assembly), 물류(Logistics), 점검(Inspection)을 수행하면 작업의 동기화(Synchronization)가 크게 향상된다.

산업 제조에서는 정밀 조립(Precision Assembly), 삽입(Insertion), 용접(Welding), 연마(Polishing), 케이블 조립(Cable Routing), 협업 조작(Collaborative Manipulation)에 활용된다. 물류에서는 자율주행, 도킹(Docking), 충전(Charging), 창고 관리에 적용되며, 의료에서는 재활, 약품 배송, 실험실 자동화에 활용된다. 농업에서는 수확(Harvesting), 가지치기(Pruning), 정밀 방제(Precision Spraying) 등에 적용된다.

ACT의 성능 평가는 단순한 예측 정확도가 아니라 시간적 부드러움(Temporal Smoothness), 궤적 연속성(Trajectory Continuity), Jerk 최소화, 제어 안정성(Control Stability), 에너지 효율(Energy Efficiency), 작업 성공률(Task Success Rate), 장기 수행 안정성(Long-Horizon Stability), 계산 지연(Inference Latency), 안전성(Safety Compliance)을 종합적으로 평가한다.

최근 로봇 파운데이션 모델은 ACT를 적극적으로 채택하고 있다. 영상, 언어, 행동 데이터를 함께 학습하여 장기간에 걸친 행동 패턴을 이해하고, 다양한 로봇 플랫폼에서도 동일한 행동 구조를 재사용할 수 있는 범용 정책을 구축하고 있다.

향후에는 ACT가 Flow Matching, 확산 정책(Diffusion Policy), 행동 토큰(Action Token), 계층형 계획(Hierarchical Planning), 세계 모델(World Model), 검색 증강 생성(RAG), 지속적 학습(Continual Learning), 외부 메모리(External Memory), 강화학습(Reinforcement Learning), 적응형 행동 청크(Dynamic Action Chunk)와 통합되어 더욱 강력한 차세대 액션 모델(Action Model)로 발전할 것으로 예상된다.

결국 ACT(Action Chunking Transformer)는 개별 행동이 아닌 연속적인 행동 청크(Action Chunk)를 생성함으로써 시간적 일관성(Temporal Consistency), 장기 계획(Long-Horizon Planning), 부드러운 제어(Smooth Control), 멀티모달 인식(Multimodal Perception), 세계 모델(World Model), 검색 증강 생성(RAG), 계층형 액션 모델(Hierarchical Action Model), 안전 검증(Safety Verification)을 하나의 통합 구조로 결합한다. 이를 통해 제조, 물류, 의료, 농업, 시설 점검, 서비스 로봇 등 다양한 물리 AI(Physical AI) 환경에서 안정적이고 자연스러운 장기 자율 행동을 실현하는 핵심 정책 기술로 자리잡고 있다.

시간적 일관성(Temporal Consistency)은 현대 로봇 정책(Robot Policy) 학습에서 가장 중요한 요소 중 하나이다. 로봇은 단순히 올바른 개별 행동(Action)을 생성하는 것만으로는 충분하지 않으며, 여러 시간(Time)에 걸쳐 부드럽고 안정적인 움직임을 유지해야 한다. 기존 정책은 매 시점마다 하나의 제어 명령만 생성하기 때문에 궤적(Trajectory)이 불연속적이거나 진동(Oscillation)이 발생하고, 장기 작업(Long-Horizon Task)에서는 오차(Error)가 누적되는 문제가 있었다. ACT(Action Chunking Transformer)는 이러한 문제를 해결하기 위해 여러 개의 미래 행동을 하나의 덩어리(Action Chunk)로 동시에 생성한다.

ACT(Action Chunking Transformer)의 기본 아이디어는 인간의 운동 제어(Human Motor Control)에서 영감을 얻었다. 사람은 매 순간 개별 근육을 독립적으로 제어하지 않는다. 걷기(Walking)는 보행 주기(Gait Cycle) 단위로 계획되고, 팔을 뻗는 동작도 하나의 연속된 궤적으로 계획된다. 물체를 잡을 때 역시 손가락을 각각 제어하는 것이 아니라 하나의 파지(Grasp) 동작 전체를 계획한다. ACT는 이러한 생물학적 제어 방식을 로봇 정책에 적용하여 여러 개의 행동을 하나의 연속된 행동 조각(Action Chunk)으로 생성한다.

기존 반응형 정책(Reactive Policy)은 현재 상태(Current State)만 보고 다음 한 개의 행동만 생성한다. 이 방식은 구현이 간단하지만 작은 예측 오차가 다음 예측으로 계속 전달되면서 장기적으로 큰 오차가 발생한다. 또한 센서 잡음(Sensor Noise)이나 환경 변화(Environment Change)가 누적되면 로봇은 불필요한 진동, 반복적인 보정 움직임, 급격한 속도 변화 등을 보일 수 있다.

ACT는 이러한 문제를 해결하기 위해 미래의 여러 제어 명령(Control Command)을 동시에 생성한다. 하나의 행동 청크(Action Chunk)는 일정 시간 동안 실행될 여러 개의 연속 행동으로 구성된다. 로봇은 이 청크를 실행하면서 동시에 주변 환경을 계속 관찰하고, 일정 시간이 지나면 새로운 상태를 반영하여 다음 행동 청크를 다시 생성한다. 이러한 반복적 계획(Receding Horizon Planning)은 장기 계획과 실시간 적응성을 동시에 제공한다.

시간적 일관성(Temporal Consistency)이란 연속된 행동들이 물리적으로 자연스럽고 논리적으로 연결되는 특성을 의미한다. 로봇의 움직임은 모터의 관성(Inertia), 기계 구조(Mechanical Constraint), 접촉(Contact), 동역학(Dynamics)을 모두 만족해야 한다. ACT는 여러 시점의 행동을 동시에 최적화하기 때문에 독립적인 행동 생성보다 훨씬 부드럽고 안정적인 움직임을 제공한다.

비전-언어-행동(VLA, Vision-Language-Action) 아키텍처에서는 RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 촉각 센서(Tactile Sensor), 힘 센서(Force Sensor), 내부 상태(Proprioception), 위치 추정(Localization), 의미 지도(Semantic Map) 등이 먼저 환경을 인식한다. 비전 인코더(Vision Encoder)는 이를 잠재 표현(Latent Representation)으로 변환하며, 대규모 언어 모델(LLM, Large Language Model)은 자연어 명령과 작업 목표(Task Goal)를 이해한다. 멀티모달 융합(Multimodal Fusion)은 이러한 정보를 통합하여 ACT의 입력으로 제공한다.

기존 액션 헤드(Action Head)는 하나의 제어 명령만 출력하지만, ACT는 여러 개의 미래 행동으로 구성된 행동 청크(Action Chunk)를 출력한다. 이 청크는 관절 궤적(Joint Trajectory), 엔드이펙터 자세(End-Effector Pose), 휠 속도(Wheel Velocity), 조향각(Steering Angle), 그리퍼 상태(Gripper State), 힘 제어(Force Control) 등으로 구성될 수 있다. 여러 행동이 동시에 생성되므로 시간적 관계가 자연스럽게 유지된다.

트랜스포머(Transformer)는 ACT에 매우 적합한 구조이다. 셀프 어텐션(Self-Attention)은 여러 미래 시점(Time Step)의 행동을 동시에 분석하여 현재 행동이 미래 행동과 일관성을 유지하도록 한다. 기존 순환 신경망(RNN, Recurrent Neural Network)보다 장기 의존성(Long-Term Dependency)을 더욱 효과적으로 학습할 수 있다.

ACT는 행동을 단순한 제어 명령이 아니라 시퀀스(Sequence)로 이해한다. 자연어 문장이 단어(Token)의 연속으로 구성되듯이, 로봇 행동도 연속적인 행동 시퀀스로 구성된다. 따라서 행동 간의 시간적 관계를 학습하여 더욱 자연스럽고 안정적인 움직임을 생성할 수 있다.

행동 청크(Action Chunk)의 길이는 매우 중요한 설계 요소이다. 짧은 청크는 빠른 환경 변화에 대응하기 쉽지만 계획 능력이 제한되고, 긴 청크는 부드러운 움직임과 장기 계획에는 유리하지만 즉각적인 반응성이 감소할 수 있다. 실제 시스템은 작업(Task), 환경(Environment), 계산 성능(Computing Power)에 따라 적절한 길이를 선택한다.

ACT는 중첩 청크(Overlapping Chunk)도 사용할 수 있다. 연속된 두 행동 청크가 일부 행동을 공유하도록 하여 청크 경계에서 발생할 수 있는 불연속적인 움직임을 줄인다. 이러한 중첩 구조는 매우 자연스러운 행동 연결을 가능하게 한다.

슬라이딩 윈도우(Sliding Window) 실행 방식도 함께 사용된다. 하나의 행동 청크를 모두 실행하지 않고 앞부분만 실행한 후 새로운 센서 정보를 반영하여 다음 청크를 다시 생성한다. 이를 통해 장기 계획을 유지하면서도 실시간 환경 변화에 적응할 수 있다.

ACT는 시간적 평활화(Temporal Smoothing)를 자연스럽게 제공한다. 여러 행동을 동시에 최적화하기 때문에 가속도(Acceleration), 속도(Velocity), 방향(Direction)이 부드럽게 변화하며, 액추에이터(Actuator)의 진동이 줄어들고 에너지 소비(Energy Consumption)도 감소한다.

오차 누적(Error Accumulation)은 기존 정책의 가장 큰 문제였다. 하나의 행동 오차가 다음 행동으로 계속 전달되면서 장기적으로 큰 실패를 유발할 수 있었다. ACT는 여러 행동을 동시에 최적화하므로 오차가 개별 행동마다 증폭되지 않고 훨씬 안정적인 장기 수행(Long-Horizon Execution)이 가능하다.

ACT는 대부분 행동 복제(Behavior Cloning) 방식으로 학습된다. 원격 조작(Teleoperation), VR 조작, 산업 자동화 데이터, 강화학습(Reinforcement Learning), 시뮬레이션(Simulation) 등을 이용하여 전문가 행동을 수집하고, 개별 행동이 아니라 전체 행동 청크를 학습 대상으로 사용한다.

손실 함수(Loss Function)도 개별 행동이 아니라 전체 궤적(Trajectory)을 기준으로 계산된다. 평균제곱오차(MSE, Mean Squared Error), Smooth L1 Loss, 속도 정규화(Velocity Regularization), 가속도 패널티(Acceleration Penalty), Jerk 최소화(Jerk Minimization), 자세 일관성(Orientation Consistency) 등을 함께 사용하여 전체 움직임의 품질을 높인다.

ACT에서는 개별 행동의 정확도보다 전체 궤적의 품질이 더욱 중요하다. 일부 시점에서 작은 오차가 있더라도 전체 움직임이 자연스럽고 작업 성공률(Task Success Rate)이 높으면 더 좋은 정책으로 평가된다.

ACT는 연속 행동 공간(Continuous Action Space)을 그대로 지원한다. 행동 토큰(Action Token)처럼 이산화(Discretization)를 수행하지 않고 관절 위치(Joint Position), 휠 속도(Wheel Velocity), 토크(Torque), 힘 제어(Force Control) 등을 연속적인 값으로 예측한다.

계층형 액션 모델(Hierarchical Action Model)과도 매우 잘 결합된다. 상위 정책(High-Level Policy)은 작업 목표를 생성하고, 중간 정책(Mid-Level Policy)은 기술(Skill)과 경로(Path)를 계획하며, ACT는 이를 실행하기 위한 행동 청크를 생성한다. 하위 제어기(Low-Level Controller)는 PID, MPC(Model Predictive Control), 역기구학(Inverse Kinematics)을 이용하여 이를 실제 모터 제어로 변환한다.

세계 모델(World Model)은 ACT의 성능을 더욱 향상시킨다. 현재 환경뿐 아니라 미래의 물체 이동(Object Motion), 사람의 움직임(Human Activity), 장애물 변화, 배터리 상태 등을 예측하여 행동 청크 생성에 반영한다. 이를 통해 미래를 고려한 계획(Predictive Planning)이 가능해진다.

어포던스 추론(Affordance Reasoning)은 실행 가능한 행동만 생성하도록 한다. 접근 가능한 물체(Reachable Object), 충돌 없는 경로(Collision-Free Path), 안정적인 파지(Stable Grasp), 이동 가능한 공간(Navigable Area)을 고려하여 실제 수행 가능한 행동 청크만 생성한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 유지보수 문서, 디지털 트윈(Digital Twin), 작업 기록, 시설 지도(Map), 기업 데이터베이스를 검색하여 행동 청크 생성에 필요한 추가 정보를 제공한다. 외부 지식이 행동 계획에 직접 활용된다.

메모리(Memory)는 장시간 작업에서 매우 중요한 역할을 한다. 에피소드 메모리(Episodic Memory)는 이전 성공 사례를 저장하고, 의미 메모리(Semantic Memory)는 환경 지식을 저장하며, 절차 메모리(Procedural Memory)는 반복 가능한 기술(Skill)을 저장한다. 작업 메모리(Working Memory)는 현재 진행 상황을 유지하여 행동 청크 생성에 활용된다.

시뮬레이션(Simulation)은 ACT 학습에서 필수적이다. 디지털 트윈(Digital Twin)은 다양한 물체 배치, 센서 잡음, 날씨, 환경 변화, 장비 고장 등을 포함한 수백만 개의 행동 청크를 생성하여 정책의 일반화 성능을 크게 향상시킨다.

Sim-to-Real 기술은 시뮬레이션에서 학습한 ACT 정책을 실제 로봇에 적용한다. 도메인 랜덤화(Domain Randomization), 물리 오차(Physics Variation), 센서 오차(Sensor Noise), 통신 지연(Latency)을 포함하여 학습한 후 실제 데이터로 미세조정(Fine-Tuning)을 수행한다.

로봇의 형태(Embodiment)에 따라 행동 청크도 달라진다. 산업용 매니퓰레이터는 관절 궤적과 그리퍼 동작을 생성하고, AMR은 속도와 조향 궤적을 생성하며, 사족보행 로봇(Quadruped)은 보행 주기(Gait Cycle)를 생성하고, 휴머노이드는 보행과 조작을 포함한 전신 움직임을 생성한다.

클라우드-엣지 협업(Cloud-Edge Collaboration)에서도 ACT는 매우 적합하다. 클라우드는 LLM 추론, 세계 모델, 시뮬레이션, 장기 계획을 수행하고, 엣지 컴퓨터는 행동 청크 생성, 실시간 제어, 안전 감시를 담당한다. 통신이 끊겨도 행동 청크를 이용하여 일정 시간 동안 자율 동작을 계속 수행할 수 있다.

ACT는 추론 효율(Inference Efficiency)도 우수하다. 하나의 순전파(Forward Pass)로 여러 개의 미래 행동을 동시에 생성하므로 매 시점마다 정책을 반복 실행하는 것보다 계산량이 크게 감소한다. GPU, Tensor Core, NPU(Neural Processing Unit), 혼합 정밀도(Mixed Precision), 양자화(Quantization)를 적용하면 실시간 성능도 충분히 확보할 수 있다.

안전(Safety)은 ACT와 독립적으로 유지된다. 생성된 행동 청크는 실행 전에 충돌 감지(Collision Detection), 운동학 검증(Kinematic Verification), 동역학 제한(Dynamic Constraint), 작업 공간 제한(Workspace Limit), 비상 정지(Emergency Stop), 런타임 모니터(Runtime Monitor)를 통해 반드시 검증된다.

다중 로봇(Multi-Robot) 협업에서도 ACT는 효과적이다. 여러 로봇이 서로 시간적으로 일관된 행동 청크를 생성하여 협업 운반(Cooperative Transport), 조립(Assembly), 물류(Logistics), 점검(Inspection)을 수행하면 작업의 동기화(Synchronization)가 크게 향상된다.

산업 제조에서는 정밀 조립(Precision Assembly), 삽입(Insertion), 용접(Welding), 연마(Polishing), 케이블 조립(Cable Routing), 협업 조작(Collaborative Manipulation)에 활용된다. 물류에서는 자율주행, 도킹(Docking), 충전(Charging), 창고 관리에 적용되며, 의료에서는 재활, 약품 배송, 실험실 자동화에 활용된다. 농업에서는 수확(Harvesting), 가지치기(Pruning), 정밀 방제(Precision Spraying) 등에 적용된다.

ACT의 성능 평가는 단순한 예측 정확도가 아니라 시간적 부드러움(Temporal Smoothness), 궤적 연속성(Trajectory Continuity), Jerk 최소화, 제어 안정성(Control Stability), 에너지 효율(Energy Efficiency), 작업 성공률(Task Success Rate), 장기 수행 안정성(Long-Horizon Stability), 계산 지연(Inference Latency), 안전성(Safety Compliance)을 종합적으로 평가한다.

최근 로봇 파운데이션 모델은 ACT를 적극적으로 채택하고 있다. 영상, 언어, 행동 데이터를 함께 학습하여 장기간에 걸친 행동 패턴을 이해하고, 다양한 로봇 플랫폼에서도 동일한 행동 구조를 재사용할 수 있는 범용 정책을 구축하고 있다.

향후에는 ACT가 Flow Matching, 확산 정책(Diffusion Policy), 행동 토큰(Action Token), 계층형 계획(Hierarchical Planning), 세계 모델(World Model), 검색 증강 생성(RAG), 지속적 학습(Continual Learning), 외부 메모리(External Memory), 강화학습(Reinforcement Learning), 적응형 행동 청크(Dynamic Action Chunk)와 통합되어 더욱 강력한 차세대 액션 모델(Action Model)로 발전할 것으로 예상된다.

결국 ACT(Action Chunking Transformer)는 개별 행동이 아닌 연속적인 행동 청크(Action Chunk)를 생성함으로써 시간적 일관성(Temporal Consistency), 장기 계획(Long-Horizon Planning), 부드러운 제어(Smooth Control), 멀티모달 인식(Multimodal Perception), 세계 모델(World Model), 검색 증강 생성(RAG), 계층형 액션 모델(Hierarchical Action Model), 안전 검증(Safety Verification)을 하나의 통합 구조로 결합한다. 이를 통해 제조, 물류, 의료, 농업, 시설 점검, 서비스 로봇 등 다양한 물리 AI(Physical AI) 환경에서 안정적이고 자연스러운 장기 자율 행동을 실현하는 핵심 정책 기술로 자리잡고 있다.

## 4.8 Multimodal Action Prediction (with Code)

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

멀티모달 행동 예측(Multi-Modal Action Prediction)은 현대 물리 AI(Physical AI)의 핵심 기술 중 하나이다. 지능형 로봇은 단순히 주변을 보는 것만으로는 충분하지 않으며, 시각(Vision), 언어(Language), 로봇 내부 상태(Proprioception), 과거 경험(Memory), 환경 정보(Environment Context)를 동시에 이해해야 한다. 기존 로봇 시스템은 인식, 계획, 제어를 각각 독립적인 모듈로 구현하였지만, 현대의 비전-언어-행동(VLA, Vision-Language-Action) 아키텍처는 모든 정보를 하나의 잠재 공간(Latent Space)으로 통합하여 행동(Action)을 생성한다. 이를 통해 범용 로봇 지능(General-Purpose Embodied Intelligence)을 구현할 수 있다.

멀티모달 행동 예측의 핵심은 하나의 정보만으로는 올바른 행동을 결정할 수 없다는 점이다. 카메라는 물체를 볼 수 있지만 왜 그 물체를 선택해야 하는지는 알지 못한다. 언어는 작업 목표를 설명하지만 실제 물체의 위치는 알 수 없다. 로봇의 내부 상태만으로는 사람의 의도를 이해할 수 없다. 따라서 다양한 정보가 동시에 결합되어야 비로소 정확한 행동을 생성할 수 있다.

인간의 인지 과정도 동일한 원리로 동작한다. 예를 들어 "검사 테이블 옆의 파란 공구함을 가져와라."라는 명령을 들으면 사람은 먼저 언어를 이해하고, 시각으로 공구함을 찾으며, 공간 관계(Spatial Relation)를 계산하고, 이전 경험(Memory)을 활용하여 공구를 인식한 후 팔과 손을 움직인다. 이 모든 과정은 독립적으로 수행되는 것이 아니라 하나의 통합된 사고 과정으로 이루어진다. 멀티모달 행동 예측은 이러한 인간의 인지 구조를 로봇에 적용한 것이다.

기존 로봇은 물체 인식(Object Detection), 위치 추정(Localization), 경로 계획(Path Planning), 제어(Control)를 각각 독립적으로 수행하였다. 이러한 구조는 산업 현장에서는 안정적이지만 새로운 물체나 새로운 명령, 예상하지 못한 상황에서는 일반화 성능이 낮았다. 멀티모달 행동 예측은 각 모듈을 별도로 연결하는 대신 하나의 통합 모델에서 의미적 관계(Semantic Relationship)를 학습한다.

비전(Vision)은 멀티모달 시스템에서 가장 중요한 입력 중 하나이다. RGB 카메라(RGB Camera)는 색상(Color), 질감(Texture), 조명(Illumination), 물체 종류(Object Category), 사람의 행동(Human Activity)을 인식한다. 깊이 카메라(Depth Camera)는 거리(Distance), 3차원 형상(3D Shape), 자유 공간(Free Space)을 제공하며, 라이다(LiDAR)는 위치 추정(Localization), 지도 작성(Mapping), 장애물 탐지(Obstacle Detection), 자율주행(Navigation)에 필요한 공간 정보를 제공한다.

최근에는 이벤트 카메라(Event Camera), 열화상 카메라(Thermal Camera), 다중분광 카메라(Multispectral Camera), 레이더(Radar), 초분광 센서(Hyperspectral Sensor)도 함께 활용된다. 다양한 센서는 서로 다른 환경 정보를 제공하므로 로봇은 더욱 정확한 환경 이해(Environment Understanding)가 가능해진다.

비전 인코더(Vision Encoder)는 이러한 다양한 센서 데이터를 잠재 특징(Latent Feature)으로 변환한다. 비전 트랜스포머(ViT, Vision Transformer), 합성곱 신경망(CNN), 마스크드 오토인코더(MAE, Masked Autoencoder), 자기지도학습(Self-Supervised Learning) 등이 사용되며, 물체의 세부 특징과 전체 장면(Scene Context)을 동시에 표현한다.

언어(Language)는 두 번째 핵심 모달리티(Modality)이다. 사람은 음성(Speech), 텍스트(Text), 작업 지시(Instruction), 매뉴얼(Manual), 유지보수 문서(Maintenance Document)를 통해 로봇에게 목표를 전달한다. 대규모 언어 모델(LLM, Large Language Model)은 이러한 자연어를 의미 표현(Semantic Representation)으로 변환하여 작업 목표(Task Goal), 절차(Procedure), 상식(Common Sense), 인과 관계(Causal Reasoning)를 이해한다.

언어 이해는 단순한 명령 해석을 넘어선다. 이전 대화(Dialog History)를 기억하고, 모호한 명령은 질문으로 확인하며, 작업 순서를 계획하고, 상식을 이용하여 안전한 행동을 선택한다. 따라서 언어는 단순한 명령이 아니라 고수준 추론(High-Level Reasoning)의 핵심 역할을 수행한다.

로봇 내부 상태(Proprioception)도 중요한 입력이다. 관절 위치(Joint Position), 휠 속도(Wheel Velocity), 조향각(Steering Angle), 토크(Torque), IMU(Inertial Measurement Unit), 배터리 상태(Battery Status), 액추에이터 온도(Actuator Temperature), 엔드이펙터 자세(End-Effector Pose)는 로봇이 현재 무엇을 할 수 있는지를 결정하는 핵심 정보이다.

환경 정보(Environment Context)도 함께 사용된다. 의미 지도(Semantic Map), 위치 추정(Localization), 작업 구역(Workspace), 제한 구역(Restricted Area), 충전 스테이션(Charging Station), 검사 지점(Inspection Point), 협업 공간(Collaboration Zone)은 현재 행동이 적절한지 판단하는 중요한 기준이 된다.

메모리(Memory)는 멀티모달 행동 예측의 또 다른 핵심 요소이다. 에피소드 메모리(Episodic Memory)는 이전 성공 사례를 저장하고, 의미 메모리(Semantic Memory)는 환경과 사물의 지식을 저장하며, 절차 메모리(Procedural Memory)는 반복 가능한 작업 절차를 저장한다. 작업 메모리(Working Memory)는 현재 작업 상태와 대화 내용을 유지하여 연속적인 작업 수행을 가능하게 한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 외부 지식을 행동 생성에 활용한다. 기업 문서, 유지보수 매뉴얼, 디지털 트윈(Digital Twin), CAD 데이터, 작업 기록, 생산 데이터베이스 등을 검색하여 모델 내부의 지식과 함께 행동 생성에 사용한다. 따라서 최신 정보도 별도의 재학습 없이 활용할 수 있다.

멀티모달 행동 예측의 가장 큰 기술적 과제는 서로 다른 정보를 하나로 통합하는 것이다. 영상은 공간 정보(Spatial Information)를 가지며, 언어는 순차적 의미(Sequential Semantics)를 가지며, 로봇 상태는 연속적인 수치(Continuous Variable)를 가진다. 이질적인 정보를 하나의 잠재 공간으로 변환하는 것이 멀티모달 융합(Multimodal Fusion)의 핵심이다.

초기 융합(Early Fusion)은 각 모달리티의 특징을 바로 연결(Concatenation)하는 방식이다. 구조는 단순하지만 서로 다른 특성을 가진 데이터가 섞이면서 중요한 의미 정보가 손실될 수 있다.

후기 융합(Late Fusion)은 각 모달리티를 독립적으로 처리한 후 마지막 단계에서 결과를 결합한다. 각 모달리티의 특성을 잘 유지할 수 있지만, 중간 단계에서 서로 영향을 주지 못하기 때문에 깊은 의미적 연결이 어려울 수 있다.

최근 가장 많이 사용되는 방식은 크로스 어텐션(Cross-Attention)이다. 영상 특징은 언어 토큰(Language Token)을 참고하고, 언어는 영상 특징을 참고하며, 로봇 상태와 메모리도 서로 영향을 주면서 필요한 정보만 선택적으로 교환한다. 이러한 동적 정보 교환이 멀티모달 학습의 핵심 기술이다.

트랜스포머(Transformer)는 멀티모달 처리에 매우 적합하다. 영상 패치(Image Patch), 언어 토큰, 로봇 상태, 메모리, 의미 지도 등을 하나의 입력 시퀀스로 처리하며, 셀프 어텐션(Self-Attention)을 이용하여 서로의 관계를 학습한다. 대규모 사전학습을 통해 다양한 모달리티가 동일한 의미 공간으로 정렬된다.

대조 학습(Contrastive Learning)은 멀티모달 정렬(Alignment)을 더욱 강화한다. 동일한 의미를 가진 영상과 문장은 잠재 공간에서 가까워지고, 관련 없는 데이터는 멀어지도록 학습한다. 이를 통해 학습하지 않은 새로운 물체나 명령도 일반화할 수 있다.

시간 정렬(Temporal Alignment)도 중요한 과제이다. 카메라는 초당 수십 프레임을 생성하지만, 언어는 몇 초에 한 번만 입력될 수 있으며, 로봇의 내부 상태는 수백 Hz로 갱신된다. 멀티모달 모델은 이러한 서로 다른 시간 해상도를 동시에 처리할 수 있어야 한다.

세계 모델(World Model)은 미래 환경까지 고려하는 멀티모달 추론을 가능하게 한다. 현재 영상뿐 아니라 사람의 이동, 물체의 움직임, 배터리 상태, 교통 흐름 등을 예측하여 행동 생성에 반영한다. 결과적으로 반응형(Reactive) 제어가 아니라 예측형(Predictive) 제어가 가능해진다.

계층형 액션 모델(Hierarchical Action Model)에서는 상위 정책(High-Level Policy)이 언어와 작업 목표를 이해하고, 중간 정책(Mid-Level Policy)이 기술(Skill)과 경로(Path)를 계획하며, 하위 정책(Low-Level Policy)은 모터를 제어한다. 모든 계층이 멀티모달 정보를 공유하지만 각자의 역할에 맞는 수준으로 활용한다.

멀티모달 임베딩(Multimodal Embedding)은 다양한 액션 헤드(Action Head)와 결합될 수 있다. 회귀 기반 정책(Regression Policy)은 연속 제어 값을 생성하고, 확산 정책(Diffusion Policy)은 행동 분포를 생성하며, Flow Matching은 연속 벡터 필드를 생성하고, 행동 토큰(Action Token)은 이산 행동을 생성하며, ACT(Action Chunking Transformer)는 여러 미래 행동을 동시에 생성한다. 어떤 정책을 사용하더라도 멀티모달 표현은 공통 기반이 된다.

어포던스 추론(Affordance Reasoning)은 행동 생성의 현실성을 높인다. 영상은 물체를 인식하고, 언어는 작업 목적을 제공하며, 메모리는 물체의 기능을 기억하고, 충돌 검사(Collision Check)는 실제 수행 가능성을 확인한다. 이 모든 정보를 통합하여 실행 가능한 행동만 선택한다.

사람과 로봇의 협업(Human-Robot Collaboration)에서도 멀티모달 추론은 매우 중요하다. 음성(Speech), 손짓(Gesture), 시선(Gaze), 얼굴 표정(Facial Expression), 자세(Posture), 이전 대화(Dialog History)를 함께 분석하여 사람의 의도를 더욱 정확하게 이해할 수 있다.

산업 제조에서는 비전으로 작업물을 인식하고, 생산 데이터베이스를 참고하며, 작업 절차 문서를 검색하고, 힘 센서로 조립 품질을 확인하면서 작업을 수행한다. 여러 정보가 동시에 활용되어 유연한 생산(Flexible Manufacturing)이 가능해진다.

물류에서는 재고 데이터베이스, 의미 지도, 위치 추정, 장애물 탐지, 배터리 상태, 작업 지시가 함께 사용된다. 의료에서는 환자 정보, 병원 규정, 생체 신호, 음성 명령을 함께 고려하며, 농업에서는 RGB 영상, 다중분광 센서, GPS, 날씨 정보, 농장 관리 시스템을 통합하여 작물 관리와 수확을 수행한다.

시뮬레이션(Simulation)은 멀티모달 데이터 생성에 매우 중요하다. 디지털 트윈은 영상, 언어, 행동, 환경 변화, 센서 데이터를 동시에 생성하며, 도메인 랜덤화(Domain Randomization)를 적용하여 실제 환경에서도 강인한 정책을 학습할 수 있도록 지원한다.

자기지도학습(Self-Supervised Learning)은 영상-언어 대응(Vision-Language Correspondence), 마스크 예측(Masked Prediction), 대조 학습(Contrastive Learning)을 이용하여 대규모 사전학습을 수행한다. 이후 실제 로봇 데이터로 미세조정(Fine-Tuning)하여 범용 멀티모달 파운데이션 모델을 구축한다.

파라미터 효율적 미세조정(PEFT, Parameter-Efficient Fine-Tuning)은 동일한 멀티모달 모델을 다양한 로봇 플랫폼에 적용할 수 있도록 한다. 상위 추론은 그대로 유지하면서 하위 제어만 로봇 구조에 맞게 수정하면 되므로 개발 비용을 크게 줄일 수 있다.

클라우드-엣지 협업(Cloud-Edge Collaboration)에서는 클라우드가 LLM 추론, 세계 모델, RAG, 장기 계획을 수행하고, 엣지는 센서 융합, 행동 생성, 안전 감시, 실시간 제어를 수행한다. 계산 자원을 효율적으로 분산할 수 있는 구조이다.

안전(Safety)은 멀티모달 추론과 독립적으로 유지된다. 생성된 행동은 충돌 감지(Collision Detection), 운동학 검증(Kinematic Verification), 힘 제한(Force Limit), 작업 공간 제한(Workspace Limit), 런타임 모니터(Runtime Monitor), 비상 정지(Emergency Stop)를 통해 항상 검증된 후 실행된다.

멀티모달 행동 예측의 평가는 단순한 인식 정확도가 아니라 작업 성공률(Task Success Rate), 조작 성공률(Manipulation Success), 자율주행 성능(Navigation Robustness), 언어 이해(Language Grounding), 일반화 성능(Generalization), 장기 작업 안정성(Long-Horizon Stability), 인간 협업(Human Collaboration), 계산 효율(Inference Efficiency), 안전성(Safety Compliance)을 종합적으로 평가한다.

향후에는 세계 모델(World Model), 지속적 학습(Continual Learning), 외부 메모리(External Memory), 검색 증강 생성(RAG), 계층형 계획(Hierarchical Planning), Flow Matching, ACT(Action Chunking Transformer), 행동 토큰(Action Token), 강화학습(Reinforcement Learning), 다중 에이전트(Multi-Agent Collaboration)가 하나의 통합 멀티모달 파운데이션 모델(Foundation Model)로 발전할 것으로 예상된다.

결국 멀티모달 행동 예측은 비전(Vision), 언어(Language), 로봇 내부 상태(Proprioception), 메모리(Memory), 환경 정보(Environment Context), 세계 모델(World Model), 검색 증강 생성(RAG), 계층형 액션 모델(Hierarchical Action Model), 다양한 액션 헤드(Action Head)를 하나의 통합 잠재 공간(Unified Latent Space)으로 결합하여 의미적으로 올바르고, 물리적으로 실행 가능하며, 시간적으로 일관된 행동을 생성한다. 이는 제조, 물류, 의료, 농업, 시설 점검, 서비스 로봇 등 다양한 물리 AI(Physical AI) 분야에서 범용 로봇 지능(General-Purpose Embodied Intelligence)을 구현하기 위한 핵심 기반 기술로 자리매김하고 있다.

## 4.9 Calibration and Uncertainty Estimation (with Code)

![](images/image9.png){width="7.268055555555556in" height="7.268055555555556in"}

액션 모델 보정(Action Model Calibration)과 불확실성 추정(Uncertainty Estimation)은 현대 물리 AI(Physical AI)에서 필수적인 기술로 자리잡고 있다. 지능형 로봇은 단순히 행동(Action)을 생성하는 것만으로는 충분하지 않으며, 생성한 행동이 얼마나 신뢰할 수 있는지까지 스스로 판단해야 한다. 기존 로봇은 정책(Policy)이 출력한 결과를 그대로 실행하는 경우가 많았지만, 실제 환경에서는 센서 오류(Sensor Failure), 환경 변화(Environment Change), 불완전한 관측(Incomplete Observation), 모호한 명령(Ambiguous Instruction) 등이 빈번하게 발생한다. 따라서 현대의 비전-언어-행동(VLA, Vision-Language-Action) 시스템은 행동을 생성하는 동시에 그 행동의 신뢰도(Confidence)를 함께 계산해야 한다.

불확실성(Uncertainty)을 고려하는 이유는 인간의 의사결정 방식과 매우 유사하다. 사람도 명확하게 보이는 물체는 자신 있게 집지만, 어두운 환경이나 비슷한 물체가 여러 개 있을 때는 망설이거나 다시 확인한다. 이러한 자기 신뢰(Self-Confidence)는 행동 수행 여부를 결정하는 중요한 기준이 된다. 로봇 역시 자신의 판단이 얼마나 확실한지를 이해해야만 안전하고 신뢰성 있는 자율 행동을 수행할 수 있다.

기존 정책은 지도학습(Supervised Learning), 모방학습(Imitation Learning), 강화학습(Reinforcement Learning)을 이용하여 행동을 생성하지만, 충분한 정보가 없는 상황에서도 항상 하나의 행동을 출력한다. 신경망은 높은 확률 값을 출력하더라도 실제로는 잘못된 판단을 할 수 있으며, 이러한 과신(Overconfidence)은 위험한 행동으로 이어질 수 있다. 따라서 행동 생성과 신뢰도 평가는 별도의 문제로 다루어져야 한다.

보정(Calibration)은 예측한 신뢰도와 실제 성공률을 일치시키는 과정이다. 예를 들어 모델이 90%의 신뢰도를 출력한 행동은 실제로 약 90% 정도 성공해야 하며, 50%의 신뢰도를 출력한 행동은 절반 정도만 성공해야 한다. 이렇게 신뢰도와 실제 성공률이 일치할 때 모델은 잘 보정(Calibrated)되었다고 말한다.

행동 생성(Action Prediction)과 불확실성 추정(Uncertainty Estimation)은 서로 다른 역할을 수행한다. 행동 생성은 "무엇을 할 것인가"를 결정하고, 불확실성 추정은 "그 행동을 얼마나 믿을 수 있는가"를 판단한다. 현대 로봇은 두 과정을 동시에 수행하여 실행 여부를 결정한다.

비전-언어-행동(VLA) 시스템에서는 불확실성이 여러 모달리티(Modality)에서 동시에 발생한다. 시각 불확실성(Visual Uncertainty)은 조명 부족, 모션 블러(Motion Blur), 가림(Occlusion), 반사체, 투명 물체, 센서 잡음 때문에 발생한다. 언어 불확실성(Language Uncertainty)은 모호한 표현, 불완전한 명령, 상충되는 대화에서 발생하며, 로봇 상태 불확실성(Robot State Uncertainty)은 위치 오차(Localization Error), 바퀴 미끄러짐(Wheel Slip), 액추에이터 열화(Actuator Degradation), 통신 지연(Latency) 등으로 인해 발생한다.

환경 불확실성(Environment Uncertainty)도 매우 중요하다. 사람의 이동, 장애물 변화, 공사 구역, 날씨 변화, 농작물 성장, 생산 라인의 변경과 같이 지속적으로 변하는 환경은 행동의 신뢰도를 크게 변화시킨다. 따라서 현대 로봇은 환경 자체의 불확실성도 함께 모델링해야 한다.

불확실성은 하나의 숫자로 표현되지 않는다. 시각 신뢰도, 언어 이해 신뢰도, 위치 추정 신뢰도, 조작 가능성(Manipulation Feasibility), 경로 안정성(Trajectory Robustness), 안전성(Safety) 등이 각각 별도로 존재하며, 이들이 함께 최종 행동의 신뢰도를 결정한다.

비전 인코더(Vision Encoder)는 물체 인식(Object Detection) 신뢰도, 깊이 추정(Depth Estimation) 오차, 자세 추정(Pose Estimation) 신뢰도 등을 함께 출력한다. 따라서 액션 모델(Action Model)은 단순한 인식 결과가 아니라 인식의 신뢰도까지 고려하여 행동을 생성할 수 있다.

대규모 언어 모델(LLM, Large Language Model)도 명령 해석의 신뢰도를 계산한다. "왼쪽에 있는 박스를 가져와."처럼 모호한 명령이 들어오면 모델은 낮은 신뢰도를 출력하고 사용자에게 다시 질문할 수 있다. 이러한 능력은 실제 협업 환경에서 매우 중요하다.

로봇 내부 상태(Proprioception)도 불확실성을 포함한다. 위치 추정(Localization)은 공분산(Covariance)을 계산하고, 힘 센서(Force Sensor)는 측정 오차를 계산하며, 배터리 관리 시스템(BMS)은 남은 사용 시간을 확률적으로 예측한다. 따라서 내부 상태 자체도 확률적으로 표현된다.

확률적 상태 추정(Probabilistic State Estimation)은 오래전부터 로봇에서 사용되어 왔다. 칼만 필터(Kalman Filter), 파티클 필터(Particle Filter), 베이지안 필터(Bayesian Filter), SLAM(Simultaneous Localization and Mapping)은 모두 상태를 하나의 값이 아니라 확률 분포(Probability Distribution)로 표현한다. 현대 액션 모델은 이러한 개념을 딥러닝 정책으로 확장하고 있다.

불확실성은 크게 두 종류로 나뉜다. 알레아토릭 불확실성(Aleatoric Uncertainty)은 센서 잡음이나 환경 변화처럼 본질적으로 제거할 수 없는 불확실성이다. 더 많은 데이터를 학습해도 완전히 사라지지 않는다.

반면 에피스테믹 불확실성(Epistemic Uncertainty)은 모델이 경험하지 못한 상황에서 발생한다. 새로운 물체, 새로운 환경, 새로운 작업은 학습 데이터가 부족하기 때문에 불확실성이 높아진다. 이러한 불확실성은 지속적 학습(Continual Learning)이나 추가 데이터 학습을 통해 감소시킬 수 있다.

두 불확실성을 구분하는 것은 매우 중요하다. 알레아토릭 불확실성이 높으면 속도를 줄이거나 추가 센서를 사용해야 하고, 에피스테믹 불확실성이 높으면 추가 학습, 시뮬레이션, 사람의 개입이 필요할 수 있다.

베이지안 신경망(Bayesian Neural Network)은 신경망의 가중치를 하나의 값이 아니라 확률 분포로 표현하여 불확실성을 계산한다. 계산량은 크지만 가장 이론적으로 타당한 방법 중 하나이다.

몬테카를로 드롭아웃(Monte Carlo Dropout)은 보다 효율적인 방법이다. 추론(Inference) 중에도 드롭아웃(Dropout)을 유지하여 동일한 입력을 여러 번 실행하고, 결과의 분산(Variance)을 계산하여 불확실성을 추정한다.

딥 앙상블(Deep Ensemble)은 여러 개의 독립적인 모델을 학습한 후 동일한 입력에 대해 서로 다른 예측을 수행한다. 모든 모델이 같은 결과를 출력하면 신뢰도가 높고, 서로 다른 결과를 출력하면 불확실성이 높다고 판단한다.

분포 기반 액션 모델(Distributional Action Model)은 하나의 행동만 생성하지 않고 여러 가능한 행동의 확률 분포를 생성한다. 따라서 기대 성능(Expected Performance)과 위험(Risk)을 동시에 고려하여 최적 행동을 선택할 수 있다.

확산 정책(Diffusion Policy)은 본질적으로 행동 분포(Action Distribution)를 생성하기 때문에 불확실성 추정에 매우 적합하다. 여러 번 샘플링(Sampling)을 수행하면 가능한 미래 행동들의 분포를 얻을 수 있으며, 분포가 넓을수록 불확실성이 크다는 의미가 된다.

Flow Matching 역시 연속적인 확률 흐름(Probability Flow)을 학습하기 때문에 행동 생성과 동시에 확률 밀도(Probability Density)를 계산할 수 있다. 최근에는 확산 정책보다 빠른 추론 속도 때문에 많은 연구가 이루어지고 있다.

ACT(Action Chunking Transformer)는 개별 행동이 아니라 행동 청크(Action Chunk)의 불확실성을 계산한다. 미래 전체 궤적(Trajectory)에 대한 신뢰도를 평가할 수 있기 때문에 장기 계획(Long-Horizon Planning)에 매우 적합하다.

행동 토큰(Action Token) 기반 정책도 토큰의 확률을 이용하여 불확실성을 계산한다. 엔트로피(Entropy)가 높으면 여러 행동이 비슷한 확률을 가지므로 불확실성이 높고, 하나의 토큰만 높은 확률을 가지면 매우 자신 있는 행동이라고 판단할 수 있다.

학습 후에는 다양한 보정(Calibration) 기법을 적용할 수 있다. Temperature Scaling, Platt Scaling, Isotonic Regression, Dirichlet Calibration 등이 대표적인 방법이며, 모델을 다시 학습하지 않고도 신뢰도를 실제 성공률에 맞게 조정할 수 있다.

보정 성능은 ECE(Expected Calibration Error)로 평가하는 경우가 가장 많다. 예측 신뢰도와 실제 성공률의 차이를 계산하여 값이 작을수록 잘 보정된 모델로 평가한다. 이 외에도 Brier Score, Negative Log Likelihood, Reliability Diagram 등이 함께 사용된다.

신뢰도 다이어그램(Reliability Diagram)은 예측 신뢰도와 실제 성공률을 시각적으로 비교하는 그래프이다. 대각선에 가까울수록 이상적인 보정 상태이며, 대각선 위쪽은 과소 신뢰(Underconfidence), 아래쪽은 과신(Overconfidence)을 의미한다.

현대 액션 모델은 위험 기반 계획(Risk-Aware Planning)을 수행한다. 신뢰도가 높으면 정상적으로 실행하고, 중간 수준이면 속도를 줄이거나 추가 센서를 사용하며, 매우 낮으면 사용자에게 확인을 요청하거나 계획을 다시 생성한다.

계층형 액션 모델(Hierarchical Action Model)에서는 각 계층이 독립적으로 불확실성을 계산한다. 상위 정책(High-Level Policy)은 작업 계획의 신뢰도를 계산하고, 중간 정책(Mid-Level Policy)은 경로 계획의 신뢰도를 계산하며, 하위 정책(Low-Level Policy)은 제어 안정성과 센서 오차를 계산한다.

세계 모델(World Model)은 미래 환경의 불확실성도 예측한다. 사람의 이동, 장애물 변화, 날씨, 배터리 상태 등을 여러 시나리오로 예측하여 가장 안전한 행동을 선택한다. 이는 단순한 현재 기반 제어보다 훨씬 안정적인 장기 계획을 가능하게 한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 불확실성을 줄이는 중요한 방법이다. 모델의 신뢰도가 낮으면 유지보수 문서, 디지털 트윈(Digital Twin), 작업 매뉴얼, 시설 지도(Map), 데이터베이스를 검색하여 부족한 정보를 보완한 후 다시 행동을 생성한다.

메모리(Memory)도 불확실성 감소에 중요한 역할을 한다. 에피소드 메모리(Episodic Memory)는 과거 성공 사례를 활용하고, 의미 메모리(Semantic Memory)는 환경 지식을 제공하며, 절차 메모리(Procedural Memory)는 검증된 작업 절차를 재사용한다. 작업 메모리(Working Memory)는 현재 불확실성의 변화를 지속적으로 관리한다.

시뮬레이션(Simulation)은 위험한 상황을 안전하게 학습할 수 있도록 한다. 디지털 트윈은 센서 고장, 액추에이터 오류, 통신 지연, 극한 환경, 예상하지 못한 장애물 등을 반복적으로 생성하여 로봇이 실제 위험 상황을 사전에 경험하도록 만든다.

Sim-to-Real 기술에서도 불확실성 모델링은 매우 중요하다. 시뮬레이션과 실제 환경의 차이를 확률적으로 표현하고, 도메인 랜덤화(Domain Randomization)를 통해 다양한 오차를 학습하면 실제 환경에서도 높은 신뢰성을 유지할 수 있다.

클라우드-엣지 협업(Cloud-Edge Collaboration)에서는 클라우드가 베이지안 추론, 앙상블 추론, 세계 모델, 대규모 시뮬레이션을 수행하고, 엣지는 실시간 신뢰도 계산, 센서 검증, 안전 제어를 담당한다. 이를 통해 계산 효율성과 실시간성을 동시에 확보할 수 있다.

안전(Safety)은 보정된 신뢰도에 크게 의존한다. 신뢰도가 높은 행동은 즉시 실행하고, 중간 수준은 추가 확인을 수행하며, 매우 낮은 경우에는 사람의 개입(Human Intervention), 비상 정지(Emergency Stop), 새로운 계획(Replanning)을 수행한다. 불확실성이 실제 안전 정책(Safety Policy)의 핵심 입력이 되는 것이다.

사람과 로봇의 협업(Human-Robot Collaboration)에서도 불확실성 표현은 매우 중요하다. 로봇은 "확신이 없습니다.", "추가 확인이 필요합니다.", "다시 설명해 주시겠습니까?"와 같이 자신의 신뢰도를 표현할 수 있어야 하며, 이러한 투명성(Transparency)은 사용자 신뢰(User Trust)를 크게 향상시킨다.

산업 제조에서는 조립, 검사, 용접, 물류, 협업 로봇 모두 높은 신뢰도의 행동 생성이 요구된다. 의료에서는 환자 안전이 최우선이므로 더욱 보수적인 신뢰도 평가가 필요하며, 자율주행, 농업, 건설, 시설 점검 등에서도 불확실성을 고려한 행동 생성이 필수 요소가 되고 있다.

액션 모델의 평가는 단순한 작업 성공률이 아니라 보정 품질(Calibration Quality), 신뢰도 정확성(Confidence Reliability), 위험 기반 의사결정(Risk-Aware Decision Making), 실패 예측(Failure Prediction), 일반화(Generalization), 사용자 신뢰(User Trust), 안전성(Safety Compliance), 장기 안정성(Long-Horizon Reliability) 등을 종합적으로 평가한다.

향후에는 멀티모달 파운데이션 모델(Multimodal Foundation Model), Flow Matching, 확산 정책(Diffusion Policy), ACT(Action Chunking Transformer), 세계 모델(World Model), 검색 증강 생성(RAG), 지속적 학습(Continual Learning), 자기 성찰(Self-Reflection), 적응형 메모리(Adaptive Memory), 강화학습(Reinforcement Learning)이 통합되어 행동 생성과 동시에 신뢰도 평가, 불확실성 설명, 외부 지식 검색, 안전 계획까지 수행하는 차세대 액션 모델(Action Model)로 발전할 것으로 예상된다.

결국 액션 모델 보정(Action Model Calibration)과 불확실성 추정(Uncertainty Estimation)은 단순한 안전 기능이 아니라 현대 물리 AI(Physical AI)의 핵심 구성 요소이다. 비전(Vision), 언어(Language), 메모리(Memory), 세계 모델(World Model), 계층형 계획(Hierarchical Planning), 행동 생성(Action Generation), 안전 검증(Safety Verification)을 하나의 통합 구조로 연결하여 **"무엇을 해야 하는가(What to Do)"**뿐 아니라 **"그 판단을 얼마나 신뢰할 수 있는가(How Much to Trust)"**까지 스스로 판단할 수 있도록 만든다. 이러한 능력은 제조, 물류, 의료, 농업, 시설 점검, 서비스 로봇 등 다양한 분야에서 신뢰 가능한 범용 물리 AI(Trustworthy Physical AI)를 구현하는 핵심 기반 기술이 될 것이다.

## 4.10 Evaluation Metrics and Benchmark Protocols (with Code)

![](images/image10.png){width="7.268055555555556in" height="7.268055555555556in"}

액션 모델 평가(Action Model Evaluation)는 현대 물리 AI(Physical AI)에서 가장 중요한 연구 분야 중 하나이다. 지능형 로봇의 성능은 단순히 하나의 정확도(Accuracy)만으로 평가할 수 없기 때문이다. 기존 머신러닝은 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score, 회귀 오차(Regression Error) 등을 사용하지만, 이러한 지표는 인식(Perception)이나 분류(Classification)와 같은 개별 문제에 적합할 뿐이다. 비전-언어-행동(VLA, Vision-Language-Action) 시스템은 인식, 추론, 계획, 제어, 실제 물리 행동이 모두 연결되어 있으므로 훨씬 종합적인 평가 체계가 필요하다.

현실에서는 동일한 작업 성공률(Task Success Rate)을 가진 두 로봇이라도 성능은 크게 다를 수 있다. 한 로봇은 부드럽고 안정적으로 작업을 수행하는 반면, 다른 로봇은 반복적인 보정 동작과 높은 에너지 소비를 동반할 수 있다. 따라서 최종 성공 여부만으로는 실제 성능을 설명할 수 없으며, 효율성(Efficiency), 안정성(Stability), 안전성(Safety), 일반화(Generalization) 등을 함께 평가해야 한다.

인간의 지능도 하나의 점수로 평가하지 않는다. 어떤 사람은 매우 정확하지만 느릴 수 있고, 다른 사람은 빠르지만 실수가 많을 수 있다. 로봇 역시 단순한 성공률뿐 아니라 추론 품질(Reasoning Quality), 실행 품질(Execution Quality), 적응성(Adaptability), 설명 가능성(Explainability), 협업 능력(Collaboration)을 함께 평가해야 진정한 성능을 판단할 수 있다.

기존 로봇 시스템은 모듈별(Module-wise) 평가를 수행하였다. 컴퓨터 비전은 물체 인식 정확도를 평가하고, 위치 추정(Localization)은 위치 오차를 측정하며, 제어기는 궤적 추종 오차(Trajectory Tracking Error)를 평가하였다. 그러나 현대의 VLA 모델은 모든 기능이 하나의 통합 모델 안에서 함께 학습되므로, 개별 모듈이 아니라 전체 시스템(End-to-End System)을 평가해야 한다.

작업 성공률(Task Success Rate)은 가장 기본적인 평가 지표이다. 목표 위치까지 이동하거나, 물체를 집거나, 검사 작업을 완료하는 등 미션을 성공적으로 수행한 비율을 의미한다. 하지만 성공률만으로는 실패 원인이나 수행 품질을 설명할 수 없기 때문에 다른 지표들과 함께 사용된다.

세부 작업 성공률(Subtask Completion Rate)은 복잡한 작업을 여러 단계로 나누어 평가한다. 예를 들어 자연어 이해, 물체 인식, 위치 추정, 이동, 조작, 보고와 같은 단계별 성공률을 측정하면 어느 단계에서 문제가 발생하는지를 쉽게 분석할 수 있다.

명령 수행 정확도(Instruction Following Accuracy)는 자연어 명령을 얼마나 정확하게 이해했는지를 평가한다. "파란 상자를 가져오고 빨간 상자는 건드리지 마라."와 같은 복합 명령에서 공간 관계(Spatial Relation), 시간 순서(Temporal Order), 안전 조건(Safety Constraint)까지 정확하게 이해했는지를 측정한다.

의미 접지 정확도(Semantic Grounding Accuracy)는 언어와 시각을 얼마나 정확하게 연결했는지를 평가한다. 예를 들어 "왼쪽의 작은 파란 박스"라는 표현을 실제 영상 속의 정확한 물체와 연결할 수 있는지를 측정한다. 이는 단순한 객체 인식이 아니라 언어와 영상의 의미 연결 능력을 평가하는 중요한 지표이다.

시각 인식 품질(Visual Perception Quality)은 물체 인식(Object Detection), 의미 분할(Semantic Segmentation), 자세 추정(Pose Estimation), 깊이 추정(Depth Estimation), 장면 이해(Scene Understanding)의 정확도를 포함한다. 최근에는 단순 정확도뿐 아니라 예측의 신뢰도(Confidence)까지 함께 평가한다.

궤적 품질(Trajectory Quality)은 실제 움직임의 품질을 평가한다. 경로 길이(Path Length), 이동 효율(Path Efficiency), 가속도 부드러움(Acceleration Smoothness), Jerk 최소화(Jerk Minimization), 자세 안정성(Orientation Stability), 최종 위치 오차(Endpoint Precision) 등을 함께 측정한다. 같은 작업을 완료하더라도 움직임이 부드러운 정책이 더 높은 평가를 받는다.

시간적 일관성(Temporal Consistency)은 ACT(Action Chunking Transformer), 확산 정책(Diffusion Policy) 등 장기 행동 생성(Long-Horizon Action Generation)에서 매우 중요한 평가 항목이다. 행동 간의 연속성, 궤적의 부드러움, 장기간 안정성을 평가하여 자연스러운 움직임을 생성하는지를 확인한다.

조작 성능(Manipulation Performance)은 파지 성공률(Grasp Success Rate), 삽입 정확도(Insertion Accuracy), 배치 오차(Placement Error), 힘 안정성(Force Stability), 물체 손상률(Object Damage Rate)을 평가한다. 특히 산업용 조립과 협동 로봇에서는 매우 중요한 성능 지표이다.

자율주행 성능(Navigation Performance)은 위치 정확도(Localization Accuracy), 경로 효율(Path Efficiency), 장애물 회피(Obstacle Avoidance), 도킹 정확도(Docking Precision), 다양한 지형(Terrain Adaptability)에 대한 적응 능력을 평가한다. 실외 로봇은 GPS 오차, 날씨 변화, 다양한 노면 환경까지 고려해야 한다.

에너지 효율(Energy Efficiency)은 모바일 로봇에서 점점 중요한 지표가 되고 있다. 동일한 작업을 수행하더라도 배터리 소비(Battery Consumption), 모터 에너지 사용량, 계산 자원 사용량, 발열(Thermal Efficiency)이 적을수록 더 우수한 정책으로 평가된다.

계산 효율(Computational Efficiency)도 매우 중요하다. VLA 모델은 비전 트랜스포머(Vision Transformer), 대규모 언어 모델(LLM), 세계 모델(World Model), RAG(Retrieval-Augmented Generation), 확산 정책(Diffusion Policy) 등을 포함하므로 추론 시간(Inference Latency), GPU 사용량, 메모리 사용량(Memory Consumption), 파라미터 수(Parameter Count)를 함께 평가해야 한다.

실시간 성능(Real-Time Performance)은 실제 로봇에서 반드시 필요한 요소이다. 아무리 정확한 행동이라도 추론 시간이 너무 길면 사용할 수 없다. 따라서 센서 입력부터 행동 생성, 안전 검증, 모터 제어까지 전체 응답 시간(End-to-End Latency)을 측정한다.

일반화 능력(Generalization Capability)은 학습하지 않은 환경이나 새로운 물체에서도 동일한 성능을 유지하는지를 평가한다. 제로샷(Zero-Shot), 퓨샷(Few-Shot), 전이학습(Transfer Learning) 성능이 중요한 평가 항목이 된다.

개방형 환경 강인성(Open-World Robustness)은 센서 고장, 환경 변화, 새로운 장애물, 통신 오류, 조명 변화, 날씨 변화와 같은 예상하지 못한 상황에서도 성능이 얼마나 유지되는지를 평가한다. 실제 산업 현장에서는 이러한 능력이 매우 중요하다.

구현체 전이(Embodiment Transfer)는 하나의 정책을 다른 로봇에서도 사용할 수 있는지를 평가한다. 매니퓰레이터, AMR, 사족보행 로봇(Quadruped), 휴머노이드(Humanoid) 등 다양한 플랫폼으로 얼마나 쉽게 이전될 수 있는지가 중요한 평가 기준이 되고 있다.

멀티모달 일관성(Multi-Modal Consistency)은 비전(Vision), 언어(Language), 촉각(Tactile), 힘 센서(Force), 내부 상태(Proprioception), 메모리(Memory)가 서로 모순 없이 동일한 결론을 도출하는지를 평가한다. 서로 다른 모달리티의 정보가 일치할수록 더욱 안정적인 행동 생성이 가능하다.

보정 품질(Calibration Quality)은 신뢰도(Confidence)가 실제 성공률과 얼마나 일치하는지를 평가한다. ECE(Expected Calibration Error), Brier Score, Negative Log Likelihood, Reliability Diagram 등이 사용되며, 잘 보정된 모델일수록 신뢰 가능한 자율 행동을 수행할 수 있다.

불확실성 추정(Uncertainty Estimation)은 모델이 자신의 한계를 얼마나 잘 인식하는지를 평가한다. 알레아토릭 불확실성(Aleatoric Uncertainty)과 에피스테믹 불확실성(Epistemic Uncertainty)을 구분하고, 위험 상황에서 적절한 대응을 수행하는지를 확인한다.

안전성(Safety)은 모든 평가 항목 중 가장 중요한 요소이다. 충돌 횟수(Collision Frequency), 최소 안전 거리(Minimum Clearance), 힘 제한(Force Limit), 비상 정지(Emergency Stop), 작업 공간 제한(Workspace Constraint), 안전 규정 준수(Regulatory Compliance)를 종합적으로 평가한다. 생산성보다 안전성이 항상 우선되어야 한다.

사람-로봇 상호작용(Human-Robot Interaction)은 협업 능력을 평가한다. 사람의 의도를 정확히 이해하고, 자신의 판단을 설명하며, 불확실성을 표현하고, 자연스럽게 대화하는 능력이 포함된다. 사용자 만족도(User Satisfaction)와 신뢰(User Trust)도 중요한 평가 요소이다.

설명 가능성(Explainability)은 산업 인증과 유지보수에서 매우 중요하다. 로봇이 왜 그런 행동을 선택했는지, 어떤 정보를 사용했는지, 어떤 대안을 고려했는지를 설명할 수 있어야 디버깅(Debugging)과 인증(Certification)이 쉬워진다.

세계 모델(World Model)은 미래 예측 능력을 평가한다. 사람의 이동, 물체의 움직임, 배터리 상태, 날씨 변화 등을 얼마나 정확하게 예측하는지가 장기 계획(Long-Horizon Planning)의 성능을 결정한다.

검색 증강 생성(RAG)은 검색 정확도(Retrieval Precision), 검색 재현율(Retrieval Recall), 문서 관련성(Relevance), 검색 지연(Retrieval Latency), 검색이 작업 성공률에 얼마나 기여했는지를 평가한다. 모델 내부 지식과 외부 지식을 구분하여 평가하는 것이 특징이다.

메모리(Memory)도 별도의 평가 대상이다. 에피소드 메모리(Episodic Memory)는 이전 경험을 얼마나 잘 재사용하는지, 의미 메모리(Semantic Memory)는 사실 정보를 얼마나 정확하게 유지하는지, 절차 메모리(Procedural Memory)는 작업 기술(Skill)을 얼마나 효율적으로 재사용하는지를 평가한다.

계층형 액션 모델(Hierarchical Action Model)은 계층별 평가를 수행한다. 상위 정책(High-Level Policy)은 작업 계획과 추론 능력을 평가하고, 중간 정책(Mid-Level Policy)은 기술 선택과 경로 계획을 평가하며, 하위 정책(Low-Level Policy)은 제어 정확도(Control Accuracy)와 안정성을 평가한다.

시뮬레이션 기반 평가(Simulation-Based Evaluation)는 실제 환경에서 발생하기 어려운 다양한 실패 상황을 반복적으로 생성한다. 디지털 트윈(Digital Twin)은 센서 오류, 장비 고장, 환경 변화, 통신 장애 등을 생성하여 정책의 강인성(Robustness)을 검증한다.

Sim-to-Real 평가는 시뮬레이션에서 학습한 정책이 실제 환경에서 얼마나 성능을 유지하는지를 측정한다. 도메인 랜덤화(Domain Randomization), 온라인 적응(Online Adaptation), 미세조정(Fine-Tuning)을 통해 성능 저하를 최소화하는 것이 목표이다.

클라우드-엣지 협업(Cloud-Edge Collaboration)에서는 클라우드 추론 성능, 엣지 지연 시간, 통신 안정성, 동기화(Synchronization), 장애 복구(Fault Tolerance), 네트워크 단절 시 성능 유지 등을 함께 평가한다.

평가에는 벤치마크 데이터셋(Benchmark Dataset)뿐 아니라 실제 로봇 실행도 함께 포함되어야 한다. 최근에는 멀티모달 데이터셋, 실제 환경, 시뮬레이션, 장기 작업(Long-Horizon Task), 인간 협업(Human Collaboration)을 모두 포함하는 통합 벤치마크가 사용되고 있다.

통계적 평가(Statistical Evaluation)도 중요하다. 평균 성능뿐 아니라 표준편차(Standard Deviation), 신뢰구간(Confidence Interval), 실패 사례(Failure Mode), 최악의 경우(Worst Case)를 함께 분석해야 실제 시스템의 신뢰성을 정확하게 평가할 수 있다.

재현성(Reproducibility)은 과학적 연구에서 필수적이다. 하드웨어, 소프트웨어, 데이터셋, 난수(Random Seed), 학습 방법, 평가 환경을 모두 공개하여 동일한 결과를 다른 연구자가 재현할 수 있어야 한다.

산업 인증(Industrial Certification)은 기능 안전(Function Safety), 사이버보안(Cybersecurity), 품질 관리(Quality Management), 검증(Verification), 검증(Validation), 추적성(Traceability)을 포함하는 표준화된 평가 절차를 요구한다. 단순한 연구 성능만으로는 실제 산업 적용이 어렵다.

향후에는 자기 평가(Self-Evaluation), 지속적 벤치마킹(Continual Benchmarking), 온라인 성능 모니터링(Online Monitoring), 평생학습(Lifelong Learning), 인간 신뢰도(Human Trust), 협업 지능(Collaborative Intelligence), 파운데이션 모델(Foundation Model) 평가가 하나의 통합 프로토콜(Evaluation Protocol)로 발전할 것으로 예상된다.

결국 액션 모델 평가(Action Model Evaluation)는 단순한 정확도 측정을 넘어 의미 이해(Semantic Understanding), 멀티모달 추론(Multi-Modal Reasoning), 물리 행동(Physical Execution), 시간적 일관성(Temporal Consistency), 불확실성 추정(Uncertainty Estimation), 안전성(Safety), 설명 가능성(Explainability), 계산 효율(Computational Efficiency), 일반화(Generalization), 구현체 전이(Embodiment Transfer), 장기 자율성(Long-Term Autonomy)까지 종합적으로 평가하는 체계로 발전하고 있다. 이러한 다차원 평가 프로토콜은 제조, 물류, 의료, 농업, 시설 점검, 서비스 로봇 등 다양한 물리 AI(Physical AI) 시스템이 실제 환경에서 안전하고 신뢰성 있게 운용되기 위한 핵심 기반이 될 것이다.
