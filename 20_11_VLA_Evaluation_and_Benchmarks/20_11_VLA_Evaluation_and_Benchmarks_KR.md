**Volume 20. Vision Language Action (VLA) Models**

# Chapter 11. VLA Evaluation and Benchmarks

## 11.1 Evaluation Framework

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

비전-언어-행동(Vision-Language-Action, VLA) 시스템은 시각(Vision), 언어(Language), 행동(Action)을 하나의 통합 모델에서 학습하는 차세대 로봇 지능 구조이다. 기존 로봇은 인식, 계획, 제어를 각각 평가했지만, VLA는 전체 의사결정 과정을 하나의 평가 체계(Evaluation Framework)에서 검증해야 한다. 따라서 단순히 작업(Task)을 성공했는지뿐 아니라, 명령 이해, 환경 적응, 일반화(Generalization), 예기치 못한 상황 대응, 실제 환경에서의 안정성까지 함께 평가해야 한다. 이러한 평가는 향후 모든 VLA 벤치마크(Benchmark)의 기반이 되는 핵심 방법론이다.

기존 로봇 평가는 위치 오차(Localization Error), 물체 파지 성공률(Grasp Success), 경로 추종 정확도(Trajectory Tracking Accuracy)와 같은 개별 성능 지표에 집중하였다. 그러나 이러한 방식은 VLA의 통합 지능을 충분히 설명하지 못한다. 로봇이 물체를 정확하게 집을 수 있어도 사용자의 명령을 잘못 이해하면 실패이며, 반대로 명령은 올바르게 이해했지만 행동 생성이 부정확해도 원하는 결과를 얻을 수 없다. 따라서 VLA 평가는 인식(Perception), 언어 이해(Language Understanding), 의미적 추론(Semantic Reasoning), 행동 생성(Action Generation), 실행 안정성(Execution Stability), 전체 임무 성공(Mission Success)을 종합적으로 분석해야 한다.

작업 성공(Task Success)은 가장 기본적인 평가 항목이지만 단순한 성공 또는 실패의 이분법으로 판단해서는 안 된다. 로봇은 목표를 달성하는 동시에 안전성(Safety), 효율성(Efficiency), 품질(Quality), 시간(Time)까지 만족해야 한다. 예를 들어 깨지기 쉬운 물체를 상자에 넣는 작업이라면 단순히 물체를 옮겼는지가 아니라, 과도한 힘을 사용하지 않았는지, 충돌이 없었는지, 물체의 방향이 적절한지, 지정된 시간 안에 수행했는지 등을 함께 평가해야 한다. 이러한 다차원적인 성공 기준이 실제 로봇의 지능 수준을 보다 정확하게 반영한다.

일반화(Generalization)는 VLA를 기존의 작업 전용(Task-Specific) 로봇과 구분하는 가장 중요한 특성이다. 우수한 VLA 모델은 학습하지 않은 새로운 작업에서도 기존 지식을 활용하여 문제를 해결해야 한다. 이를 평가하기 위해 새로운 물체(Object), 새로운 환경(Environment), 새로운 명령(Instruction), 새로운 작업(Task Composition)을 제공하고 성능을 측정한다. 진정한 일반화 능력을 가진 모델은 익숙하지 않은 상황에서도 급격한 성능 저하 없이 안정적으로 동작하며, 단순 암기가 아닌 개념적 이해를 바탕으로 행동을 생성한다.

작업 간 일반화(Cross-Task Generalization)는 하나의 작업에서 학습한 기술(Skill)을 다른 작업에 얼마나 효과적으로 재사용하는지를 평가한다. 예를 들어 창고에서 습득한 물체 집기 기술이 가정 환경의 정리 작업에도 적용되고, 사무실에서 학습한 자율주행 기술이 공장 환경에서도 활용될 수 있어야 한다. 평가 과정에서는 다양한 작업을 조합하여 새로운 문제를 제시하며, 로봇이 기존 기술을 적절히 결합하여 새로운 행동을 만들어내는지를 분석한다. 이는 모델이 단순한 행동 시퀀스를 외운 것이 아니라 물리적 개념과 작업 원리를 이해했는지를 확인하는 중요한 기준이다.

명령 일반화(Instruction Generalization)는 자연어(Natural Language)에 대한 이해 능력을 평가한다. 실제 사용자는 동일한 명령을 항상 같은 방식으로 말하지 않는다. 따라서 평가에서는 다양한 문장 구조, 동의어, 우회적인 표현, 다단계 명령(Multi-Step Instruction), 문맥(Context)이 포함된 명령을 제공한다. 우수한 VLA 모델은 표현 방식이 달라도 동일한 의도를 정확하게 파악해야 하며, 미리 정의된 명령어만 인식하는 것이 아니라 언어의 의미 자체를 이해해야 한다. 이러한 능력은 사람과 로봇의 자연스러운 상호작용(Human-Robot Interaction)을 가능하게 한다.

환경 일반화(Environment Generalization)는 실제 환경 변화에 대한 적응 능력을 평가한다. 조명(Lighting), 배경(Background), 물체 배치(Object Arrangement), 센서 노이즈(Sensor Noise), 날씨(Weather), 동적 장애물(Dynamic Obstacle) 등이 변화해도 로봇은 안정적으로 동작해야 한다. 이를 위해 평가에서는 동일한 작업을 다양한 환경 조건에서 반복 수행하며 성능 변화를 측정한다. 환경 변화에도 높은 성능을 유지하는 모델은 단순한 시각적 특징이 아니라 환경의 본질적인 의미를 이해하고 있음을 의미한다.

구현체 일반화(Embodiment Generalization)는 서로 다른 로봇 플랫폼 사이에서 학습 결과를 얼마나 효과적으로 이전할 수 있는지를 평가한다. 하나의 로봇팔에서 학습한 조작 기술이 다른 제조사의 로봇에도 적용될 수 있어야 하며, 이동 로봇(AMR)의 자율주행 정책이 휴머노이드(Humanoid)나 다족 로봇(Quadruped)에도 확장될 수 있어야 한다. 이러한 평가는 로봇의 하드웨어 차이를 넘어 추상적인 작업 개념을 학습했는지를 판단하는 중요한 요소이며, 범용 로봇 정책(Generalist Robot Policy)의 핵심 성능 지표가 된다.

강건성(Robustness) 평가는 예기치 못한 환경 변화와 외란(Disturbance) 속에서도 안정적인 성능을 유지하는 능력을 측정한다. 실제 환경에서는 센서 오류, 부분 가림(Occlusion), 통신 지연, 물체 이동, 제어 오차, 기계적 진동 등이 지속적으로 발생한다. 평가에서는 이러한 조건을 의도적으로 추가하여 성능 저하 정도를 분석한다. 우수한 모델은 작은 외란에도 쉽게 실패하지 않으며, 문제가 발생하면 스스로 복구(Recovery)하거나 재계획(Replanning)을 수행하여 작업을 계속 진행한다.

인지 강건성(Perception Robustness)은 조명 변화, 그림자, 반사광, 모션 블러(Motion Blur), 카메라 노이즈(Camera Noise), 복잡한 배경(Clutter), 부분 가림 등을 포함한 다양한 시각적 변화에 대한 내성을 평가한다. 언어 강건성(Language Robustness)은 음성 인식 오류, 문법 오류, 불완전한 명령, 다국어 표현, 모호한 표현 등에 대한 대응 능력을 측정한다. 실행 강건성(Action Robustness)은 물체가 움직이거나 바퀴가 미끄러지는 상황에서도 행동을 수정하고 안정적으로 작업을 이어가는 능력을 평가한다. 이러한 세 가지 강건성은 실제 산업 현장에서 매우 중요한 평가 요소이다.

평가는 시뮬레이션(Simulation)과 실제 환경(Real World)을 함께 활용해야 한다. 시뮬레이션은 수천 개의 실험을 빠르게 반복할 수 있고 다양한 조건을 쉽게 생성할 수 있다는 장점이 있다. 반면 실제 환경에서는 센서 오차, 기계적 편차, 마찰, 충돌, 사람과의 상호작용 등 현실적인 요소를 검증할 수 있다. 따라서 가장 이상적인 평가 체계는 대규모 시뮬레이션을 통해 알고리즘을 검증한 후 실제 로봇 실험으로 최종 성능을 확인하는 하이브리드(Hybrid) 방식이다.

평가의 신뢰성을 확보하기 위해서는 통계적 분석(Statistical Analysis)이 반드시 필요하다. 한두 번의 성공 사례만으로 모델의 성능을 판단해서는 안 되며, 다양한 환경과 여러 개의 무작위 초기 조건(Random Seed)에서 반복 실험을 수행해야 한다. 평균 성공률(Average Success Rate)뿐 아니라 분산(Variance), 신뢰구간(Confidence Interval), 강건성 곡선(Robustness Curve) 등을 함께 분석해야 공정한 모델 비교가 가능하다. 이러한 통계적 접근은 모델 간의 실제 성능 차이를 객관적으로 보여준다.

안전성(Safety)은 모든 평가 항목과 함께 고려되어야 하는 핵심 요소이다. 충돌(Collision), 과도한 힘(Excessive Force), 작업 공간 이탈(Workspace Violation), 사람과의 위험한 접근(Human Proximity), 비상 정지(Emergency Stop), 정책 우회(Policy Override) 등을 지속적으로 기록하여 평가해야 한다. 아무리 높은 작업 성공률을 달성하더라도 안전 기준을 만족하지 못하는 모델은 실제 산업 현장에 적용하기 어렵다. 따라서 현대 VLA 평가 체계는 성능과 안전을 동시에 검증하는 방향으로 발전하고 있다.

궁극적으로 VLA 평가 프레임워크(VLA Evaluation Framework)의 목적은 단순히 벤치마크 점수(Benchmark Score)를 비교하는 것이 아니다. 다양한 환경과 작업, 서로 다른 로봇 플랫폼에서 신뢰성 있게 동작하는 범용 물리 AI(Physical AI)를 개발하기 위한 기준을 제공하는 데 있다. 작업 성공(Task Success), 언어 이해(Language Understanding), 인식 품질(Perception Quality), 행동 생성(Action Generation), 일반화(Generalization), 강건성(Robustness), 안전성(Safety), 실제 적용성(Real-World Adaptability)을 종합적으로 평가함으로써 연구자와 개발자는 모델의 실제 능력을 객관적으로 분석할 수 있다. 이러한 통합 평가 체계는 차세대 VLA 모델의 발전을 촉진하고, 산업 자동화, 물류, 의료, 서비스 로봇, 가정용 로봇 등 다양한 분야에서 신뢰할 수 있는 범용 로봇 지능을 구현하는 핵심 기반이 될 것이다.

## 11.2 LIBERO Benchmark (with Code)

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

LIBERO 벤치마크(LIBERO Benchmark)는 현대 로봇 학습(Robot Learning), 특히 비전-언어-행동(Vision-Language-Action, VLA) 모델과 모방학습(Imitation Learning), 로봇 파운데이션 모델(Robot Foundation Model)을 평가하기 위한 대표적인 벤치마크이다. 기존 로봇 평가는 개별 작업(Task)의 성능을 측정하는 데 집중했지만, LIBERO는 로봇이 시간이 지나면서 지속적으로 새로운 기술(Skill)을 습득하고 기존 지식을 유지하는 능력을 평가하는 것을 목표로 한다. 따라서 VLA 평가 체계에서 LIBERO는 평생학습(Lifelong Learning) 기반의 범용 로봇 지능을 측정하는 핵심 기준으로 활용된다.

LIBERO는 \'평생 로봇 학습 벤치마크(Lifelong Benchmark for Robot Learning)\'를 의미한다. 가장 큰 특징은 모든 데이터를 한 번에 학습하는 기존 방식과 달리 새로운 작업이 순차적으로 주어지는 연속 학습(Continual Learning) 환경을 제공한다는 점이다. 실제 서비스 로봇이나 산업용 로봇은 운영 중에도 새로운 작업과 새로운 환경을 계속 경험하게 된다. 따라서 LIBERO는 새로운 작업을 얼마나 잘 학습하는지뿐 아니라, 이전에 학습한 작업을 계속 유지할 수 있는지도 함께 평가한다.

연속 학습(Continual Learning)은 미래 로봇에서 반드시 필요한 능력이다. 기존의 지도학습(Supervised Learning)은 고정된 데이터셋으로 학습한 후 모델을 배포하지만, 실제 환경은 지속적으로 변화한다. 새로운 도구, 새로운 작업 공간, 새로운 고객 요구사항이 계속 발생하기 때문에 로봇은 기존 모델을 모두 다시 학습하지 않고도 새로운 지식을 추가할 수 있어야 한다. LIBERO는 이러한 지속적인 학습 능력을 정량적으로 평가하여 장기간 운용 가능한 로봇 지능 개발을 지원한다.

LIBERO가 중점적으로 해결하려는 문제 가운데 하나는 치명적 망각(Catastrophic Forgetting)이다. 신경망(Neural Network)은 새로운 작업을 학습하면 기존에 학습한 내용을 잊어버리는 현상이 자주 발생한다. 예를 들어 주방 정리 작업을 잘 수행하던 로봇이 물류 작업을 학습한 후 이전 능력을 잃을 수도 있다. LIBERO는 새로운 작업을 학습하는 과정에서도 기존 작업의 성능을 반복적으로 측정하여 지식 유지(Knowledge Retention) 정도를 평가하며, 재생학습(Replay), 정규화(Regularization), 어댑터(Adapter) 기반 학습 등의 다양한 방법을 비교할 수 있도록 지원한다.

LIBERO는 난이도가 점진적으로 증가하는 다양한 작업 집합(Task Suite)을 제공한다. 작업은 물체 집기(Grasping), 서랍 열기(Drawer Opening), 버튼 누르기(Button Pressing), 물체 배치(Object Placement), 용기(Container) 조작, 다단계 조립(Assembly) 등 일상적인 조작 작업으로 구성된다. 각각의 작업은 비교적 단순해 보이지만, 다양한 물체와 환경, 공간 관계(Spatial Relation), 언어 명령을 조합하여 단순한 제어기가 아니라 범용 조작 정책(Generalized Manipulation Policy)을 학습하도록 설계되어 있다.

작업 다양성(Task Diversity)은 LIBERO의 핵심 장점이다. 일부 작업은 시각 인식(Visual Recognition)을 중심으로 평가하고, 다른 작업은 언어 이해(Language Understanding), 공간 추론(Spatial Reasoning), 장기 계획(Long-Horizon Planning), 조작 제어(Manipulation Control)를 중점적으로 평가한다. 이처럼 서로 다른 능력을 요구하는 작업을 함께 제공함으로써 특정 작업에 최적화된 모델이 아니라 다양한 상황에서 활용 가능한 범용 로봇 지능을 평가할 수 있다.

최근 VLA 모델이 발전하면서 자연어 기반 평가(Language Conditioning)의 중요성도 크게 증가하였다. LIBERO에서는 로봇이 미리 정의된 명령이 아니라 자연어(Natural Language) 명령을 입력받아 이를 실제 행동으로 변환해야 한다. 명령에는 물체의 종류, 색상, 위치, 공간 관계, 작업 순서 등이 포함될 수 있으며, 로봇은 시각 정보와 관절 정보(Proprioception)를 함께 활용하여 올바른 행동을 생성해야 한다. 이는 언어와 행동을 연결하는 의미적 접지(Semantic Grounding) 능력을 평가하는 중요한 요소이다.

일반화(Generalization) 평가는 LIBERO의 가장 중요한 특징 가운데 하나이다. 평가 과정에서는 학습 과정과 다른 물체 배치, 새로운 환경, 새로운 조명, 새로운 언어 표현 등을 제공한다. 우수한 모델은 새로운 상황에서도 추가 학습 없이 기존 지식을 활용하여 문제를 해결할 수 있어야 한다. 이러한 평가는 단순히 시범 데이터를 암기한 것이 아니라 작업의 본질적인 개념을 이해했는지를 확인하는 중요한 기준이 된다.

장기 작업(Long-Horizon Task)은 여러 단계의 행동을 연속적으로 수행해야 하는 작업을 의미한다. 예를 들어 서랍을 열고 물체를 꺼낸 후 다른 위치로 이동하여 용기에 넣고 다시 서랍을 닫는 작업은 하나의 행동으로 해결할 수 없다. 로봇은 이전 행동의 결과를 기억하면서 다음 행동을 계획해야 하며, 전체 작업의 목표를 지속적으로 유지해야 한다. LIBERO는 이러한 장기 계획(Long-Horizon Planning) 능력을 평가하여 실제 산업 및 서비스 환경에서 필요한 복합 작업 수행 능력을 측정한다.

LIBERO는 조합 학습(Compositional Learning)도 중요하게 평가한다. 새로운 작업마다 모든 기술을 새롭게 학습하는 것이 아니라, 기존에 습득한 잡기(Grasp), 밀기(Push), 회전(Rotate), 배치(Place), 정렬(Align) 등의 기본 기술을 조합하여 새로운 문제를 해결해야 한다. 이러한 재사용 능력은 범용 로봇 정책(Generalist Robot Policy)의 핵심 요소이며, 학습 효율성과 확장성을 크게 향상시킨다.

LIBERO는 주로 시뮬레이션(Simulation) 환경에서 실행된다. 이를 통해 동일한 초기 조건과 동일한 물리 환경에서 수많은 실험을 반복 수행할 수 있으며, 전 세계 연구자들이 동일한 조건에서 알고리즘을 비교할 수 있다. 이러한 높은 재현성(Reproducibility)은 공정한 벤치마크(Benchmark)를 가능하게 하며, 다양한 연구 결과를 객관적으로 비교하는 기반을 제공한다.

비록 시뮬레이션 기반이지만 LIBERO의 작업은 실제 산업 환경과 매우 유사하게 구성되어 있다. 제조 공장, 물류 창고, 연구실, 서비스 로봇, 가정용 로봇에서 수행하는 조작 작업을 반영하고 있으며, 시뮬레이션-실환경 전이(Sim-to-Real Transfer), 도메인 랜덤화(Domain Randomization), 합성 데이터(Synthetic Data)와 결합하면 실제 로봇에서도 높은 활용 가능성을 보여준다.

LIBERO의 평가 지표(Evaluation Metric)는 단순한 성공률(Success Rate)에 그치지 않는다. 연속 학습 효율(Continual Learning Efficiency), 역전이(Backward Transfer), 순전이(Forward Transfer), 망각률(Forgetting Rate), 적응 속도(Adaptation Speed), 정책 안정성(Policy Stability), 샘플 효율성(Sample Efficiency), 계산 비용(Computational Cost) 등 다양한 지표를 함께 분석한다. 이를 통해 알고리즘의 장점과 한계를 다각도로 평가할 수 있으며, 단순한 성공률 경쟁이 아니라 균형 잡힌 성능 개선을 유도한다.

최근 로봇 파운데이션 모델(Robot Foundation Model)의 발전으로 LIBERO의 중요성은 더욱 커지고 있다. 인터넷 규모의 데이터셋과 멀티모달(Multimodal) 학습을 기반으로 하는 VLA 모델은 단순한 조작 성능보다 일반화와 지속적인 학습 능력이 더욱 중요하다. 따라서 많은 최신 연구에서는 RLBench, Language Table, DROID, SimplerEnv와 함께 LIBERO를 핵심 벤치마크로 사용하여 모델의 범용성과 적응 능력을 검증하고 있다.

LIBERO는 공정하고 재현 가능한 연구를 지원한다는 점에서도 큰 의미를 가진다. 표준화된 작업 정의(Standardized Task Definition), 동일한 평가 프로토콜(Evaluation Protocol), 공개된 시뮬레이션 환경을 제공함으로써 연구 기관마다 동일한 조건에서 실험을 수행할 수 있다. 이를 통해 모방학습, 강화학습(Reinforcement Learning), 트랜스포머(Transformer), 확산 정책(Diffusion Policy), 행동 청킹(Action Chunking), VLA 모델 등의 성능을 객관적으로 비교할 수 있다.

산업적인 측면에서도 LIBERO는 매우 중요한 의미를 가진다. 제조, 물류, 의료, 연구, 서비스 로봇은 장기간 운영되면서 새로운 작업을 지속적으로 학습해야 한다. LIBERO의 평가 결과는 로봇이 실제 환경에서 얼마나 안정적으로 새로운 작업을 습득하고 기존 지식을 유지할 수 있는지를 보여주는 중요한 지표가 된다. 따라서 기업은 LIBERO 결과를 활용하여 모델의 장기적인 확장성과 유지보수성을 객관적으로 평가할 수 있다.

궁극적으로 LIBERO 벤치마크(LIBERO Benchmark)는 로봇 학습을 개별 작업 최적화가 아니라 지속적으로 성장하는 지능 시스템으로 바라보는 새로운 평가 철학을 제시한다. 연속 학습(Continual Learning), 치명적 망각(Catastrophic Forgetting), 일반화(Generalization), 언어 접지(Language Grounding), 조합 학습(Compositional Learning), 장기 계획(Long-Horizon Planning), 지식 유지(Knowledge Retention)를 하나의 통합된 평가 체계에서 분석함으로써 현대 VLA 시스템과 로봇 파운데이션 모델의 실제 지능 수준을 종합적으로 평가할 수 있다. 앞으로 범용 물리 AI(Physical AI)가 발전할수록 LIBERO는 적응성과 신뢰성을 갖춘 차세대 로봇을 개발하기 위한 가장 중요한 평가 기준 가운데 하나로 자리 잡을 것이다.

## 11.3 RLBench Evaluation (with Code)

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

RLBench(Robot Learning Benchmark)는 로봇 조작(Robot Manipulation), 모방학습(Imitation Learning), 강화학습(Reinforcement Learning), 비전-언어-행동(Vision-Language-Action, VLA) 모델, 그리고 로봇 파운데이션 모델(Robot Foundation Model)을 평가하기 위한 대표적인 벤치마크(Benchmark)이다. 최근 로봇 기술이 단일 작업(Task) 중심에서 범용 물리 AI(Physical AI)로 발전하면서 단순한 조작 정확도뿐 아니라 언어 이해(Language Understanding), 장기 계획(Long-Horizon Planning), 일반화(Generalization), 환경 적응(Adaptation)까지 함께 평가할 필요성이 커졌다. RLBench는 이러한 요구를 만족하는 표준화된 시뮬레이션 기반 평가 환경을 제공한다.

RLBench는 CoppeliaSim 시뮬레이터(Simulator)를 기반으로 구축되었으며, 현실적인 물리 엔진(Physics Engine)을 이용하여 로봇과 물체 사이의 상호작용을 사실적으로 재현한다. 다양한 조작 작업이 동일한 시뮬레이션 환경에서 수행되므로 연구자들은 하드웨어 차이나 환경 차이에 영향을 받지 않고 동일한 조건에서 알고리즘을 비교할 수 있다. 이러한 높은 재현성(Reproducibility)은 RLBench가 전 세계적으로 널리 사용되는 이유 중 하나이다.

RLBench 평가 프로토콜(Evaluation Protocol)의 핵심 목적은 사람이 직접 프로그래밍한 제어 로직이 아니라 학습된 정책(Learned Policy)이 다양한 조작 작업을 얼마나 성공적으로 수행하는지를 평가하는 것이다. 기존 로봇은 물체 위치, 이동 경로, 예외 처리 등을 사람이 모두 설계해야 했지만, VLA 모델은 시범 데이터(Demonstration), 시각 정보(Vision), 언어(Language)를 이용하여 스스로 행동(Action)을 생성한다. RLBench는 이러한 학습 기반 로봇 제어의 실제 성능을 객관적으로 검증하는 역할을 수행한다.

RLBench의 가장 큰 특징 가운데 하나는 매우 다양한 작업(Task Diversity)을 제공한다는 점이다. 물체 집기(Grasping), 밀기(Pushing), 당기기(Pulling), 쌓기(Stacking), 삽입(Inserting), 문 열기(Open), 문 닫기(Close), 버튼 누르기(Button Pressing), 회전(Rotation), 분류(Sorting), 조립(Assembly), 물체 이동(Object Relocation) 등 실제 산업 및 서비스 환경에서 자주 발생하는 조작 작업이 포함되어 있다. 각각의 작업은 서로 다른 인식과 제어 능력을 요구하므로 로봇의 종합적인 지능을 평가할 수 있다.

각 작업은 단순히 물체를 이동시키는 수준이 아니라 현실적인 물리 상호작용(Physical Interaction)을 포함한다. 서랍(Drawer), 문(Door), 버튼(Button), 캐비닛(Cabinet), 용기(Container), 도구(Tool) 등 관절을 가진 물체를 조작해야 하며, 단순한 위치 제어만으로는 성공하기 어렵다. 로봇은 물체의 어포던스(Affordance)를 이해하고 접촉(Contact Dynamics)을 고려하면서 안정적으로 작업을 수행해야 한다. 이러한 특성은 실제 산업용 로봇의 작업과 매우 유사하다.

최근에는 자연어 기반(Language Conditioned) 평가도 RLBench에서 매우 중요한 요소가 되었다. 하나의 작업에는 여러 개의 자연어 명령(Natural Language Instruction)이 함께 제공되며, 로봇은 사람이 말한 명령을 이해하고 장면(Scene) 속에서 적절한 물체를 찾아 올바른 행동을 생성해야 한다. 따라서 RLBench는 단순한 조작 벤치마크를 넘어 VLA 시스템의 언어 이해(Language Grounding) 능력까지 평가하는 중요한 플랫폼으로 발전하고 있다.

시각 인식(Visual Perception)은 RLBench 평가의 핵심 요소이다. 로봇은 RGB 카메라(Camera), 깊이 카메라(Depth Camera), 분할 이미지(Segmentation Image), 관절 정보(Proprioception) 등을 입력받는다. 비전 인코더(Vision Encoder)는 물체를 인식하고 위치를 추정하며, 조작 대상을 선택해야 한다. 따라서 최종 성능은 조작 정책뿐 아니라 시각 특징 추출(Feature Extraction)과 센서 융합(Multi-Sensor Fusion)의 품질에도 크게 영향을 받는다.

RLBench는 시범학습(Demonstration Learning)을 적극적으로 활용한다. 대부분의 작업에는 전문가가 수행한 시범 데이터가 포함되어 있으며, 이를 이용하여 모방학습(Imitation Learning), 행동 복제(Behavior Cloning), 역강화학습(Inverse Reinforcement Learning), 확산 정책(Diffusion Policy), 트랜스포머 정책(Transformer Policy) 등을 학습할 수 있다. 동일한 시범 데이터를 사용하므로 다양한 알고리즘의 학습 효율성과 성능을 공정하게 비교할 수 있다.

일반화(Generalization) 평가는 RLBench의 핵심 기능 중 하나이다. 평가 시에는 물체의 위치, 방향, 초기 상태, 카메라 시점(Camera Viewpoint), 조명(Lighting), 환경 구성을 무작위(Randomization)로 변경한다. 우수한 모델은 이러한 변화에도 추가 학습 없이 안정적으로 작업을 수행해야 한다. 이는 단순히 시범 데이터를 암기한 모델과 실제 개념을 이해한 모델을 구별하는 중요한 기준이 된다.

장기 작업(Long-Horizon Task)은 여러 단계의 행동을 순차적으로 수행해야 하는 작업이다. 예를 들어 물체를 찾고, 집고, 장애물을 피하며 이동한 후, 문을 열고, 용기에 넣는 과정은 하나의 행동만으로 해결되지 않는다. 로봇은 전체 작업의 목표를 기억하면서 이전 행동의 결과를 다음 단계에 반영해야 한다. RLBench는 이러한 장기 계획(Long-Horizon Planning)과 순차적 추론(Sequential Reasoning) 능력을 평가하는 데 매우 적합하다.

RLBench는 최종 성공 여부뿐 아니라 움직임의 품질(Motion Quality)도 함께 평가한다. 부드러운 경로(Smooth Trajectory), 충돌 회피(Collision Avoidance), 안정적인 파지(Stable Grasp), 효율적인 이동(Path Efficiency), 자연스러운 조작(Motion Smoothness) 등이 모두 평가 대상이다. 같은 작업을 성공했더라도 움직임이 불안정하거나 위험하다면 더 낮은 평가를 받게 된다. 이러한 품질 평가는 실제 로봇 적용 시 매우 중요한 요소이다.

시뮬레이션의 높은 재현성(Reproducibility)은 RLBench의 가장 큰 장점이다. 동일한 초기 조건에서 반복 실험을 수행할 수 있으며, 다양한 환경 변수만 선택적으로 변경할 수 있다. 따라서 서로 다른 연구 기관에서도 동일한 실험을 수행하여 결과를 직접 비교할 수 있다. 이러한 표준화(Standardization)는 최신 트랜스포머 기반 정책, 확산 정책(Diffusion Policy), 행동 청킹(Action Chunking), VLA 모델 평가의 기준으로 활용되고 있다.

비록 RLBench는 시뮬레이션 기반이지만 실제 로봇 적용(Sim-to-Real Transfer)을 고려하여 설계되었다. 학습 과정에서는 질감(Texture), 조명(Lighting), 카메라 보정(Camera Calibration), 물리 파라미터(Physics Parameter), 물체 외형(Object Appearance) 등을 무작위로 변경하는 도메인 랜덤화(Domain Randomization)를 적용한다. 이를 통해 학습된 정책은 실제 로봇에서도 환경 변화에 강한 일반화 능력을 가질 수 있다.

RLBench의 평가 지표(Evaluation Metrics)는 단순 성공률(Success Rate)만 포함하지 않는다. 작업 완료 시간(Task Completion Time), 경로 효율(Path Efficiency), 궤적 부드러움(Trajectory Smoothness), 충돌 횟수(Collision Frequency), 조작 정확도(Manipulation Precision), 파지 안정성(Grasp Stability), 샘플 효율성(Sample Efficiency), 계산 비용(Computational Cost), 추론 지연(Inference Latency), 정책 일관성(Policy Consistency) 등을 함께 평가한다. 최근에는 언어 이해(Language Following Accuracy), 의미 접지(Semantic Grounding), 멀티모달 추론(Multimodal Reasoning)까지 포함하여 보다 종합적인 평가를 수행하고 있다.

최신 로봇 파운데이션 모델(Robot Foundation Model)은 RLBench와 함께 LIBERO, Language Table, DROID, SimplerEnv, CALVIN 등의 벤치마크를 함께 사용한다. 각각의 벤치마크는 서로 다른 능력을 평가한다. RLBench는 다양한 조작(Task Diversity), 시범학습(Demonstration Learning), 시뮬레이션 기반 일반화(Simulation Generalization), 멀티모달 정책(Multimodal Policy) 수행 능력을 평가하는 데 가장 강점을 가진다. 여러 벤치마크를 함께 활용하면 모델의 실제 범용성을 더욱 정확하게 분석할 수 있다.

RLBench는 다양한 학습 알고리즘을 공정하게 비교할 수 있도록 표준 프로토콜(Standard Protocol)을 제공한다. 강화학습(Reinforcement Learning), 모방학습(Imitation Learning), 오프라인 강화학습(Offline RL), 확산 기반 정책(Diffusion Policy), 트랜스포머(Transformer), 계층형 계획(Hierarchical Planning), VLA 파운데이션 모델 등을 동일한 환경에서 비교할 수 있다. 이를 통해 연구자들은 알고리즘 자체의 성능 차이를 객관적으로 분석할 수 있으며, 연구 개발 속도도 크게 향상된다.

산업적인 관점에서도 RLBench는 매우 중요한 의미를 가진다. 제조 자동화, 물류 창고, 제약 연구소, 전자 조립, 식품 처리, 서비스 로봇 등은 기존처럼 고정된 프로그램이 아니라 변화하는 환경에 적응할 수 있는 지능형 로봇을 요구하고 있다. RLBench의 평가 결과는 실제 로봇을 제작하기 전에 정책의 강건성(Robustness), 조작 능력(Manipulation Capability), 일반화 성능(Generalization)을 미리 검증할 수 있는 중요한 기준이 된다.

VLA 아키텍처(Vision-Language-Action Architecture)가 발전할수록 RLBench의 중요성은 더욱 커지고 있다. RLBench는 시각 인식(Perception), 언어 이해(Language Grounding), 조작 계획(Manipulation Planning), 순차 추론(Sequential Reasoning), 실제 행동(Action Execution)을 하나의 통합된 평가 체계에서 분석한다. 이러한 구조는 미래의 가정용 로봇, 산업용 협동로봇(Collaborative Robot), 의료 로봇, 연구 자동화 시스템 등 다양한 물리 AI(Physical AI)의 요구사항과 매우 잘 부합한다.

궁극적으로 RLBench VLA 평가 프로토콜(RLBench VLA Evaluation Protocol)은 단순한 조작 작업 모음이 아니라 현대 로봇 지능 전체를 평가하기 위한 과학적이고 표준화된 평가 체계이다. 현실적인 물리 시뮬레이션(Physical Simulation), 멀티모달 인식(Multimodal Perception), 자연어 이해(Natural Language Understanding), 시범학습(Demonstration Learning), 장기 작업(Long-Horizon Task), 일반화(Generalization), 움직임 품질(Motion Quality)을 하나의 프레임워크에서 종합적으로 분석함으로써 VLA 모델과 로봇 파운데이션 모델의 실제 성능을 객관적으로 평가할 수 있다. 앞으로 범용 물리 AI가 발전할수록 RLBench는 신뢰성 있고 적응력이 뛰어난 자율 로봇을 개발하기 위한 가장 중요한 국제 표준 벤치마크 가운데 하나로 계속 활용될 것이다.

## 11.4 SimplerEnv Benchmark (with Code)

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

SimplerEnv는 현대 비전-언어-행동(Vision-Language-Action, VLA) 시스템, 로봇 파운데이션 모델(Robot Foundation Model), 그리고 구현형 인공지능(Embodied AI) 정책을 평가하기 위해 개발된 대표적인 벤치마크(Benchmark)이다. 기존의 많은 로봇 학습 벤치마크는 시뮬레이션(Simulation)에 의존하지만, SimplerEnv는 실제 로봇 환경과의 차이를 줄이고 현실적인 성능을 평가하는 것을 목표로 한다. 따라서 시뮬레이션 성능뿐 아니라 실제 환경에서 얼마나 안정적으로 동작하는지를 검증하는 중요한 평가 체계로 활용된다.

SimplerEnv가 개발된 가장 큰 이유는 높은 시뮬레이션 성능이 반드시 실제 로봇 성능으로 이어지지 않기 때문이다. 시뮬레이션에서는 우수한 결과를 보이는 정책도 실제 환경에서는 카메라 보정(Camera Calibration), 조명(Lighting), 센서 노이즈(Sensor Noise), 물체 외형(Object Appearance), 마찰(Friction), 구동기 특성(Actuator Dynamics) 등의 차이로 인해 성능이 크게 저하될 수 있다. SimplerEnv는 이러한 시뮬레이션-실환경 전이(Sim-to-Real Transfer) 문제를 줄이기 위한 평가 환경을 제공한다.

SimplerEnv는 접지(Grounding) 중심의 평가 철학을 채택하고 있다. 여기서 접지(Grounding)는 로봇이 시각 정보(Vision), 언어(Language), 행동(Action), 그리고 환경 변화(Environmental Feedback)를 하나의 연속된 감각-운동 루프(Sensorimotor Loop) 안에서 이해하고 연결하는 능력을 의미한다. 단순히 물체를 인식하거나 행동을 생성하는 것이 아니라, 자신의 행동이 환경에 어떤 영향을 미치는지를 이해하고 다음 행동에 반영하는 것이 핵심이다. 따라서 SimplerEnv는 인식과 행동이 지속적으로 연결되는 폐루프 제어(Closed-Loop Control)를 평가한다.

기존의 복잡한 시뮬레이션 환경과 달리 SimplerEnv는 불필요한 복잡성을 줄이고 실제 물리 상호작용(Physical Interaction)에 집중한다. 수천 개의 물체를 포함한 복잡한 가상 환경 대신, 핵심적인 조작 작업을 중심으로 현실적인 환경을 구성하였다. 이를 통해 알고리즘의 실제 성능을 방해하는 불필요한 요소를 줄이고, 로봇의 인식 능력과 조작 능력을 보다 정확하게 평가할 수 있다. 이러한 단순하지만 현실적인 설계가 SimplerEnv의 가장 큰 특징 가운데 하나이다.

환경은 단순하지만 작업 다양성(Task Diversity)은 충분히 유지된다. 물체 집기(Grasping), 놓기(Placing), 밀기(Pushing), 당기기(Pulling), 문이나 서랍 열기(Open Drawer), 부품 삽입(Insertion), 물체 정렬(Arrangement), 다단계 조작(Sequential Manipulation) 등 실제 산업과 서비스 로봇에서 자주 수행하는 작업이 포함되어 있다. 각 작업은 시각 인식, 공간 추론(Spatial Reasoning), 조작 계획(Manipulation Planning), 저수준 제어(Low-Level Control)를 모두 요구한다.

SimplerEnv는 자연어 기반(Language Conditioned) 조작 평가를 중요하게 다룬다. 사용자는 미리 정의된 명령어가 아니라 자연어(Natural Language)로 작업을 지시하며, 로봇은 이를 이해하여 적절한 물체를 찾고 공간 관계를 해석한 뒤 행동을 생성해야 한다. 색상(Color), 크기(Size), 위치(Location), 형태(Shape)와 같은 의미적 속성을 정확하게 이해해야 하므로, 언어 이해(Language Understanding)와 행동 생성(Action Generation)이 동시에 평가된다. 이러한 특성은 최신 VLA 모델 평가에 매우 적합하다.

시각 인식 강건성(Visual Perception Robustness)은 SimplerEnv의 중요한 평가 요소이다. 실제 환경에서는 조명 변화(Lighting Variation), 그림자(Shadow), 반사(Reflection), 카메라 노출(Camera Exposure), 모션 블러(Motion Blur), 부분 가림(Occlusion), 물체의 질감(Texture) 등이 지속적으로 변한다. SimplerEnv는 이러한 현실적인 시각 조건을 체계적으로 변경하면서 모델의 성능을 측정한다. 이를 통해 실제 환경에서도 안정적인 시각 표현(Visual Representation)을 유지할 수 있는지를 평가한다.

일반화(Generalization)는 새로운 물체, 새로운 위치, 새로운 장면 구성(Scene Layout), 다양한 자연어 표현 등을 통해 평가된다. 모델은 추가적인 미세조정(Fine-Tuning) 없이도 기존에 학습한 지식을 활용하여 새로운 환경에 적응해야 한다. 이러한 평가는 단순한 시범 데이터 암기가 아니라 물리적 개념과 의미를 실제로 이해하고 있는지를 확인하는 중요한 기준이 된다.

또 다른 중요한 특징은 구현 일관성(Embodied Consistency) 평가이다. 로봇은 RGB 영상, 깊이 정보(Depth), 관절 정보(Proprioception), 언어 명령 등을 지속적으로 입력받으며, 환경 변화에 따라 행동을 안정적으로 수정해야 한다. 우수한 정책은 환경이 변해도 갑작스러운 진동이나 불안정한 움직임 없이 부드럽게 작업을 계속 수행한다. 이러한 시간적 일관성(Temporal Consistency)은 여러 단계의 작업(Long-Horizon Task)에서 특히 중요한 요소이다.

SimplerEnv는 개방 루프(Open-Loop) 방식이 아닌 폐루프 정책(Closed-Loop Policy)을 평가한다. 로봇은 미리 계산된 경로를 그대로 수행하는 것이 아니라 실행 결과를 지속적으로 관찰하고 다음 행동을 수정해야 한다. 물체가 조금 움직이거나 파지가 실패해도 스스로 복구(Recovery)하거나 재계획(Replanning)을 수행하여 작업을 계속 진행하는 능력이 중요하게 평가된다. 이는 실제 자율 로봇에서 반드시 필요한 기능이다.

강건성(Robustness) 평가는 시각 요소뿐 아니라 물리적인 불확실성까지 포함한다. 물체 위치 변화(Object Displacement), 접촉 오차(Contact Variation), 구동기 노이즈(Actuator Noise), 마찰 변화(Friction Difference), 파지 불안정성(Grasp Instability), 센서 지연(Sensor Latency) 등을 의도적으로 추가하여 정책의 안정성을 평가한다. 이러한 환경 변화에서도 일정한 성능을 유지하는 모델이 실제 현장에서 더욱 높은 활용 가치를 가진다.

움직임 품질(Motion Quality) 역시 중요한 평가 항목이다. 동일한 작업을 수행하더라도 경로가 부드럽고(Smooth Trajectory), 이동이 효율적이며(Path Efficiency), 파지가 안정적이고(Stable Grasp), 불필요한 움직임이 적은 정책이 더 높은 평가를 받는다. 따라서 단순한 작업 성공률뿐 아니라 실제 산업 현장에서 사용할 수 있는 안전하고 자연스러운 움직임도 함께 평가된다.

SimplerEnv에서도 시뮬레이션은 중요한 역할을 수행하지만 목적은 기존 벤치마크와 다르다. 최대한 복잡한 환경을 만드는 것이 아니라 실제 환경과 유사한 물체 모델(Object Model), 접촉 특성(Contact Property), 카메라 설정(Camera Configuration)을 사용하여 전이 가능성을 높인다. 또한 도메인 랜덤화(Domain Randomization)를 적용하여 다양한 조건에서 학습하도록 함으로써 실제 환경에서의 일반화 능력을 향상시킨다.

SimplerEnv의 가장 큰 특징 중 하나는 실제 로봇 검증(Real Robot Validation)을 포함한다는 점이다. 시뮬레이션에서 우수한 성능을 보인 정책은 동일한 조건의 실제 로봇에서도 평가된다. 이를 통해 시뮬레이션-실환경 전이 효율(Sim-to-Real Transfer Efficiency), 환경 적응성(Adaptation), 실제 운용 안정성(Robustness)을 직접 측정할 수 있다. 동일한 평가 프로토콜을 사용하기 때문에 시뮬레이션과 실제 환경의 차이를 객관적으로 비교할 수 있다.

평가 지표(Evaluation Metrics)는 기존 로봇 성능 지표와 최신 VLA 평가 지표를 모두 포함한다. 작업 성공률(Success Rate), 수행 시간(Completion Time), 조작 정확도(Manipulation Precision), 경로 효율(Path Efficiency), 충돌 빈도(Collision Frequency), 파지 안정성(Grasp Stability), 실행 일관성(Execution Consistency)을 기본적으로 측정한다. 추가적으로 언어 명령 수행 정확도(Instruction Following Accuracy), 의미 접지(Semantic Grounding), 멀티모달 추론(Multimodal Reasoning), 복구 성능(Recovery Performance), 강건성(Robustness), 시뮬레이션-실환경 전이 성능(Sim-to-Real Transfer Effectiveness)까지 함께 분석한다.

대규모 멀티모달 데이터셋(Multimodal Dataset)으로 학습한 로봇 파운데이션 모델의 발전과 함께 SimplerEnv의 활용도도 크게 증가하고 있다. 이러한 모델은 제로샷(Zero-Shot) 능력이 뛰어나지만 실제 환경에서의 성능 검증이 반드시 필요하다. SimplerEnv는 작업 일반화(Cross-Task Transfer), 환경 적응(Cross-Environment Adaptation), 명령 일반화(Instruction Generalization), 강건한 조작(Robust Manipulation)을 표준화된 방식으로 평가할 수 있어 최신 구현형 AI 연구에서 널리 활용되고 있다.

SimplerEnv는 공개된 평가 환경과 표준화된 프로토콜(Standardized Protocol)을 제공하여 연구 결과의 재현성(Reproducibility)을 높인다. 강화학습(Reinforcement Learning), 모방학습(Imitation Learning), 확산 정책(Diffusion Policy), 트랜스포머(Transformer), 행동 청킹(Action Chunking), 계층형 계획(Hierarchical Planning), VLA 모델 등을 동일한 조건에서 비교할 수 있다. 이러한 공정한 비교 환경은 연구 개발 속도를 높이고 알고리즘의 실제 성능을 객관적으로 검증하는 기반이 된다.

산업 현장에서도 SimplerEnv는 매우 실용적인 평가 기준으로 활용될 수 있다. 제조 자동화, 물류 창고, 연구실 자동화, 전자 조립, 식품 처리, 의료 서비스 등은 지나치게 복잡한 환경보다 일정한 구조를 가진 환경에서 높은 안정성과 신뢰성을 요구한다. SimplerEnv는 이러한 실제 환경을 잘 반영하기 때문에 실제 배포(Deployment) 성능과 높은 상관관계를 가지며, 산업용 로봇 개발 과정에서 매우 유용한 검증 도구가 된다.

궁극적으로 SimplerEnv Real Robot Grounded Benchmark는 단순히 시뮬레이션 성능을 평가하는 벤치마크가 아니라 실제 물리 AI(Physical AI)의 구현 가능성을 검증하기 위한 현실 중심 평가 체계이다. 시각 인식(Perception), 자연어 이해(Language Understanding), 순차 추론(Sequential Reasoning), 물리 상호작용(Physical Interaction), 폐루프 제어(Closed-Loop Control), 강건성(Robustness), 실제 로봇 검증(Real Robot Validation)을 하나의 통합된 프레임워크에서 평가함으로써 VLA 시스템과 로봇 파운데이션 모델의 실제 활용 가능성을 객관적으로 측정할 수 있다. 앞으로 범용 자율 로봇이 다양한 산업과 일상 환경으로 확산될수록 SimplerEnv는 실제 환경 적응 능력을 검증하는 가장 중요한 국제 벤치마크 가운데 하나로 자리매김할 것이다.

## 11.5 Language Table (with Code)

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

Language Table은 자연어 기반 조작(Language-Conditioned Manipulation)을 평가하기 위해 개발된 대표적인 벤치마크(Benchmark)로, 비전-언어-행동(Vision-Language-Action, VLA) 시스템, 로봇 파운데이션 모델(Robot Foundation Model), 모방학습(Imitation Learning), 구현형 인공지능(Embodied AI) 연구에서 널리 활용되고 있다. 기존 로봇 벤치마크가 미리 정의된 목표나 기호(Symbolic Goal)를 기반으로 조작 성능을 평가했다면, Language Table은 사람이 자연어(Natural Language)로 지시한 명령을 로봇이 이해하고 실제 조작 행동으로 수행할 수 있는지를 평가하는 새로운 패러다임을 제시하였다.

Language Table의 핵심 목적은 인식(Perception), 언어(Language), 행동(Action)을 하나의 통합된 파이프라인(Pipeline)으로 평가하는 것이다. 사용자는 자연어로 작업 목표를 전달하며, 로봇은 주변 환경을 관찰하고, 관련 물체를 찾고, 공간 관계를 이해하며, 작업의 의미를 추론한 후 적절한 조작 계획을 생성하여 실행해야 한다. 따라서 단순한 물체 인식이나 조작 정확도만이 아니라 전체 언어-행동(Language-to-Action) 과정이 평가 대상이 된다.

Language Table이라는 이름은 평평한 작업대(Tabletop) 위에서 수행되는 조작 실험 환경에서 유래하였다. 환경 자체는 비교적 단순하지만, 다양한 색상(Color), 형태(Shape), 크기(Size), 위치(Position), 방향(Orientation)을 가진 여러 물체를 포함하고 있어 충분한 난이도를 제공한다. 로봇은 이러한 물체들을 대상으로 자연어 명령을 수행해야 하며, 미리 정의된 기호가 아닌 실제 언어 표현을 이해하는 능력이 요구된다.

Language Table에서 가장 중요한 평가 항목은 언어 접지(Language Grounding)이다. 언어 접지는 자연어의 의미를 실제 환경 속 물체와 공간 관계에 연결하는 과정을 의미한다. 예를 들어 "빨간 블록을 파란 원통 옆으로 옮겨라"라는 명령을 받으면 로봇은 색상을 인식하고, 물체 종류를 구분하며, 공간 관계를 이해한 후 올바른 조작 행동을 수행해야 한다. 이러한 과정은 시각 인식, 언어 이해, 행동 계획이 긴밀하게 결합되어야만 가능하다.

자연어 이해(Natural Language Understanding)는 단순한 키워드 검색이 아니다. Language Table은 다양한 문장 구조, 동의어(Synonym), 문장 바꿔쓰기(Paraphrase), 문맥(Context)을 포함하는 명령을 제공한다. 예를 들어 "옆에 놓아라", "가까이에 배치해라", "인접하게 이동시켜라"와 같은 서로 다른 표현도 동일한 의미로 이해해야 한다. 따라서 모델은 문장을 암기하는 것이 아니라 의미 자체를 이해하는 능력을 갖추어야 한다.

공간 추론(Spatial Reasoning)도 Language Table의 핵심 평가 요소이다. 많은 작업은 절대 좌표가 아니라 물체 간의 상대적인 위치를 기반으로 수행된다. "왼쪽", "오른쪽", "앞", "뒤", "사이", "안쪽", "바깥", "가까이", "멀리"와 같은 공간 개념을 정확하게 이해해야 한다. 이를 위해서는 장면(Scene)의 기하학적 구조와 의미적 관계를 함께 분석하는 능력이 필요하며, 이는 현대 VLA 시스템의 중요한 지능 요소로 평가된다.

물체 구분(Object Disambiguation) 역시 중요한 평가 대상이다. 동일한 모양을 가진 여러 물체가 존재할 경우 색상, 크기, 위치와 같은 추가적인 속성을 이용하여 정확한 대상을 선택해야 한다. 자연어 명령은 이러한 속성을 이용하여 특정 물체를 지시하는 경우가 많기 때문에, 로봇은 언어 정보와 시각 정보를 동시에 활용하여 모호성을 해결해야 한다. 이는 실제 사람과 로봇의 상호작용에서 매우 자주 발생하는 상황이다.

Language Table은 개방 루프(Open-Loop)가 아니라 폐루프 조작(Closed-Loop Manipulation)을 평가한다. 로봇은 작업 수행 중에도 계속해서 환경을 관찰하고 자신의 행동 결과를 확인해야 한다. 물체가 예상과 다르게 움직이거나 파지가 실패하면 새로운 정보를 바탕으로 행동을 수정해야 한다. 이러한 적응적 제어(Adaptive Control)는 실제 자율 로봇에서 필수적인 능력이며, Language Table은 이를 중요한 평가 기준으로 사용한다.

멀티모달 표현 학습(Multimodal Representation Learning)은 Language Table의 또 다른 핵심 요소이다. 최신 VLA 모델은 비전 인코더(Vision Encoder), 언어 모델(Language Model), 행동 정책(Action Policy)을 결합하여 동작한다. Language Table은 이러한 서로 다른 정보 표현이 얼마나 효과적으로 융합(Fusion)되어 정확한 행동 생성으로 이어지는지를 평가한다. 시각과 언어 사이의 정렬(Alignment)이 우수한 모델일수록 높은 성능을 보이는 경향이 있다.

Language Table은 시범학습(Demonstration Learning)도 적극적으로 활용한다. 전문가가 수행한 성공적인 조작 사례가 데이터셋으로 제공되며, 이를 이용하여 행동 복제(Behavior Cloning), 모방학습(Imitation Learning), 확산 정책(Diffusion Policy), 트랜스포머 기반 정책(Transformer Policy) 등을 학습할 수 있다. 동일한 시범 데이터를 사용하기 때문에 서로 다른 알고리즘의 학습 효율성과 일반화 능력을 공정하게 비교할 수 있다.

일반화(Generalization)는 Language Table에서 매우 중요한 평가 항목이다. 평가 과정에서는 기존에 학습하지 않은 새로운 물체 조합, 새로운 문장 구조, 새로운 공간 관계, 다양한 문장 표현, 변경된 작업 환경 등이 제공된다. 우수한 모델은 추가적인 미세조정(Fine-Tuning) 없이도 이러한 새로운 상황에서 정확하게 작업을 수행해야 한다. 이러한 평가는 모델이 실제 의미를 이해하고 있는지를 확인하는 중요한 기준이 된다.

장기 작업(Long-Horizon Manipulation)은 여러 단계의 조작을 연속적으로 수행하는 능력을 평가한다. 예를 들어 먼저 하나의 물체를 이동하여 공간을 확보하고, 다른 물체를 옮긴 뒤, 여러 개의 물체를 원하는 형태로 배열하는 작업은 단일 행동으로 해결할 수 없다. 로봇은 중간 목표(Intermediate Goal)를 기억하면서 전체 작업 목표를 유지해야 하며, 계획(Planning)과 순차 추론(Sequential Reasoning) 능력이 요구된다.

강건성(Robustness) 평가는 다양한 환경 변화 속에서도 안정적으로 작업을 수행할 수 있는지를 측정한다. 조명 변화(Lighting Variation), 카메라 시점(Camera Viewpoint), 센서 노이즈(Sensor Noise), 물체 위치 변화(Object Displacement), 부분 가림(Occlusion) 등이 포함되며, 언어 측면에서도 문법 변화, 동의어 사용, 문장 순서 변경, 대화체 표현 등이 함께 평가된다. 이러한 다양한 변화에도 성능을 유지하는 모델이 실제 환경에서 더욱 높은 신뢰성을 가진다.

Language Table의 평가 지표(Evaluation Metrics)는 단순한 성공률(Success Rate)에 그치지 않는다. 언어 이해 정확도(Language Understanding Accuracy), 언어 접지 정확도(Grounding Precision), 조작 정확도(Manipulation Accuracy), 공간 관계 수행 정확도(Spatial Relation Accuracy), 작업 수행 효율(Execution Efficiency), 경로 부드러움(Trajectory Smoothness), 충돌 빈도(Collision Frequency), 파지 안정성(Grasp Stability), 복구 능력(Recovery Capability), 추론 지연(Inference Latency) 등을 함께 평가한다. 최근에는 멀티모달 정렬(Multimodal Alignment), 의미 일관성(Semantic Consistency), 정책 신뢰도(Policy Confidence)까지 포함하여 더욱 종합적인 분석을 수행한다.

Language Table은 RLBench, LIBERO, SimplerEnv, DROID와 같은 다른 벤치마크와 상호 보완적인 역할을 한다. RLBench는 다양한 조작 기술을, LIBERO는 연속 학습(Continual Learning)을, SimplerEnv는 실제 환경 기반 평가를, DROID는 대규모 실제 로봇 데이터를 중점적으로 다룬다. 반면 Language Table은 자연어 기반 의미 접지(Language Grounding)와 언어 조건 조작(Language-Conditioned Manipulation)을 전문적으로 평가하는 벤치마크라는 점에서 차별성을 가진다.

최근 로봇 파운데이션 모델(Robot Foundation Model)의 발전으로 Language Table의 중요성은 더욱 커지고 있다. 대규모 멀티모달(Multimodal) 사전학습 모델은 뛰어난 언어 이해 능력을 보이지만, 실제 로봇에서는 이를 물리적인 행동으로 정확하게 연결해야 한다. Language Table은 이러한 의미 이해(Semantic Understanding)에서 실제 행동(Action)으로의 변환 과정을 표준화된 환경에서 평가할 수 있기 때문에 VLA 시스템 검증에 매우 적합한 벤치마크로 평가받고 있다.

학술적인 측면에서도 Language Table은 높은 재현성(Reproducibility)을 제공한다. 공개된 데이터셋, 표준화된 작업(Task), 일관된 평가 프로토콜(Evaluation Protocol), 공통 시범 데이터(Demonstration Dataset)를 제공함으로써 전 세계 연구자들이 동일한 조건에서 강화학습(Reinforcement Learning), 모방학습(Imitation Learning), 트랜스포머(Transformer), 확산 정책(Diffusion Policy), 행동 청킹(Action Chunking), VLA 모델 등을 객관적으로 비교할 수 있다.

산업적인 측면에서도 Language Table의 활용 가치는 매우 높다. 미래의 서비스 로봇(Service Robot), 물류 로봇(Warehouse Robot), 협동 로봇(Collaborative Robot), 연구실 자동화(Laboratory Automation), 의료 로봇(Healthcare Robot), 가정용 로봇(Home Robot)은 대부분 자연어를 통해 사람과 상호작용하게 될 것이다. 따라서 자연어 기반 조작 능력을 평가하는 Language Table은 실제 제품 개발과 상용화 과정에서 매우 중요한 검증 도구로 활용될 수 있다.

궁극적으로 Language Table은 자연어 이해를 중심으로 로봇 조작 능력을 평가하는 새로운 평가 방법론을 제시하였다. 단순한 물체 인식이나 조작 정확도를 넘어 언어 이해(Language Understanding), 의미 접지(Language Grounding), 공간 추론(Spatial Reasoning), 멀티모달 표현 학습(Multimodal Representation Learning), 폐루프 제어(Closed-Loop Control), 일반화(Generalization)를 하나의 통합된 프레임워크에서 평가함으로써 현대 VLA 시스템의 실제 지능 수준을 종합적으로 분석할 수 있다. 앞으로 범용 물리 AI(Physical AI)가 발전할수록 Language Table은 사람과 자연스럽게 소통하며 작업을 수행하는 차세대 로봇을 평가하는 가장 중요한 국제 벤치마크 가운데 하나로 지속적으로 활용될 것이다.

## 11.6 DROID Dataset Evaluation (with Code)

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

DROID(Diverse Robot Interaction Dataset)는 비전-언어-행동(Vision-Language-Action, VLA) 시스템, 로봇 파운데이션 모델(Robot Foundation Model), 모방학습(Imitation Learning), 그리고 구현형 인공지능(Embodied AI)을 평가하기 위한 대표적인 대규모 실제 데이터셋(Dataset) 기반 벤치마크이다. 기존의 많은 로봇 벤치마크는 시뮬레이션이나 통제된 실험실 환경에 의존하지만, DROID는 실제 환경에서 수집된 대규모 데이터를 기반으로 로봇의 일반화 능력과 실제 배포 가능성을 평가하는 것을 목표로 한다. 따라서 DROID는 현대 VLA 평가 체계에서 실제 환경 적응성을 검증하는 핵심 벤치마크로 자리잡고 있다.

DROID는 \'다양한 로봇 상호작용 데이터셋(Diverse Robot Interaction Dataset)\'이라는 이름처럼 다양한 환경, 다양한 작업, 다양한 물체, 다양한 사용자로부터 수집된 실제 데이터를 제공한다. 기존 벤치마크가 제한된 작업만 평가했다면, DROID는 환경의 다양성(Environment Diversity) 자체를 로봇 지능의 중요한 요소로 간주한다. 미래의 서비스 로봇, 물류 로봇, 산업용 로봇, 의료 로봇, 가정용 로봇은 모두 서로 다른 환경에서 동작해야 하므로, 이러한 다양성을 포함한 데이터가 필수적이다.

DROID의 가장 큰 특징은 실제 환경(In-the-Wild)에서 데이터를 수집한다는 점이다. 여기서 실제 환경(In-the-Wild)은 연구실에서 인위적으로 구성한 환경이 아니라, 실제 사람이 생활하고 작업하는 환경을 의미한다. 물체는 다양한 조명(Lighting), 부분 가림(Occlusion), 복잡한 배경(Clutter), 여러 시점(Viewpoint), 다양한 배치(Layout) 상태에서 존재하며, 사용자는 특별히 학습을 위해 최적화하지 않은 자연스러운 방식으로 로봇을 조작한다. 따라서 데이터는 실제 로봇이 현장에서 경험하게 될 상황을 매우 잘 반영한다.

대규모 데이터 수집(Large-Scale Data Collection)은 DROID의 또 다른 핵심 특징이다. 최신 로봇 파운데이션 모델은 수많은 경험 데이터를 기반으로 학습해야 범용적인 물리 지식(Physical Knowledge)을 습득할 수 있다. DROID는 다양한 작업(Task), 환경(Environment), 물체(Object), 사용자(User)를 포함하는 방대한 데이터를 제공하여 지도학습(Supervised Learning), 모방학습(Imitation Learning), 자기지도학습(Self-Supervised Learning), 멀티모달 사전학습(Multimodal Pretraining), VLA 정책 학습에 활용될 수 있다.

사람의 시범(Human Demonstration)은 DROID에서 매우 중요한 역할을 한다. 기존처럼 사람이 설계한 이상적인 로봇 경로가 아니라 실제 사람이 자연스럽게 로봇을 조작하는 과정을 기록한다. 따라서 속도(Speed), 이동 경로(Trajectory), 파지 방법(Grasp Strategy), 수정 동작(Correction), 복구 행동(Recovery) 등이 사람마다 서로 다르게 나타난다. 이러한 다양성은 오히려 실제 환경에서 발생하는 자연스러운 행동 패턴을 학습할 수 있도록 해주며, 보다 강건한(Robust) 정책을 만드는 데 도움이 된다.

DROID는 멀티모달 관측(Multimodal Observation)을 제공한다. 하나의 작업에는 RGB 영상, 깊이 정보(Depth), 관절 상태(Proprioception), 말단장치 상태(End-Effector State), 관절 위치(Joint Position), 행동 명령(Action Command), 시간 정보(Temporal Sequence), 작업 주석(Task Annotation) 등이 함께 저장된다. 경우에 따라 자연어 설명(Natural Language Description)도 포함되어 있어 VLA 모델은 시각, 언어, 행동 사이의 관계를 동시에 학습할 수 있다.

일반화(Generalization)는 DROID에서 가장 중요한 평가 요소 가운데 하나이다. 데이터는 하나의 실험실에서 수집된 것이 아니라 매우 다양한 환경에서 획득되었기 때문에 물체의 모양, 크기, 질감(Texture), 조명, 배경, 배치가 지속적으로 변화한다. 우수한 모델은 특정 환경을 암기하는 것이 아니라 이러한 다양한 조건에서도 공통적인 물리 개념을 학습해야 한다. 따라서 DROID는 모델의 표현 학습(Representation Learning) 능력과 환경 간 전이(Cross-Domain Transfer) 성능을 평가하는 데 매우 적합하다.

DROID는 다양한 사용자(Cross-Operator Diversity)의 데이터를 포함한다. 동일한 작업이라도 사람마다 수행 방식이 크게 다르다. 어떤 사람은 천천히 움직이고, 다른 사람은 빠르게 수행하며, 파지 위치나 이동 경로도 서로 다르다. DROID는 이러한 차이를 단순한 노이즈(Noise)가 아니라 여러 가지 성공적인 작업 전략(Manipulation Strategy)으로 간주한다. 따라서 모델은 사람의 스타일을 암기하는 것이 아니라 작업의 본질적인 목적을 이해하는 능력을 학습해야 한다.

작업 다양성(Task Diversity) 역시 매우 크다. 물체 집기(Pick), 놓기(Place), 분류(Sort), 문 열기(Open), 용기 닫기(Close), 가정용 물체 정리(Household Organization), 도구 사용(Tool Manipulation), 부품 조립(Assembly), 물체 재배치(Object Rearrangement), 다단계 조작(Long-Horizon Manipulation) 등 실제 산업과 서비스 환경에서 자주 발생하는 작업들이 포함되어 있다. 이러한 폭넓은 작업 분포는 범용 로봇 정책(General-Purpose Robot Policy)의 학습을 가능하게 한다.

시간적 일관성(Temporal Consistency)도 DROID의 중요한 특징이다. 조작 작업은 하나의 이미지가 아니라 시간에 따라 연속적으로 진행된다. 따라서 로봇은 이전 행동과 현재 상태를 기억하면서 작업을 이어가야 한다. DROID는 이러한 전체 시퀀스(Sequence)를 저장하므로 트랜스포머(Transformer), 확산 정책(Diffusion Policy), 행동 청킹(Action Chunking), 장기 계획(Long-Horizon Planning)과 같은 최신 알고리즘을 효과적으로 평가할 수 있다.

최근에는 자연어 기반 학습(Language-Conditioned Learning)의 중요성도 높아지고 있다. DROID의 일부 데이터에는 자연어 설명이 함께 제공되며, 이를 이용하여 로봇은 사람이 왜 특정 행동을 수행했는지까지 함께 학습할 수 있다. 이러한 의미 정보(Semantic Information)는 명령 수행(Instruction Following), 제로샷(Zero-Shot) 작업 수행, 언어 기반 정책 적응(Language-Guided Policy Adaptation), 멀티모달 추론(Multimodal Reasoning)에 매우 중요한 역할을 한다.

강건성(Robustness) 평가는 DROID의 자연스러운 데이터 다양성 덕분에 매우 현실적으로 이루어진다. 기존 벤치마크는 인위적으로 조명이나 노이즈를 추가하지만, DROID는 실제 환경에서 발생한 조명 변화, 물체 이동, 복잡한 배경, 센서 오차, 사용자 차이 등을 그대로 포함하고 있다. 따라서 실제 환경에서의 강건성을 보다 현실적으로 평가할 수 있으며, 실제 배포 환경과 높은 상관관계를 가진다.

DROID의 가장 큰 장점 가운데 하나는 파운데이션 모델(Foundation Model)의 사전학습(Pretraining)에 매우 적합하다는 점이다. 현대 VLA 모델은 수백만 개 이상의 멀티모달 데이터를 이용하여 먼저 일반적인 물리 지식을 학습한 후 특정 작업에 미세조정(Fine-Tuning)하는 방식으로 개발된다. DROID는 이러한 사전학습을 위한 매우 풍부한 실제 데이터를 제공하므로 범용적인 물리 표현(General Physical Representation)을 학습하는 데 큰 도움이 된다.

또한 DROID는 전이학습(Transfer Learning) 평가에도 활용된다. 먼저 DROID에서 모델을 사전학습한 후 RLBench, LIBERO, Language Table, SimplerEnv, CALVIN 등과 같은 다른 데이터셋으로 미세조정을 수행하면 적은 데이터만으로도 높은 성능을 달성할 수 있다. 이러한 전이 성능은 사전학습된 표현의 품질을 평가하는 중요한 기준이 되며, 실제 산업용 응용에서도 매우 유용하다.

DROID의 평가 지표(Evaluation Metrics)는 단순한 작업 성공률(Success Rate)에 머무르지 않는다. 조작 성공률(Manipulation Success), 이동 경로 품질(Trajectory Quality), 행동 일관성(Action Consistency), 시간적 연속성(Temporal Coherence), 파지 안정성(Grasp Stability), 실행 부드러움(Execution Smoothness), 샘플 효율성(Sample Efficiency), 표현 전이성(Representation Transferability), 정책 강건성(Policy Robustness), 적응 효율(Adaptation Efficiency), 일반화 성능(Generalization Performance) 등을 종합적으로 평가한다. 최근에는 언어 접지(Language Grounding), 멀티모달 정렬(Multimodal Alignment), 제로샷 수행 능력(Zero-Shot Capability), 명령 수행 정확도(Instruction Following Accuracy)까지 포함하는 평가가 확대되고 있다.

비록 실제 환경 데이터를 중심으로 하지만 DROID는 연구 재현성(Reproducibility)도 고려하여 설계되었다. 표준화된 데이터 형식(Standardized Data Format), 동기화된 멀티모달 데이터(Synchronized Multimodal Data), 일관된 주석 체계(Annotation), 통합된 평가 프로토콜(Evaluation Protocol)을 제공하므로 전 세계 연구자들이 동일한 조건에서 알고리즘을 비교할 수 있다. 이를 통해 모방학습, 강화학습(Reinforcement Learning), 트랜스포머, 확산 정책, 행동 청킹, VLA 모델 등을 공정하게 비교할 수 있다.

산업적인 관점에서 DROID는 실제 배포 가능성(Deployment Readiness)을 평가하는 매우 중요한 기준이다. 실제 산업용 로봇은 실험실과 달리 다양한 조명, 다양한 물체, 다양한 사용자, 변화하는 작업 환경을 지속적으로 경험한다. DROID는 이러한 현실적인 데이터를 제공하기 때문에 서비스 로봇(Service Robot), 협동 로봇(Collaborative Robot), 물류 자동화(Warehouse Automation), 제조 자동화(Manufacturing Automation), 의료 로봇(Healthcare Robot) 등 실제 제품 개발 과정에서 모델 선택과 성능 검증에 매우 유용하게 활용된다.

미래의 구현형 인공지능(Embodied AI)에서는 모델 구조(Model Architecture)만큼 데이터 다양성(Data Diversity)이 중요해질 것으로 예상된다. 시뮬레이션만으로는 실제 물리 세계의 복잡성과 사람의 다양한 행동을 충분히 학습하기 어렵다. DROID는 실제 환경에서 발생하는 다양한 상황과 사람의 자연스러운 조작 데이터를 제공함으로써 범용 물리 AI(Physical AI)의 핵심 기반 데이터를 제공한다는 점에서 매우 큰 의미를 가진다.

궁극적으로 DROID는 환경의 다양성을 제거하는 것이 아니라 오히려 적극적으로 활용하여 로봇 지능을 평가하는 새로운 접근 방식을 제시하였다. 대규모 실제 데이터(Large-Scale Real-World Data), 멀티모달 시범(Multimodal Demonstration), 자연어 접지(Language Grounding), 시간적 일관성(Temporal Consistency), 다양한 사용자(Cross-Operator Diversity), 강건한 조작(Robust Manipulation), 전이학습(Transfer Learning), 파운데이션 모델 사전학습(Foundation Model Pretraining)을 하나의 통합된 데이터셋으로 제공함으로써 VLA 시스템을 평가하는 가장 중요한 벤치마크 가운데 하나로 자리매김하고 있다. 앞으로 범용 자율 로봇 시대가 도래할수록 DROID는 실제 환경 일반화(Real-World Generalization)를 측정하고 차세대 물리 AI의 발전을 이끄는 핵심 데이터셋으로 지속적으로 활용될 것이다.

## 11.7 Designing Custom Evaluation Protocols (with Code)

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

맞춤형 작업 평가 프로토콜(Custom Task Evaluation Protocol)은 비전-언어-행동(Vision-Language-Action, VLA) 시스템이 실제 산업, 의료, 물류, 농업, 서비스, 가정용 로봇 환경에서 요구되는 성능을 충족하는지를 검증하기 위해 설계되는 평가 체계이다. RLBench, LIBERO, Language Table, SimplerEnv, DROID와 같은 공개 벤치마크는 범용 로봇 지능을 평가하는 데 유용하지만, 특정 산업 현장의 요구사항을 모두 반영하기는 어렵다. 따라서 실제 제품 개발에서는 응용 분야에 맞는 맞춤형 평가 프로토콜을 별도로 설계해야 하며, 이를 통해 기능성(Functionality), 안전성(Safety), 신뢰성(Reliability), 효율성(Efficiency)을 종합적으로 검증할 수 있다.

맞춤형 평가 프로토콜의 가장 중요한 목적은 실제 운영 요구사항(Operational Requirement)을 정량적인 성능 지표(Performance Metric)로 변환하는 것이다. 창고 물류 로봇, 병원 안내 로봇, 농업용 자율주행 로봇, 산업용 검사 로봇은 모두 사용하는 VLA 구조는 유사하지만 요구되는 성능은 크게 다르다. 예를 들어 병원 로봇은 안전성과 정확성이 우선이며, 물류 로봇은 작업 처리량(Throughput)과 이동 효율이 더욱 중요하다. 따라서 평가 항목 역시 실제 업무 목표에 맞추어 설계되어야 한다.

평가 설계의 첫 번째 단계는 작업 분해(Task Decomposition)이다. 복잡한 임무(Mission)는 여러 개의 세부 작업(Subtask)으로 나누어야 한다. 예를 들어 자율 검사 로봇은 이동(Navigation), 위치 추정(Localization), 검사 대상 인식(Target Detection), 검사 위치 정렬(Positioning), 센서 동작(Sensor Activation), 이상 탐지(Anomaly Detection), 보고서 생성(Report Generation), 복귀(Return-to-Base) 등으로 구분할 수 있다. 이러한 분해를 통해 각 단계의 성능을 개별적으로 분석하면서도 전체 임무의 성공 여부를 함께 평가할 수 있다.

운영 시나리오(Operational Scenario)는 평가 프로토콜의 핵심이다. 단순한 실험실 환경이 아니라 실제 운영 환경을 최대한 그대로 재현해야 한다. 작업 공간의 구조(Layout), 물체 배치(Object Distribution), 조명(Lighting), 사람과의 상호작용(Human Interaction), 날씨(Weather), 통신 상태(Communication Quality), 지형(Terrain), 센서 제약(Sensor Limitation), 운영 시간(Operation Schedule) 등을 실제 현장과 유사하게 구성해야 한다. 이러한 시나리오가 실제 환경을 잘 반영할수록 평가 결과의 신뢰성이 높아진다.

작업 정의(Task Definition)는 명확하고 측정 가능해야 한다. 작업 시작 조건(Initial Condition), 입력(Input), 기대 결과(Expected Output), 수행 시간(Time Constraint), 환경 조건(Environment Assumption), 안전 규칙(Safety Constraint), 성공 기준(Success Threshold)을 구체적으로 명시해야 한다. 이러한 표준화는 여러 버전의 로봇 정책(Policy)이나 서로 다른 시스템을 동일한 기준으로 비교할 수 있도록 해주며, 자동화된 평가 시스템 구축에도 도움이 된다.

최근에는 자연어 기반 평가(Language-Conditioned Evaluation)의 중요성이 크게 증가하고 있다. VLA 시스템은 기존의 고정 명령 대신 자연어를 통해 작업을 수행하므로, 평가에서는 산업 현장에서 사용하는 전문 용어(Domain-Specific Terminology), 작업 절차(Procedure), 상황별 명령(Contextual Instruction), 대화형 표현(Conversational Command) 등을 포함해야 한다. 예를 들어 제조 공장, 병원, 물류센터마다 사용하는 용어가 다르므로 해당 환경에 맞는 언어 데이터셋(Language Dataset)을 구축하는 것이 중요하다.

성능 지표(Performance Metrics)는 맞춤형 평가 프로토콜의 가장 중요한 요소이다. 단순한 성공률(Success Rate)만으로는 실제 성능을 충분히 설명할 수 없다. 작업 완료율(Task Completion Ratio), 수행 시간(Execution Time), 경로 효율(Path Efficiency), 조작 정확도(Manipulation Precision), 위치 추정 정확도(Localization Accuracy), 인식 정확도(Perception Accuracy), 언어 이해(Language Understanding), 복구 성공률(Recovery Success), 에너지 소비(Energy Consumption), 추론 지연(Inference Latency), 정책 안정성(Policy Stability), 임무 신뢰성(Mission Reliability) 등을 함께 측정해야 한다. 산업별 우선순위에 따라 중요 지표도 달라질 수 있다.

성공 기준(Success Criteria)은 반드시 충족해야 하는 필수 조건(Mandatory Requirement)과 성능 향상을 위한 최적화 항목(Optimization Objective)을 구분해야 한다. 충돌 방지(Collision Avoidance), 안전 규정 준수(Regulatory Compliance), 작업 성공률과 같은 요소는 반드시 만족해야 하는 기준이다. 반면 이동 속도, 에너지 효율, 경로 부드러움(Trajectory Smoothness), 사용자 만족도(User Satisfaction)는 성능 개선을 위한 요소로 구분할 수 있다. 이러한 구분은 시스템 설계 시 우선순위를 명확하게 해준다.

강건성(Robustness) 평가는 실제 환경의 다양한 불확실성을 반영해야 한다. 조명 변화(Lighting Variation), 센서 노이즈(Sensor Noise), 통신 지연(Communication Delay), 이동 장애물(Moving Obstacle), 물체 위치 변화(Object Displacement), 날씨 변화(Weather Change), 복잡한 환경(Clutter), 위치 오차(Localization Drift), 구동기 오차(Actuator Error), 사람의 예기치 않은 개입(Human Intervention) 등을 체계적으로 포함해야 한다. 이러한 외란에서도 안정적인 성능을 유지하는 정책이 실제 현장에서 높은 가치를 가진다.

일반화(Generalization) 평가는 동일한 시나리오만 반복하는 것이 아니라 새로운 환경에서도 성능을 유지하는지를 검증해야 한다. 새로운 물체(New Object), 새로운 환경(New Environment), 새로운 언어 표현(New Instruction), 새로운 사용자(New Operator), 새로운 작업 조합(New Task Combination), 계절 변화(Seasonal Change) 등을 포함하여 평가를 수행한다. 높은 일반화 성능은 모델이 특정 환경을 암기한 것이 아니라 실제 작업 개념을 이해하고 있음을 의미한다.

장기 임무(Long-Horizon Mission) 평가는 실제 자율 로봇에서 매우 중요하다. 산업용 검사 로봇, 물류 로봇, 배송 로봇, 농업 로봇, 병원 서비스 로봇은 수십 분에서 수 시간 동안 작업을 수행한다. 따라서 단일 조작이나 단일 이동이 아니라 전체 임무 동안 발생하는 누적 오차(Cumulative Error), 시간적 일관성(Temporal Consistency), 자원 사용(Resource Utilization), 복구 행동(Recovery Behavior), 장시간 신뢰성(Long-Term Reliability)을 함께 평가해야 한다.

맞춤형 평가 프로토콜은 폐루프 평가(Closed-Loop Evaluation)를 기본으로 해야 한다. 로봇은 환경 변화를 지속적으로 인식하고 새로운 정보를 바탕으로 행동을 수정해야 한다. 따라서 재계획(Replanning), 적응 행동(Adaptive Behavior), 복구 효율(Recovery Efficiency), 정책 일관성(Policy Consistency), 예기치 않은 이벤트 대응(Response to Unexpected Events) 등을 함께 평가해야 한다. 이는 미리 정의된 경로만 수행하는 개방 루프(Open-Loop) 방식보다 실제 환경을 훨씬 잘 반영한다.

사람-로봇 상호작용(Human-Robot Interaction)도 중요한 평가 요소이다. 많은 서비스 로봇과 협동 로봇은 사람과 함께 작업한다. 따라서 명령 이해(Instruction Interpretation), 응답 시간(Response Time), 사용자 만족도(User Satisfaction), 공유 작업 공간(Shared Workspace), 작업 중단 처리(Interruption Handling), 협업 효율(Collaboration Efficiency), 안전성(Safety)을 종합적으로 평가해야 한다. 특히 의료, 서비스, 교육 분야에서는 이러한 요소의 중요성이 더욱 크다.

시뮬레이션(Simulation)과 실제 로봇(Real Robot) 평가를 통합하는 전략도 필요하다. 시뮬레이션은 빠른 알고리즘 개발과 대규모 반복 실험에 적합하며, 실제 로봇은 물리적 상호작용과 기계적 신뢰성을 검증하는 데 적합하다. 따라서 맞춤형 평가 프로토콜은 시뮬레이션 결과와 실제 로봇 검증을 연계하여 개발 효율성과 실제 적용성을 동시에 확보할 수 있도록 설계되어야 한다.

데이터 수집(Data Collection)과 기록(Logging)은 평가 품질을 결정하는 중요한 요소이다. 모든 실험에서는 RGB 영상, 깊이 정보(Depth), 언어 입력(Language Input), 로봇 상태(Robot State), 행동(Action), 위치 추정(Localization), 센서 데이터(Sensor Data), 시간 정보(Timestamp), 환경 상태(Environment Condition), 시스템 로그(System Log), 안전 이벤트(Safety Event)를 동기화하여 저장해야 한다. 이러한 데이터는 실패 원인 분석, 알고리즘 개선, 모델 재학습에 매우 중요한 자산이 된다.

통계 분석(Statistical Analysis)은 평가 결과의 신뢰성을 높인다. 동일한 시나리오를 여러 번 반복하고 초기 조건(Random Initial Condition)을 변경하여 평균 성능(Mean), 분산(Variance), 신뢰구간(Confidence Interval), 실패 분포(Failure Distribution), 강건성 곡선(Robustness Curve), 민감도 분석(Sensitivity Analysis)을 수행해야 한다. 이러한 통계적 접근은 단일 실험 결과보다 훨씬 객관적인 성능 비교를 가능하게 한다.

안전성(Safety)은 작업 성공과 별도로 독립적으로 평가되어야 한다. 아무리 높은 작업 성공률을 달성하더라도 충돌(Collision), 과도한 힘(Excessive Force), 위험 속도(Unsafe Velocity), 작업 공간 이탈(Workspace Violation), 비상 정지(Emergency Stop), 정책 우회(Policy Override), 사람과의 위험한 접근(Human Proximity)이 발생하면 실제 배포는 어렵다. 따라서 안전 지표는 항상 성능 지표와 함께 관리되어야 한다.

맞춤형 평가 프로토콜은 지속적인 평가(Continuous Evaluation)를 지원해야 한다. 소프트웨어 업데이트, 파운데이션 모델 개선, 센서 변경, 하드웨어 업그레이드가 이루어질 때마다 자동 회귀 시험(Regression Test), 지속적 통합(Continuous Integration), 버전 비교(Version Comparison), 성능 추이 분석(Performance Trend Analysis)을 수행해야 한다. 이를 통해 새로운 기능이 기존 성능을 저하시켰는지 여부를 지속적으로 확인할 수 있다.

산업 현장에서는 맞춤형 평가 프로토콜이 제품 인증(Product Qualification), 고객 인수 시험(Customer Acceptance Test), 규제 인증(Regulatory Certification), 품질 보증(Quality Assurance), 운영 준비도 평가(Operational Readiness Assessment)에 직접 활용된다. 따라서 이러한 평가 체계는 단순한 연구용 벤치마크가 아니라 제품 개발과 유지보수 전 과정에서 활용되는 핵심 엔지니어링 자산이라고 할 수 있다.

궁극적으로 맞춤형 작업 평가 프로토콜(Custom Task Evaluation Protocol)은 공개 벤치마크와 실제 산업 적용을 연결하는 핵심적인 다리 역할을 수행한다. 공개 벤치마크가 범용 로봇 지능을 평가한다면, 맞춤형 평가 프로토콜은 특정 산업 환경에서 요구되는 실제 성능을 검증한다. 작업 분해(Task Decomposition), 현실적인 시나리오(Realistic Scenario), 멀티모달 평가(Multimodal Evaluation), 언어 이해(Language Understanding), 강건성(Robustness), 장기 임무(Long-Horizon Mission), 통계 분석(Statistical Analysis), 사람-로봇 상호작용(Human-Robot Interaction), 안전성(Safety), 지속적 평가(Continuous Evaluation)를 통합함으로써 VLA 시스템이 실제 현장에서 신뢰성 있게 동작할 수 있는지를 종합적으로 검증하는 가장 중요한 평가 방법론이 될 것이다.

## 11.8 Robustness Evaluation (with Code)

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

비전-언어-행동(Vision-Language-Action, VLA) 시스템에서 강건성 평가(Robustness Evaluation)는 실제 환경에서의 신뢰성을 검증하기 위한 가장 중요한 평가 항목 가운데 하나이다. 대부분의 연구용 데이터셋은 일정한 조명(Lighting), 깨끗한 배경(Clean Background), 규칙적인 물체 배치(Object Arrangement)를 사용하지만, 실제 환경에서는 조명 변화, 복잡한 배경(Clutter), 부분 가림(Occlusion), 이동 장애물(Dynamic Obstacle), 센서 노이즈(Sensor Noise)와 같은 다양한 외란이 지속적으로 발생한다. 따라서 VLA 평가 체계는 이러한 환경 변화 속에서도 안정적으로 동작하는지를 체계적으로 검증해야 하며, 이를 통해 실제 배포 가능성(Deployment Readiness)을 판단할 수 있다.

강건성 평가의 핵심 목적은 모델이 단순한 시각적 특징(Visual Feature)을 암기한 것이 아니라 안정적인 의미 표현(Semantic Representation)을 학습했는지를 확인하는 것이다. 제한된 환경에서만 학습한 모델은 벤치마크에서는 높은 성능을 보이더라도 새로운 조명이나 복잡한 환경에서는 급격히 성능이 저하될 수 있다. 이는 모델이 물체의 본질적인 의미가 아니라 외형만 학습했음을 의미한다. 따라서 강건성 평가는 인식(Perception), 언어 접지(Language Grounding), 계획(Planning), 행동 생성(Action Generation)이 환경 변화에도 얼마나 안정적으로 유지되는지를 측정한다.

조명 변화(Lighting Variation)는 로봇 인식에 가장 큰 영향을 주는 요소 가운데 하나이다. 공장, 물류센터, 병원, 연구실, 사무실, 가정, 농업 환경은 시간과 장소에 따라 조명이 지속적으로 변한다. 자연광은 시간과 계절, 날씨에 따라 변화하며, 실내에서는 조명의 밝기, 색온도(Color Temperature), 그림자, 반사광 등이 계속 달라진다. 우수한 VLA 모델은 이러한 변화에도 별도의 재학습이나 재보정(Recalibration) 없이 안정적으로 물체를 인식하고 작업을 수행할 수 있어야 한다.

평가에서는 다양한 조명 조건을 체계적으로 구성해야 한다. 밝은 조명(Bright Lighting), 어두운 조명(Dim Lighting), 역광(Backlighting), 측광(Side Lighting), 색상이 있는 조명(Colored Lighting), 혼합 광원(Mixed Lighting), 깜빡이는 조명(Flickering Light), 급격한 밝기 변화(Rapid Brightness Change) 등을 포함해야 한다. 단순히 평균 성능만 평가하는 것이 아니라 조명 조건이 어려워질수록 성능이 얼마나 감소하는지를 분석해야 한다. 이러한 성능 저하 곡선(Performance Degradation Curve)은 모델의 실제 안정성을 보여주는 중요한 지표이다.

그림자(Shadow)는 단순한 밝기 변화보다 더욱 복잡한 문제를 만든다. 그림자는 물체의 형태, 질감(Texture), 색상(Color), 경계(Boundary)를 변화시키며, 실제 존재하지 않는 구조를 생성하여 인식 알고리즘을 혼동시킬 수 있다. 사람이나 차량, 다른 로봇이 움직이면서 생성하는 동적 그림자(Dynamic Shadow)는 작업 중에도 지속적으로 변화한다. 따라서 강건성 평가는 로봇이 실제 물체와 그림자를 정확하게 구분하여 안정적으로 작업을 수행할 수 있는지를 확인해야 한다.

반사(Reflection)와 눈부심(Glare)도 실제 환경에서 매우 중요한 문제이다. 금속(Metal), 유리(Glass), 액체(Liquid), 광택이 있는 플라스틱(Glossy Plastic) 등은 강한 반사를 발생시킨다. 이로 인해 카메라는 과다 노출(Overexposure), 대비 감소(Low Contrast), 잘못된 물체 경계(False Boundary)를 생성할 수 있다. 강건성 평가는 다양한 반사 물체와 여러 관찰 각도(Viewpoint)를 이용하여 이러한 광학적 외란에서도 안정적으로 인식이 가능한지를 평가해야 한다.

색상 일관성(Color Consistency)은 언어 기반 조작(Language-Conditioned Manipulation)에서 매우 중요하다. "파란 상자를 집어라" 또는 "빨간 부품을 초록 상자 옆에 놓아라"와 같은 명령은 정확한 색상 인식을 요구한다. 그러나 조명 종류와 카메라 화이트 밸런스(White Balance), 노출(Exposure)에 따라 동일한 물체도 서로 다른 색으로 보일 수 있다. 따라서 강건성 평가는 다양한 조명 환경에서도 의미적인 색상(Color Semantics)을 안정적으로 유지할 수 있는지를 확인해야 한다.

카메라 노출(Camera Exposure) 변화도 중요한 평가 요소이다. 카메라는 주변 밝기에 따라 자동 노출(Auto Exposure)을 수행하며, 밝은 공간에서 어두운 공간으로 이동하거나 그 반대의 경우 일시적으로 과다 노출 또는 부족 노출(Underexposure)이 발생할 수 있다. 평가에서는 이러한 노출 변화를 의도적으로 포함하여 비전 인코더(Vision Encoder)가 환경 변화에도 안정적인 특징 표현(Feature Representation)을 유지하는지를 분석해야 한다.

복잡한 배경(Clutter)은 실제 로봇 환경에서 매우 일반적인 상황이다. 실제 작업 공간은 깨끗한 배경이 아니라 여러 개의 물체, 공구(Tool), 포장재(Packaging), 케이블(Cable), 문서(Document), 설비(Equipment), 사람 등이 함께 존재한다. 이러한 시각적 복잡성은 물체 인식과 언어 접지(Language Grounding)를 어렵게 만든다. 따라서 강건성 평가는 복잡한 작업 공간에서도 목표 물체를 정확하게 식별할 수 있는지를 검증해야 한다.

부분 가림(Object Occlusion)은 복잡한 환경에서 가장 자주 발생하는 문제이다. 목표 물체는 다른 물체나 사람, 장비에 의해 일부가 가려질 수 있으며, 항상 전체 형태를 볼 수 있는 것은 아니다. 우수한 VLA 모델은 일부만 보이는 물체도 문맥(Context), 기하학적 구조(Geometry), 의미 관계(Semantic Relationship)를 이용하여 정확하게 추론해야 한다. 따라서 평가에서는 다양한 가림 정도(Occlusion Level)를 적용하여 성능 변화를 측정한다.

배경 복잡도(Background Complexity)는 인식 성능에 직접적인 영향을 준다. 연구용 데이터셋은 단순한 배경을 사용하지만 실제 환경은 벽면, 선반(Shelf), 장비, 케이블, 기계, 사람 등이 함께 존재한다. 유사한 색상과 반복적인 패턴은 물체 분할(Segmentation)과 인식을 더욱 어렵게 만든다. 따라서 배경의 복잡성을 단계적으로 증가시키면서 모델의 안정성을 평가해야 한다.

장면 밀도(Scene Density)는 작업 공간 안에 존재하는 물체 수를 의미한다. 몇 개의 물체만 있는 환경과 물류 창고, 조립 라인, 실험실, 가정의 수납장처럼 많은 물체가 밀집된 환경은 난이도가 크게 다르다. 물체가 많아질수록 목표 물체를 찾기 어려워지고 조작 공간도 줄어든다. 따라서 장면 밀도를 변화시키면서 조작 성능이 얼마나 유지되는지를 평가하는 것이 중요하다.

복잡한 환경에서는 언어 접지(Language Grounding)의 중요성도 더욱 커진다. "파란 상자 옆의 드라이버를 집어라" 또는 "큰 박스 뒤에 있는 물체를 이동시켜라"와 같은 명령은 시각적 복잡성 속에서도 공간 관계와 물체 의미를 동시에 이해해야 한다. 따라서 강건성 평가는 복잡한 장면에서도 언어와 시각을 정확하게 결합하여 목표를 찾을 수 있는지를 함께 분석해야 한다.

시간적 강건성(Temporal Robustness)도 고려해야 한다. 실제 환경은 정적인 환경이 아니라 작업 중에도 사람이나 다른 로봇이 이동하고, 장비가 추가되며, 조명이 바뀌고, 작업 공간이 재배치된다. 따라서 평가는 실험 시작 전에만 환경을 변경하는 것이 아니라 작업 수행 중에도 지속적으로 환경을 변화시켜 모델의 적응 능력을 측정해야 한다.

강건성 평가는 시뮬레이션(Simulation)과 실제 로봇(Real Robot)을 함께 활용해야 한다. 시뮬레이션은 수천 가지의 조명과 배경 조건을 반복적으로 시험할 수 있지만, 실제 광학 특성이나 센서의 물리적 특성까지 완벽하게 재현할 수는 없다. 따라서 대규모 시뮬레이션 평가 후 실제 로봇을 이용한 검증을 수행하는 통합 평가 방식이 가장 효과적이다.

평가 지표(Evaluation Metrics)는 단순한 작업 성공률(Success Rate)보다 훨씬 다양해야 한다. 인식 정확도(Perception Accuracy), 물체 탐지 안정성(Object Detection Stability), 분할 일관성(Segmentation Consistency), 언어 접지 정확도(Language Grounding Precision), 조작 성공률(Manipulation Success), 복구 능력(Recovery Capability), 추론 지연(Inference Latency), 정책 신뢰도(Policy Confidence), 경로 품질(Trajectory Quality), 충돌 빈도(Collision Frequency), 파지 안정성(Grasp Stability), 환경 난이도에 따른 성능 저하 곡선(Performance Degradation Curve) 등을 함께 분석해야 한다.

실패 분석(Failure Analysis)은 강건성 평가에서 매우 중요한 과정이다. 실패가 발생하면 원인이 인식 오류(Perception Error), 언어 이해(Language Understanding Error), 계획 실패(Planning Failure), 조작 불안정(Manipulation Instability), 위치 오차(Localization Drift), 제어(Control Failure) 가운데 무엇인지 구분해야 한다. 이러한 분석은 향후 VLA 모델을 개선하기 위한 핵심 자료가 되며, 약한 부분을 정확하게 보완할 수 있도록 해준다.

파운데이션 모델 사전학습(Foundation Model Pretraining)은 강건성에 큰 영향을 미친다. 다양한 조명, 다양한 환경, 다양한 물체, 다양한 작업을 포함한 대규모 멀티모달 데이터(Multimodal Data)로 학습한 모델은 제한된 연구실 데이터만 사용한 모델보다 훨씬 높은 강건성을 보인다. 따라서 강건성 평가는 사전학습 데이터의 다양성과 표현 학습(Representation Learning)의 품질을 간접적으로 평가하는 중요한 기준이기도 하다.

산업 현장에서는 강건성이 무엇보다 중요하다. 제조 공장, 물류센터, 병원, 연구소, 농업, 건설 현장, 서비스 환경은 모두 조명과 작업 환경이 지속적으로 변화하며, 로봇은 장시간 사람의 개입 없이 작업해야 한다. 따라서 조명과 복잡한 배경에서의 강건성 평가는 실제 제품의 배포 준비도(Deployment Readiness)와 장기 운용 신뢰성(Long-Term Reliability)을 판단하는 핵심 지표가 된다.

궁극적으로 **VLA 강건성 평가(VLA Robustness Evaluation)**는 연구실 수준의 성능과 실제 현장에서의 자율성 사이의 차이를 검증하는 가장 중요한 평가 방법이다. 조명 변화(Lighting Variation), 그림자(Shadow), 반사(Reflection), 노출 변화(Exposure Adaptation), 복잡한 배경(Clutter), 부분 가림(Occlusion), 배경 복잡도(Background Complexity), 장면 밀도(Scene Density), 언어 접지(Language Grounding), 시간적 적응(Temporal Adaptation), 실패 분석(Failure Analysis), 실제 로봇 검증(Real Robot Validation)을 하나의 통합된 평가 체계에서 분석함으로써 VLA 시스템의 신뢰성과 안정성을 종합적으로 평가할 수 있다. 앞으로 범용 물리 AI(Physical AI)가 다양한 실제 환경으로 확산될수록 이러한 강건성 평가는 안전하고 신뢰할 수 있는 자율 로봇을 개발하기 위한 필수적인 국제 평가 기준으로 자리매김할 것이다.

## 11.9 Human Evaluation Protocols

![](images/image9.png){width="7.268055555555556in" height="7.268055555555556in"}

사람 평가(Human Evaluation)는 비전-언어-행동(Vision-Language-Action, VLA) 시스템을 평가하는 데 있어 점점 더 중요한 요소가 되고 있다. 기존의 로봇 평가는 작업 성공률(Task Completion Rate), 조작 정확도(Manipulation Precision), 이동 경로(Trajectory Accuracy), 수행 시간(Execution Time)과 같은 정량적인 지표를 중심으로 이루어졌다. 그러나 이러한 수치만으로는 로봇이 사람의 의도를 얼마나 정확하게 이해하고, 자연스럽고 안전하게 행동하며, 사용자가 만족할 만한 결과를 제공하는지를 충분히 평가하기 어렵다. 따라서 사람 평가는 객관적인 벤치마크를 보완하여 실제 사용자의 관점에서 로봇의 행동 품질을 평가하는 핵심 요소로 활용된다.

사람 평가의 가장 중요한 목적은 로봇의 행동이 단순히 작업을 완료하는 것이 아니라 사용자의 의도(User Intention)에 얼마나 부합하는지를 확인하는 것이다. 예를 들어 로봇이 작업 자체는 성공적으로 수행했더라도 불필요하게 위험한 움직임을 보이거나, 비효율적인 경로를 선택하거나, 사람의 상식(Common Sense)에 어긋나는 행동을 한다면 실제 사용자 만족도는 낮아질 수 있다. 반대로 작은 오차가 있더라도 사람의 의도를 정확히 이해하고 자연스럽게 행동하는 로봇은 실제 환경에서 더 높은 평가를 받을 수 있다.

명령 수행(Instruction Following)은 정확한 언어 이해(Language Understanding)에서 시작된다. 실제 사용자는 연구용 데이터셋처럼 완전한 문장을 사용하지 않는다. 일상 대화에서는 생략된 표현, 모호한 지시, 대화체 표현, 이전 문맥(Context), 전문 용어(Domain-Specific Terminology)를 자연스럽게 사용한다. 따라서 사람 평가는 로봇이 이러한 자연스러운 언어를 별도의 형식화 없이도 올바르게 이해할 수 있는지를 평가한다. 이는 실제 VLA 시스템의 활용성을 결정하는 매우 중요한 요소이다.

의미 이해(Semantic Understanding)는 단순히 단어를 인식하는 수준을 넘어선다. 사람은 동일한 의미를 다양한 표현으로 전달한다. 예를 들어 "컵을 접시 옆에 놓아라", "머그컵을 접시 가까이에 두어라", "커피잔을 접시 옆으로 이동시켜라"는 모두 같은 의미를 가진다. 사람 평가는 로봇이 이러한 다양한 문장 표현을 동일한 작업 목표로 이해하는지를 평가한다. 이는 단순한 키워드 검색이 아니라 의미 자체를 이해하는 능력을 검증하는 과정이다.

문맥 인식(Context Awareness)은 사람 평가의 또 다른 핵심 요소이다. 사람은 "다른 공구를 가져와", "원래 위치로 다시 놓아", "방금 사용한 것을 치워"와 같이 이전 대화나 주변 환경을 전제로 하는 명령을 자주 사용한다. 이러한 명령을 수행하기 위해서는 로봇이 시각 정보(Vision), 작업 이력(Task History), 물체 정보(Object Identity), 현재 상황(Context)을 종합적으로 이해해야 한다. 사람 평가는 이러한 문맥 기반 추론 능력을 중점적으로 평가한다.

평가에서는 최종 결과뿐 아니라 작업 수행 과정도 중요하게 평가한다. 예를 들어 작업 공간을 정리하라는 명령을 받으면 로봇은 필요한 물체를 구분하고, 불필요한 이동을 최소화하며, 기존 정리 상태를 유지하고, 안전하게 조작해야 한다. 따라서 사람 평가는 작업 완료 여부뿐 아니라 계획 과정(Planning Process), 절차적 추론(Procedural Reasoning), 행동의 일관성(Behavioral Consistency)까지 함께 고려한다.

행동의 자연스러움(Naturalness)은 사람 중심 평가에서 매우 중요한 요소이다. 작업을 성공했더라도 움직임이 부자연스럽거나, 지나치게 느리거나, 복잡하거나, 예측하기 어렵다면 사람은 로봇을 지능적이라고 느끼지 않는다. 부드러운 이동(Smooth Motion), 안정적인 조작(Stable Manipulation), 효율적인 경로(Path Efficiency), 이해하기 쉬운 행동 순서(Logical Action Sequence)는 사람의 신뢰와 만족도를 크게 높인다. 이러한 요소는 기존의 정량적 지표만으로는 평가하기 어렵다.

안전성에 대한 사람의 인식(Safety Perception)도 중요한 평가 항목이다. 공학적인 안전 기준은 충돌, 속도, 비상 정지 등을 객관적으로 측정하지만, 사람은 로봇의 움직임이 예측 가능하고, 신중하며, 편안하게 느껴지는지도 함께 판단한다. 갑작스러운 가속, 사람 가까이에서의 빠른 움직임, 공격적으로 보이는 동작은 실제 위험이 없더라도 불안감을 줄 수 있다. 따라서 사람 평가는 객관적인 안전성과 함께 주관적인 신뢰감(Trust)도 함께 측정한다.

설명 가능성(Explainability)과 투명성(Transparency)은 최근 VLA 시스템에서 매우 중요한 평가 요소가 되었다. 사용자는 로봇이 왜 특정 행동을 선택했는지를 알고 싶어 한다. 특히 예상과 다른 행동을 수행할 경우에는 이유를 설명하거나, 명령이 모호할 경우에는 추가 질문을 하는 능력이 필요하다. 사람 평가는 로봇이 자신의 판단 과정을 이해하기 쉽게 설명하고, 불확실성을 적절히 표현하며, 잘못된 명령은 확인하는지를 평가한다. 이러한 기능은 사람과 로봇 사이의 신뢰를 크게 향상시킨다.

오류 복구(Error Recovery) 능력도 중요한 평가 요소이다. 실제 환경에서는 인식 오류, 조작 실패, 물체 이동, 예기치 않은 상황이 자주 발생한다. 사람 평가는 로봇이 이러한 실패를 스스로 감지하고, 필요한 경우 사용자에게 질문하거나, 작업을 다시 계획하여 자연스럽게 복구하는지를 평가한다. 실제 사용자는 완벽한 성공보다 실수를 얼마나 잘 복구하는지를 더 중요하게 생각하는 경우도 많다.

사람과의 대화 품질(Conversational Interaction)도 함께 평가된다. 최신 VLA 시스템은 단순히 명령을 수행하는 것이 아니라 진행 상황을 알려주고, 모호한 명령을 확인하며, 작업 지연이나 문제를 설명할 수 있다. 사람 평가는 이러한 대화가 이해하기 쉽고, 간결하며, 상황에 적절하고, 사회적으로 자연스러운지를 평가한다. 이러한 상호작용 능력은 협업 환경에서 매우 중요한 요소이다.

반복적인 상호작용에서의 일관성(Consistency)도 중요한 평가 항목이다. 동일한 명령과 동일한 환경에서는 비슷한 행동을 수행해야 사용자는 로봇을 신뢰할 수 있다. 같은 상황에서 매번 다른 행동을 보이면 예측 가능성이 낮아지고 신뢰도도 감소한다. 사람 평가는 반복된 실험에서 정책(Policy)이 안정적으로 동작하는지를 확인하며, 동시에 환경 변화에는 적절히 적응하는지도 함께 평가한다.

일반화(Generalization)는 새로운 명령과 새로운 환경에서도 평가된다. 평가자는 기존에 학습하지 않은 문장 표현, 새로운 물체 조합, 새로운 공간 구성, 새로운 상황을 제시한다. 로봇은 이를 암기한 것이 아니라 의미를 이해하여 대응해야 한다. 이러한 사람 평가는 로봇 파운데이션 모델(Robot Foundation Model)이 실제로 추상적인 개념(Abstract Concept)을 학습했는지를 확인하는 중요한 방법이다.

사람 평가는 여러 명의 평가자(Multiple Evaluators)가 함께 수행하는 것이 일반적이다. 사람마다 행동의 자연스러움, 안전성, 효율성에 대한 판단 기준이 다르기 때문에 여러 평가자의 의견을 종합하여 편향(Bias)을 줄인다. 서로 다른 문화적 배경, 경험, 전문성을 가진 평가자들의 결과를 통계적으로 분석하면 보다 객관적이고 신뢰성 높은 평가를 수행할 수 있다.

평가의 재현성을 높이기 위해 구조화된 평가 기준(Structured Evaluation Rubric)을 사용한다. 일반적으로 언어 이해(Language Understanding), 의미 정확성(Semantic Correctness), 작업 성공(Task Completion), 계획 품질(Planning Quality), 움직임 자연스러움(Motion Naturalness), 안전성(Safety), 의사소통(Communication), 강건성(Robustness), 오류 복구(Recovery Capability), 협업 품질(Collaboration Quality), 사용자 만족도(User Satisfaction), 실제 배포 가능성(Deployment Readiness) 등을 일정한 척도(Rating Scale)로 평가한다. 또한 평가자의 의견을 서술형으로 함께 기록하여 개선 방향을 분석한다.

사람 평가는 자동 평가(Automated Evaluation)를 대체하는 것이 아니라 상호 보완한다. 자동 평가는 조작 정확도, 이동 경로, 추론 시간과 같은 정량적 성능을 정확하게 측정할 수 있지만, 행동의 자연스러움이나 사용자 만족도는 평가하기 어렵다. 반대로 사람 평가는 주관적이지만 실제 사용 경험을 반영할 수 있다. 따라서 두 가지 평가를 함께 수행하면 VLA 시스템의 성능을 훨씬 종합적으로 분석할 수 있다.

사람 평가는 VLA 시스템의 지속적인 개선에도 중요한 역할을 한다. 평가 과정에서 수집된 피드백은 언어 이해 오류, 부자연스러운 행동, 의사소통 문제, 계획 실패 등을 발견하는 데 활용된다. 개발자는 이러한 정보를 바탕으로 언어 접지(Language Grounding), 멀티모달 추론(Multimodal Reasoning), 정책 최적화(Policy Optimization), 안전 전략(Safety Strategy), 상호작용 설계(Interaction Design)를 지속적으로 개선할 수 있다. 따라서 사람 평가는 개발 과정 전체에서 반복적으로 수행되는 중요한 절차이다.

산업 현장에서는 사람 중심 평가의 중요성이 더욱 커지고 있다. 물류센터, 병원, 제조 공장, 연구소, 매장, 사무실, 가정에서는 로봇이 사람과 함께 작업해야 한다. 따라서 다양한 사용자의 명령을 이해하고, 직관적이며 신뢰할 수 있는 행동을 수행하는 능력이 필수적이다. 사람 평가는 실제 상용화 이전에 이러한 협업 능력을 검증하는 가장 중요한 과정으로 활용된다.

최근 대규모 로봇 파운데이션 모델(Robot Foundation Model)의 발전으로 사람 평가의 중요성은 더욱 높아지고 있다. 최신 멀티모달 모델은 뛰어난 언어 이해 능력을 보이지만, 실제 행동에서는 의미 접지(Language Grounding) 오류나 문맥(Context) 해석 오류로 인해 잘못된 행동을 생성하는 경우가 있다. 사람은 이러한 미묘한 오류를 쉽게 발견할 수 있으므로, 사람 평가는 현재까지도 VLA 시스템을 평가하는 가장 신뢰할 수 있는 방법 가운데 하나로 인정받고 있다.

궁극적으로 **VLA 명령 수행을 위한 사람 평가(Human Evaluation for VLA Instruction Following)**는 단순한 작업 성공률을 넘어 사람과 로봇의 실제 협업 품질을 평가하는 핵심 방법론이다. 언어 이해(Language Understanding), 의미 추론(Semantic Reasoning), 문맥 인식(Context Awareness), 행동 자연스러움(Naturalness), 안전성(Safety), 설명 가능성(Explainability), 오류 복구(Error Recovery), 협업(Collaboration), 사용자 만족도(User Satisfaction)를 하나의 통합된 평가 체계에서 분석함으로써 VLA 시스템이 실제 사람과 함께 안전하고 효율적으로 작업할 수 있는지를 종합적으로 검증할 수 있다. 앞으로 구현형 인공지능(Embodied AI)이 다양한 산업과 일상생활로 확산될수록 사람 중심 평가는 신뢰할 수 있는 자율 로봇 개발을 위한 가장 중요한 국제 평가 기준 가운데 하나가 될 것이다.

## 11.10 Benchmark Leaderboards and Model Comparison

![](images/image10.png){width="7.268055555555556in" height="7.268055555555556in"}

비전-언어-행동(Vision-Language-Action, VLA) 시스템이 범용 구현형 인공지능(Embodied AI)으로 발전하면서 벤치마크 리더보드(Benchmark Leaderboard)는 연구 성과를 객관적으로 비교하는 핵심 도구가 되었다. 리더보드는 동일한 평가 기준을 적용하여 다양한 로봇 파운데이션 모델(Robot Foundation Model)의 성능을 비교하고 순위를 제공한다. 이를 통해 연구자와 기업은 각 모델의 강점과 약점을 객관적으로 분석할 수 있으며, 실제 산업 현장에서 사용할 수 있는 배포 준비도(Deployment Readiness)까지 평가할 수 있다.

VLA 리더보드의 가장 중요한 목적은 공정하고 재현 가능한(Fair and Reproducible) 비교를 제공하는 것이다. VLA 모델은 구조(Architecture), 학습 방식(Training Method), 파운데이션 모델(Foundation Model), 데이터셋(Dataset), 정책 표현(Policy Representation), 계산 복잡도(Computational Complexity) 등이 서로 다르다. 이러한 차이 때문에 동일한 기준이 없다면 성능 비교가 어렵다. 리더보드는 동일한 데이터셋과 동일한 평가 프로토콜(Evaluation Protocol)을 사용함으로써 연구 결과를 객관적으로 비교할 수 있도록 한다.

현대의 VLA 리더보드는 하나의 벤치마크만 사용하는 것이 아니라 여러 개의 대표적인 벤치마크를 함께 활용한다. RLBench는 다양한 조작 능력을 평가하고, LIBERO는 연속학습(Continual Learning)을, Language Table은 자연어 기반 조작(Language-Conditioned Manipulation)을, SimplerEnv는 실제 환경 기반 강건성(Robustness)을, DROID는 대규모 실제 환경 일반화(Real-World Generalization)를 평가한다. 또한 산업용 맞춤형 벤치마크(Custom Benchmark)를 함께 활용하여 실제 응용 환경에서의 성능도 검증한다.

리더보드는 하나의 점수만 제공하는 것이 아니라 다양한 성능 요소를 동시에 평가해야 한다. 기본적으로 작업 성공률(Task Completion Rate)을 포함하지만, 조작 정확도(Manipulation Precision), 이동 경로 효율(Path Efficiency), 언어 이해(Language Understanding), 의미 접지(Language Grounding), 공간 추론(Spatial Reasoning), 계획 안정성(Planning Consistency), 오류 복구(Recovery Behavior), 환경 변화에 대한 강건성(Robustness), 일반화(Generalization), 추론 지연(Inference Latency), 계산 효율(Computational Efficiency), 에너지 소비(Energy Consumption), 장기 작업(Long-Horizon Execution) 신뢰성 등을 함께 분석해야 한다.

일반화(Generalization)는 최신 VLA 모델을 평가하는 가장 중요한 항목 가운데 하나이다. 기존의 로봇은 학습한 작업만 수행할 수 있었지만, 최신 파운데이션 모델은 새로운 환경에서도 동작할 수 있어야 한다. 따라서 리더보드는 학습에 포함되지 않은 새로운 물체(Unseen Object), 새로운 환경(New Environment), 새로운 자연어 명령(New Instruction), 새로운 공간 구조(New Spatial Configuration), 새로운 작업 조합(New Task Combination)을 이용하여 제로샷(Zero-Shot) 및 퓨샷(Few-Shot) 성능을 함께 평가한다.

언어 이해(Language Understanding) 역시 중요한 비교 요소이다. VLA 시스템은 다양한 문장 구조, 동의어(Synonym), 문맥(Context), 대화체 표현(Conversational Expression)을 이해해야 한다. 리더보드는 명령 수행 정확도(Instruction Following Accuracy), 의미 일관성(Semantic Consistency), 언어 접지(Language Grounding), 문맥 추론(Contextual Reasoning), 모호성 해결(Ambiguity Resolution), 멀티모달 정렬(Multimodal Alignment) 등을 평가하여 실제 사람과의 상호작용 능력을 측정한다.

강건성(Robustness)은 실제 환경 적용을 위해 반드시 필요한 평가 항목이다. 연구실에서는 높은 성능을 보이더라도 실제 환경에서는 조명 변화(Lighting Variation), 복잡한 배경(Clutter), 센서 노이즈(Sensor Noise), 이동 장애물(Dynamic Obstacle), 부분 가림(Occlusion), 통신 지연(Communication Delay), 구동기 오차(Actuator Variation) 등에 의해 성능이 크게 저하될 수 있다. 따라서 리더보드는 이러한 환경 변화 속에서도 안정적으로 성능을 유지하는지를 평가한다.

장기 작업(Long-Horizon Planning) 능력도 최근 매우 중요한 비교 요소가 되었다. 산업용 검사, 물류 자동화, 연구실 자동화, 의료 서비스, 가정용 로봇은 수십 단계 이상의 연속적인 작업을 수행해야 한다. 따라서 리더보드는 장기 계획(Long-Term Planning), 시간적 일관성(Temporal Consistency), 누적 오차(Cumulative Error), 중간 목표 유지(Intermediate Goal Maintenance), 장시간 정책 안정성(Policy Stability)을 함께 평가하여 실제 운영 능력을 분석한다.

추론 효율(Inference Efficiency)은 실제 제품 개발에서 점점 더 중요한 요소가 되고 있다. 대규모 파운데이션 모델은 매우 높은 계산 자원을 요구하기 때문에 임베디드(Embedded) 로봇에서는 실행하기 어려운 경우가 많다. 따라서 리더보드는 추론 시간(Inference Latency), 처리량(Throughput), GPU 메모리 사용량(GPU Memory Usage), 에너지 소비(Energy Consumption), 모델 크기(Parameter Count), 엣지 컴퓨팅(Edge Computing) 적합성 등을 함께 공개하여 실제 적용 가능성을 평가한다.

파운데이션 모델의 확장성(Scalability)도 비교 대상이 된다. 일부 모델은 수십억 개의 파라미터(Billion Parameters)를 사용하지만, 일부 모델은 임베디드 환경을 위해 소형 구조를 사용한다. 따라서 단순한 정확도뿐 아니라 파라미터당 성능(Performance per Parameter), 전력당 성능(Performance per Watt), 샘플 효율(Sample Efficiency), 계산 비용(Computational Cost) 등을 함께 공개하여 알고리즘 자체의 효율성을 평가하는 것이 중요하다.

시뮬레이션-실환경 전이(Simulation-to-Real Transfer)는 실제 배포를 위한 핵심 비교 요소이다. 리더보드는 시뮬레이션에서 학습한 정책이 실제 로봇에서도 별도의 재학습 없이 얼마나 잘 동작하는지를 측정한다. 전이 효율(Transfer Efficiency), 환경 적응성(Adaptation Capability), 실제 조작 정확도(Physical Manipulation Accuracy), 인식 안정성(Perception Stability), 물리적 상호작용 신뢰성(Physical Interaction Reliability) 등을 평가하여 실제 활용 가능성을 분석한다.

최근에는 사람 평가(Human Evaluation)도 리더보드에 함께 포함되고 있다. 자동 평가는 재현성이 뛰어나지만, 행동의 자연스러움(Naturalness), 명령 이해 품질(Instruction Understanding), 안전성(Safety Perception), 의사소통 품질(Communication Quality), 설명 가능성(Explainability), 사용자 만족도(User Satisfaction)는 사람만이 평가할 수 있다. 따라서 정량 평가와 사람 평가를 함께 수행하면 실제 VLA 시스템의 성능을 더욱 정확하게 분석할 수 있다.

통계적 신뢰성(Statistical Reliability)은 리더보드의 중요한 요소이다. 단 한 번의 성공적인 실험이 아니라 반복 실험을 수행하여 평균 성능(Mean), 분산(Variance), 신뢰구간(Confidence Interval), 성공률 분포(Success Distribution), 강건성 곡선(Robustness Curve), 통계적 유의성(Statistical Significance)을 함께 제공해야 한다. 이러한 정보는 모델 간의 차이가 실제로 의미 있는지를 객관적으로 판단하는 근거가 된다.

실패 분석(Failure Analysis)은 단순한 순위보다 더욱 중요한 정보를 제공한다. 리더보드는 인식 오류(Perception Error), 언어 접지 실패(Language Grounding Failure), 계획 오류(Planning Error), 조작 실패(Manipulation Failure), 위치 추정 오차(Localization Drift), 복구 실패(Recovery Failure), 안전 위반(Safety Violation) 등을 함께 분석한다. 이를 통해 연구자는 모델의 약점을 정확하게 파악하고 개선 방향을 결정할 수 있으며, 기업은 자신의 응용 분야에 적합한 모델을 선택할 수 있다.

공개 리더보드(Open Benchmark Leaderboard)는 연구 재현성(Reproducibility)을 크게 향상시킨다. 데이터셋, 모델 구조, 학습 방법, 하이퍼파라미터(Hyperparameter), 평가 환경을 공개함으로써 전 세계 연구자들이 동일한 조건에서 성능을 비교할 수 있다. 이러한 공개 문화는 연구의 투명성을 높이고 VLA 분야 전체의 기술 발전을 가속화하는 중요한 역할을 한다.

산업 현장에서는 리더보드가 기술 선정(Technology Selection)의 중요한 기준으로 활용된다. 물류 자동화, 제조 로봇, 의료 로봇, 검사 로봇, 농업 로봇, 협동 로봇, 서비스 로봇을 도입하는 기업은 실제 구매 전에 객관적인 성능 자료가 필요하다. 리더보드는 다양한 모델의 장단점을 비교하여 가장 적합한 VLA 시스템을 선택할 수 있도록 지원하는 중요한 의사결정 도구가 된다.

그러나 리더보드는 로봇 지능 전체를 완벽하게 평가하는 것은 아니다. 특정 벤치마크에서 높은 성능을 얻기 위해 과도하게 최적화된 모델은 실제 산업 환경에서는 기대만큼의 성능을 보이지 않을 수도 있다. 따라서 공개 벤치마크 결과와 함께 실제 응용 분야에 맞는 맞춤형 평가(Custom Evaluation)를 병행하는 것이 중요하다. 두 가지 평가를 함께 활용해야 실제 활용 가능성을 정확하게 판단할 수 있다.

미래의 VLA 리더보드는 더욱 다양한 능력을 평가하게 될 것으로 예상된다. 멀티모달 추론(Multimodal Reasoning), 평생학습(Lifelong Learning), 적응형 계획(Adaptive Planning), 협업(Collaboration), 자율 탐사(Autonomous Exploration), 장기 자율성(Long-Term Autonomy), 안전 인증(Safety Certification), 설명 가능성(Explainability), 에너지 효율(Energy Efficiency), 실제 환경 강건성(Real-World Robustness) 등을 종합적으로 평가하는 다차원 리더보드(Multi-Dimensional Leaderboard)가 등장할 것으로 전망된다.

또한 정적인 평가가 아니라 지속적 벤치마킹(Continuous Benchmarking)도 중요한 방향이 될 것이다. 파운데이션 모델은 지속적으로 업데이트되고 데이터셋도 계속 확장된다. 따라서 새로운 버전이 출시될 때마다 자동으로 평가하고 이전 버전과 성능을 비교하는 지속적인 리더보드가 구축될 것이다. 이러한 시스템은 지속적 통합(Continuous Integration), 자동 회귀 시험(Regression Test), 장기 성능 모니터링(Long-Term Performance Monitoring)과 자연스럽게 연계될 수 있다.

궁극적으로 **VLA 벤치마크 리더보드 및 모델 비교(VLA Benchmark Leaderboard and Model Comparison)**는 단순한 순위표가 아니라 구현형 인공지능(Embodied AI)의 발전을 측정하는 핵심 인프라이다. 표준화된 평가 프로토콜(Standardized Evaluation Protocol), 다양한 벤치마크(Benchmark Suite), 강건성(Robustness), 일반화(Generalization), 사람 평가(Human Evaluation), 추론 효율(Inference Efficiency), 통계 분석(Statistical Validation), 실패 분석(Failure Analysis), 시뮬레이션-실환경 전이(Simulation-to-Real Transfer), 실제 배포 준비도(Deployment Readiness)를 하나의 통합 체계에서 비교함으로써 VLA 시스템의 실제 활용 가능성을 객관적으로 평가할 수 있다. 앞으로 범용 물리 AI(Physical AI)가 발전할수록 이러한 리더보드는 연구와 산업을 연결하는 가장 중요한 국제 평가 기준으로 지속적으로 활용될 것이다.
