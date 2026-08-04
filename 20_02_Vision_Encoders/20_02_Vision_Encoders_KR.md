**Volume 20. Vision Language Action (VLA) Models**

# Chapter 2. Vision Encoders

## 2.1 Role of Vision Encoders in VLA Feature Extraction

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

비전-언어-행동(Vision-Language-Action, VLA) 시스템에서 비전 인코더(Vision Encoder)는 로봇의 \'눈(Eyes)\' 역할을 수행하는 가장 중요한 구성 요소이다. 사람의 시각 피질(Visual Cortex)이 빛을 의미 있는 정보로 변환하여 주변 환경을 이해하듯이, 비전 인코더는 카메라와 다양한 센서로부터 입력된 원시 데이터(Raw Data)를 인공지능이 이해할 수 있는 특징 표현(Feature Representation)으로 변환한다. 아무리 뛰어난 언어 모델(Language Model)이나 행동 정책(Action Policy)을 사용하더라도, 환경을 올바르게 인식하지 못하면 정확한 판단과 행동은 불가능하다. 따라서 비전 인코더는 VLA 전체 지능의 출발점이며, 이후의 추론과 행동 생성 품질을 결정하는 핵심 기술이다.

현대의 로봇에서는 비전 인식이 독립적인 컴퓨터 비전(Computer Vision) 문제가 아니라 통합된 다중모달(Multimodal) 지능의 첫 번째 단계로 간주된다. 비전 인코더는 하나 이상의 센서로부터 입력된 영상을 분석하여 의미적 정보(Semantic Information)와 공간 정보(Spatial Information)를 포함하는 잠재 표현(Latent Representation)을 생성한다. 이러한 특징 벡터는 이후 언어(Language), 고유감각(Proprioception), 메모리(Memory), 작업 목표(Task Goal)와 결합되어 최종적으로 행동(Action)을 생성하는 기반이 된다.

초기의 로봇 비전 시스템은 대부분 사람이 직접 설계한 특징 추출(Handcrafted Feature Extraction)에 의존하였다. 에지 검출기(Edge Detector), 코너 검출기(Corner Detector), 색상 히스토그램(Color Histogram), SIFT, SURF, ORB, HOG와 같은 기법이 대표적이다. 이러한 알고리즘은 계산량이 적고 해석이 쉬웠지만, 조명 변화(Illumination Variation), 시점 변화(Viewpoint Change), 물체 가림(Occlusion), 물체 변형(Object Deformation)과 같은 실제 환경 변화에 매우 취약하였다. 따라서 실험실에서는 잘 동작하더라도 실제 산업 현장에서는 성능이 크게 저하되는 경우가 많았다.

딥러닝(Deep Learning)의 등장으로 비전 기술은 근본적으로 변화하였다. 합성곱 신경망(Convolutional Neural Network, CNN)은 사람이 직접 특징을 설계하지 않고, 데이터를 이용하여 계층적 특징(Hierarchical Feature)을 자동으로 학습하였다. 초기 계층은 에지와 질감을 학습하고, 중간 계층은 물체의 일부 구조를 이해하며, 깊은 계층에서는 완전한 물체와 장면(Scene)의 의미를 이해한다. 이러한 계층적 표현은 기존 수작업 특징보다 훨씬 높은 일반화 성능(Generalization)을 제공하였다.

그러나 CNN 역시 한계를 가지고 있었다. 합성곱은 기본적으로 지역적(Local) 연산이기 때문에 이미지 전체의 관계(Global Context)를 이해하려면 매우 깊은 네트워크가 필요하다. 하지만 실제 로봇은 멀리 떨어진 물체 간의 관계나 장면 전체의 구조를 이해해야 하는 경우가 많다. 따라서 전역적인 문맥(Global Context)을 보다 효율적으로 처리할 수 있는 새로운 구조가 필요하게 되었다.

이러한 요구를 해결하기 위해 등장한 것이 비전 트랜스포머(Vision Transformer, ViT)이다. ViT는 이미지를 작은 패치(Image Patch) 단위로 분할한 뒤 자기 어텐션(Self-Attention)을 이용하여 모든 패치 간의 관계를 동시에 계산한다. 이 방식은 CNN보다 훨씬 효과적으로 장면 전체를 이해할 수 있으며, 물체 간의 거리, 위치, 공간적 관계를 표현하는 데 매우 유리하다. 특히 조작(Manipulation), 자율주행(Navigation), 휴머노이드(Humanoid)와 같이 복잡한 환경을 이해해야 하는 VLA에서 매우 중요한 기술로 자리잡았다.

비전 인코더의 가장 중요한 목적은 단순히 이미지를 압축하는 것이 아니라 의미를 유지하면서 차원을 축소(Dimensionality Reduction)하는 것이다. 원시 영상에는 조명 변화, 센서 노이즈(Sensor Noise), 배경(Background), 불필요한 질감(Texture) 등 작업과 무관한 정보도 매우 많이 포함되어 있다. 비전 인코더는 이러한 불필요한 요소를 제거하면서 물체의 의미, 공간 구조, 형태, 위치, 관계와 같은 중요한 정보만을 특징 벡터로 추출한다.

효과적인 특징 추출은 단순한 압축이 아니라 의미 보존(Semantic Preservation)을 목표로 한다. 예를 들어 같은 컵(Cup)을 서로 다른 조명이나 각도에서 촬영하더라도 비슷한 특징 벡터를 생성해야 한다. 반대로 외형은 비슷하지만 기능적으로 다른 물체는 명확하게 구분되어야 한다. 이러한 특징 공간(Feature Space)을 만드는 것이 현대 비전 인코더 설계의 핵심 목표이다.

현대 VLA에서는 비전 인코더가 객체 이름이나 분할 결과를 직접 출력하는 것이 아니라, 고차원 특징 토큰(Feature Token)을 생성하는 것이 일반적이다. 이러한 토큰은 물체의 의미와 공간 구조를 모두 포함하고 있으며, 이후 트랜스포머가 현재 작업 목적에 맞게 해석한다. 따라서 동일한 특징 표현은 객체 인식(Object Recognition), 장면 이해(Scene Understanding), 조작 계획(Manipulation Planning), 자율주행(Navigation), 이상 탐지(Anomaly Detection) 등 다양한 작업에서 동시에 활용될 수 있다.

최근의 비전 인코더는 RGB 영상뿐 아니라 다양한 센서를 함께 처리한다. 깊이 카메라(Depth Camera)는 거리 정보를 제공하고, 열화상 카메라(Thermal Camera)는 온도 분포를 제공하며, 이벤트 카메라(Event Camera)는 빠른 움직임을 정확하게 측정한다. 또한 라이다(LiDAR), 레이더(Radar), 다중분광(Multispectral), 초분광(Hyperspectral) 센서까지 함께 활용되면서 로봇은 사람보다 더 풍부한 환경 정보를 인식할 수 있게 되었다.

다중모달 융합(Multimodal Fusion)은 이러한 다양한 센서 정보를 하나의 특징 공간으로 통합하는 기술이다. 융합은 원시 데이터 단계(Early Fusion), 특징 단계(Feature-Level Fusion), 의사결정 단계(Late Fusion)에서 수행될 수 있으며, 센서 종류와 응용 분야에 따라 적절한 구조를 선택한다. 최종 목표는 개별 센서보다 훨씬 풍부하고 신뢰성 높은 환경 표현(Environment Representation)을 생성하는 것이다.

공간 이해(Spatial Understanding)는 비전 인코더의 가장 중요한 기능 가운데 하나이다. 단순히 물체를 인식하는 것만으로는 로봇이 작업할 수 없다. 물체가 어디에 있는지, 어떤 방향을 향하고 있는지, 접근 가능한지, 집을 수 있는지 등을 함께 이해해야 한다. 따라서 특징 표현에는 위치(Position), 자세(Pose), 크기(Scale), 깊이(Depth), 표면 형상(Surface Geometry), 가림 관계(Occlusion), 이동 가능 공간(Free Space), 조작 가능성(Affordance) 등이 함께 포함된다.

최근에는 3차원 이해(3D Understanding)의 중요성이 더욱 커지고 있다. 단일 RGB 영상만으로는 정확한 거리와 공간 구조를 알기 어렵기 때문이다. 따라서 스테레오 비전(Stereo Vision), 깊이 추정(Depth Estimation), 포인트 클라우드(Point Cloud), 신경 방사장(Neural Radiance Field, NeRF), 가우시안 스플래팅(Gaussian Splatting) 등 다양한 기술이 비전 인코더에 적용되고 있으며, 더욱 정밀한 공간 표현을 가능하게 하고 있다.

시간적 이해(Temporal Understanding) 역시 중요한 기능이다. 실제 로봇은 정지된 사진이 아니라 연속적인 영상(Video)을 처리한다. 움직이는 사람, 이동하는 물체, 로봇 자신의 움직임은 모두 시간 정보(Time Information)를 포함한다. 따라서 최근 비전 인코더는 개별 이미지가 아니라 영상 시퀀스(Video Sequence)를 입력으로 받아 시간적 특징(Temporal Feature)을 함께 학습하는 방향으로 발전하고 있다.

움직임(Motion)은 로봇에게 매우 중요한 정보이다. 사람의 이동 방향을 예측하여 충돌을 방지할 수 있으며, 물체의 움직임을 이용하여 작업 진행 상황(Task Progress)을 판단할 수도 있다. 또한 일시적으로 물체가 가려지더라도 이전 프레임의 정보를 기억하여 작업을 계속 수행할 수 있다. 이러한 시간 기반 표현은 장시간 작업(Long-Horizon Task)에서 매우 중요한 역할을 한다.

비전과 언어를 연결하는 시각적 그라운딩(Visual Grounding)은 VLA에서 핵심 기술이다. 사용자가 "빨간 컵을 집어라."라고 말하면 비전 인코더는 영상 속의 빨간 컵을 언어 표현과 연결해야 한다. 이를 위해 이미지와 텍스트가 동일한 의미 공간(Shared Embedding Space)에 표현되도록 학습한다. 이러한 구조 덕분에 학습하지 않은 새로운 물체도 언어 설명만으로 인식할 수 있게 된다.

이러한 교차모달 정렬(Cross-Modal Alignment)은 현대 다중모달 AI의 가장 중요한 기술 가운데 하나이다. 컵 이미지는 "Cup"이라는 텍스트와 가까운 위치에 표현되고, 의자는 "Chair"라는 텍스트와 가까운 특징 공간을 형성한다. 따라서 언어 모델과 비전 모델은 동일한 의미 공간을 공유하면서 자연스럽게 상호작용할 수 있다.

최근에는 범용 비전 기반 모델(Vision Foundation Model)이 매우 큰 성공을 거두고 있다. 인터넷 규모의 데이터셋에서 사전학습된 모델은 로봇 데이터가 거의 없어도 뛰어난 일반화 능력을 제공한다. 따라서 대부분의 VLA는 이러한 모델을 초기화(Initialization)한 후, 로봇 환경에 맞게 미세조정(Fine-Tuning), 어댑터(Adapter), 프롬프트 튜닝(Prompt Tuning), 저랭크 적응(Low-Rank Adaptation, LoRA) 등을 적용한다.

전이학습(Transfer Learning)이 가능한 이유는 대부분의 시각적 개념이 공통적으로 존재하기 때문이다. 인터넷 이미지에서 학습한 에지, 질감, 형태, 물체 구조는 산업용 로봇, 서비스 로봇, 자율주행차에서도 동일하게 나타난다. 따라서 처음부터 모든 것을 학습하는 것보다 기존 표현을 활용하는 것이 훨씬 효율적이다.

그러나 로봇 환경은 일반 이미지와 상당히 다르다. 손목 카메라(Wrist Camera), 이동 카메라(Mobile Camera), 심한 모션 블러(Motion Blur), 다양한 조명, 작업 공간의 복잡한 배경 등이 존재하기 때문에 추가적인 도메인 적응(Domain Adaptation)이 필요하다. 이를 통해 일반 비전 모델을 실제 로봇 환경에 최적화할 수 있다.

데이터 증강(Data Augmentation)은 강인한 비전 인코더를 만드는 중요한 기술이다. 회전(Rotation), 크기 변경(Scaling), 색상 변화(Color Jitter), 조명 변화, 노이즈 추가(Noise Injection), 가림(Occlusion), 기하학적 왜곡(Geometric Distortion), 도메인 랜덤화(Domain Randomization) 등을 이용하여 다양한 환경을 학습시키면 실제 환경에서도 높은 일반화 성능을 얻을 수 있다.

최근에는 자기지도학습(Self-Supervised Learning)이 매우 중요해지고 있다. 로봇 데이터를 모두 사람이 라벨링(Labeling)하는 것은 비용이 매우 크기 때문이다. 자기지도학습은 이미지 자체의 구조를 이용하여 의미 있는 특징을 자동으로 학습하며, 대규모 비라벨(Unlabeled) 데이터를 효과적으로 활용할 수 있다. 대조학습(Contrastive Learning), 마스킹 이미지 모델링(Masked Image Modeling), 예측 부호화(Predictive Coding) 등이 대표적인 방법이다.

비전 인코더의 계산 효율성(Computational Efficiency)은 실제 로봇에서 매우 중요한 요소이다. 여러 대의 카메라와 고해상도 영상을 동시에 처리하면 계산량이 급격히 증가한다. 그러나 이동 로봇은 전력(Power), 발열(Thermal), 지연(Latency)에 제한이 있기 때문에 경량 트랜스포머(Lightweight Transformer), 토큰 프루닝(Token Pruning), 동적 해상도(Dynamic Resolution), 하드웨어 최적화(Hardware-Aware Optimization) 등이 함께 사용된다.

메모리 사용량(Memory Consumption) 역시 중요한 설계 요소이다. 다중 카메라와 긴 영상 시퀀스를 처리하면 GPU 메모리 사용량이 급격히 증가한다. 이를 해결하기 위해 혼합 정밀도(Mixed Precision), 특징 압축(Feature Compression), 메모리 효율 어텐션(Memory-Efficient Attention), 양자화(Quantization) 등이 널리 적용되고 있다.

설명 가능성(Interpretability)은 비전 인코더 연구의 중요한 분야이다. 특징 벡터 내부에 어떤 정보가 저장되어 있는지 이해하기 위해 어텐션 맵(Attention Map), 특징 시각화(Feature Visualization), 활성화 분석(Activation Analysis), 특징 역변환(Feature Inversion) 등이 사용된다. 이러한 분석은 모델의 디버깅(Debugging), 안전성(Safety), 신뢰성(Reliability)을 높이는 데 중요한 역할을 한다.

비전 인코더의 평가는 단순한 이미지 분류 정확도로 이루어지지 않는다. 객체 위치 추정(Object Localization), 조작 성공률(Manipulation Success), 자율주행 성능(Navigation Performance), 공간 추론(Spatial Reasoning), 언어 그라운딩(Language Grounding), 계산 속도(Computational Efficiency), 일반화(Generalization), 강인성(Robustness) 등을 종합적으로 평가한다. 결국 비전 인코더의 품질은 최종 로봇 성능으로 평가된다.

크로스 엠바디먼트(Cross-Embodiment) 전이도 중요한 연구 분야이다. 하나의 비전 인코더가 매니퓰레이터(Manipulator), AMR, 사족보행 로봇(Quadruped), 드론(Drone), 휴머노이드(Humanoid) 등 다양한 플랫폼에서 동일하게 활용될 수 있어야 한다. 이를 통해 로봇 하드웨어가 달라져도 동일한 지능을 재사용할 수 있다.

안전(Safety) 역시 비전 인코더와 밀접한 관련이 있다. 잘못된 인식은 곧 잘못된 행동으로 이어질 수 있기 때문에 최신 비전 인코더는 신뢰도 추정(Confidence Estimation), 이상 탐지(Anomaly Detection), 분포 외 탐지(Out-of-Distribution Detection), 센서 일관성 검사(Sensor Consistency Check)를 함께 수행하여 위험 상황을 조기에 발견하도록 설계된다.

지속적 학습(Continual Learning)은 미래 비전 인코더의 중요한 방향이다. 실제 환경은 계속 변화하며 새로운 물체와 새로운 작업이 지속적으로 등장한다. 따라서 비전 인코더는 기존 지식을 잊지 않으면서(Catastrophic Forgetting 방지) 새로운 환경을 계속 학습할 수 있어야 한다. 이는 장기적으로 자율적으로 성장하는 물리 인공지능(Physical AI)의 핵심 기술이 될 것이다.

앞으로의 비전 인코더는 단순히 이미지를 특징 벡터로 변환하는 수준을 넘어, 비전(Vision), 언어(Language), 메모리(Memory), 물리 법칙(Physics), 상호작용(Interaction), 예측(Prediction)을 통합한 월드 모델(World Model)을 구축하는 방향으로 발전할 것이다. 이러한 세계 표현은 장기 계획(Long-Horizon Planning), 인과 추론(Causal Reasoning), 협업(Collaboration), 자율 탐사(Autonomous Exploration), 범용 조작(General Manipulation)을 지원하는 핵심 기반이 된다.

결국 비전 인코더는 현실 세계(Physical World)와 인공지능(Artificial Intelligence)을 연결하는 가장 중요한 관문이다. 원시 센서 데이터를 의미 있는 특징 표현으로 변환함으로써 객체 인식(Object Recognition), 공간 이해(Spatial Understanding), 언어 그라운딩(Language Grounding), 환경 예측(Environment Prediction), 추론(Reasoning), 행동 생성(Action Generation)의 기반을 제공한다. 미래의 VLA와 물리 인공지능(Physical AI)이 더욱 높은 수준의 지능을 갖추기 위해서는 강인하고 효율적이며 범용적인 비전 인코더의 발전이 무엇보다 중요한 핵심 요소가 될 것이다.

## 2.2 Vision Transformer (ViT) Architecture and Variants (with Code)

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

비전 트랜스포머(Vision Transformer, ViT)는 현대 컴퓨터 비전(Computer Vision)의 가장 중요한 발전 가운데 하나이며, 오늘날 비전-언어-행동(Vision-Language-Action, VLA) 시스템의 핵심 구성 요소로 자리 잡았다. ViT는 자연어 처리(Natural Language Processing)의 트랜스포머(Transformer) 구조를 영상 처리에 적용한 모델로, 기존의 합성곱 신경망(Convolutional Neural Network, CNN) 중심 구조를 근본적으로 변화시켰다. 이미지를 단순한 픽셀 집합이 아니라 시각 토큰(Visual Token)의 시퀀스(Sequence)로 처리함으로써 언어 모델과 동일한 방식의 표현 학습과 추론이 가능해졌다.

ViT의 등장은 단순한 모델 개선이 아니라 영상 인식 방식 자체의 패러다임 전환이었다. 기존 CNN은 국소 영역(Local Region)의 특징을 단계적으로 확장하면서 장면을 이해했지만, ViT는 자기 어텐션(Self-Attention)을 이용하여 모든 이미지 영역이 서로 직접 정보를 교환하도록 설계되었다. 따라서 이미지 전체의 문맥(Global Context)을 초기 단계부터 동시에 이해할 수 있으며, 복잡한 장면을 처리해야 하는 로봇 환경에서 매우 뛰어난 성능을 보인다.

VLA 시스템에서 ViT는 가장 중요한 비전 인코더(Vision Encoder) 역할을 수행한다. 카메라로부터 입력된 영상을 의미 있는 특징 표현(Feature Representation)으로 변환한 뒤, 이를 언어(Language), 로봇 상태(Proprioception), 메모리(Memory), 행동 정책(Action Policy)과 결합하여 최종 행동을 생성한다. 따라서 비전 표현의 품질은 이후의 추론(Reasoning), 계획(Planning), 행동(Action) 성능을 직접적으로 결정하며, 대부분의 최신 로봇 기반 모델은 ViT를 기본 백본(Backbone)으로 채택하고 있다.

ViT의 핵심 아이디어는 매우 단순하면서도 강력하다. 입력 이미지를 일정한 크기의 패치(Image Patch)로 나누고, 각 패치를 하나의 토큰(Token)으로 변환한다. 이후 각 패치는 선형 투영(Linear Projection)을 통해 고차원 임베딩(Embedding)으로 변환되며, 자연어 처리에서 단어(Token)를 처리하는 것과 동일한 방식으로 트랜스포머에 입력된다. 즉, ViT는 이미지를 하나의 문장처럼 이해하는 구조라고 볼 수 있다.

패치 임베딩(Patch Embedding)은 ViT의 첫 번째 단계이다. 예를 들어 224×224 크기의 이미지를 16×16 패치로 나누면 총 196개의 패치가 생성된다. 각 패치는 독립적인 시각 토큰으로 변환되며, 이후 모든 토큰은 동일한 트랜스포머 구조에서 처리된다. 패치 크기는 매우 중요한 설계 변수이다. 작은 패치는 세밀한 정보를 유지하지만 계산량이 증가하고, 큰 패치는 계산량은 줄지만 세부 구조가 손실될 수 있다. 따라서 로봇 조작과 같이 정밀한 작업에서는 적절한 패치 크기 선택이 매우 중요하다.

위치 임베딩(Positional Embedding)은 ViT에서 반드시 필요한 요소이다. 트랜스포머는 입력 순서를 스스로 알 수 없기 때문에 각 패치의 위치 정보를 별도로 추가해야 한다. 이를 통해 모델은 어떤 패치가 위쪽에 있는지, 어느 패치가 서로 인접한지를 이해하게 된다. 위치 임베딩이 없으면 이미지의 공간적 구조를 학습할 수 없기 때문에 시각적 의미를 제대로 이해하기 어렵다.

패치 임베딩 이후에는 여러 개의 트랜스포머 인코더(Transformer Encoder)가 반복적으로 적용된다. 각 인코더는 멀티헤드 자기 어텐션(Multi-Head Self-Attention), 피드포워드 네트워크(Feed Forward Network), 잔차 연결(Residual Connection), 계층 정규화(Layer Normalization)로 구성된다. 이러한 구조는 깊은 네트워크에서도 안정적인 학습을 가능하게 하며, 점진적으로 더 높은 수준의 의미 표현을 생성한다.

멀티헤드 자기 어텐션(Multi-Head Self-Attention)은 ViT의 핵심 기술이다. 기존 CNN에서는 서로 멀리 떨어진 영역이 정보를 교환하기 위해 많은 계층을 거쳐야 했지만, ViT에서는 모든 패치가 서로 직접 관계를 계산한다. 예를 들어 로봇 그리퍼(Gripper)가 이미지의 한쪽 끝에 있고 목표 물체가 반대편에 있더라도 즉시 관계를 파악할 수 있다. 이러한 전역 문맥 이해(Global Context Understanding)는 로봇 작업에서 매우 큰 장점을 제공한다.

멀티헤드(Multi-Head) 구조는 다양한 정보를 동시에 학습하도록 만든다. 일부 어텐션 헤드(Attention Head)는 물체의 경계를 학습하고, 다른 헤드는 질감(Texture), 의미(Semantics), 공간 구조(Spatial Structure), 물체 간 관계(Relationship)를 학습한다. 여러 헤드가 서로 다른 관점을 동시에 분석함으로써 훨씬 풍부한 시각 표현을 생성할 수 있다.

피드포워드 네트워크(Feed Forward Network)는 어텐션 이후 각 토큰을 비선형적으로 변환한다. 자기 어텐션이 정보의 흐름을 결정한다면, 피드포워드 네트워크는 각 토큰의 의미를 더욱 풍부하게 확장하는 역할을 수행한다. 또한 잔차 연결과 계층 정규화는 깊은 모델에서도 안정적인 학습과 빠른 수렴을 가능하게 한다.

CNN과 가장 큰 차이점은 ViT에는 지역성(Locality)이나 이동 불변성(Translation Invariance)에 대한 사전 가정(Inductive Bias)이 없다는 것이다. 이러한 특성은 모두 데이터로부터 학습된다. 초기에는 이러한 구조가 성능이 떨어질 것으로 예상되었지만, 대규모 데이터셋에서 충분히 학습할 경우 CNN보다 더 높은 성능을 달성한다는 것이 입증되었다. 이 발견은 컴퓨터 비전 분야의 연구 방향을 완전히 바꾸는 계기가 되었다.

ViT는 대규모 사전학습(Large-Scale Pretraining)에 크게 의존한다. CNN은 비교적 적은 데이터에서도 잘 학습되지만, ViT는 내재된 구조적 가정이 적기 때문에 충분한 데이터가 필요하다. 따라서 일반적으로 수백만에서 수십억 장의 이미지로 먼저 사전학습한 후, 특정 로봇 작업에 맞게 미세조정(Fine-Tuning)하여 사용한다.

전이학습(Transfer Learning)은 ViT가 로보틱스에서 성공한 가장 큰 이유 가운데 하나이다. 인터넷 규모의 이미지 데이터에는 다양한 물체(Object), 재질(Material), 질감(Texture), 조명(Lighting), 시점(Viewpoint)이 포함되어 있다. 이러한 다양한 경험을 학습한 ViT는 산업용 검사, 물류 자동화, 자율주행, 서비스 로봇, 농업 로봇, 휴머노이드 등 매우 다양한 분야에 쉽게 적용될 수 있다.

ViT의 특징 추출 방식은 기존 객체 인식과 다르다. 과거에는 객체 분류(Classification)나 검출(Detection)이 목적이었다면, ViT는 다양한 작업에서 공통으로 사용할 수 있는 고차원 잠재 표현(Latent Embedding)을 생성한다. 동일한 특징 표현은 물체 인식(Object Recognition), 장면 이해(Scene Understanding), 조작 계획(Manipulation Planning), 자율주행(Navigation), 이상 탐지(Anomaly Detection) 등 여러 작업에 동시에 활용될 수 있다.

공간 추론(Spatial Reasoning)은 ViT의 가장 큰 장점 가운데 하나이다. 모든 패치가 서로 연결되어 있기 때문에 물체 간 거리, 장애물, 작업 공간, 이동 가능 영역 등을 동시에 이해할 수 있다. 특히 복잡한 작업대 위에서 여러 개의 물체를 다루는 로봇은 이러한 전역적인 공간 이해 능력을 통해 더욱 안정적인 작업을 수행할 수 있다.

하지만 자기 어텐션(Self-Attention)은 계산량이 매우 크다는 단점도 가지고 있다. 모든 패치가 서로 계산하기 때문에 계산 복잡도는 토큰 수의 제곱(Quadratic Complexity)에 비례한다. 이미지 해상도가 높아질수록 연산량과 메모리 사용량이 급격히 증가하므로 실제 로봇에서는 이를 해결하기 위한 다양한 개선 구조가 개발되고 있다.

이를 해결하기 위해 다양한 ViT 변형 모델(Variant)이 등장하였다. 대표적으로 계층적 비전 트랜스포머(Hierarchical Vision Transformer)는 토큰을 단계적으로 병합하여 계산량을 줄이면서 다중 해상도(Multi-Scale) 정보를 유지한다. 이는 기존 CNN의 피처 피라미드(Feature Pyramid)와 유사한 효과를 제공하면서도 트랜스포머의 장점을 그대로 유지한다.

윈도우 기반 어텐션(Window-Based Attention)도 중요한 개선 방식이다. 이미지 전체가 아니라 작은 윈도우(Window) 안에서만 자기 어텐션을 수행하고, 일정 주기마다 윈도우를 이동(Shifted Window)시켜 전체 정보를 연결한다. 이를 통해 계산량을 크게 줄이면서도 전역 문맥을 효과적으로 유지할 수 있다.

하이브리드 구조(Hybrid Architecture)는 CNN과 ViT를 함께 사용하는 방식이다. 초기 계층에서는 CNN이 에지와 질감을 효율적으로 추출하고, 이후에는 ViT가 전체 문맥을 이해한다. 이러한 구조는 계산 효율성과 표현 능력을 동시에 확보할 수 있기 때문에 실제 산업용 로봇에서 많이 활용되고 있다.

계층적 ViT(Hierarchical ViT)는 로봇 비전에 매우 적합하다. 초기 계층에서는 정밀한 공간 정보를 유지하고, 깊은 계층에서는 점차 의미 중심의 특징을 생성한다. 이러한 다중 해상도(Multi-Scale) 표현은 객체 검출(Object Detection), 의미 분할(Semantic Segmentation), 깊이 추정(Depth Estimation), 위치 추정(Localization) 등에 효과적으로 활용된다.

자기지도학습(Self-Supervised Learning)은 ViT 발전에 큰 영향을 주었다. 사람이 직접 라벨링하지 않아도 이미지 자체의 구조를 이용하여 특징을 학습할 수 있기 때문이다. 마스킹 이미지 모델링(Masked Image Modeling), 대조학습(Contrastive Learning), 예측 부호화(Predictive Coding), 지식 증류(Knowledge Distillation) 등이 대표적인 학습 방법이며, 이를 통해 대규모 비라벨 데이터(Unlabeled Data)를 효과적으로 활용할 수 있다.

대조학습(Contrastive Learning)은 같은 이미지의 다양한 변형은 가까운 특징 공간에, 다른 이미지는 먼 공간에 위치하도록 학습한다. 이러한 방식은 조명 변화, 시점 변화, 색상 변화에도 동일한 의미를 유지하는 강인한 특징 표현(Robust Feature Representation)을 생성하며, 실제 로봇 환경에서 매우 높은 일반화 성능을 제공한다.

마스킹 이미지 모델링(Masked Image Modeling)은 일부 패치를 가리고 이를 복원하도록 학습한다. 이를 위해 모델은 주변 문맥(Context)을 이해해야 하므로 단순한 픽셀 복원이 아니라 장면의 의미(Semantics)를 학습하게 된다. 이러한 방식은 로봇 작업에서도 매우 뛰어난 전이학습 성능을 제공한다.

지식 증류(Knowledge Distillation)는 대형 ViT의 성능을 작은 모델로 전달하는 기술이다. 이를 통해 모바일 로봇(Mobile Robot), 드론(Drone), 협동로봇(Cobot)과 같이 계산 자원이 제한된 엣지 장치(Edge Device)에서도 ViT를 사용할 수 있게 되었다.

ViT는 다중모달(Multimodal) AI와도 매우 잘 결합된다. 이미지 패치와 언어 토큰이 모두 동일한 토큰(Token) 구조를 가지므로 비전과 언어를 동일한 트랜스포머 안에서 처리할 수 있다. 이러한 구조 덕분에 CLIP, PaLM-E, LLaVA, GPT-4V, Gemini, Qwen-VL, InternVL, RT-2와 같은 대부분의 최신 비전-언어 모델(Vision-Language Model, VLM)이 ViT를 비전 백본으로 사용하고 있다.

VLA에서는 ViT가 RGB 카메라, 손목 카메라(Wrist Camera), 스테레오 카메라(Stereo Camera), 다중 카메라(Multi-Camera)의 영상을 통합하여 하나의 시각 표현을 생성한다. 이후 언어, 로봇 상태, 힘 센서(Force Sensor), 메모리와 결합하여 행동 생성(Action Generation)의 입력으로 사용된다.

멀티카메라(Multi-Camera) 환경에서도 ViT는 뛰어난 성능을 보인다. 헤드 카메라, 손목 카메라, 측면 카메라, 깊이 센서 등 여러 시점의 정보를 자기 어텐션으로 자연스럽게 융합하여 하나의 장면(Scene)으로 이해할 수 있다. 이는 기존의 복잡한 센서 융합 알고리즘보다 훨씬 유연한 구조를 제공한다.

시간 정보를 처리하는 비디오 트랜스포머(Video Transformer)는 공간뿐 아니라 시간(Temporal Dimension)까지 함께 학습한다. 사람의 움직임, 물체의 이동, 작업 진행 과정 등을 이해할 수 있으며, 장시간 작업(Long-Horizon Task)을 수행하는 VLA에서 중요한 역할을 한다.

최근에는 3차원 비전 트랜스포머(3D Vision Transformer)도 활발히 연구되고 있다. 포인트 클라우드(Point Cloud), 복셀(Voxel), 신경 방사장(Neural Radiance Field, NeRF), RGB-D 데이터를 직접 처리하여 실제 공간과 더욱 가까운 표현을 생성한다. 이러한 기술은 조작, 자율주행, 산업 검사 등에서 매우 중요한 역할을 수행한다.

설명 가능성(Interpretability)은 ViT 연구의 중요한 주제이다. 어텐션 맵(Attention Map), 토큰 유사도(Token Similarity), 특징 시각화(Feature Visualization) 등을 이용하면 모델이 어떤 영역에 주목하는지 분석할 수 있다. 이는 모델 디버깅(Debugging), 안전성(Safety), 신뢰성(Reliability)을 향상시키는 데 큰 도움이 된다.

실제 로봇에서는 지연 시간(Latency), 메모리(Memory), 소비 전력(Power), 발열(Thermal) 제약이 존재한다. 따라서 양자화(Quantization), 프루닝(Pruning), 토큰 감소(Token Reduction), 혼합 정밀도(Mixed Precision), 컴파일러 최적화(Compiler Optimization), 전용 AI 가속기(AI Accelerator)를 이용하여 ViT를 실시간으로 실행할 수 있도록 최적화한다.

미래의 ViT는 단순히 이미지를 인코딩하는 수준을 넘어, 시간(Time), 공간(Space), 메모리(Memory), 물리 법칙(Physics), 상호작용(Interaction)을 통합한 월드 모델(World Model)로 발전할 것으로 예상된다. 이를 통해 인과 추론(Causal Reasoning), 장기 계획(Long-Horizon Planning), 지속적 학습(Continual Learning), 자율 탐사(Autonomous Exploration)를 수행하는 진정한 물리 인공지능(Physical AI)의 핵심 기술이 될 것이다.

결국 비전 트랜스포머(Vision Transformer)는 기존 CNN 기반 국소 특징 추출(Local Feature Extraction)을 전역 문맥 기반 의미 표현(Global Semantic Representation)으로 발전시킨 혁신적인 기술이다. 자기 어텐션(Self-Attention), 대규모 사전학습(Large-Scale Pretraining), 전이학습(Transfer Learning), 다중모달 융합(Multimodal Fusion), 다양한 경량화 구조(Variant)를 통해 오늘날 대부분의 VLA와 비전-언어 모델(VLM)의 핵심 비전 백본으로 자리 잡았다. 앞으로도 ViT와 그 후속 기술은 로봇이 현실 세계를 이해하고, 언어를 해석하며, 추론하고, 안전하게 행동하도록 만드는 가장 중요한 기반 기술로 지속적으로 발전할 것이다.

## 2.3 CLIP Visual Encoder for Language-Aligned Features (with Code)

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

CLIP(Contrastive Language-Image Pretraining)는 현대 다중모달 인공지능(Multimodal AI)의 발전을 이끈 가장 중요한 기술 가운데 하나이며, 비전-언어-행동(Vision-Language-Action, VLA) 시스템에서 비전과 언어를 연결하는 핵심 기반 모델이다. 기존의 컴퓨터 비전(Computer Vision)은 이미지 분류(Image Classification)만 수행하고, 자연어 처리(Natural Language Processing)는 텍스트만 이해하였다. CLIP은 이미지(Image)와 텍스트(Text)를 하나의 의미 공간(Shared Semantic Embedding Space)에서 동시에 학습함으로써, 로봇이 사람의 언어를 이해하고 이를 실제 시각 정보와 연결할 수 있도록 만들었다.

VLA에서 CLIP은 단순한 객체 인식(Object Recognition) 모델이 아니라 의미 중심의 비전 인코더(Vision Encoder) 역할을 수행한다. 기존 모델처럼 미리 정의된 객체(Object Category)만 인식하는 것이 아니라, 자연어 설명과 시각 정보를 동시에 이해한다. 따라서 로봇은 처음 보는 물체라도 언어 설명(Language Description)을 이용하여 인식할 수 있으며, 다양한 환경에서도 높은 일반화(Generalization) 성능을 유지할 수 있다.

CLIP이 등장하기 전의 지도학습(Supervised Learning) 기반 컴퓨터 비전은 사람이 직접 라벨(Label)을 붙인 데이터셋에 크게 의존하였다. 예를 들어 고양이(Cat), 자동차(Car), 의자(Chair)와 같이 정해진 클래스(Class)만 인식할 수 있었으며, 새로운 물체를 추가하려면 새로운 데이터를 수집하고 모델을 다시 학습해야 했다. 이러한 방식은 다양한 환경에서 동작해야 하는 범용 로봇(General-Purpose Robot)에는 매우 큰 제약이었다.

반면 인터넷에는 수많은 이미지와 설명(Caption)이 이미 존재한다. CLIP은 이러한 이미지-텍스트(Image-Text Pair)를 그대로 활용하여 학습한다. 수억 개 이상의 이미지와 문장을 동시에 학습하면서 특정 객체뿐 아니라 장소, 재질(Material), 색상(Color), 행동(Action), 직업(Profession), 환경(Environment) 등 매우 다양한 개념을 함께 이해하게 된다. 이러한 학습 방식은 기존 지도학습보다 훨씬 풍부한 의미 표현(Semantic Representation)을 생성한다.

CLIP의 핵심 아이디어는 두 개의 독립적인 인코더(Encoder)를 동시에 학습하는 것이다. 하나는 이미지를 처리하는 비전 인코더(Vision Encoder)이고, 다른 하나는 문장을 처리하는 언어 인코더(Language Encoder)이다. 동일한 이미지와 설명은 특징 공간에서 서로 가까워지도록 학습하고, 관련 없는 이미지와 문장은 서로 멀어지도록 학습한다. 이러한 대조학습(Contrastive Learning)을 반복하면서 이미지와 언어는 하나의 공통 의미 공간을 형성하게 된다.

대조학습(Contrastive Learning)은 기존 분류(Classification) 방식과 근본적으로 다르다. 기존에는 "이것은 고양이이다."처럼 정답(Label)을 맞추는 것이 목적이었다면, CLIP은 "이 이미지와 이 문장은 서로 얼마나 비슷한가?"를 학습한다. 따라서 모델은 특정 클래스만 기억하는 것이 아니라 의미적 유사성(Semantic Similarity)을 스스로 학습하게 되며, 새로운 개념도 자연스럽게 이해할 수 있다.

CLIP의 비전 인코더는 다양한 구조를 사용할 수 있다. 초기에는 합성곱 신경망(Convolutional Neural Network, CNN)과 비전 트랜스포머(Vision Transformer, ViT)를 모두 사용하였지만, 최근에는 대부분 ViT가 기본 구조로 사용된다. ViT는 이미지 전체의 문맥(Global Context)을 이해하는 능력이 뛰어나며, 언어 모델과 구조적으로도 매우 잘 맞기 때문에 현재 대부분의 VLA에서 표준 비전 인코더로 활용되고 있다.

언어 인코더(Language Encoder)는 트랜스포머(Transformer)를 기반으로 문장을 처리한다. 단순히 단어를 인식하는 것이 아니라 문장의 의미(Context)를 함께 이해한다. 예를 들어 "빨간 세라믹 컵이 나무 테이블 위에 있다."라는 문장은 컵(Cup), 색(Color), 재질(Material), 위치(Relation)를 모두 포함한 하나의 의미 벡터로 표현된다. 이러한 풍부한 언어 표현은 이미지와의 의미 정렬(Semantic Alignment)을 가능하게 한다.

학습이 완료되면 이미지와 텍스트는 동일한 잠재 공간(Latent Space)에 표현된다. 예를 들어 컵 사진과 "Cup"이라는 문장은 매우 가까운 위치에 존재하고, 자동차와 "Car" 역시 서로 가까운 특징 공간을 형성한다. 이러한 공통 임베딩(Shared Embedding)은 이미지와 언어를 직접 비교할 수 있게 하며, 다양한 응용 분야에서 매우 높은 활용성을 제공한다.

CLIP의 가장 큰 장점 가운데 하나는 제로샷 분류(Zero-Shot Classification)이다. 새로운 객체를 학습하지 않아도 텍스트만 입력하면 해당 객체를 인식할 수 있다. 예를 들어 "A red toolbox", "A fire extinguisher", "An emergency exit sign"과 같은 문장을 입력하면 모델은 이를 이미지와 직접 비교하여 가장 유사한 물체를 찾아낸다. 이러한 기능은 새로운 물체가 계속 등장하는 실제 로봇 환경에서 매우 큰 장점을 제공한다.

로보틱스에서는 제로샷 인식(Zero-Shot Recognition)의 가치가 매우 크다. 가정(Home), 공장(Factory), 병원(Hospital), 창고(Warehouse)에서는 새로운 물체가 지속적으로 등장한다. 기존 시스템은 새로운 물체마다 다시 학습해야 했지만, CLIP 기반 로봇은 사용자가 "파란 공구 상자를 집어라."와 같이 설명만 해도 해당 물체를 찾아낼 수 있다. 이는 범용 로봇(General Robot) 구현에 매우 중요한 기술이다.

CLIP은 단순히 객체를 구분하는 것이 아니라 의미적 관계(Semantic Relationship)도 이해한다. 의자(Chair), 소파(Sofa), 벤치(Bench)는 모두 앉는 가구라는 공통 의미를 가지므로 특징 공간에서도 서로 가까운 위치를 형성한다. 또한 주방(Kitchen), 병원(Hospital), 공장(Factory)과 같은 장면(Scene)도 각각 의미적으로 구분된다. 이러한 구조는 로봇의 고수준 추론(High-Level Reasoning)을 가능하게 한다.

시각적 그라운딩(Visual Grounding)은 CLIP이 제공하는 또 다른 핵심 기능이다. 사용자가 "공구함 옆의 초록색 상자를 가져와."라고 말하면, 로봇은 언어와 이미지가 동일한 특징 공간에 존재하기 때문에 해당 물체를 자연스럽게 찾을 수 있다. 별도의 객체 ID나 규칙을 만들 필요 없이 자연어만으로 물체를 지정할 수 있는 것이 CLIP의 큰 장점이다.

CLIP은 공간 추론(Spatial Reasoning)에도 간접적으로 기여한다. CLIP 자체는 거리나 자세를 직접 계산하지는 않지만, "컵은 보통 테이블 위에 있다.", "전자레인지는 주방에 있다."와 같은 의미적 지식을 학습한다. 이러한 의미 정보는 깊이 센서(Depth Sensor)나 3차원 비전(3D Vision)과 결합될 때 더욱 강력한 공간 이해 능력을 제공한다.

전이학습(Transfer Learning)은 CLIP의 가장 큰 장점 가운데 하나이다. 하나의 CLIP 모델은 객체 검출(Object Detection), 의미 분할(Semantic Segmentation), 조작(Manipulation), 자율주행(Navigation), 산업 검사(Industrial Inspection), 이상 탐지(Anomaly Detection), 인간-로봇 상호작용(Human-Robot Interaction) 등 매우 다양한 응용 분야에서 그대로 사용할 수 있다. 따라서 대부분의 VLA는 CLIP을 기반으로 추가 학습만 수행한다.

CLIP을 로봇에 적용하는 방법은 여러 가지가 있다. 모든 파라미터를 다시 학습하는 전체 미세조정(Full Fine-Tuning)도 가능하지만 계산량이 매우 크다. 최근에는 어댑터(Adapter), 프롬프트 튜닝(Prompt Tuning), 저랭크 적응(Low-Rank Adaptation, LoRA)과 같은 경량화 방법을 이용하여 적은 데이터만으로도 새로운 환경에 빠르게 적응하는 방식이 널리 사용된다.

프롬프트 엔지니어링(Prompt Engineering)도 CLIP 성능에 큰 영향을 준다. 단순히 "Cup"이라고 입력하는 것보다 "A photo of a ceramic coffee cup"처럼 자연스러운 문장을 사용할 경우 인식 성능이 더욱 높아진다. 최근에는 사람이 직접 문장을 작성하지 않고, 학습을 통해 최적의 프롬프트를 자동으로 생성하는 프롬프트 학습(Prompt Learning) 기술도 활발히 연구되고 있다.

로봇 조작(Robotic Manipulation)은 CLIP의 대표적인 응용 분야이다. 기존 시스템은 특정 물체만 집을 수 있었지만, CLIP은 "필기할 수 있는 물건을 가져와."와 같은 개념적 명령도 이해할 수 있다. 따라서 펜(Pen), 연필(Pencil), 마커(Marker)처럼 서로 다른 물체도 동일한 의미 그룹으로 인식하여 작업을 수행할 수 있다.

오픈 보캐뷸러리(Open-Vocabulary) 인식은 CLIP이 만든 새로운 패러다임이다. 기존 객체 검출기는 미리 정의된 클래스만 인식했지만, CLIP 기반 시스템은 사람이 언어로 표현할 수 있는 거의 모든 개념을 인식할 수 있다. 이러한 구조를 기반으로 최근에는 오픈 보캐뷸러리 객체 검출(Open-Vocabulary Object Detection)과 의미 분할(Open-Vocabulary Segmentation)이 활발히 연구되고 있다.

장면 이해(Scene Understanding) 역시 CLIP의 중요한 기능이다. 단순히 물체를 인식하는 것이 아니라 공장, 병원, 연구실, 창고, 사무실과 같은 환경 전체를 의미적으로 이해한다. 이러한 장면 정보는 자율주행, 작업 계획(Task Planning), 서비스 로봇(Service Robot)에서 매우 중요한 역할을 수행한다.

인간-로봇 상호작용(Human-Robot Interaction)은 CLIP 덕분에 더욱 자연스러워졌다. 사용자는 복잡한 명령어 대신 평범한 자연어(Natural Language)를 사용할 수 있으며, 로봇은 동의어(Synonym), 속성(Attribute), 설명(Context)을 모두 이해할 수 있다. 따라서 비전문가도 쉽게 로봇을 사용할 수 있게 되었다.

CLIP은 데이터셋 구축 방식도 변화시켰다. 과거에는 객체 이름만 라벨링했지만, 최근에는 자유 형식의 문장(Free-Form Caption)을 함께 저장한다. 이러한 언어 정보는 단순한 객체 분류를 넘어 의미 추론과 언어 기반 로봇 제어를 가능하게 한다.

그러나 CLIP에도 한계는 존재한다. 학습 데이터 대부분이 인터넷에서 수집되기 때문에 문화적 편향(Bias), 지역 편향(Regional Bias), 특정 직업이나 사물에 대한 데이터 불균형이 존재할 수 있다. 따라서 실제 산업용 로봇에서는 추가적인 데이터 수집과 검증을 통해 이러한 편향을 최소화해야 한다.

또한 CLIP은 의미 중심 모델이기 때문에 기하학적 정보(Geometric Information)는 상대적으로 부족하다. 물체의 거리, 자세(Pose), 깊이(Depth), 파지 가능성(Graspability) 등은 정확하게 표현하지 못한다. 따라서 실제 VLA에서는 RGB-D 카메라(RGB-D Camera), 포인트 클라우드(Point Cloud), 스테레오 비전(Stereo Vision) 등과 함께 사용하여 의미와 공간 정보를 동시에 확보한다.

시간적 이해(Temporal Understanding)도 CLIP의 한계 가운데 하나이다. 원래 CLIP은 정적인 이미지(Static Image)를 대상으로 설계되었기 때문에 움직임(Motion)이나 시간에 따른 변화(Time Series)는 직접 처리하지 못한다. 따라서 비디오 트랜스포머(Video Transformer)나 월드 모델(World Model)과 결합하여 장시간 작업(Long-Horizon Task)을 지원하는 방식으로 발전하고 있다.

실제 로봇에서는 계산 효율성(Computational Efficiency)도 중요하다. 원본 CLIP은 GPU 자원을 많이 사용하기 때문에 모바일 로봇(Mobile Robot)에서는 경량화 모델(Lightweight Model), 양자화(Quantization), 프루닝(Pruning), 지식 증류(Knowledge Distillation)를 적용하여 추론 속도와 전력 소비를 최적화한다.

CLIP은 현재 대부분의 비전-언어 모델(Vision-Language Model, VLM)의 기본 비전 인코더로 사용된다. 다양한 멀티모달 모델은 CLIP의 시각 특징을 언어 모델과 연결하여 대화형 AI, 로봇 계획(Robot Planning), 비전-언어-행동(VLA) 시스템을 구축한다. 즉, CLIP은 독립적인 모델이라기보다 현대 멀티모달 AI의 핵심 기반 기술이라고 할 수 있다.

VLA에서는 CLIP이 생성한 시각 특징을 로봇 상태(Proprioception), 힘 센서(Force Sensor), 언어(Language), 메모리(Memory), 환경 지도(Environment Map)와 융합한다. 이후 상위 추론 모델(Reasoning Model)이 이를 분석하여 행동 정책(Action Policy)을 생성하며, 최종적으로 로봇이 실제 동작을 수행하게 된다.

앞으로의 CLIP은 단순히 이미지와 언어를 연결하는 수준을 넘어, 3차원 공간(3D Geometry), 시간(Temporal Memory), 물리 법칙(Physics), 인과 추론(Causal Reasoning), 월드 모델(World Model)까지 통합하는 방향으로 발전할 것으로 예상된다. 이를 통해 로봇은 단순히 "무엇이 있는가"를 이해하는 수준을 넘어 "어떻게 움직이고", "어떻게 조작하며", "앞으로 어떻게 변화할 것인가"까지 예측할 수 있게 될 것이다.

결국 CLIP은 이미지와 언어를 하나의 의미 공간으로 통합한 최초의 대규모 성공 사례이며, 현대 비전-언어-행동(VLA) 시스템의 핵심 기반 기술이다. 대조학습(Contrastive Learning), 공통 임베딩 공간(Shared Embedding Space), 제로샷 학습(Zero-Shot Learning), 오픈 보캐뷸러리(Open Vocabulary), 뛰어난 전이학습(Transfer Learning)을 통해 로봇은 사람의 언어를 이해하고, 새로운 물체를 인식하며, 의미 기반 추론을 수행할 수 있게 되었다. 앞으로도 CLIP에서 시작된 언어 정렬(Language-Aligned) 비전 표현은 차세대 물리 인공지능(Physical AI)과 범용 로봇(General-Purpose Robot)을 실현하는 핵심 기술로 계속 발전할 것이다.

## 2.4 DINOv2 Self-Supervised Vision Encoder (with Code)

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

DINOv2는 자기지도학습(Self-Supervised Learning) 기반의 대표적인 비전 기반 모델(Vision Foundation Model)로, 현대 비전-언어-행동(Vision-Language-Action, VLA) 시스템에서 가장 강력한 비전 인코더(Vision Encoder) 가운데 하나로 평가받는다. 기존의 지도학습(Supervised Learning) 기반 모델이 사람이 직접 라벨(Label)을 붙인 데이터를 필요로 했다면, DINOv2는 라벨이 없는(Unlabeled) 대규모 이미지 데이터만으로도 의미 있는 시각 표현(Visual Representation)을 학습할 수 있다. 이는 데이터 수집 비용을 크게 줄이고, 다양한 환경에서 활용 가능한 범용 시각 지능(General Visual Intelligence)을 구축할 수 있도록 한다.

DINOv2가 등장한 배경에는 인간의 시각 학습 방식에 대한 중요한 통찰이 있다. 사람은 태어나면서 모든 사물의 이름을 배우는 것이 아니라, 주변 환경을 지속적으로 관찰하면서 물체의 형태, 질감, 구조, 관계를 스스로 학습한다. 마찬가지로 로봇도 공장, 병원, 창고, 농업, 서비스 환경 등에서 끊임없이 새로운 영상을 획득한다. DINOv2는 이러한 방대한 비라벨(Unlabeled) 데이터를 활용하여 의미 있는 특징을 자동으로 학습하는 구조를 제공하며, 범용 로봇 지능의 핵심 기술로 주목받고 있다.

기존 지도학습 기반 컴퓨터 비전은 뛰어난 성능을 보였지만 여러 한계가 존재하였다. 데이터에 사람이 직접 객체 이름과 라벨을 붙여야 하며, 산업 현장에서는 새로운 부품이나 장비가 지속적으로 등장하기 때문에 데이터셋을 계속 수정하고 다시 학습해야 한다. 또한 공장마다 사용하는 제품과 공정이 다르기 때문에 공개 데이터셋만으로는 실제 산업 환경을 충분히 반영하기 어렵다. 이러한 문제를 해결하기 위해 자기지도학습(Self-Supervised Learning)이 중요한 대안으로 떠오르게 되었다.

자기지도학습은 사람이 정답(Label)을 제공하지 않고, 데이터 자체에서 학습 목표(Learning Objective)를 생성하는 방식이다. 모델은 이미지 속의 구조와 패턴을 스스로 발견하며, 이를 통해 일반적인 시각 표현을 학습한다. 이후 이 표현은 객체 검출(Object Detection), 의미 분할(Semantic Segmentation), 자율주행(Navigation), 조작(Manipulation), 위치 추정(Localization), 이상 탐지(Anomaly Detection), 산업 검사(Industrial Inspection) 등 다양한 로봇 응용 분야에 활용될 수 있다.

DINO(Self-DIstillation with NO Labels)는 사람의 라벨 없이도 지식을 전달하는 자기 증류(Self-Distillation) 구조를 최초로 제안하였다. DINOv2는 이를 더욱 발전시켜 더 큰 데이터셋, 향상된 최적화 알고리즘(Optimization), 확장된 모델 구조(Architecture), 뛰어난 일반화 성능(Generalization)을 제공한다. 그 결과 다양한 비전 작업에서 기존 지도학습 모델에 필적하거나 이를 뛰어넘는 성능을 달성하였다.

DINOv2의 핵심 원리는 자기 증류(Self-Distillation)이다. 두 개의 동일한 신경망(Network)을 사용하며, 하나는 교사 모델(Teacher Network), 다른 하나는 학생 모델(Student Network) 역할을 수행한다. 두 모델은 동일한 이미지를 서로 다른 방식으로 변형(Augmentation)하여 입력받고, 학생 모델은 교사 모델이 생성한 특징 표현을 따라가도록 학습한다. 이 과정에서 사람의 라벨은 전혀 사용되지 않으며, 의미 있는 시각 표현이 자연스럽게 형성된다.

교사-학생(Teacher-Student) 구조는 DINOv2의 가장 중요한 특징이다. 교사 모델은 직접 학습되는 것이 아니라 학생 모델의 파라미터(Parameter)를 지수 이동 평균(Exponential Moving Average, EMA) 방식으로 천천히 반영한다. 따라서 교사 모델은 항상 안정적인 목표(Target)를 제공하며, 학생 모델은 이를 지속적으로 따라가면서 더욱 강인한 특징 표현을 학습하게 된다.

이미지 증강(Image Augmentation)은 DINOv2 학습에서 매우 중요한 역할을 한다. 동일한 이미지를 회전(Rotation), 확대 및 축소(Scaling), 색상 변화(Color Jitter), 밝기 변화(Brightness), 블러(Blur), 자르기(Cropping) 등 다양한 방식으로 변형한다. 비록 픽셀(Pixel)은 크게 달라지지만 의미는 동일하기 때문에, 모델은 조명이나 시점 변화와 무관하게 동일한 의미를 유지하는 특징을 학습하게 된다. 이러한 특성은 실제 로봇 환경에서 매우 높은 강인성(Robustness)을 제공한다.

기존의 대조학습(Contrastive Learning)은 서로 다른 이미지 간의 거리까지 함께 계산해야 했지만, DINOv2는 명시적인 음성 샘플(Negative Sample)을 사용하지 않는다. 대신 교사 모델이 생성한 특징을 학생 모델이 그대로 따라가도록 만드는 자기 증류 방식을 사용한다. 이 구조는 계산 효율성을 높이고, 더욱 안정적인 학습을 가능하게 하며, 우수한 시각 표현을 생성한다.

DINOv2는 비전 트랜스포머(Vision Transformer, ViT)를 기본 구조로 사용한다. ViT는 이미지 전체를 하나의 시각 토큰(Visual Token) 시퀀스로 처리하며, 자기 어텐션(Self-Attention)을 이용하여 장면 전체의 관계를 동시에 이해할 수 있다. 이러한 구조는 물체 간 관계, 장면 구성(Scene Composition), 공간 구조(Spatial Structure)를 효과적으로 학습할 수 있어 로봇 비전에서 매우 적합한 기반 모델로 활용된다.

DINOv2의 가장 큰 특징은 사람이 라벨을 제공하지 않았음에도 의미적 구조(Semantic Structure)를 스스로 학습한다는 점이다. 예를 들어 비슷한 형태나 기능을 가진 물체들은 자연스럽게 특징 공간(Feature Space)에서 가까운 위치를 형성한다. 이는 지도학습 없이도 고수준 의미 표현(High-Level Semantic Representation)이 형성될 수 있음을 보여주는 대표적인 사례이다.

DINOv2는 특정 작업을 위한 모델이 아니라 범용 특징 추출기(General-Purpose Feature Extractor)이다. 학습된 특징은 객체 인식(Object Recognition), 의미 분할(Semantic Segmentation), 자율주행(Navigation), 산업 검사(Industrial Inspection), 사람 행동 인식(Human Activity Recognition), 장면 이해(Scene Understanding), 이상 탐지(Anomaly Detection) 등 매우 다양한 작업에 그대로 사용할 수 있다.

이러한 범용성 덕분에 전이학습(Transfer Learning)이 매우 효과적으로 이루어진다. DINOv2는 대규모 이미지에서 미리 학습한 시각 지식을 그대로 유지한 채, 적은 양의 데이터만 이용하여 새로운 작업에 빠르게 적응할 수 있다. 따라서 로봇 개발자는 방대한 데이터를 다시 수집하지 않고도 새로운 응용 분야를 효율적으로 구축할 수 있다.

DINOv2는 특히 밀집 예측(Dense Prediction) 작업에서 매우 뛰어난 성능을 보인다. 객체의 경계(Object Boundary), 표면 구조(Surface Structure), 공간 관계(Spatial Relationship)를 세밀하게 유지하기 때문에 의미 분할, 객체 분할(Instance Segmentation), 표면 이해(Surface Understanding), 픽셀 단위 대응(Dense Correspondence)과 같은 작업에서 우수한 결과를 제공한다. 이러한 특성은 로봇 조작과 산업 검사에서 매우 중요한 역할을 수행한다.

로봇 조작(Robotic Manipulation)은 DINOv2의 대표적인 응용 분야이다. 로봇은 물체의 미세한 형상 차이를 정확하게 구분해야 하며, 동시에 조명이나 배경 변화에는 영향을 받지 않아야 한다. DINOv2는 이러한 요구를 만족하는 특징 표현을 생성하여 파지 계획(Grasp Planning), 자세 추정(Pose Estimation), 조작 계획(Manipulation Planning)의 성능을 크게 향상시킨다.

자율주행(Navigation) 역시 DINOv2의 중요한 응용 분야이다. 이동 로봇은 계절 변화, 조명 변화, 가구 배치 변경과 같은 다양한 환경 변화 속에서도 동일한 장소를 인식해야 한다. DINOv2는 이러한 변화에 강인한 특징을 생성하기 때문에 장소 인식(Place Recognition), 위치 추정(Visual Localization), 루프 클로저(Loop Closure)에서 매우 높은 성능을 제공한다.

산업 검사(Industrial Inspection)는 DINOv2가 특히 강점을 가지는 분야이다. 대부분의 공장은 공개 데이터셋에 존재하지 않는 제품과 설비를 사용한다. DINOv2는 라벨 없는 공장 데이터를 이용하여 일반적인 시각 표현을 먼저 학습한 후, 적은 양의 검사 데이터만으로 결함 검출(Defect Detection), 품질 검사(Quality Inspection), 예지보전(Predictive Maintenance) 등에 빠르게 적용할 수 있다.

이상 탐지(Anomaly Detection) 역시 자기지도학습의 대표적인 응용 분야이다. 실제 산업 현장에서는 정상 제품은 많지만 불량 제품은 매우 적다. DINOv2는 정상 제품의 특징을 충분히 학습한 후, 이와 크게 다른 특징을 이상으로 판단한다. 따라서 불량 데이터가 거의 없어도 높은 수준의 이상 탐지가 가능하다.

깊이 추정(Depth Estimation)과 3차원 장면 이해(3D Scene Understanding)에도 DINOv2의 특징 표현이 활용된다. DINOv2 자체는 RGB 영상만 처리하지만, 학습된 특징에는 물체 경계, 표면 연속성(Surface Continuity), 장면 구조(Scene Structure)에 대한 정보가 포함되어 있기 때문에 깊이 추정 모델의 초기화(Initialization)로 매우 효과적이다.

시각 위치 추정(Visual Localization)은 이동 로봇에게 필수적인 기술이다. DINOv2의 특징은 조명, 계절, 카메라 시점이 변해도 비교적 안정적으로 유지되므로 장기간 운영되는 자율주행 시스템에서 매우 높은 신뢰성을 제공한다. 이러한 특성은 실외 로봇과 물류 AMR에서 특히 중요한 장점이다.

DINOv2는 도메인 간 전이(Cross-Domain Transfer) 능력이 매우 뛰어나다. 자연 이미지(Natural Image)로 학습한 모델이 산업 검사, 의료 영상(Medical Imaging), 농업(Agriculture), 원격 탐사(Remote Sensing), 자율주행(Autonomous Driving) 등 다양한 분야에서도 높은 성능을 유지한다. 이는 자기지도학습이 매우 범용적인 시각 지식을 학습한다는 것을 보여준다.

특징 시각화(Feature Visualization)를 수행하면 DINOv2가 매우 의미 있는 영역에 자연스럽게 주목하는 것을 확인할 수 있다. 사람의 라벨이 전혀 없었음에도 물체, 재질, 질감, 형태, 장면 등을 스스로 구분하며, 내부 특징 공간이 의미적으로 잘 조직되어 있음을 보여준다.

지도학습 모델은 미리 정의된 클래스(Class)를 중심으로 특징 공간을 형성하지만, DINOv2는 연속적인 의미 공간(Continuous Semantic Space)을 생성한다. 따라서 서로 유사한 개념은 자연스럽게 가까운 위치를 형성하고, 의미 차이에 따라 점진적으로 분리된다. 이러한 구조는 검색(Retrieval), 클러스터링(Clustering), 유사도 분석(Similarity Search)에 매우 적합하다.

멀티카메라(Multi-Camera) 시스템에서도 DINOv2는 뛰어난 성능을 보인다. 손목 카메라(Wrist Camera), 헤드 카메라(Head Camera), 스테레오 카메라(Stereo Camera), 전방 카메라 등 서로 다른 시점에서도 일관된 특징을 생성하므로 센서 융합(Sensor Fusion)이 훨씬 쉬워진다.

DINOv2는 사전학습 단계에서는 매우 큰 계산 자원이 필요하지만, 학습이 완료된 이후에는 추론(Inference) 단계에서 효율적으로 사용할 수 있다. 양자화(Quantization), 혼합 정밀도(Mixed Precision), 하드웨어 가속(Hardware Acceleration)을 적용하면 모바일 로봇이나 산업용 엣지 컴퓨터(Edge Computer)에서도 충분히 활용 가능하다.

실제 로봇에서는 전체 모델을 다시 학습하는 대신 경량 적응(Parameter-Efficient Fine-Tuning)이 널리 사용된다. 어댑터(Adapter), 프롬프트 튜닝(Prompt Tuning), 저랭크 적응(Low-Rank Adaptation, LoRA) 등을 적용하면 적은 데이터와 적은 계산량으로 새로운 작업에 빠르게 적응할 수 있다.

지속적 학습(Continual Learning)은 DINOv2 이후의 중요한 연구 방향이다. 로봇은 운영 과정에서 계속 새로운 환경을 경험하므로 기존 지식을 유지하면서 새로운 정보를 지속적으로 학습해야 한다. 이러한 온라인 자기지도학습(Online Self-Supervised Learning)은 미래 범용 로봇의 핵심 기술이 될 것으로 예상된다.

DINOv2는 다중모달(Multimodal) AI와도 매우 잘 결합된다. 원래는 비전 모델이지만, 생성된 특징 표현은 언어 모델(Language Model), 월드 모델(World Model), 메모리 시스템(Memory System), 계획 모델(Planning Model), 비전-언어-행동(VLA) 정책과 자연스럽게 융합될 수 있다. 따라서 DINOv2는 단순한 비전 모델을 넘어 범용 AI의 핵심 구성 요소로 활용되고 있다.

VLA에서는 DINOv2가 RGB 카메라 입력을 처리하는 비전 인코더 역할을 수행한다. 생성된 특징은 언어(Language), 로봇 상태(Proprioception), 힘 센서(Force Sensor), 환경 지도(Environment Map), 메모리와 결합되어 행동 정책(Action Policy)의 입력으로 사용된다. 즉, DINOv2는 행동을 직접 생성하지는 않지만 전체 로봇 지능의 기반이 되는 시각 표현을 제공한다.

DINOv2는 CLIP과 자주 함께 사용된다. CLIP은 이미지와 언어의 의미 정렬(Language Alignment)에 강점을 가지며, DINOv2는 구조적 이해(Structural Understanding), 공간 표현(Geometric Representation), 밀집 특징(Dense Feature)에 강점을 가진다. 따라서 최신 VLA에서는 두 모델을 함께 사용하여 의미 이해와 공간 이해를 동시에 확보하는 경우가 많다.

DINOv2의 등장은 라벨링(Labeling)이 강력한 비전 모델을 만드는 유일한 방법이 아니라는 사실을 보여주었다. 앞으로 로봇은 운영 과정에서 수집하는 방대한 비라벨 데이터를 스스로 학습하며 지속적으로 성능을 향상시키는 방향으로 발전할 것이다. 이는 자기 성장(Self-Improving AI)을 실현하는 중요한 기반 기술이 된다.

향후 DINO 계열 모델은 시간 정보(Temporal Information), 3차원 공간(3D Geometry), 물리 상호작용(Physical Interaction), 월드 모델(World Model), 다중모달 추론(Multimodal Reasoning), 지속적 학습(Continual Learning)을 통합하는 방향으로 발전할 것으로 예상된다. 단순한 이미지 특징 추출을 넘어 현실 세계 전체를 이해하는 시각 지능으로 진화할 것이다.

결국 DINOv2는 대규모 자기지도학습(Self-Supervised Learning)을 통해 지도학습을 뛰어넘는 범용 시각 표현을 생성할 수 있음을 입증한 대표적인 비전 기반 모델이다. 교사-학생(Self-Distillation) 구조, 비전 트랜스포머(Vision Transformer), 뛰어난 전이학습(Transfer Learning), 강력한 밀집 특징(Dense Feature), 높은 일반화 성능(Generalization)을 바탕으로 현대 로보틱스와 비전-언어-행동(VLA) 시스템의 핵심 비전 인코더로 자리 잡았다. 앞으로도 DINOv2 계열의 자기지도학습 기반 모델은 물리 인공지능(Physical AI)과 범용 로봇(General-Purpose Robot)의 핵심 시각 기술로 지속적으로 발전할 것이다.

## 2.5 SigLIP: Sigmoid Loss for Vision-Language Pretraining (with Code)

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

SigLIP(Sigmoid Loss for Language-Image Pre-training)는 CLIP 이후 등장한 대표적인 비전-언어(Vision-Language) 기반 모델로, 이미지와 언어를 하나의 의미 공간(Shared Semantic Space)으로 정렬하는(Language Alignment) 새로운 학습 방식을 제안하였다. CLIP이 대조학습(Contrastive Learning)을 통해 이미지와 텍스트를 연결하였다면, SigLIP은 시그모이드 손실(Sigmoid Loss)을 이용하여 더욱 효율적이고 확장 가능한 학습 구조를 제공한다. 이 구조는 대규모 비전-언어 기반 모델(Vision-Language Foundation Model)의 학습 효율을 크게 향상시키며, 최신 비전-언어-행동(Vision-Language-Action, VLA) 시스템의 핵심 비전 인코더(Vision Encoder) 가운데 하나로 활용되고 있다.

SigLIP가 등장한 배경은 기존 CLIP의 한계를 개선하기 위한 것이었다. CLIP은 InfoNCE 기반의 대조학습을 사용하여 하나의 미니배치(Mini-Batch)에 포함된 모든 이미지와 텍스트를 서로 비교한다. 올바른 이미지-텍스트 쌍은 가깝게 만들고 나머지 모든 조합은 멀어지도록 학습한다. 이러한 방식은 매우 뛰어난 성능을 제공하지만, 많은 음성 샘플(Negative Sample)을 확보하기 위해 매우 큰 배치 크기(Batch Size)가 필요하며, 수천 개의 GPU를 사용하는 대규모 분산 학습 환경에서만 최고의 성능을 발휘하는 문제가 있었다.

모델 규모가 점점 커지면서 이러한 대규모 배치 의존성은 중요한 병목(Bottleneck)이 되었다. CLIP에서는 전체 배치에 포함된 모든 이미지와 텍스트 간의 유사도를 계산해야 하므로 GPU 간 통신량(Communication Overhead)이 매우 커진다. 또한 배치가 커질수록 메모리 사용량과 동기화(Synchronization) 비용도 급격히 증가한다. 이러한 이유로 더 효율적인 학습 방법이 요구되었으며, 그 결과 SigLIP이 제안되었다.

SigLIP은 이러한 문제를 해결하기 위해 학습 목표 자체를 변경하였다. 기존처럼 모든 샘플을 동시에 비교하는 대신, 각각의 이미지-텍스트 쌍(Image-Text Pair)을 독립적인 이진 분류(Binary Classification) 문제로 처리한다. 올바른 쌍은 긍정(Positive), 잘못된 쌍은 부정(Negative)으로 판단하며, 각각을 시그모이드 손실(Sigmoid Cross Entropy Loss)로 학습한다. 따라서 하나의 샘플이 다른 샘플과 경쟁하지 않아도 되며, 전체 배치 크기에 대한 의존성이 크게 줄어든다.

겉으로 보기에는 단순한 손실 함수(Loss Function)의 변경처럼 보이지만, 실제 효과는 매우 크다. CLIP에서는 모든 이미지와 텍스트를 동시에 정규화(Softmax Normalization)해야 하지만, SigLIP에서는 각각의 이미지-텍스트 관계를 독립적으로 평가한다. 따라서 대규모 GPU 클러스터에서도 통신량이 크게 감소하고, 분산 학습(Distributed Training)의 효율이 크게 향상된다.

SigLIP의 전체 구조는 CLIP과 매우 유사하다. 하나의 비전 인코더(Vision Encoder)가 이미지를 처리하고, 하나의 언어 인코더(Language Encoder)가 문장을 처리한다. 각각의 인코더는 이미지와 문장을 고차원 임베딩(Embedding)으로 변환한 후 동일한 의미 공간으로 투영한다. 가장 큰 차이는 모델 구조가 아니라 이미지와 텍스트를 연결하는 학습 방법(Learning Objective)에 있다.

비전 인코더는 일반적으로 비전 트랜스포머(Vision Transformer, ViT)를 사용한다. 이미지는 여러 개의 패치(Image Patch)로 분할되고, 각각이 시각 토큰(Visual Token)으로 변환된 뒤 트랜스포머를 거쳐 의미 표현(Semantic Representation)을 생성한다. 이러한 특징은 물체(Object), 장면(Scene), 재질(Material), 질감(Texture), 공간 구조(Spatial Structure)와 같은 다양한 정보를 포함하며, 이후 언어와 자연스럽게 결합된다.

언어 인코더(Language Encoder)는 트랜스포머(Transformer)를 이용하여 문장을 의미 벡터(Semantic Embedding)로 변환한다. 단순히 단어를 인식하는 것이 아니라 색상(Color), 재질(Material), 위치(Relation), 행동(Action), 속성(Attribute) 등 다양한 문맥(Context)을 함께 이해한다. 이러한 풍부한 언어 표현 덕분에 이미지와의 의미 정렬(Language Alignment)이 더욱 정확하게 이루어진다.

SigLIP 역시 이미지와 텍스트를 하나의 공통 임베딩 공간(Shared Embedding Space)에 표현한다. 컵(Cup) 이미지는 "Cup"이라는 단어와 가까운 위치를 형성하며, 자동차(Car)는 "Car"와 가까운 특징 공간에 위치한다. 이러한 공통 의미 공간은 이미지와 언어를 직접 비교할 수 있도록 하며, 로봇이 자연어 명령을 이해하는 핵심 기반이 된다.

SigLIP의 핵심은 시그모이드 손실(Sigmoid Loss)에 있다. CLIP에서는 소프트맥스(Softmax)를 이용하여 전체 배치 안에서 가장 유사한 쌍을 찾지만, SigLIP에서는 각 이미지와 텍스트의 관계를 독립적으로 예측한다. 각각의 관계가 올바른지 여부만 판단하면 되므로 학습이 훨씬 단순하며, 전체 배치 크기에 덜 민감하다.

이러한 독립적인 최적화(Independent Optimization)는 다양한 배치 크기에서도 안정적인 성능을 제공한다. CLIP은 배치가 작아질수록 음성 샘플이 줄어들어 성능이 떨어질 수 있지만, SigLIP은 개별 샘플을 독립적으로 학습하므로 상대적으로 작은 배치에서도 우수한 성능을 유지한다. 따라서 다양한 하드웨어 환경에서 효율적으로 학습할 수 있다.

학습 효율성(Training Efficiency)은 SigLIP의 가장 큰 장점 가운데 하나이다. GPU 간 통신량이 크게 감소하고, 전체 유사도 행렬(Similarity Matrix)을 계산하지 않아도 되므로 학습 속도가 향상된다. 또한 GPU 활용률(Utilization)이 높아지고, 대규모 분산 학습에서 계산 자원을 더욱 효율적으로 사용할 수 있다.

최적화 안정성(Optimization Stability)도 향상된다. Softmax 기반 학습에서는 하나의 샘플 변화가 전체 배치의 손실에 영향을 주지만, SigLIP은 각 샘플이 독립적으로 학습되므로 다른 샘플의 영향을 적게 받는다. 이러한 구조는 더욱 부드러운 학습 곡선과 안정적인 수렴(Convergence)을 제공한다.

SigLIP 역시 대규모 이미지-텍스트(Image-Text Pair) 데이터를 이용하여 학습된다. 인터넷에는 수억 개 이상의 이미지와 설명(Caption)이 존재하며, 이를 통해 물체(Object), 행동(Action), 장소(Environment), 재질(Material), 예술(Art), 과학(Science), 산업(Industry) 등 다양한 의미를 함께 학습한다. 이러한 대규모 데이터는 범용 시각 표현(General Visual Representation)의 기반이 된다.

학습이 완료되면 의미 정렬(Semantic Alignment)이 자연스럽게 이루어진다. 예를 들어 커피잔(Coffee Mug)은 컵(Cup), 음료(Drink), 주방(Kitchen), 세라믹(Ceramic)과 같은 다양한 언어 표현과 가까운 특징 공간을 형성한다. 자동차는 차량(Vehicle), 이동(Mobility), 도로(Road)와 연결되며, 산업 장비는 그 기능(Function)에 따라 의미적으로 그룹화된다.

SigLIP의 가장 중요한 장점 가운데 하나는 제로샷 인식(Zero-Shot Recognition)이다. 새로운 물체가 등장하더라도 추가 학습 없이 자연어 설명만으로 인식할 수 있다. 사용자가 "파란 공구 상자(Blue Toolbox)" 또는 "비상구 표지판(Emergency Exit Sign)"과 같은 문장을 입력하면, 로봇은 가장 유사한 이미지를 찾아낼 수 있다.

오픈 보캐뷸러리(Open Vocabulary) 인식도 SigLIP의 중요한 특징이다. 기존의 지도학습 모델은 미리 정의된 객체만 인식할 수 있었지만, SigLIP은 사람이 언어로 표현할 수 있는 거의 모든 개념을 인식할 수 있다. 따라서 새로운 제품과 새로운 환경이 계속 등장하는 실제 산업 현장에서 매우 높은 활용성을 가진다.

로봇 조작(Robotic Manipulation)은 SigLIP의 대표적인 응용 분야이다. 예를 들어 "나사를 조일 수 있는 도구를 가져와."와 같은 명령을 받으면, SigLIP은 드라이버(Screwdriver), 전동 드라이버(Power Driver), 정비 도구(Maintenance Tool)를 모두 의미적으로 이해할 수 있다. 이는 단순한 객체 인식을 넘어 개념 기반 추론(Concept-Based Reasoning)을 가능하게 한다.

언어 그라운딩(Language Grounding)은 SigLIP의 핵심 기능이다. 사람은 객체 ID 대신 "컨베이어 옆의 파란 플라스틱 상자"와 같이 자연어를 사용한다. SigLIP은 이러한 설명을 이미지와 직접 연결하여 로봇이 원하는 물체를 정확하게 찾을 수 있도록 한다.

장면 이해(Scene Understanding)도 SigLIP의 중요한 응용 분야이다. 공장(Factory), 병원(Hospital), 연구실(Laboratory), 창고(Warehouse), 사무실(Office), 농장(Farm) 등 환경 전체를 의미적으로 이해하며, 이를 기반으로 자율주행(Navigation), 작업 계획(Task Planning), 이상 탐지(Anomaly Detection)를 수행할 수 있다.

전이학습(Transfer Learning)은 SigLIP의 강력한 장점이다. 사전학습된 모델은 객체 검출(Object Detection), 의미 분할(Semantic Segmentation), 자율주행, 시각 위치 추정(Visual Localization), 산업 검사(Industrial Inspection), 인간 행동 인식(Human Activity Recognition), 시각 검색(Visual Retrieval) 등 매우 다양한 작업에 활용될 수 있다. 적은 양의 데이터만으로도 새로운 작업에 빠르게 적응할 수 있다.

실제 로봇에서는 전체 모델을 다시 학습하기보다 파라미터 효율적 미세조정(Parameter-Efficient Fine-Tuning)을 사용한다. 어댑터(Adapter), 프롬프트 튜닝(Prompt Tuning), 저랭크 적응(Low-Rank Adaptation, LoRA)을 이용하면 적은 계산량으로도 새로운 산업 환경에 효과적으로 적응할 수 있다.

프롬프트 엔지니어링(Prompt Engineering) 역시 SigLIP의 성능을 향상시키는 중요한 요소이다. 단순한 단어보다 "공장에서 사용하는 산업용 드라이버(A photo of an industrial screwdriver)"와 같은 자연스러운 설명을 사용할 경우 인식 성능이 향상된다. 최근에는 자동 프롬프트 최적화(Prompt Optimization)도 활발히 연구되고 있다.

SigLIP은 최신 비전-언어 모델(Vision-Language Model, VLM)의 핵심 비전 인코더로 활용된다. 생성된 시각 특징은 대규모 언어 모델(Large Language Model, LLM)과 연결되어 대화형 AI, 로봇 계획(Robot Planning), 비전-언어-행동(VLA) 시스템을 구성한다. 즉, SigLIP은 단순한 이미지 인식 모델이 아니라 현대 멀티모달 AI의 핵심 기반 기술이다.

VLA에서는 SigLIP이 RGB 카메라, 손목 카메라(Wrist Camera), 스테레오 카메라(Stereo Camera) 등의 영상을 의미 특징(Semantic Feature)으로 변환한다. 이후 언어(Language), 로봇 상태(Proprioception), 힘 센서(Force Sensor), 환경 지도(Environment Map), 메모리(Memory)와 결합되어 행동 정책(Action Policy)의 입력으로 사용된다.

CLIP과 비교하면 SigLIP은 더욱 효율적인 학습 구조를 제공한다. Softmax 기반 경쟁 구조 대신 Sigmoid 기반 독립 학습을 사용하기 때문에 대규모 분산 학습에서 계산량과 통신량이 감소한다. 반면 두 모델 모두 이미지와 언어를 동일한 의미 공간으로 정렬한다는 기본 철학은 동일하며, 실제 응용에서는 목적에 따라 두 모델을 선택하거나 함께 사용하는 경우가 많다.

DINOv2와 비교하면 SigLIP은 언어 정렬(Language Alignment)에 특화되어 있다. DINOv2는 구조적 이해(Structural Understanding)와 기하학적 특징(Geometric Representation)에 강점을 가지며, SigLIP은 이미지와 언어를 연결하는 의미 표현(Semantic Representation)에 강점을 가진다. 최신 로봇에서는 두 모델을 함께 사용하여 구조 정보와 의미 정보를 동시에 활용하는 경우가 많다.

실제 로봇에서는 계산 효율성도 중요하다. 대형 ViT 모델은 메모리와 연산량이 크기 때문에 양자화(Quantization), 혼합 정밀도(Mixed Precision), 프루닝(Pruning), 토큰 감소(Token Reduction), TensorRT 최적화 등을 적용하여 모바일 로봇과 산업용 엣지 컴퓨터에서도 실시간 추론이 가능하도록 한다.

설명 가능성(Interpretability)은 SigLIP 연구의 중요한 분야이다. 어텐션 맵(Attention Map), 임베딩 시각화(Embedding Visualization), 특징 분석(Feature Analysis)을 통해 모델이 어떤 의미 구조를 학습했는지 확인할 수 있으며, 이는 안전성(Safety), 신뢰성(Reliability), 모델 디버깅(Debugging)에 큰 도움이 된다.

하지만 SigLIP도 인터넷 기반 데이터셋에서 학습되므로 데이터 편향(Bias), 산업 분야의 데이터 부족, 문화적 차이, 잘못된 캡션 등의 영향을 받을 수 있다. 따라서 실제 산업용 로봇에서는 추가적인 도메인 적응(Domain Adaptation), 지속적 학습(Continual Learning), 현장 데이터 수집이 매우 중요하다.

향후 SigLIP은 단순한 이미지-언어 정렬을 넘어 시간 정보(Temporal Information), 3차원 공간(3D Geometry), 물리 상호작용(Physical Interaction), 인과 추론(Causal Reasoning), 메모리(Memory), 월드 모델(World Model)을 통합하는 방향으로 발전할 것으로 예상된다. 이는 더욱 높은 수준의 물리 인공지능(Physical AI)을 구현하는 핵심 기반이 될 것이다.

결국 SigLIP은 CLIP 이후의 대표적인 비전-언어 기반 모델로서, Softmax 기반 대조학습을 Sigmoid 기반 독립 학습으로 발전시켜 더욱 효율적이고 확장 가능한 언어-영상 정렬(Language-Image Alignment)을 실현하였다. 뛰어난 학습 효율성(Training Efficiency), 우수한 전이학습(Transfer Learning), 제로샷 인식(Zero-Shot Recognition), 오픈 보캐뷸러리(Open Vocabulary), 강력한 의미 표현(Semantic Representation)을 바탕으로 현재의 비전-언어-행동(VLA) 시스템과 차세대 물리 인공지능(Physical AI)의 핵심 비전 인코더로 자리 잡고 있으며, 앞으로도 범용 로봇 지능을 위한 중요한 기반 기술로 지속적으로 발전할 것이다.

## 2.6 Spatial Feature Extraction for Robotic Manipulation (with Code)

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

공간 특징 추출(Spatial Feature Extraction)은 로봇 조작(Robotic Manipulation)에서 가장 핵심적인 기술 가운데 하나이며, 시각 인식(Visual Perception)과 실제 물리적 행동(Physical Interaction)을 연결하는 중요한 역할을 수행한다. 일반적인 이미지 분류(Image Classification)는 "무엇이 있는가"를 판단하는 것이 목적이지만, 조작을 위한 비전은 "어디에 있는가", "어떤 자세(Pose)를 가지고 있는가", "어떻게 잡아야 하는가", "주변 물체와 어떤 관계를 가지는가"까지 이해해야 한다. 따라서 비전-언어-행동(Vision-Language-Action, VLA) 시스템에서는 의미 정보(Semantics)뿐 아니라 기하학적 구조(Geometry)와 공간 관계(Spatial Relationship)를 함께 표현하는 특징이 필요하다.

조작을 위한 공간 이해는 일반적인 객체 인식보다 훨씬 복잡하다. 예를 들어 컵(Cup)을 인식하는 것은 비교적 쉬운 작업이지만, 실제 로봇이 컵을 집기 위해서는 컵의 위치(Position), 방향(Orientation), 손잡이 위치(Handle Position), 주변 장애물(Obstacle), 접근 방향(Approach Direction), 안정적인 파지(Stable Grasp) 가능성까지 모두 판단해야 한다. 따라서 조작용 비전은 단순한 객체 분류가 아니라 물리적인 상호작용을 위한 공간 정보를 생성하는 것이 목적이다.

공간 특징 추출은 센서(Sensor)로부터 획득한 데이터를 로봇이 이해할 수 있는 구조화된 표현으로 변환하는 과정이다. 이러한 특징은 객체의 위치, 형태, 공간 구조, 표면, 물리적 관계 등을 포함하며, 이후 행동 정책(Action Policy), 경로 계획(Motion Planning), 파지 계획(Grasp Planning)의 입력으로 사용된다. 즉, 공간 특징은 로봇이 실제 환경과 상호작용하기 위한 핵심 정보이다.

현대 로봇은 다양한 센서를 동시에 활용하여 공간 정보를 생성한다. RGB 카메라(RGB Camera)는 색상(Color), 질감(Texture), 의미 정보(Semantic Information)를 제공하며, 깊이 카메라(Depth Camera)는 거리(Distance)와 3차원 구조(3D Geometry)를 제공한다. 스테레오 비전(Stereo Vision)은 양안 시차(Binocular Disparity)를 이용하여 깊이를 계산하고, LiDAR는 대규모 3차원 포인트 클라우드(Point Cloud)를 생성한다. 여기에 힘 센서(Force Sensor), 촉각 센서(Tactile Sensor), 이벤트 카메라(Event Camera)까지 결합하여 더욱 정밀한 공간 표현을 생성한다.

VLA 시스템에서는 공간 특징 추출이 일반적으로 비전 인코더(Vision Encoder)에서 시작된다. 입력 영상은 비전 트랜스포머(Vision Transformer, ViT), 합성곱 신경망(Convolutional Neural Network, CNN), 또는 하이브리드(Hybrid) 구조를 통해 처리된다. 초기 계층에서는 에지(Edge), 코너(Corner), 질감(Texture)과 같은 저수준 특징을 추출하고, 중간 계층에서는 물체의 형태와 공간 구조를 학습하며, 상위 계층에서는 의미 정보와 공간 관계를 동시에 표현하는 고수준 특징을 생성한다.

조작을 위한 비전에서는 공간 대응성(Spatial Correspondence)을 유지하는 것이 매우 중요하다. 일반 이미지 분류 모델은 해상도를 크게 줄여도 문제가 없지만, 조작에서는 물체의 경계(Object Boundary), 접촉면(Contact Surface), 파지 위치(Grasp Point)를 정확하게 유지해야 한다. 따라서 조작용 비전 모델은 가능한 한 높은 공간 해상도(Spatial Resolution)를 유지하면서 특징을 추출하도록 설계된다.

특징 피라미드(Feature Pyramid)는 다양한 공간 해상도를 동시에 활용하는 대표적인 방법이다. 높은 해상도에서는 작은 부품과 경계를 정확하게 표현하고, 중간 해상도에서는 물체의 구조를 이해하며, 낮은 해상도에서는 장면 전체의 문맥(Global Context)을 이해한다. 이러한 다중 해상도(Multi-Scale) 특징을 결합하면 정밀성과 전체적인 이해를 동시에 확보할 수 있다.

비전 트랜스포머(Vision Transformer)는 공간 특징 추출에서 큰 장점을 가진다. 자기 어텐션(Self-Attention)을 이용하여 이미지의 모든 영역이 서로 직접 정보를 교환하기 때문에 멀리 떨어진 물체 간 관계도 쉽게 이해할 수 있다. 이는 복잡한 작업 공간에서 여러 물체를 동시에 다루는 조작 작업에 매우 유리하다.

트랜스포머에서는 위치 임베딩(Positional Embedding)이 매우 중요하다. 트랜스포머는 입력 순서를 자동으로 알 수 없기 때문에 각 패치(Patch)의 위치 정보를 별도로 추가해야 한다. 절대 위치 임베딩(Absolute Positional Embedding), 상대 위치 임베딩(Relative Positional Embedding), 회전 위치 임베딩(Rotary Positional Embedding) 등을 이용하여 공간 구조를 유지하며, 이를 통해 로봇은 물체의 위치와 방향을 정확하게 이해할 수 있다.

깊이 인식(Depth Perception)은 조작 성능을 크게 향상시킨다. RGB 영상은 물체의 색과 형태는 알 수 있지만 정확한 거리 정보는 제공하지 못한다. 깊이 카메라와 RGB-D(RGB-Depth) 센서를 함께 사용하면 물체까지의 거리, 표면 형태, 접근 방향, 충돌 여부를 정확하게 계산할 수 있다. 이러한 기하학적 정보는 안정적인 파지와 충돌 회피에 매우 중요하다.

RGB-D 기반 비전은 현재 가장 널리 사용되는 조작 방식이다. RGB 영상은 의미 정보와 언어 정렬(Language Alignment)을 제공하고, 깊이 영상은 공간 구조와 거리 정보를 제공한다. 두 정보를 특징 융합(Feature Fusion)하면 의미 정보와 기하학 정보를 동시에 포함하는 강력한 공간 표현을 생성할 수 있으며, 대부분의 최신 조작 시스템이 이러한 구조를 채택하고 있다.

포인트 클라우드(Point Cloud)는 3차원 공간을 직접 표현하는 대표적인 방식이다. 이미지가 2차원 픽셀(Pixel)로 구성되는 반면, 포인트 클라우드는 수많은 3차원 좌표(Point)로 이루어진다. 각 점은 색상(Color), 법선 벡터(Surface Normal), 의미 정보(Semantic Label) 등을 함께 포함할 수 있어 실제 공간을 매우 정확하게 표현한다.

포인트 클라우드는 일반 이미지와 구조가 다르기 때문에 전용 네트워크가 필요하다. PointNet, PointNet++, Dynamic Graph CNN, Point Transformer와 같은 구조는 불규칙한 포인트 데이터를 직접 처리하며, 공간 구조를 유지하면서 특징을 추출한다. 이러한 모델은 로봇 조작에서 매우 널리 사용된다.

표면 법선(Surface Normal)은 물체 표면의 방향을 나타내는 벡터이다. 로봇은 일반적으로 표면에 수직으로 접근할 때 가장 안정적인 파지가 가능하므로, 표면 법선은 파지 계획(Grasp Planning)에서 매우 중요한 역할을 수행한다. 또한 삽입(Insertion), 조립(Assembly), 충돌 회피(Collision Avoidance)에서도 필수적인 정보이다.

객체 자세 추정(Object Pose Estimation)은 물체의 위치(Position)와 회전(Rotation)을 함께 추정하는 기술이다. 일반적으로 6자유도(6 Degrees of Freedom, 6DoF)를 계산하며, 이를 통해 로봇은 물체를 정확하게 집거나 조립할 수 있다. 다중 시점(Multi-View) 카메라를 함께 사용하면 가려짐(Occlusion) 문제를 줄이고 더욱 정확한 자세를 추정할 수 있다.

키포인트 검출(Keypoint Detection)은 물체 전체를 이해하기보다 중요한 지점을 찾는 방식이다. 손잡이(Handle), 모서리(Corner), 구멍(Hole), 힌지(Hinge), 연결부(Connection) 등을 검출하여 로봇이 쉽게 조작할 수 있도록 한다. 이러한 방식은 계산량이 적으면서도 매우 높은 조작 성능을 제공한다.

어포던스(Affordance) 학습은 공간 특징을 기능(Function) 중심으로 확장한 개념이다. 손잡이는 잡을 수 있고, 버튼은 누를 수 있으며, 문은 밀거나 당길 수 있다. 이러한 기능적 의미를 학습함으로써 로봇은 처음 보는 물체라도 "어떻게 사용할 수 있는지"를 추론할 수 있게 된다.

밀집 대응(Dense Correspondence)은 서로 다른 물체 사이의 픽셀 또는 포인트를 연결하는 기술이다. 이를 통해 서로 다른 형태의 물체라도 동일한 조작 전략을 적용할 수 있으며, 변형 물체(Deformable Object)나 조립 작업에서도 매우 효과적으로 활용된다.

의미 분할(Semantic Segmentation)은 이미지의 모든 픽셀에 의미 정보를 부여하는 기술이다. 객체 검출(Object Detection)보다 훨씬 정밀한 경계를 제공하며, 인스턴스 분할(Instance Segmentation)은 동일한 종류의 여러 물체도 개별적으로 구분한다. 이러한 정보는 파지 위치와 충돌 회피를 결정하는 데 매우 중요하다.

가려짐(Occlusion) 문제는 실제 로봇 환경에서 자주 발생한다. 여러 물체가 서로 겹쳐 있을 경우 일부만 보이더라도 로봇은 숨겨진 부분을 추정해야 한다. 이를 위해 다중 카메라(Multi-Camera), 능동 시각(Active Perception), 점유 지도(Occupancy Map), 신경 장면 표현(Neural Scene Representation) 등이 함께 활용된다.

공간 어텐션(Spatial Attention)은 작업에 필요한 영역만 집중적으로 분석하는 기술이다. 복잡한 작업 공간에서는 모든 물체를 동일하게 처리할 필요가 없으며, 조작 대상(Target Object)만 정확하게 분석하면 된다. 특히 자연어 명령과 결합하면 "파란 상자"와 같은 특정 객체에만 집중하는 언어 기반 공간 어텐션(Language-Guided Attention)을 구현할 수 있다.

멀티카메라(Multi-Camera) 시스템은 공간 특징의 품질을 크게 향상시킨다. 헤드 카메라(Head Camera)는 전체 환경을 관찰하고, 손목 카메라(Wrist Camera)는 조작 대상의 세부 정보를 제공하며, 스테레오 카메라는 깊이를 계산하고, 천장 카메라는 전체 작업 공간을 모니터링한다. 이러한 정보를 융합하면 가려짐과 시점 변화에 매우 강한 공간 표현을 생성할 수 있다.

3차원 장면 재구성(3D Scene Reconstruction)은 순간적인 영상뿐 아니라 환경 전체를 지속적으로 모델링하는 기술이다. SLAM(Simultaneous Localization and Mapping), Neural Radiance Field(NeRF), Gaussian Splatting, Signed Distance Field(SDF) 등을 이용하여 장시간 사용할 수 있는 공간 모델을 생성하며, 장기적인 조작 계획에도 활용된다.

시간 일관성(Temporal Consistency)은 움직이는 물체나 장시간 작업에서 매우 중요하다. 단일 프레임만으로는 정보가 부족할 수 있기 때문에 여러 프레임을 함께 분석하여 공간 정보를 안정적으로 유지한다. 이를 위해 순환 신경망(Recurrent Neural Network), 시간 트랜스포머(Temporal Transformer), 메모리(Memory) 모듈 등이 활용된다.

파지 검출(Grasp Detection)은 공간 특징 추출의 대표적인 응용 분야이다. 로봇은 영상만 보고도 그리퍼(Gripper)의 위치, 방향, 폭(Gripper Width), 접근 방향, 성공 확률까지 예측해야 한다. 최근에는 RGB-D, 포인트 클라우드, 비전 트랜스포머를 함께 사용하여 매우 높은 파지 성공률을 달성하고 있다.

변형 물체(Deformable Object) 조작은 더욱 어려운 문제이다. 천(Fabric), 케이블(Cable), 음식(Food), 생체 조직(Biological Tissue)과 같은 물체는 형태가 계속 변하기 때문에 기존 강체(Rigid Body) 기반 모델만으로는 처리하기 어렵다. 이를 해결하기 위해 물리 시뮬레이션(Physics Simulation)과 신경망 기반 변형 모델(Neural Deformation Model)이 함께 연구되고 있다.

산업용 로봇은 서브밀리미터(Sub-Millimeter) 수준의 정밀도를 요구하는 경우가 많다. 조립, 용접, 반도체 공정에서는 매우 정확한 공간 특징이 필요하며, 이를 위해 구조광(Structured Light), 레이저 삼각측량(Laser Triangulation), 고정밀 카메라와 힘 센서를 함께 사용한다.

언어 기반 공간 이해(Language Grounding)는 VLA의 핵심 기능이다. "파란 공구함 옆의 드라이버를 집어라."와 같은 명령을 수행하려면 로봇은 언어의 의미와 공간 위치를 동시에 이해해야 한다. 공간 특징과 언어 임베딩(Language Embedding)을 함께 사용하는 언어 조건부 공간 어텐션(Language-Conditioned Attention)이 이러한 문제를 해결한다.

최근에는 DINOv2, CLIP, SigLIP, SAM(Segment Anything Model)과 같은 비전 기반 모델(Vision Foundation Model)이 공간 특징 추출의 기본 모델로 사용된다. 대규모 사전학습을 통해 학습된 일반적인 시각 지식을 활용한 후, 조작 작업에 맞게 미세조정(Fine-Tuning)하면 적은 데이터만으로도 매우 높은 성능을 얻을 수 있다.

시뮬레이션(Simulation)은 공간 특징 학습에서 매우 중요한 역할을 한다. 가상 환경에서는 RGB, 깊이, 포인트 클라우드, 자세 정보(Pose), 접촉 정보(Contact), 성공한 조작 데이터까지 무한히 생성할 수 있다. 도메인 랜덤화(Domain Randomization)를 적용하면 실제 환경에서도 높은 일반화 성능을 확보할 수 있다.

특징 융합(Feature Fusion)은 현대 조작 시스템의 핵심 설계 원칙이다. 비전 특징, 깊이 정보, 포인트 클라우드, 촉각(Tactile), 힘 센서, 자기 위치(Proprioception), 환경 지도(Environment Map)를 모두 통합하여 하나의 공간 표현을 생성한다. 이러한 다중모달 융합(Multimodal Fusion)은 최신 로봇 시스템의 기본 구조가 되고 있다.

실시간 처리(Real-Time Processing)는 조작 시스템에서 매우 중요하다. 공간 특징 추출이 늦어지면 로봇의 제어(Control) 성능이 크게 저하된다. 따라서 경량 트랜스포머(Lightweight Transformer), 토큰 감소(Token Pruning), 양자화(Quantization), TensorRT 최적화, 전용 AI 가속기(AI Accelerator)를 이용하여 높은 정확도와 낮은 지연 시간을 동시에 달성한다.

공간 특징 추출은 조명 변화, 반사체(Reflective Surface), 투명 물체(Transparent Object), 센서 노이즈, 가려짐, 시점 변화 등 다양한 환경에서도 강인하게 동작해야 한다. 이를 위해 자기지도학습(Self-Supervised Learning), 데이터 증강(Data Augmentation), 도메인 적응(Domain Adaptation), 지속적 학습(Continual Learning), 불확실성 추정(Uncertainty Estimation)이 함께 활용된다.

미래의 공간 특징 추출은 의미(Semantics), 기하학(Geometry), 물리 법칙(Physics), 언어(Language), 메모리(Memory), 행동(Action)을 하나의 모델 안에서 동시에 표현하는 방향으로 발전할 것으로 예상된다. 즉, 물체의 위치뿐 아니라 기능, 물리적 특성, 상호작용 방식, 미래의 변화까지 함께 이해하는 통합 월드 모델(World Model)이 차세대 조작 시스템의 핵심이 될 것이다.

결국 공간 특징 추출은 로봇 조작의 가장 기본이 되는 인식 기술이다. 단순한 객체 인식만으로는 실제 물리적 작업을 수행할 수 없으며, 로봇은 깊이(Depth), 자세(Pose), 표면(Surface), 어포던스(Affordance), 공간 관계(Spatial Relationship), 시간 정보(Temporal Information), 환경 구조(Environment Structure)를 동시에 이해해야 한다. 현대의 비전 트랜스포머(Vision Transformer), 다중모달 융합(Multimodal Fusion), 자기지도학습(Self-Supervised Learning), 3차원 공간 인식(3D Spatial Perception)은 이러한 공간 특징을 효과적으로 생성하며, 산업 자동화, 서비스 로봇, 의료 로봇, 물류, 농업, 범용 물리 인공지능(Physical AI)의 핵심 기반 기술로 지속적으로 발전하고 있다.

## 2.7 Multi-Camera Vision Encoding: Wrist, Head, and Stereo Cameras (with Code)

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

멀티카메라 비전 인코딩(Multi-Camera Vision Encoding)은 현대 로보틱스에서 가장 중요한 시각 인식 기술 가운데 하나이다. 하나의 카메라만으로는 복잡한 환경을 완전하게 이해하기 어렵기 때문에, 여러 위치에 장착된 카메라를 동시에 활용하여 공간을 인식한다. 사람도 두 눈과 머리의 움직임을 함께 사용하여 입체적으로 세상을 인식하는 것처럼, 로봇 역시 헤드 카메라(Head Camera), 손목 카메라(Wrist Camera), 스테레오 카메라(Stereo Camera), 측면 카메라(Side Camera), 천장 카메라(Overhead Camera) 등을 조합하여 환경을 이해한다. 이러한 구조는 비전-언어-행동(Vision-Language-Action, VLA) 시스템에서 공간 이해와 행동 생성의 핵심 기반이 된다.

멀티카메라 시스템이 필요한 가장 큰 이유는 단일 카메라(Single Camera)의 한계 때문이다. 하나의 RGB 카메라는 3차원 공간을 2차원 영상으로 투영하기 때문에 깊이(Depth), 가려짐(Occlusion), 숨겨진 물체(Hidden Surface), 공간 구조(Spatial Structure)를 완전히 이해하기 어렵다. 물체 뒤쪽은 보이지 않고, 접근 가능한 파지 위치도 확인하기 어렵다. 여러 카메라를 서로 다른 위치에 배치하면 동일한 물체를 다양한 시점(Viewpoint)에서 관찰할 수 있으므로 이러한 문제를 크게 줄일 수 있다.

인간의 조작 능력은 멀티뷰(Multi-View) 시각의 중요성을 잘 보여준다. 사람은 두 눈을 이용해 깊이를 인식하고, 머리를 움직여 가려진 부분을 확인하며, 손이 물체에 가까워질수록 시선을 집중시킨다. 로봇도 마찬가지로 전역(Global) 시야와 근거리(Local) 시야를 동시에 활용해야 한다. 헤드 카메라는 전체 작업 공간을 관찰하고, 손목 카메라는 실제 파지와 조작 과정을 가까운 거리에서 관찰한다. 이러한 협력 구조는 안정적인 조작 성공률을 제공한다.

VLA 시스템에서 멀티카메라는 각각의 역할이 명확하게 구분된다. 헤드 카메라는 환경 전체를 이해하고, 손목 카메라는 정밀 조작을 담당하며, 스테레오 카메라는 깊이를 계산한다. 측면 카메라는 사각지대(Blind Spot)를 줄이고, 천장 카메라는 전체 작업 공간을 모니터링한다. 이동 로봇은 여기에 전방 카메라(Forward Camera), 후방 카메라(Rear Camera), 파노라마 카메라(Panoramic Camera), 하향 카메라(Downward Camera)까지 추가하여 더욱 넓은 환경을 인식한다.

헤드 카메라(Head Camera)는 사람의 시선과 가장 유사한 역할을 수행한다. 높은 위치에서 넓은 시야(Field of View)를 확보하여 작업 공간 전체를 관찰한다. 여러 개의 물체, 작업대, 사람, 이동 경로, 장애물을 동시에 인식하며, 물체 탐지(Object Detection), 의미 분할(Semantic Segmentation), 장면 이해(Scene Understanding), 자율주행(Navigation), 작업 계획(Task Planning)에 필요한 정보를 제공한다.

헤드 카메라는 전역 문맥(Global Context)을 이해하는 데 매우 적합하다. 예를 들어 공장에서는 작업대와 부품 위치를 파악하고, 병원에서는 의료 장비와 사람을 구분하며, 창고에서는 선반과 물류 박스를 동시에 인식한다. 이러한 전역 정보는 이후 로봇의 행동 계획(Action Planning)과 이동 계획(Motion Planning)의 기반이 된다.

손목 카메라(Wrist Camera)는 로봇 팔 끝단이나 그리퍼(Gripper) 근처에 장착된다. 조작 대상과 매우 가까운 거리에서 영상을 획득하기 때문에 파지 위치(Grasp Point), 삽입 구멍(Insertion Hole), 조립 위치(Assembly Position), 접촉면(Contact Surface) 등을 매우 정확하게 인식할 수 있다. 실제 산업용 로봇에서는 가장 중요한 센서 가운데 하나이다.

손목 카메라는 로봇 팔과 함께 움직이기 때문에 조작 과정에서도 항상 목표 물체를 관찰할 수 있다. 헤드 카메라는 로봇 팔에 의해 시야가 가려질 수 있지만, 손목 카메라는 대상과 함께 이동하므로 가려짐(Occlusion)에 매우 강하다. 이러한 특징은 조립, 삽입, 나사 체결, 정밀 검사와 같은 작업에서 매우 큰 장점을 제공한다.

비주얼 서보잉(Visual Servoing)은 손목 카메라의 대표적인 응용 기술이다. 로봇은 미리 계산된 궤적만 따라가는 것이 아니라, 손목 카메라의 영상을 실시간으로 분석하여 위치 오차를 지속적으로 수정한다. 이러한 폐루프 제어(Closed-Loop Control)는 조립 오차, 물체 이동, 기계 오차 등을 자동으로 보정하여 매우 높은 작업 성공률을 제공한다.

스테레오 비전(Stereo Vision)은 두 개의 카메라를 이용하여 깊이를 계산하는 방식이다. 두 카메라는 서로 약간 떨어진 위치에서 동일한 물체를 촬영하며, 좌우 영상의 차이(Disparity)를 계산하여 물체까지의 거리를 추정한다. 이는 인간의 양안 시각(Binocular Vision)과 동일한 원리이다.

스테레오 비전은 단안(Monocular) 카메라보다 훨씬 정확한 공간 정보를 제공한다. 물체의 거리(Distance), 크기(Size), 표면 방향(Surface Orientation), 작업 공간의 구조를 직접 계산할 수 있기 때문에 파지 계획(Grasp Planning), 충돌 회피(Collision Avoidance), 자율주행(Navigation), 환경 모델링(Environment Modeling)의 정확도가 크게 향상된다.

스테레오 매칭(Stereo Matching)은 스테레오 비전의 핵심 알고리즘이다. 좌우 영상에서 동일한 특징점을 찾아야 거리 계산이 가능하다. 초기에는 블록 매칭(Block Matching)과 같은 전통적인 알고리즘이 사용되었지만, 최근에는 딥러닝 기반 스테레오 네트워크(Deep Stereo Network)와 트랜스포머 기반 모델(Transformer-Based Stereo Model)이 매우 높은 정확도를 제공하고 있다.

멀티카메라 시스템에서는 캘리브레이션(Calibration)이 매우 중요하다. 내부 파라미터(Intrinsic Parameter)는 렌즈와 초점거리(Focal Length), 왜곡(Distortion)을 보정하며, 외부 파라미터(Extrinsic Parameter)는 여러 카메라 간의 위치와 방향을 계산한다. 이러한 보정이 정확해야 서로 다른 카메라의 정보를 하나의 좌표계(Coordinate System)에서 사용할 수 있다.

시간 동기화(Time Synchronization)도 매우 중요하다. 움직이는 로봇이나 사람을 여러 카메라가 동시에 관찰할 경우, 촬영 시점이 다르면 서로 다른 장면을 보게 된다. 이를 방지하기 위해 하드웨어 트리거(Hardware Trigger), 정밀 시간 프로토콜(Precision Time Protocol, PTP), GNSS 시간 동기화 등을 이용하여 모든 카메라를 마이크로초(Microsecond) 수준으로 동기화한다.

좌표 변환(Coordinate Transformation)은 여러 카메라의 정보를 하나의 좌표계로 통합하는 과정이다. 각각의 카메라는 자신의 좌표계를 가지므로, 이를 로봇 좌표계(Robot Coordinate), 월드 좌표계(World Coordinate), 작업 좌표계(Task Coordinate)로 변환해야 한다. 이러한 과정이 완료되어야 로봇은 모든 센서 정보를 일관되게 사용할 수 있다.

멀티카메라 특징 융합(Multi-Camera Feature Fusion)은 가장 중요한 연구 분야 가운데 하나이다. 헤드 카메라는 전역 문맥을 제공하고, 손목 카메라는 세밀한 기하 정보를 제공하며, 스테레오 카메라는 깊이를 제공한다. 이러한 서로 다른 특징을 효과적으로 결합해야 최종적인 공간 표현(Spatial Representation)이 생성된다.

크로스 어텐션(Cross-Attention)은 멀티카메라 융합에서 가장 널리 사용되는 방법이다. 헤드 카메라가 목표 물체를 찾으면 손목 카메라는 해당 영역의 세부 구조를 분석하고, 스테레오는 깊이 정보를 추가한다. 이러한 상호 참조를 통해 하나의 카메라만 사용할 때보다 훨씬 풍부한 특징을 생성할 수 있다.

계층적 특징 융합(Hierarchical Feature Fusion)은 여러 단계에서 특징을 결합하는 방식이다. 초기 단계(Early Fusion)는 원시 특징(Raw Feature)을 합치고, 중간 단계(Mid-Level Fusion)는 부분적으로 처리된 특징을 결합하며, 마지막 단계(Late Fusion)는 의미 정보(Semantic Feature)를 통합한다. 최근에는 이러한 방식을 모두 사용하는 하이브리드(Hybrid Fusion) 구조가 많이 사용된다.

비전 트랜스포머(Vision Transformer)는 멀티카메라 처리에 매우 적합하다. 서로 다른 카메라에서 생성된 비전 토큰(Visual Token)을 하나의 트랜스포머에서 동시에 처리할 수 있으며, 자기 어텐션(Self-Attention)을 이용하여 카메라 간 정보를 자연스럽게 교환한다. 이러한 구조는 최신 멀티카메라 VLA 시스템의 기본 구조가 되고 있다.

공간 일관성(Spatial Consistency)은 멀티카메라에서 매우 중요한 목표이다. 동일한 물체는 어느 카메라에서 보더라도 동일한 의미 표현을 가져야 한다. 이를 위해 자기지도학습(Self-Supervised Learning), 대조학습(Contrastive Learning), 기하학 제약(Geometric Constraint), 다중 시점 일관성(Multi-View Consistency) 학습이 활용된다.

가려짐(Occlusion)은 멀티카메라가 가장 큰 장점을 가지는 부분이다. 하나의 카메라에서 보이지 않는 물체라도 다른 카메라에서는 보일 가능성이 매우 높다. 따라서 여러 시점을 동시에 사용하면 물체의 전체 구조를 안정적으로 복원할 수 있으며, 조작 성공률도 크게 향상된다.

조작 계획(Manipulation Planning)은 멀티카메라의 대표적인 응용 분야이다. 헤드 카메라는 목표 물체를 찾고, 스테레오는 거리와 자세를 계산하며, 손목 카메라는 최종 파지를 수행한다. 측면 카메라는 주변 장애물을 감시하여 충돌을 방지한다. 이러한 계층적인 협력 구조가 현대 조작 시스템의 기본 구조이다.

픽앤플레이스(Pick-and-Place)는 멀티카메라 시스템의 대표적인 예이다. 헤드 카메라는 물체를 탐색하고, 이동 로봇은 접근하며, 스테레오는 깊이를 계산하고, 손목 카메라는 최종 파지를 수행한다. 물체를 들어 올린 이후에도 손목 카메라는 물체의 안정성을 확인하고, 배치 과정에서도 위치를 지속적으로 수정한다.

산업용 조립(Industrial Assembly)은 더욱 높은 정밀도를 요구한다. 헤드 카메라는 전체 작업을 관리하고, 스테레오는 조립 위치를 계산하며, 손목 카메라는 삽입 직전의 정렬을 수행한다. 힘 센서(Force Sensor)와 함께 사용하면 서브밀리미터(Sub-Millimeter) 수준의 조립 정확도를 달성할 수 있다.

이동 조작(Mobile Manipulation)은 자율주행과 조작이 동시에 이루어진다. 전방 카메라는 이동 경로를 관찰하고, 헤드 카메라는 작업 대상을 찾으며, 스테레오는 장애물을 인식하고, 손목 카메라는 물체를 집는다. 이러한 복합적인 시각 시스템은 이동과 조작을 동시에 수행하는 범용 로봇의 핵심 기술이다.

최근에는 DINOv2, CLIP, SigLIP, SAM(Segment Anything Model)과 같은 비전 기반 모델(Vision Foundation Model)이 멀티카메라 시스템의 기본 인코더로 사용된다. 대규모 사전학습을 통해 얻은 일반적인 시각 표현을 각 카메라에 적용한 후, 멀티카메라 환경에 맞게 미세조정(Fine-Tuning)하면 적은 데이터만으로도 매우 높은 성능을 얻을 수 있다.

시뮬레이션(Simulation)은 멀티카메라 학습에서 매우 중요한 역할을 한다. 가상 환경에서는 다양한 카메라 배치를 자유롭게 구성할 수 있으며, RGB, 깊이, 자세(Pose), 의미 분할, 포인트 클라우드 등을 동시에 생성할 수 있다. 도메인 랜덤화(Domain Randomization)를 적용하면 실제 환경에서도 높은 일반화 성능을 얻을 수 있다.

멀티카메라 시스템은 계산량이 매우 크기 때문에 엣지 컴퓨팅(Edge Computing) 최적화가 필수적이다. 공유 백본(Shared Backbone), 토큰 프루닝(Token Pruning), 양자화(Quantization), 혼합 정밀도(Mixed Precision), TensorRT 최적화 등을 적용하여 여러 카메라 영상을 실시간으로 처리한다.

메모리 관리(Memory Management)도 중요한 연구 분야이다. 모든 영상을 그대로 저장하는 대신, 환경 전체를 하나의 잠재 표현(Latent World Representation)으로 유지하고 필요한 정보만 저장한다. 이러한 에피소드 메모리(Episodic Memory)는 장기적인 작업 수행에서 매우 중요한 역할을 한다.

불확실성 추정(Uncertainty Estimation)은 멀티카메라 시스템의 안정성을 높여준다. 특정 카메라가 조명 변화나 오염으로 성능이 떨어질 경우, 시스템은 해당 카메라의 신뢰도를 낮추고 다른 카메라의 정보를 더 많이 활용한다. 이러한 확률 기반 센서 융합(Probabilistic Sensor Fusion)은 실제 산업 환경에서 매우 중요하다.

미래의 멀티카메라 시스템은 영상뿐 아니라 깊이(Depth), 촉각(Tactile), 자기 위치(Proprioception), 언어(Language), 메모리(Memory), 월드 모델(World Model)을 모두 하나의 트랜스포머 기반 구조로 통합할 것으로 예상된다. 즉, 단순한 카메라 융합을 넘어 로봇 전체의 인지 시스템(Cognitive System)으로 발전하게 된다.

또한 능동 시각(Active Perception)도 중요한 연구 분야이다. 고정된 카메라만 사용하는 것이 아니라, 로봇이 머리나 팔을 움직여 가장 좋은 시점을 직접 선택한다. 이러한 정보 획득 최대화(Maximum Information Gain) 전략은 사람의 시각 탐색 방식과 매우 유사하며, 미래 범용 로봇의 핵심 기술이 될 것으로 전망된다.

결국 멀티카메라 비전 인코딩(Multi-Camera Vision Encoding)은 단일 카메라의 한계를 극복하여 전역 환경(Global Context), 정밀 조작(Local Manipulation), 깊이 정보(Depth), 공간 구조(Spatial Structure), 의미 이해(Semantic Understanding)를 하나의 통합 표현으로 결합하는 기술이다. 헤드 카메라, 손목 카메라, 스테레오 비전, 다중모달 특징 융합(Multimodal Feature Fusion), 비전 기반 모델(Vision Foundation Model)이 결합됨으로써 현대 비전-언어-행동(VLA) 시스템은 사람과 유사한 수준의 공간 인식과 조작 능력을 갖추게 되었으며, 이러한 기술은 범용 로봇(General-Purpose Robot)과 물리 인공지능(Physical AI)의 핵심 기반 기술로 지속적으로 발전할 것이다.

## 2.8 3D Vision Encoding Using Depth and Point Clouds (with Code)

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

3차원 비전 인코딩(3D Vision Encoding)은 현대 로보틱스에서 가장 중요한 핵심 기술 가운데 하나이다. 로봇이 실제 물리 세계와 상호작용하기 위해서는 단순히 2차원 영상만 인식하는 것이 아니라, 물체의 거리(Depth), 형태(Geometry), 공간 구조(Spatial Structure), 크기(Size), 방향(Orientation)까지 이해해야 한다. 따라서 비전-언어-행동(Vision-Language-Action, VLA) 시스템에서는 RGB 영상뿐 아니라 깊이(Depth)와 포인트 클라우드(Point Cloud)를 함께 사용하여 보다 정확한 공간 표현(Spatial Representation)을 생성한다. 이러한 3차원 정보는 조작(Manipulation), 자율주행(Navigation), 계획(Planning), 추론(Reasoning)의 핵심 기반이 된다.

2차원 영상은 실제 3차원 공간을 하나의 평면(Image Plane)에 투영한 결과이다. 이 과정에서 깊이 정보가 사라지므로 물체까지의 거리, 실제 크기, 높이 차이, 표면 방향 등을 직접 알 수 없다. 사람은 양안 시각(Binocular Vision), 머리 움직임(Motion Parallax), 경험(Prior Knowledge)을 이용하여 이를 보완하지만, 로봇은 별도의 3차원 인식 기술을 이용해야 한다. 따라서 3차원 비전은 실제 환경에서 물리적인 행동을 수행하기 위한 필수 기술이 된다.

깊이 인식(Depth Perception)은 RGB 영상에 부족한 공간 정보를 제공한다. 영상의 모든 픽셀(Pixel)은 색상뿐 아니라 카메라와의 거리(Distance)도 함께 가지게 된다. 이를 통해 로봇은 물체의 위치(Position), 자세(Pose), 표면 구조(Surface Geometry), 충돌 여부(Collision), 접근 방향(Approach Direction)을 계산할 수 있다. 오늘날 대부분의 산업용 로봇, 서비스 로봇, 물류 로봇, 의료 로봇, 휴머노이드(Humanoid)는 이러한 깊이 정보를 기본적으로 활용하고 있다.

깊이를 측정하는 방법은 여러 가지가 존재한다. 스테레오 비전(Stereo Vision)은 두 개의 카메라를 이용하여 거리 정보를 계산하며, 구조광(Structured Light)은 적외선 패턴을 투사하여 표면 형상을 복원한다. ToF(Time-of-Flight)는 빛이 왕복하는 시간을 측정하여 거리를 계산하고, LiDAR는 레이저를 이용하여 매우 정확한 거리 정보를 생성한다. 최근에는 이벤트 카메라(Event Camera)와 같은 새로운 센서도 등장하여 다양한 환경에서 활용되고 있다.

스테레오 비전은 가장 널리 사용되는 깊이 측정 방식이다. 두 개의 카메라가 동일한 장면을 서로 다른 위치에서 촬영하면, 동일한 물체가 좌우 영상에서 서로 다른 위치에 나타난다. 이러한 시차(Disparity)를 계산하여 삼각측량(Triangulation)을 수행하면 물체까지의 거리를 계산할 수 있다. 최근에는 딥러닝 기반 스테레오 매칭(Stereo Matching) 모델이 기존 알고리즘보다 훨씬 높은 정확도를 제공하고 있다.

구조광(Structured Light)은 적외선(IR) 패턴을 물체에 투사한 후, 패턴의 변형 정도를 분석하여 깊이를 계산한다. 매우 높은 정밀도를 제공하기 때문에 산업 검사(Industrial Inspection), 로봇 조작(Robotic Manipulation), 증강현실(Augmented Reality) 등에서 널리 사용된다. 다만 강한 햇빛에서는 적외선이 간섭을 받아 실외 환경에서는 사용이 제한될 수 있다.

ToF(Time-of-Flight) 카메라는 적외선을 발사한 후 반사되어 돌아오는 시간을 측정하여 거리를 계산한다. 모든 픽셀이 동시에 거리 정보를 생성하기 때문에 매우 빠른 속도로 깊이 영상을 획득할 수 있으며, 실시간성이 중요한 로봇 응용 분야에서 널리 활용된다. 구조광보다 다양한 환경에서 안정적으로 사용할 수 있다는 장점도 가진다.

LiDAR(Light Detection and Ranging)는 레이저를 이용하여 주변 공간을 스캔한다. RGB-D 카메라처럼 깊이 영상을 생성하는 대신, 수백만 개의 3차원 점(Point)을 생성하는 포인트 클라우드(Point Cloud)를 출력한다. LiDAR는 매우 높은 거리 정확도와 긴 측정 거리(Long Range)를 제공하므로 자율주행 자동차, 실외 자율주행 로봇, 드론, 측량 시스템에서 필수적인 센서로 사용된다.

깊이 맵(Depth Map)은 가장 기본적인 3차원 표현 방식이다. RGB 영상과 동일한 해상도를 가지며, 각 픽셀에는 색상 대신 거리 값이 저장된다. 이러한 구조는 기존 RGB 영상과 쉽게 결합할 수 있으므로 RGB-D 기반 비전 시스템의 핵심 데이터 형식으로 사용된다. VLA 시스템에서도 RGB와 Depth를 함께 입력으로 사용하는 경우가 많다.

포인트 클라우드(Point Cloud)는 실제 공간을 직접 표현하는 대표적인 3차원 데이터 구조이다. 각각의 점(Point)은 X, Y, Z 좌표를 가지며, 색상(Color), 반사 강도(Intensity), 표면 법선(Surface Normal), 의미 정보(Semantic Label) 등을 함께 저장할 수 있다. 이미지가 2차원 공간을 표현하는 반면, 포인트 클라우드는 실제 3차원 환경 자체를 표현한다는 점에서 큰 차이가 있다.

포인트 클라우드는 다양한 방법으로 생성된다. LiDAR는 직접 포인트를 측정하며, 스테레오 비전은 깊이를 계산한 후 포인트로 변환한다. RGB-D 카메라도 깊이 영상을 이용하여 컬러 포인트 클라우드를 생성할 수 있다. 여러 시점에서 획득한 포인트 클라우드를 합치면 하나의 완전한 3차원 환경 모델을 구축할 수 있다.

포인트 클라우드는 이미지처럼 규칙적인 격자(Grid) 구조가 아니기 때문에 일반 CNN을 사용할 수 없다. 이를 위해 PointNet, PointNet++, Dynamic Graph CNN(DGCNN), Point Transformer와 같은 전용 신경망이 개발되었다. 이러한 모델은 점들의 순서와 관계를 유지하면서 효과적으로 특징을 추출할 수 있도록 설계되었다.

복셀(Voxel)은 3차원 공간을 작은 정육면체(Cell)로 분할하는 방법이다. 2차원 이미지의 픽셀(Pixel)에 해당하는 개념으로, 각 복셀에는 점유 여부(Occupancy), 밀도(Density), 색상(Color), 의미 정보(Semantic Feature) 등이 저장된다. 이후 3차원 CNN을 이용하여 공간 정보를 처리할 수 있다. 그러나 공간 해상도가 높아질수록 메모리 사용량이 급격히 증가하는 단점이 있다.

표면 재구성(Surface Reconstruction)은 포인트 클라우드를 연속적인 표면(Surface)으로 변환하는 과정이다. 삼각형 메쉬(Triangle Mesh), Signed Distance Field(SDF), Occupancy Network, Neural Radiance Field(NeRF), Gaussian Splatting 등이 대표적인 기술이다. 이러한 표현은 충돌 검사(Collision Checking), 시뮬레이션(Simulation), 조작 계획(Manipulation Planning)에 매우 유용하다.

표면 법선(Surface Normal)은 각 표면이 어느 방향을 향하고 있는지를 나타내는 벡터(Vector)이다. 로봇은 일반적으로 표면에 수직으로 접근해야 안정적인 파지가 가능하므로, 표면 법선은 파지 계획(Grasp Planning), 조립(Assembly), 삽입(Insertion), 지형 분석(Terrain Analysis)에서 매우 중요한 정보가 된다.

객체 자세 추정(Object Pose Estimation)은 물체의 위치(Position)와 방향(Orientation)을 함께 계산하는 기술이다. 일반적으로 6자유도(6 Degrees of Freedom, 6DoF)를 추정하며, 로봇은 이를 이용하여 물체를 집거나 조립하고 검사 작업을 수행한다. 3차원 정보는 2차원 영상보다 훨씬 높은 자세 추정 정확도를 제공한다.

3차원 의미 분할(3D Semantic Segmentation)은 이미지가 아니라 포인트 또는 복셀 각각에 의미 정보를 부여하는 기술이다. 이를 통해 로봇은 물체뿐 아니라 바닥(Floor), 벽(Wall), 계단(Stair), 작업대(Table)까지 모두 이해할 수 있다. 이러한 기술은 자율주행, 창고 관리, 시설 점검, 농업 로봇 등에서 매우 중요하다.

포인트 클라우드 정합(Point Cloud Registration)은 서로 다른 위치에서 획득한 3차원 데이터를 하나로 합치는 기술이다. 기존에는 ICP(Iterative Closest Point)가 주로 사용되었지만, 최근에는 딥러닝 기반 정합(Registration) 기술이 등장하여 더욱 정확한 결과를 제공한다. 이는 SLAM, 환경 지도(Environment Mapping), 객체 추적(Object Tracking)의 핵심 기술이다.

동시 위치 추정 및 지도 작성(Simultaneous Localization and Mapping, SLAM)은 3차원 비전 인코딩의 대표적인 응용 분야이다. 이동 로봇은 이동하면서 자신의 위치를 추정하는 동시에 환경 지도를 구축해야 한다. 포인트 클라우드와 깊이 정보는 매우 정확한 기하학 정보를 제공하므로 위치 추정(Localization)과 환경 모델링(Environment Modeling)의 정확도를 크게 향상시킨다.

3차원 정보는 로봇 조작(Robotic Manipulation)의 성능도 크게 향상시킨다. 로봇은 표면 형태, 접근 방향, 충돌 가능성, 파지 안정성 등을 직접 계산할 수 있으며, 작업 도중 환경이 바뀌더라도 새로운 경로를 다시 계산할 수 있다. 이러한 기하학적 추론(Geometric Reasoning)은 RGB 영상만 사용하는 시스템보다 훨씬 높은 성공률을 제공한다.

언어 기반 공간 이해(Language Grounding) 역시 3차원 정보의 도움을 받는다. 사람은 "파란 상자 위에 있는 컵" 또는 "의자 뒤에 있는 공구"와 같이 공간 관계를 포함한 명령을 자주 사용한다. 이러한 관계는 2차원 영상에서는 모호하지만, 3차원 공간에서는 정확하게 표현할 수 있으므로 VLA 시스템의 언어 이해 능력이 크게 향상된다.

어포던스(Affordance) 학습도 3차원 기하학과 밀접한 관련이 있다. 손잡이는 잡을 수 있고, 평평한 면은 물건을 올려놓을 수 있으며, 원통형 손잡이는 회전시킬 수 있다. 이러한 기능적 의미는 단순한 색상이 아니라 물체의 기하학적 구조를 이해해야만 학습할 수 있다.

최근의 비전 기반 모델(Vision Foundation Model)은 RGB뿐 아니라 깊이와 포인트 클라우드도 함께 처리한다. 비전 트랜스포머(Vision Transformer, ViT)는 깊이 토큰(Depth Token), 포인트 토큰(Point Token), 복셀 임베딩(Voxel Embedding)을 함께 처리할 수 있으며, RGB와 3차원 기하학을 동시에 학습하는 멀티모달(Multimodal) 구조로 발전하고 있다.

멀티카메라 시스템(Multi-Camera System)은 3차원 비전의 정확도를 더욱 향상시킨다. 헤드 카메라는 전역 환경을 관찰하고, 손목 카메라는 조작 대상을 근거리에서 관찰하며, 스테레오는 깊이를 계산한다. 여러 시점의 정보를 융합하면 하나의 센서보다 훨씬 완전한 공간 표현을 생성할 수 있다.

시간적 통합(Temporal Integration)도 중요한 요소이다. 단일 프레임에서는 노이즈나 가려짐이 존재할 수 있으므로, 여러 시점과 여러 시간의 데이터를 누적하여 보다 완전한 환경 모델을 생성한다. 시간 트랜스포머(Temporal Transformer), 메모리(Memory), 점유 지도(Occupancy Map)가 이러한 역할을 수행한다.

시뮬레이션(Simulation)은 3차원 비전 학습에서 매우 중요한 역할을 한다. 가상 환경에서는 깊이 맵, 포인트 클라우드, 자세(Pose), 접촉 정보(Contact), 의미 분할(Semantic Segmentation) 등을 완벽하게 생성할 수 있다. 도메인 랜덤화(Domain Randomization)를 적용하면 실제 환경에서도 높은 일반화 성능을 확보할 수 있다.

3차원 비전은 매우 많은 계산량을 요구하기 때문에 효율적인 처리 기술이 필요하다. 희소 포인트(Sparse Point), 적응형 복셀(Adaptive Voxel), 효율적인 트랜스포머(Efficient Transformer), 혼합 정밀도(Mixed Precision), AI 가속기(AI Accelerator), TensorRT 최적화 등이 실시간 처리를 가능하게 한다.

실제 환경에서는 반사체(Reflective Surface), 투명 물체(Transparent Object), 비(Rain), 안개(Fog), 먼지(Dust), 직사광선(Sunlight) 등이 깊이 센서의 성능을 저하시킨다. 이를 해결하기 위해 RGB, LiDAR, 레이더(Radar), 열화상(Thermal Camera), IMU 등을 함께 사용하는 다중모달 센서 융합(Multimodal Sensor Fusion)이 활발히 연구되고 있다.

3차원 월드 모델(3D World Model)은 미래 로봇의 핵심 기술이다. 단순히 현재 보이는 장면만 저장하는 것이 아니라, 환경 전체의 기하학, 의미 정보, 물리 특성, 시간 변화, 객체의 움직임을 하나의 지속적인 공간 표현으로 유지한다. 이러한 월드 모델은 장기 계획(Long-Horizon Planning)과 자율 행동(Autonomous Action)의 핵심이 될 것이다.

최근에는 생성형 AI(Generative AI)와 3차원 비전도 결합되고 있다. 미래의 모델은 보이지 않는 부분까지 예측하고, 물체의 내부 구조와 물리적 움직임을 추론하며, 행동 이후의 결과까지 미리 시뮬레이션할 수 있을 것으로 기대된다. 이는 물리 인공지능(Physical AI)의 중요한 발전 방향이다.

휴머노이드(Humanoid)는 특히 고성능 3차원 비전이 필수적이다. 계단 오르기(Stair Climbing), 문 열기(Door Opening), 공구 사용(Tool Use), 사람과의 협업(Human-Robot Collaboration)은 모두 복잡한 3차원 공간을 이해해야 수행할 수 있다. 여러 대의 카메라와 깊이 센서를 결합한 멀티카메라 시스템은 이러한 기능의 핵심 기반이 된다.

차세대 범용 로봇(General-Purpose Robot)은 RGB 영상뿐 아니라 깊이, 포인트 클라우드, 촉각(Tactile), 힘 센서(Force Sensor), 자기 위치(Proprioception), 언어(Language), 메모리(Memory)를 하나의 통합 모델에서 처리하는 방향으로 발전하고 있다. 이러한 통합 멀티모달 구조는 사람과 유사한 수준의 환경 이해와 행동 생성을 가능하게 할 것이다.

결국 3차원 비전 인코딩(3D Vision Encoding)은 2차원 영상의 한계를 극복하고 실제 물리 세계의 기하학적 구조를 이해하기 위한 핵심 기술이다. 깊이 맵(Depth Map)은 거리 정보를 제공하고, 포인트 클라우드(Point Cloud)는 실제 3차원 공간을 직접 표현한다. 여기에 비전 트랜스포머(Vision Transformer), 비전 기반 모델(Vision Foundation Model), 자기지도학습(Self-Supervised Learning), 멀티모달 융합(Multimodal Fusion)이 결합되면서 현대 비전-언어-행동(VLA) 시스템은 사람 수준에 가까운 공간 이해 능력을 갖추게 되었으며, 이러한 기술은 앞으로 범용 로봇과 물리 인공지능(Physical AI)의 핵심 기반 기술로 지속적으로 발전할 것이다.

## 2.9 Fine-Tuning Vision Encoders for Robotic Applications (with Code)

![](images/image9.png){width="7.268055555555556in" height="7.268055555555556in"}

비전 인코더 미세조정(Vision Encoder Fine-Tuning)은 현대 비전-언어-행동(Vision-Language-Action, VLA) 시스템에서 가장 중요한 기술 가운데 하나이다. 비전 인코더(Vision Encoder)는 카메라로 입력된 영상을 의미 있는 특징 벡터(Feature Embedding)로 변환하여 언어 모델(Language Model), 계획기(Planner), 행동 정책(Action Policy)이 이해할 수 있도록 만든다. 최근에는 Vision Transformer(ViT), CLIP, DINOv2, SigLIP, Segment Anything Model(SAM)과 같은 대규모 비전 기반 모델(Vision Foundation Model)이 널리 사용되고 있으며, 이들은 인터넷의 수십억 장 이상의 이미지로 사전학습(Pre-training)되어 뛰어난 일반화 성능을 제공한다.

그러나 이러한 비전 기반 모델은 일반적인 인터넷 이미지를 학습한 것이므로 실제 로봇 환경과는 상당한 차이가 존재한다. 공장(Factory), 물류 창고(Warehouse), 병원(Hospital), 농업(Agriculture), 연구실(Laboratory), 가정(Home)과 같은 로봇 환경은 인터넷 사진보다 훨씬 복잡하며, 카메라의 시점(Viewpoint), 조명(Lighting), 물체 배치(Object Arrangement), 가려짐(Occlusion), 반사체(Reflective Surface), 투명 물체(Transparent Object) 등이 매우 다르다. 따라서 사전학습된 모델을 그대로 사용하기보다는 로봇 환경에 맞게 미세조정(Fine-Tuning)하는 과정이 반드시 필요하다.

비전 인코더 미세조정의 목적은 기존 모델을 처음부터 다시 학습하는 것이 아니라, 이미 학습된 일반적인 시각 지식을 유지하면서 로봇 환경에 적합하도록 일부 표현을 수정하는 것이다. 기존 모델은 에지(Edge), 색상(Color), 질감(Texture), 형태(Shape), 의미 정보(Semantics)를 이미 잘 학습하고 있으므로, 이러한 기반 지식을 유지하면서 공간 구조(Spatial Structure), 파지 위치(Grasp Point), 물체 자세(Object Pose), 작업 문맥(Task Context), 조작 가능성(Affordance)과 같은 로봇 특화 정보를 더욱 잘 표현하도록 만드는 것이 핵심 목표이다.

인터넷 이미지와 로봇 데이터 사이에는 큰 도메인 차이(Domain Gap)가 존재한다. 인터넷 사진은 대부분 조명이 균일하고 물체가 잘 보이는 환경에서 촬영되지만, 로봇은 복잡한 작업 공간, 매우 가까운 거리, 다양한 시점, 반복적인 산업 부품, 강한 반사, 그림자, 움직임이 있는 환경을 지속적으로 관찰한다. 손목 카메라(Wrist Camera)는 매우 가까운 거리에서 물체를 촬영하고, 이동 로봇은 이동하면서 지속적으로 환경을 관찰한다. 이러한 차이는 비전 인코더를 로봇 환경에 맞게 적응시켜야 하는 가장 중요한 이유이다.

일반적인 컴퓨터 비전(Computer Vision)은 이미지 분류(Image Classification), 객체 인식(Object Detection), 의미 분할(Semantic Segmentation) 등을 목표로 하지만, 로봇 비전은 실제 행동(Action)을 위한 정보를 제공해야 한다. 따라서 단순히 물체를 인식하는 것이 아니라 파지 가능한 영역(Grasp Region), 접촉면(Contact Surface), 충돌 가능성(Collision Constraint), 자유 공간(Free Space), 조립 위치(Insertion Point), 공구 인터페이스(Tool Interface) 등을 유지하는 특징 표현이 필요하다. 이러한 공간 정보는 성공적인 조작과 자율행동을 위한 핵심 요소이다.

비전 인코더 미세조정에는 다양한 방법이 존재한다. 전체 미세조정(Full Fine-Tuning)은 모델의 모든 파라미터(Parameter)를 업데이트하는 방식으로 가장 높은 성능을 기대할 수 있지만, 매우 많은 GPU 자원과 대규모 데이터셋이 필요하다. 또한 지나치게 특정 작업에 적응하면 기존에 학습한 일반적인 시각 지식을 잃어버리는 문제도 발생할 수 있다.

이러한 문제를 파국적 망각(Catastrophic Forgetting)이라고 한다. 특정 산업 데이터만 반복적으로 학습하면 기존에 학습했던 수많은 일반 객체와 장면에 대한 이해 능력이 감소할 수 있다. 따라서 현대의 미세조정은 기존의 일반화 능력을 유지하면서도 새로운 환경에 적응하는 균형(Balance)을 매우 중요하게 고려한다.

부분 미세조정(Partial Fine-Tuning)은 이러한 문제를 해결하기 위한 대표적인 방법이다. 비전 트랜스포머(Vision Transformer)의 초기 계층(Lower Layer)은 에지와 질감처럼 모든 환경에서 공통적으로 사용되는 특징을 학습하고 있으므로 그대로 유지한다. 대신 상위 계층(Upper Layer)만 업데이트하여 로봇 환경에 특화된 의미 표현을 학습한다. 이 방법은 계산량이 적고 일반화 성능도 잘 유지된다.

레이어 고정(Layer Freezing)은 가장 널리 사용되는 기법이다. 초기 레이어는 완전히 고정(Frozen)하고 마지막 몇 개의 트랜스포머 블록(Transformer Block)만 학습한다. 데이터셋의 크기와 목표 작업에 따라 일부 정규화 계층(Normalization Layer)이나 어텐션 모듈(Attention Module)만 선택적으로 학습하는 방식도 많이 사용된다.

최근에는 파라미터 효율적 미세조정(Parameter-Efficient Fine-Tuning, PEFT)이 매우 중요한 연구 분야가 되었다. 전체 모델을 수정하지 않고 아주 작은 수의 파라미터만 학습하기 때문에 GPU 메모리 사용량과 학습 시간이 크게 감소한다. 엣지 AI(Edge AI)와 산업용 로봇에서는 이러한 방식이 사실상 표준이 되어가고 있다.

어댑터(Adapter)는 가장 성공적인 PEFT 기법 가운데 하나이다. 기존 트랜스포머 계층 사이에 작은 신경망을 삽입하고, 이 부분만 학습한다. 원래의 비전 인코더는 그대로 유지되므로 여러 로봇 플랫폼이 동일한 기본 모델을 공유하면서도 산업용, 의료용, 농업용, 물류용 등 각각 다른 어댑터만 교체하여 사용할 수 있다.

LoRA(Low-Rank Adaptation)는 또 다른 대표적인 PEFT 기술이다. 거대한 가중치 행렬(Weight Matrix)을 직접 수정하지 않고, 저차원(Low-Rank) 행렬만 학습하여 전체 모델을 적응시킨다. 전체 파라미터의 1% 이하만 학습해도 높은 성능을 유지할 수 있기 때문에 최근 VLA와 대규모 언어 모델(LLM)에서 가장 널리 사용되는 방법 가운데 하나이다.

프롬프트 튜닝(Prompt Tuning)은 언어 모델의 프롬프트 개념을 비전에 적용한 기술이다. 학습 가능한 시각 프롬프트(Visual Prompt)를 입력 영상 앞에 추가하여 모델의 주의를 특정 작업에 집중시킨다. 예를 들어 파지 위치, 공구 인터페이스, 위험 영역과 같은 정보를 더욱 강조하도록 유도할 수 있으며, 매우 적은 계산량으로도 높은 성능 향상을 얻을 수 있다.

최근에는 언어 기반 미세조정(Language-Aware Fine-Tuning)도 활발하게 연구되고 있다. 단순히 이미지만 학습하는 것이 아니라, 자연어 명령(Natural Language Instruction), 행동(Action), 환경(Context)을 함께 학습하여 비전 특징과 언어 임베딩(Language Embedding)을 자연스럽게 정렬한다. 이를 통해 VLA 시스템의 언어 이해와 작업 계획 능력이 크게 향상된다.

데이터 품질(Data Quality)은 비전 인코더 미세조정에서 가장 중요한 요소이다. 실제 배포 환경과 유사한 카메라 시점, 조명 조건, 작업 공간, 물체 종류, 로봇 플랫폼, 작업 순서를 포함하는 데이터셋이 필요하다. 단순히 데이터의 양을 늘리는 것보다 실제 환경을 잘 반영하는 데이터셋을 구축하는 것이 훨씬 중요한 경우가 많다.

멀티카메라(Multi-Camera) 데이터는 미세조정 성능을 더욱 향상시킨다. 헤드 카메라(Head Camera)는 전역 환경을 제공하고, 손목 카메라(Wrist Camera)는 조작 대상의 세부 정보를 제공하며, 스테레오 카메라는 깊이 정보를 생성한다. 여러 시점을 동시에 학습하면 시점 변화(Viewpoint Variation)에 강한 특징 표현을 얻을 수 있다.

깊이 정보(Depth Information)도 매우 중요한 입력이다. 기존 비전 모델은 RGB 영상만 처리하지만, 로봇은 RGB와 깊이(Depth), 포인트 클라우드(Point Cloud)를 함께 사용한다. 이러한 다중모달(Multimodal) 학습을 통해 의미 정보와 기하학적 정보를 동시에 학습할 수 있으며, 조작과 자율주행 성능이 크게 향상된다.

포인트 클라우드(Point Cloud)를 함께 사용하는 미세조정도 증가하고 있다. RGB 영상과 3차원 기하학을 동시에 학습하면 물체의 형태, 공간 관계, 표면 구조, 접근 가능성 등을 더욱 정확하게 이해할 수 있다. 이러한 특징은 3차원 공간 추론(3D Spatial Reasoning)에 매우 효과적이다.

시간 정보(Temporal Information)를 활용하는 미세조정도 중요하다. 로봇은 정적인 이미지가 아니라 연속적인 비디오(Video)를 관찰한다. 따라서 시간 트랜스포머(Temporal Transformer), 순환 신경망(Recurrent Neural Network), 메모리(Memory)를 사용하여 시간에 따라 변화하는 환경을 이해하는 특징을 학습한다. 이는 장기 작업(Long-Horizon Task)에서 매우 중요하다.

자기지도학습(Self-Supervised Learning)은 라벨(Label)이 부족한 로봇 환경에서 매우 유용하다. 대조학습(Contrastive Learning), 마스크 이미지 모델링(Masked Image Modeling), 다중 시점 일관성(Multi-View Consistency), 미래 예측(Predictive Learning) 등을 이용하여 사람이 직접 라벨을 붙이지 않아도 의미 있는 특징을 학습할 수 있다.

시범 데이터(Demonstration Data)도 중요한 학습 자원이다. 사람의 원격 조작(Teleoperation), 직접 교시(Kinesthetic Teaching), 모방학습(Imitation Learning)을 통해 수집된 데이터는 성공적인 행동과 영상을 함께 제공한다. 이를 통해 비전 인코더는 단순한 이미지 인식이 아니라 실제 행동과 관련된 특징을 학습할 수 있다.

작업(Task)에 따라 미세조정 방식도 달라진다. 파지 검출(Grasp Detection)은 접촉면과 물체 방향을 강조하고, 비주얼 서보잉(Visual Servoing)은 위치 정확도를 중요하게 학습한다. 산업 검사(Industrial Inspection)는 결함 탐지를 중심으로 학습하며, 자율주행은 장애물과 주행 가능 영역을 중심으로 최적화된다.

산업용 로봇(Industrial Robot)은 매우 특수한 환경을 가진다. 금속 표면(Metal Surface), 반복적인 부품, 반사체, 미세한 조립 공정 등이 많기 때문에 일반 비전 모델만으로는 충분하지 않다. 산업 환경에 맞게 미세조정하면 유사한 부품을 구별하고, 조립 상태를 정확하게 판단하며, 생산 품질을 향상시킬 수 있다.

물류 로봇(Warehouse Robot)은 다양한 박스(Box), 바코드(Barcode), 선반(Shelf), 포장재(Package)를 인식해야 한다. 창고 데이터로 미세조정된 비전 인코더는 복잡한 적재 상태와 다양한 조명 환경에서도 높은 정확도로 물체를 인식하고 피킹(Picking) 작업을 수행할 수 있다.

농업 로봇(Agricultural Robot)은 계절 변화, 날씨, 토양, 작물 성장에 따라 환경이 계속 변한다. 일반 이미지 데이터셋으로는 이러한 변화를 충분히 학습할 수 없기 때문에 농업 전용 데이터셋을 이용한 미세조정이 필요하다. 이를 통해 과일 수확, 잡초 제거, 병충해 탐지 등의 성능을 향상시킬 수 있다.

의료 로봇(Medical Robot)은 내시경 영상(Endoscopic Image), 생체 조직(Biological Tissue), 수술 기구(Surgical Instrument)와 같이 일반 이미지와 매우 다른 환경을 다룬다. 의료 데이터로 미세조정된 비전 인코더는 조직 분할, 기구 추적, 수술 보조 등에서 높은 정확도를 제공한다.

휴머노이드(Humanoid)는 가장 다양한 환경을 경험하는 로봇이다. 가정, 사무실, 병원, 공장 등에서 사람과 함께 생활해야 하므로, 매우 넓은 범위의 시각 정보를 이해해야 한다. 따라서 일반적인 의미 이해와 로봇 특화 능력을 동시에 유지하는 균형 잡힌 미세조정이 매우 중요하다.

시뮬레이션(Simulation)은 비전 인코더 학습에서 매우 중요한 역할을 한다. 가상 환경에서는 RGB, 깊이, 포인트 클라우드, 자세(Pose), 의미 분할(Semantic Segmentation), 조작 데이터까지 무한히 생성할 수 있다. 이후 실제 데이터로 소량의 추가 미세조정을 수행하는 Sim-to-Real 전략은 데이터 수집 비용을 크게 줄이면서 높은 성능을 제공한다.

도메인 적응(Domain Adaptation)은 시뮬레이션과 실제 환경의 차이를 줄이는 기술이다. 특징 정렬(Feature Alignment), 적대적 학습(Adversarial Adaptation), 스타일 변환(Style Transfer), 자기학습(Self-Training) 등을 이용하여 시뮬레이션에서 학습한 모델이 실제 환경에서도 높은 성능을 유지하도록 만든다.

비전 인코더의 평가는 단순한 이미지 정확도만으로는 충분하지 않다. 실제 로봇에서는 조작 성공률(Manipulation Success Rate), 파지 안정성(Grasp Stability), 자율주행 성능(Navigation Performance), 충돌 회피(Collision Avoidance), 언어 이해(Language Grounding), 자세 추정(Pose Estimation), 일반화(Generalization) 등을 종합적으로 평가해야 한다.

실시간 추론(Real-Time Inference)을 위해서는 계산 효율성도 매우 중요하다. 양자화(Quantization), 혼합 정밀도(Mixed Precision), 프루닝(Pruning), TensorRT 최적화, ONNX 변환, AI 가속기(AI Accelerator)를 적용하여 엣지 컴퓨터에서도 빠른 추론이 가능하도록 최적화한다.

지속적 학습(Continual Learning)은 차세대 로봇 비전의 핵심 방향이다. 로봇은 한 번 학습하고 끝나는 것이 아니라 새로운 환경, 새로운 작업, 새로운 물체를 계속 경험하면서 자신의 비전 인코더를 지속적으로 개선한다. 이를 위해 리플레이 메모리(Replay Memory), 적응형 정규화(Adaptive Regularization), 모듈 확장(Modular Expansion) 등의 기술이 연구되고 있다.

미래의 비전 기반 모델은 시각(Vision), 언어(Language), 월드 모델(World Model), 행동(Action)을 하나의 통합 구조에서 공동으로 학습하게 될 것이다. 비전 인코더는 더 이상 독립적인 인식 모듈이 아니라, 추론(Reasoning), 계획(Planning), 행동(Action)을 동시에 지원하는 핵심 표현 생성기로 발전할 것이다.

또한 플랫폼 독립적(Cross-Embodiment) 비전 모델도 중요한 연구 방향이다. 하나의 비전 인코더가 이동 로봇(AMR), 매니퓰레이터(Manipulator), 휴머노이드(Humanoid), 사족보행 로봇(Quadruped), 드론(Drone) 등 다양한 로봇에서 공통으로 사용되고, 어댑터(Adapter)나 LoRA만 변경하여 각 플랫폼에 맞게 적응하는 방식이 점차 확대될 것으로 예상된다.

결국 비전 인코더 미세조정(Vision Encoder Fine-Tuning)은 대규모 비전 기반 모델(Vision Foundation Model)의 일반적인 시각 지식과 로봇 환경의 특수성을 연결하는 핵심 기술이다. 사전학습은 광범위한 의미 이해를 제공하고, 미세조정은 조작(Manipulation), 자율주행(Navigation), 공간 추론(Spatial Reasoning), 안전(Safety), 인간-로봇 상호작용(Human-Robot Interaction)에 필요한 표현을 강화한다. 파라미터 효율적 학습(Parameter-Efficient Learning), 멀티모달 데이터(Multimodal Data), 시뮬레이션(Simulation), 지속적 학습(Continual Learning)을 결합함으로써 미래의 VLA 시스템은 산업 자동화, 서비스 로봇, 의료, 물류, 농업, 범용 물리 인공지능(Physical AI)을 위한 더욱 강력한 시각 인식 능력을 갖추게 될 것이다.

## 2.10 Real-Time Vision Encoder Optimization (with Code)

![](images/image10.png){width="7.268055555555556in" height="7.268055555555556in"}

실시간(Real-Time) 비전은 현대 로봇 시스템의 핵심 요구사항 가운데 하나이다. 로봇은 주변 환경을 지속적으로 관찰하고, 상황을 이해하며, 행동을 계획하고, 안전한 제어 명령을 매우 짧은 시간 안에 생성해야 한다. 아무리 높은 정확도를 가진 비전 인코더(Vision Encoder)라도 추론 지연(Latency)이 지나치게 크면 실제 로봇에서는 사용할 수 없다. 장애물을 1초 뒤에 정확하게 인식하는 로봇은 이미 충돌했을 가능성이 높기 때문이다. 따라서 비전-언어-행동(Vision-Language-Action, VLA) 시스템에서는 비전 인코더의 지연 시간을 최소화하는 것이 매우 중요한 연구 분야가 되었다.

비전 인코더는 대부분의 VLA 시스템에서 가장 앞단에 위치한다. RGB 카메라, 스테레오 카메라(Stereo Camera), 깊이 카메라(Depth Camera), 열화상 카메라(Thermal Camera), 이벤트 카메라(Event Camera), LiDAR 등으로부터 입력받은 데이터를 의미 있는 특징 벡터(Feature Embedding)로 변환하여 언어 모델(Language Model), 월드 모델(World Model), 작업 계획기(Task Planner), 행동 정책(Action Policy)에 전달한다. 이후의 모든 처리 과정은 비전 인코더의 결과에 의존하므로, 비전 인코더의 추론 속도는 전체 로봇 시스템의 응답성을 결정하는 핵심 요소가 된다.

일반적인 컴퓨터 비전은 한 장의 이미지를 처리하는 데 수백 밀리초가 걸려도 문제가 되지 않는 경우가 많다. 그러나 로봇은 초당 30\~120장의 영상을 지속적으로 처리하면서 동시에 움직여야 한다. 각 프레임(Frame)은 객체 탐지(Object Detection), 의미 분할(Semantic Segmentation), 깊이 추정(Depth Estimation), 장면 이해(Scene Understanding), 언어 정렬(Language Grounding), 행동 추론(Action Inference)을 수행해야 하므로 허용 가능한 계산 시간이 매우 제한적이다. 비전 처리 시간이 길어질수록 조작 정확도와 주행 안정성은 급격히 감소하게 된다.

실시간 최적화는 전체 인식 파이프라인(Perception Pipeline)을 이해하는 것에서 시작된다. 일반적인 로봇은 여러 대의 카메라에서 영상을 수집하고, 크기 조정(Image Resize), 정규화(Normalization), 색상 변환(Color Conversion), 왜곡 보정(Distortion Correction), 센서 동기화(Sensor Synchronization)를 수행한다. 이후 비전 인코더가 특징을 추출하고, 언어 모델과 계획기가 이를 이용하여 행동을 생성한 후 제어기가 모터 명령을 출력한다. 이 모든 과정이 합쳐져 전체 지연 시간이 결정되므로, 특정 모델만 빠르게 만드는 것이 아니라 시스템 전체를 최적화해야 한다.

실시간 시스템에서는 처리량(Throughput)과 지연 시간(Latency)을 구분해야 한다. 처리량은 초당 몇 장의 이미지를 처리할 수 있는지를 의미하며, 지연 시간은 하나의 영상이 입력된 후 결과가 출력되기까지 걸리는 시간을 의미한다. 클라우드 AI는 높은 처리량을 중요하게 생각하지만, 로봇은 하나의 프레임이라도 가능한 한 빨리 처리해야 하므로 지연 시간이 훨씬 중요한 성능 지표가 된다.

제어 주기(Control Frequency)는 허용 가능한 지연 시간을 결정한다. 산업용 로봇은 일반적으로 수백 Hz 이상의 제어 주기를 사용하며, 비전 제어는 30\~100Hz 정도로 수행된다. 이동 로봇은 약 10\~50Hz, 휴머노이드(Humanoid)는 균형 제어를 위해 30ms 이하의 시각 응답이 요구되는 경우가 많다. 따라서 대부분의 VLA 시스템은 수 밀리초에서 수십 밀리초 이내의 비전 추론을 목표로 설계된다.

비전 인코더의 구조(Model Architecture)는 지연 시간을 결정하는 가장 중요한 요소이다. 기존 CNN(Convolutional Neural Network)은 계산 효율이 높지만 전역 문맥(Global Context)을 이해하는 능력이 제한적이었다. 반면 비전 트랜스포머(Vision Transformer, ViT)는 긴 거리의 의미 관계를 잘 학습하지만, 자기 어텐션(Self-Attention)의 계산량이 입력 토큰(Token)의 제곱에 비례하여 증가한다. 따라서 최신 비전 인코더는 표현 능력과 계산 효율을 동시에 고려한 구조를 채택한다.

입력 영상의 해상도(Image Resolution)는 계산량에 직접적인 영향을 준다. 해상도가 두 배가 되면 처리해야 하는 토큰 수는 약 네 배로 증가하며, 트랜스포머의 계산량도 크게 증가한다. 따라서 고해상도 영상은 정밀 조작에만 사용하고, 일반적인 이동이나 장애물 회피에는 낮은 해상도를 사용하는 것이 일반적인 최적화 방법이다.

적응형 해상도(Adaptive Resolution)는 작업 상황에 따라 입력 해상도를 동적으로 변경하는 방법이다. 장거리 자율주행에서는 저해상도 영상만 사용하고, 물체에 가까워졌을 때만 고해상도 처리를 수행한다. 이렇게 하면 전체 평균 계산량을 크게 줄이면서도 정밀 조작 성능은 유지할 수 있다.

관심 영역(Region of Interest, ROI) 기반 처리는 전체 이미지를 모두 처리하지 않고 중요한 부분만 고해상도로 분석하는 기술이다. 먼저 가벼운 객체 탐지기가 관심 영역을 찾고, 이후 대형 비전 인코더가 해당 영역만 정밀하게 분석한다. 사람의 시각이 중요한 대상에만 집중하는 것과 매우 유사한 방식이다.

비전 트랜스포머에서는 패치 선택(Patch Selection)이 매우 중요한 최적화 기술이다. 이미지는 작은 패치(Patch) 단위로 나누어 처리되는데, 모든 패치가 동일하게 중요한 것은 아니다. 토큰 프루닝(Token Pruning)은 중요도가 낮은 패치를 제거하고, 중요한 패치에만 계산 자원을 집중함으로써 추론 시간을 크게 줄일 수 있다.

동적 토큰 라우팅(Dynamic Token Routing)은 더욱 발전된 방식이다. 모든 토큰이 모든 트랜스포머 계층을 통과하는 것이 아니라, 중요한 토큰만 깊은 계산을 수행하고 나머지는 일부 계층을 건너뛴다. 이를 통해 계산량은 감소하지만 인식 성능은 거의 유지할 수 있다.

지식 증류(Knowledge Distillation)는 대형 비전 모델의 지식을 작은 학생 모델(Student Model)로 전달하는 방법이다. 학생 모델은 훨씬 적은 파라미터(Parameter)를 사용하지만 교사 모델(Teacher Model)의 특징 표현을 학습하므로, 실시간 로봇 시스템에서도 높은 정확도를 유지하면서 빠른 추론이 가능하다.

모델 프루닝(Model Pruning)은 필요 없는 채널(Channel), 뉴런(Neuron), 어텐션 헤드(Attention Head), 트랜스포머 블록(Transformer Block)을 제거하는 기술이다. 특히 구조적 프루닝(Structured Pruning)은 실제 하드웨어에서 높은 가속 효과를 제공하기 때문에 산업용 로봇에서 많이 사용된다.

양자화(Quantization)는 가장 널리 사용되는 최적화 기법이다. 일반적인 FP32(Float32) 계산 대신 INT8, FP16, BF16, INT4와 같은 저정밀도 계산을 사용하면 메모리 사용량과 연산량이 크게 감소한다. 최신 GPU와 AI 가속기(AI Accelerator)는 이러한 저정밀도 연산을 매우 빠르게 수행할 수 있다.

양자화 인식 학습(Quantization-Aware Training)은 학습 과정부터 저정밀도 연산을 고려하는 방법이다. 추론 단계에서 갑자기 양자화를 적용하는 것보다 정확도 손실이 훨씬 적으며, 실제 산업용 AI 모델에서는 매우 많이 활용된다.

혼합 정밀도(Mixed Precision)는 계층마다 서로 다른 계산 정밀도를 사용하는 기술이다. 대부분의 연산은 FP16으로 수행하고, 정밀도가 중요한 계층만 FP32를 유지한다. NVIDIA Tensor Core는 이러한 혼합 정밀도 계산을 매우 빠르게 수행하여 비전 인코더의 속도를 크게 향상시킨다.

TensorRT는 NVIDIA 플랫폼에서 가장 널리 사용되는 추론 최적화 프레임워크이다. 연산 그래프(Graph)를 분석하여 불필요한 연산을 제거하고, 연산 결합(Operator Fusion), 메모리 최적화(Memory Optimization), 커널 선택(Kernel Selection)을 수행한다. 동일한 모델이라도 TensorRT를 적용하면 상당한 수준의 지연 시간 감소를 얻을 수 있다.

연산 결합(Operator Fusion)은 여러 개의 연속된 연산을 하나의 커널(Kernel)로 통합하는 기술이다. 예를 들어 컨볼루션(Convolution), 정규화(Normalization), 활성화 함수(Activation Function)를 하나로 합치면 메모리 접근 횟수와 커널 실행 횟수가 줄어들어 추론 속도가 향상된다.

ONNX(Open Neural Network Exchange)는 다양한 하드웨어에서 동일한 모델을 사용할 수 있도록 하는 표준 포맷이다. TensorRT, OpenVINO, ARM AI, Qualcomm AI Engine 등 다양한 추론 엔진에서 사용할 수 있으므로 로봇 플랫폼 간 이식성이 매우 높다.

배치 처리(Batch Processing)는 서버 AI에서는 매우 중요하지만 로봇에서는 다르게 접근해야 한다. 서버는 여러 이미지를 한꺼번에 처리하여 처리량을 높이지만, 로봇은 가장 최근의 영상 하나를 가능한 빨리 처리해야 한다. 따라서 대부분의 로봇은 배치 크기(Batch Size)를 1로 유지하면서 지연 시간을 최소화하는 방향으로 최적화된다.

파이프라인 병렬화(Pipeline Parallelism)는 인식, 추론, 계획, 제어를 동시에 수행하는 방식이다. 비전 인코더가 다음 프레임을 처리하는 동안 언어 모델은 이전 프레임을 분석하고, 제어기는 현재 명령을 실행한다. 이러한 비동기 처리(Asynchronous Processing)는 전체 시스템의 응답 속도를 크게 향상시킨다.

멀티스레드(Multi-threading)도 중요한 최적화 방법이다. 카메라 입력, 이미지 전처리, AI 추론, 센서 융합, 지도 작성, 제어 등을 각각 독립적인 스레드(Thread)에서 수행하면 CPU와 GPU를 효율적으로 사용할 수 있으며 전체 지연 시간을 줄일 수 있다.

메모리 최적화(Memory Optimization)는 계산 최적화만큼 중요하다. 트랜스포머는 매우 큰 특징 맵(Feature Map)과 가중치를 반복적으로 읽고 쓰기 때문에 메모리 병목(Memory Bottleneck)이 자주 발생한다. 메모리 재사용(Memory Reuse), 캐시(Cache) 최적화, 비동기 전송(Asynchronous Transfer)을 적용하면 추론 속도를 크게 향상시킬 수 있다.

특징 캐싱(Feature Caching)은 연속된 프레임 사이의 중복 계산을 줄이는 기술이다. 로봇이 정지해 있거나 환경 변화가 거의 없는 경우에는 이전 프레임의 특징을 재사용하고 변경된 부분만 다시 계산하여 연산량을 크게 줄일 수 있다.

시간적 중복성(Temporal Redundancy) 역시 중요한 최적화 대상이다. 연속된 영상은 대부분 비슷하므로 모든 프레임을 처음부터 계산할 필요가 없다. 시간 트랜스포머(Temporal Transformer), 상태 공간 모델(State Space Model), 특징 전파(Feature Propagation)를 이용하면 이전 계산 결과를 활용하여 빠른 추론이 가능하다.

멀티카메라(Multi-Camera) 시스템에서는 여러 대의 카메라를 각각 독립적으로 처리하면 계산량이 크게 증가한다. 따라서 공유 백본(Shared Backbone)을 이용하여 공통 특징을 추출한 후, 헤드 카메라(Head Camera), 손목 카메라(Wrist Camera), 스테레오 카메라(Stereo Camera)의 특징만 추가적으로 처리하는 방식이 많이 사용된다.

센서 융합(Sensor Fusion)도 지연 시간에 영향을 미친다. RGB 카메라, LiDAR, IMU, 힘 센서(Force Sensor)는 서로 다른 주기로 데이터를 생성한다. 모든 센서가 도착할 때까지 기다리면 지연이 증가하므로, 최근에는 비동기 센서 융합(Asynchronous Sensor Fusion)을 통해 즉시 처리 가능한 데이터를 먼저 사용하는 방식이 많이 연구되고 있다.

엣지 컴퓨팅(Edge Computing)은 로봇에서 매우 중요한 제약 조건이다. 이동 로봇은 제한된 배터리와 제한된 GPU를 사용해야 하므로 전력(Power), 발열(Thermal), 메모리(Memory), 크기(Size)를 모두 고려해야 한다. 따라서 지연 시간을 줄이는 것은 배터리 사용 시간까지 증가시키는 효과를 가진다.

AI 가속기(AI Accelerator)는 이러한 문제를 해결하기 위한 핵심 기술이다. GPU는 범용성이 높고, NPU(Neural Processing Unit)는 전력 효율이 뛰어나며, FPGA는 결정론적(Deterministic) 지연 시간을 제공하고, ASIC은 특정 모델에서 최고의 성능을 제공한다. 따라서 하드웨어 선택도 지연 시간 최적화의 중요한 요소이다.

ROS 2(Robot Operating System 2)는 통신 과정에서도 추가적인 지연이 발생한다. 메시지 직렬화(Serialization), DDS(Data Distribution Service), 콜백(Callback), QoS(Quality of Service) 설정 등에 따라 실제 시스템의 지연 시간이 크게 달라질 수 있으므로, 공유 메모리(Shared Memory), Zero-Copy 통신 등을 활용한 최적화가 필요하다.

스케줄링(Scheduling) 전략도 중요하다. 일정한 주기로 항상 비전을 수행하는 고정 주기 방식(Fixed-Rate Scheduling)도 있지만, 환경 변화가 있을 때만 AI를 실행하는 이벤트 기반(Event-Driven) 방식이나, 로봇 속도와 환경 복잡도에 따라 처리 빈도를 조절하는 적응형 스케줄링(Adaptive Scheduling)도 활발히 연구되고 있다.

실시간 최적화에서는 속도만 빠른 것이 아니라 정확도와의 균형도 중요하다. 지나친 프루닝, 과도한 양자화, 지나치게 낮은 해상도는 추론 속도는 향상시키지만 인식 성능과 조작 성공률을 떨어뜨릴 수 있다. 따라서 실제 시스템에서는 정확도와 지연 시간 사이의 최적 균형점(Pareto Optimal Point)을 찾는 것이 핵심이다.

실시간 비전 인코더는 단순히 AI 모델의 추론 속도만 측정해서는 안 된다. 카메라 노출(Camera Exposure), 영상 획득(Image Acquisition), 전처리, AI 추론, 특징 전달, 언어 추론, 행동 생성, 모터 제어까지 포함한 전체 End-to-End 지연 시간을 평가해야 실제 로봇의 성능을 정확하게 판단할 수 있다.

프로파일링(Profiling)은 최적화 과정에서 반드시 필요한 단계이다. GPU 프로파일러(GPU Profiler)는 병목 구간을 찾고, 메모리 분석기는 캐시 사용과 메모리 접근을 분석하며, 전력 측정기는 소비 전력을 평가한다. 이러한 분석 결과를 기반으로 가장 효과적인 최적화 지점을 찾을 수 있다.

미래에는 적응형 계산(Adaptive Computation)이 더욱 중요해질 것으로 예상된다. 단순한 장면에서는 적은 계산만 수행하고, 복잡한 작업이나 위험 상황에서는 더 많은 연산을 수행하는 방식이다. 이는 사람의 시각 시스템이 어려운 상황일수록 더 집중하는 방식과 매우 유사하다.

또한 하드웨어-소프트웨어 공동 설계(Hardware-Software Co-design)도 중요한 연구 분야이다. AI 모델, GPU, NPU, 메모리 구조, 운영체제를 함께 설계하면 기존 방식보다 훨씬 높은 지연 시간 감소와 전력 효율을 얻을 수 있다.

결국 비전 인코더 지연 시간 최적화(Vision Encoder Latency Optimization)는 신경망 구조 설계(Neural Architecture Design), 모델 압축(Model Compression), 양자화(Quantization), TensorRT 최적화, 메모리 관리(Memory Management), 비동기 처리(Asynchronous Processing), 센서 융합(Sensor Fusion), 운영체제 최적화(OS Optimization), AI 가속기(AI Accelerator)를 모두 포함하는 종합적인 시스템 엔지니어링 분야이다. 정확한 인식만으로는 실제 로봇을 구현할 수 없으며, 빠르고 안정적인 실시간 추론이 함께 이루어질 때 비로소 산업용 로봇, 자율주행 로봇, 휴머노이드(Humanoid), 범용 물리 인공지능(Physical AI)을 위한 VLA 시스템이 안전하고 효율적으로 동작할 수 있다.
