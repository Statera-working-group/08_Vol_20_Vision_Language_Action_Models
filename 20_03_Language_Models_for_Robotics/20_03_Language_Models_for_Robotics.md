**Volume 20. Vision Language Action (VLA) Models**


# Chapter 3. Language Models for Robotics

##  

## 3.1 The Role of Large Language Models in Robotics

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

\`42_03_01_LLM_Role_in_Robotics_Task_Understanding_Planning\` is an appropriate chapter under **Volume 42 → 42.03 Language Models for Robotics**, focusing on how Large Language Models serve as cognitive reasoning engines within Vision-Language-Action (VLA) systems for robotic task understanding and planning. This aligns with the uploaded library structure.

Large Language Models have fundamentally changed the role of artificial intelligence in robotics by enabling robots to understand human intentions, reason about complex tasks, generate executable plans, and communicate naturally with users. Unlike traditional robotic software, where every task must be explicitly programmed as a sequence of predefined operations, LLM-powered robots interpret high-level natural language instructions and transform them into structured plans that can ultimately be executed by robot control systems. This capability bridges the long-standing gap between human language and robotic actions, making robots significantly more flexible, adaptive, and capable of operating in dynamic environments.

In modern Vision-Language-Action architectures, the Large Language Model functions primarily as a cognitive planning module rather than a low-level controller. The LLM does not directly generate motor torques or servo commands. Instead, it receives multimodal information from perception modules, interprets user intentions, reasons about the current environment, determines an appropriate sequence of actions, and generates semantic task plans that are later translated into executable robot skills by downstream planning and control systems.

The overall robotic cognitive pipeline begins with multimodal perception. Cameras, depth sensors, LiDAR systems, force sensors, tactile sensors, joint encoders, and environmental maps continuously provide information describing the robot\'s surroundings and internal state. Vision encoders transform raw sensor observations into compact semantic embeddings that describe objects, spatial relationships, scene context, and environmental conditions. These representations become part of the prompt provided to the Large Language Model together with user instructions and robot status information.

Human instructions are generally expressed in natural language rather than robot programming languages. A user may instruct a robot to bring a bottle of water from the kitchen, inspect a warehouse shelf, organize scattered tools, or deliver a package to a specific location. Such instructions are inherently ambiguous because humans naturally omit assumptions that other humans easily infer from context. The LLM serves as an interpreter that fills these missing semantic gaps by leveraging its extensive prior knowledge acquired during pretraining.

Task understanding begins by identifying the user\'s primary objective. Rather than interpreting words individually, the LLM analyzes the complete instruction, extracting task intent, desired outcomes, constraints, environmental assumptions, temporal ordering, and implicit requirements. It identifies verbs representing actions, nouns representing target objects, adjectives describing properties, locations indicating spatial goals, and conditions that constrain execution.

For example, the instruction "Please bring me the red coffee mug from the kitchen table" requires considerably more reasoning than a simple object detection task. The robot must understand that the final goal is delivering the mug to the user rather than merely locating it. The phrase "red coffee mug" specifies semantic object properties, while "kitchen table" provides spatial context. The LLM organizes these concepts into a structured representation that later guides perception, navigation, manipulation, and interaction.

Context understanding represents another important responsibility of the LLM. Human instructions frequently depend on shared context rather than explicit descriptions. Commands such as "Pick up the one I used yesterday" or "Return it to where you found it" require memory of previous interactions. By maintaining conversational history and contextual information, the LLM allows robots to perform long-term reasoning across multiple interactions rather than treating each instruction independently.

Modern robotic systems also require commonsense reasoning that cannot easily be encoded using traditional programming. Humans naturally understand that fragile objects should be handled carefully, liquids should remain upright, doors must be opened before passing through, and heavy objects require stable grasps. LLMs possess extensive commonsense knowledge acquired during large-scale language pretraining, allowing robots to infer many of these unstated requirements during task planning.

After understanding user intent, the LLM proceeds to hierarchical task decomposition. Complex missions rarely consist of a single robotic action. Instead, they are decomposed into a sequence of smaller subgoals that collectively achieve the desired objective. A household cleaning task may involve locating cleaning supplies, navigating to multiple rooms, identifying dirty surfaces, selecting appropriate cleaning tools, performing cleaning operations, verifying completion, and returning equipment to storage. The LLM organizes these subtasks into an ordered plan while preserving dependencies among individual operations.

Hierarchical planning significantly improves robustness because each subgoal can be independently monitored, verified, and corrected if necessary. Rather than attempting to solve an entire mission as one monolithic optimization problem, robots incrementally accomplish smaller objectives that gradually converge toward the overall task goal.

Semantic planning differs fundamentally from geometric planning. Traditional motion planners compute collision-free trajectories through configuration space using mathematical optimization techniques. LLMs instead reason about semantic relationships among tasks, objects, and goals. They determine what should be done rather than precisely how every joint should move. This separation of responsibilities enables the LLM to operate at a high level while lower-level planning algorithms generate feasible robot trajectories.

The planning process frequently incorporates external knowledge. Many robotic tasks require understanding of objects, environments, or procedures beyond what can be directly observed. LLMs integrate world knowledge, procedural knowledge, and commonsense reasoning to generate more intelligent plans. A warehouse robot, for example, understands that fragile packages require gentle handling, refrigerated products belong in cold storage, hazardous materials require special precautions, and heavy objects should be transported using appropriate equipment.

Environmental understanding also plays a crucial role. The LLM reasons about scene descriptions generated by Vision-Language Models or visual encoders, allowing it to understand spatial relationships such as "the box is under the table," "the screwdriver is next to the toolbox," or "the charging station is behind the robot." These semantic relationships support more effective planning than relying solely on numerical coordinates.

Constraint reasoning represents another essential capability. Robot tasks often include physical, temporal, safety, or operational constraints. The LLM considers these constraints while generating plans. Battery capacity, payload limitations, joint range restrictions, collision avoidance requirements, workspace boundaries, human safety regulations, and task priorities all influence the resulting action sequence.

Temporal reasoning enables robots to execute tasks in appropriate order. Certain operations cannot begin until prerequisite actions have completed successfully. A robot cannot place an object before grasping it, cannot inspect an enclosed container before opening it, and cannot navigate through a closed door without first opening it. The LLM naturally captures these dependencies through sequential reasoning.

Goal refinement frequently occurs during planning. Human instructions are often intentionally brief because people assume contextual understanding. The LLM expands these abbreviated requests into complete operational objectives. A request such as "Set the table" becomes a structured plan involving plate placement, utensil arrangement, cup positioning, napkin organization, and verification of completion according to learned dining conventions.

The LLM also generates intermediate symbolic representations that simplify downstream processing. Instead of directly producing motor commands, it creates structured task graphs, behavior trees, symbolic action sequences, planning trees, JSON representations, or domain-specific planning languages that subsequent software modules can interpret and execute.

Task planning increasingly incorporates uncertainty estimation. Real-world environments contain incomplete observations, ambiguous language, occluded objects, and dynamic changes. Modern LLM-based planning systems explicitly identify uncertain assumptions and request additional information when necessary. Rather than making unsupported decisions, the robot may ask clarification questions such as "Which blue box do you mean?" or "Should I place the package inside the room or outside the door?"

Interactive planning significantly improves overall system reliability. Human-robot collaboration becomes a dialogue rather than a one-way command interface. The robot explains its interpretation of the task, requests clarification when ambiguity exists, confirms critical decisions, and updates plans as new information becomes available.

Modern robotic systems often employ retrieval-augmented planning. Rather than relying exclusively on pretrained knowledge, the LLM retrieves documentation, robot capabilities, environmental maps, equipment specifications, previous task histories, or company operating procedures from external databases. This retrieval process greatly improves planning accuracy while reducing hallucination.

Skill selection represents another major responsibility of the LLM. Robots typically possess libraries containing reusable motion primitives such as navigation, grasping, opening doors, pressing buttons, docking, visual inspection, object recognition, or charging. The LLM selects appropriate skills based on task requirements and determines the optimal execution sequence.

Task planning also integrates closely with behavior trees and hierarchical finite state machines. The LLM generates semantic task structures that are subsequently converted into executable robotic workflows. This combination preserves the flexibility of language-based reasoning while maintaining the deterministic execution characteristics required for industrial robotics.

Reasoning over multiple objectives is increasingly important in autonomous robots. A service robot may simultaneously optimize task completion time, energy consumption, safety, user comfort, and operational efficiency. The LLM evaluates these competing objectives and generates plans that balance multiple optimization criteria rather than pursuing a single metric.

In collaborative robotics, the LLM reasons about human intentions, predicts future human actions, and coordinates robot behavior accordingly. It determines when humans should perform specific subtasks, when robots should assist, when communication is required, and how responsibilities should be distributed among team members.

Dynamic replanning is another critical capability. Real-world environments continuously change due to moving people, relocated objects, unexpected obstacles, or equipment failures. The LLM monitors execution progress and updates task plans whenever environmental conditions deviate significantly from original assumptions. Rather than restarting the entire mission, the planner modifies only the affected portions of the execution sequence.

Memory plays an increasingly important role in long-duration robotic missions. Episodic memory records previous experiences, semantic memory stores general knowledge, procedural memory captures learned skills, and working memory maintains current task context. Together, these memory systems enable robots to improve planning quality across repeated interactions.

Modern VLA architectures frequently employ a hierarchical reasoning framework in which perception modules describe the environment, the LLM performs semantic reasoning, task planners generate executable workflows, motion planners compute feasible trajectories, controllers produce actuator commands, and feedback systems continuously monitor execution outcomes. This layered organization separates cognitive reasoning from physical control while maintaining efficient information flow between abstraction levels.

Despite remarkable progress, several challenges remain. LLMs may occasionally generate logically plausible but physically impossible plans, overlook safety constraints, misunderstand spatial relationships, or hallucinate nonexistent objects. Consequently, modern robotic systems incorporate verification modules, affordance estimation, geometric validation, simulation testing, and safety controllers that validate LLM-generated plans before execution.

Latency also presents practical challenges for real-time robotics. Large foundation models may require hundreds of milliseconds or even seconds to generate responses. Consequently, many robotic systems employ hierarchical architectures in which the LLM performs relatively infrequent high-level planning while low-latency local controllers execute continuous feedback control loops independently.

Future robotic systems are expected to integrate increasingly capable multimodal foundation models capable of jointly reasoning over language, images, depth maps, point clouds, audio, tactile feedback, force sensing, robot state, environmental memory, and long-term mission history. Rather than functioning as isolated language processors, these models will become comprehensive cognitive engines capable of understanding complex environments, predicting future events, coordinating multiple robots, adapting to unfamiliar situations, and generating robust long-horizon plans. As Vision-Language-Action systems continue to evolve, the Large Language Model will remain the central reasoning component that transforms human intentions into structured, executable robotic behaviors, enabling autonomous robots to operate safely, intelligently, and collaboratively across industrial, commercial, service, healthcare, logistics, construction, agricultural, and domestic environments.

대규모 언어 모델(LLM, Large Language Model)은 로봇공학(Robotics)에서 단순한 자연어 처리 도구를 넘어, 로봇의 인지(Cognition), 추론(Reasoning), 계획(Planning)을 담당하는 핵심 두뇌 역할을 수행한다. 기존 로봇은 모든 작업 절차를 사람이 미리 프로그래밍해야 했지만, LLM을 적용한 로봇은 사람의 자연어(Natural Language)를 이해하고 이를 실행 가능한 작업(Task) 계획으로 변환할 수 있다. 이러한 변화는 사람의 의도와 로봇의 행동 사이의 간극을 줄여 보다 유연하고 지능적인 자율 시스템을 가능하게 한다.

비전-언어-행동(VLA, Vision-Language-Action) 아키텍처(Architecture)에서 LLM은 저수준 제어(Low-Level Control)를 담당하지 않는다. 모터 토크(Torque)나 관절 명령(Joint Command)을 직접 생성하는 대신, 센서 정보를 이해하고 사용자의 의도를 분석하며, 전체 작업을 계획하고 실행 순서를 결정하는 고수준 인지 계층(High-Level Cognitive Layer)의 역할을 수행한다. 이후 생성된 계획은 하위 계획기(Planner)와 제어기(Controller)에 전달되어 실제 움직임으로 변환된다.

전체 인지 과정은 다중모달 인식(Multimodal Perception)으로 시작된다. 카메라(Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 힘 센서(Force Sensor), 촉각 센서(Tactile Sensor), 관절 엔코더(Joint Encoder) 등의 다양한 센서가 지속적으로 환경과 로봇의 상태를 측정한다. 비전 인코더(Vision Encoder)는 이러한 원시 데이터를 의미적 특징(Semantic Feature)으로 변환하며, 객체(Object), 위치(Location), 공간 관계(Spatial Relation), 장면(Scene)의 의미를 추출하여 LLM이 이해할 수 있는 형태로 제공한다.

사용자는 일반적으로 프로그래밍 언어가 아닌 자연어를 이용하여 로봇에게 작업을 지시한다. 예를 들어 "주방에서 빨간 컵을 가져와."와 같은 명령에는 목적(Objective), 대상(Object), 위치(Location), 최종 목표(Goal)가 모두 포함되어 있지만 구체적인 절차는 생략되어 있다. LLM은 이러한 문장을 해석하여 필요한 세부 작업을 자동으로 추론하고, 실행 가능한 구조화된 작업(Task Structure)으로 변환한다.

작업 이해(Task Understanding)의 첫 단계는 사용자의 의도(Intent)를 정확하게 파악하는 것이다. LLM은 단어를 개별적으로 해석하지 않고 문장 전체의 의미를 분석하여 작업 목적, 원하는 결과, 시간적 순서, 환경 조건, 제약 사항 등을 동시에 이해한다. 동사(Verb)는 수행할 행동을, 명사(Noun)는 대상 객체를, 형용사(Adjective)는 객체의 속성을, 장소 표현은 목표 위치를 의미하는 요소로 분석된다.

예를 들어 "주방 테이블 위의 빨간 머그컵을 가져와."라는 명령은 단순한 객체 탐지(Object Detection)가 아니다. 로봇은 최종 목표가 컵을 찾는 것이 아니라 사용자에게 전달하는 것임을 이해해야 하며, 빨간색이라는 속성과 주방 테이블이라는 공간적 정보를 동시에 고려해야 한다. LLM은 이러한 의미를 종합하여 로봇이 수행해야 할 작업 흐름을 생성한다.

LLM은 대화의 문맥(Context)도 함께 이해한다. 사람은 "어제 사용했던 것을 가져와.", "원래 있던 곳에 다시 놔."와 같이 이전 대화를 전제로 명령하는 경우가 많다. 이러한 문장은 단독으로는 의미가 불완전하지만, LLM은 이전 대화 기록과 작업 이력을 기억하여 현재 명령과 연결한다. 이를 통해 장기간 작업(Long-Term Task)과 연속적인 인간-로봇 상호작용(Human-Robot Interaction)이 가능해진다.

또한 LLM은 상식(Common Sense)을 활용하여 사람이 명시하지 않은 조건도 추론한다. 예를 들어 유리컵은 조심해서 들어야 하고, 액체가 담긴 컵은 기울이지 않아야 하며, 문을 통과하기 위해서는 먼저 문을 열어야 한다는 사실은 일반적인 로봇 프로그램에는 포함되어 있지 않을 수 있다. 하지만 LLM은 대규모 학습을 통해 이러한 상식을 내재화하고 있어 보다 자연스럽고 안전한 계획을 생성할 수 있다.

의도를 이해한 이후에는 계층적 작업 분해(Hierarchical Task Decomposition)가 수행된다. 대부분의 실제 작업은 하나의 행동으로 끝나지 않는다. 예를 들어 청소 작업이라면 청소 도구 찾기, 이동하기, 오염 위치 확인, 청소 수행, 결과 확인, 장비 정리 등의 여러 하위 작업(Subtask)으로 구성된다. LLM은 이러한 작업을 논리적인 순서로 분해하고 각 단계의 의존성(Dependency)을 고려하여 전체 계획을 구성한다.

계층적 계획(Hierarchical Planning)은 작업의 안정성을 크게 향상시킨다. 각각의 하위 작업은 독립적으로 실행 및 검증될 수 있으며, 특정 단계에서 문제가 발생하면 전체 작업을 처음부터 다시 시작하는 것이 아니라 해당 단계만 수정하거나 재계획(Replanning)하면 된다. 이는 실제 산업용 로봇과 서비스 로봇 모두에서 매우 중요한 특성이다.

LLM이 수행하는 의미 기반 계획(Semantic Planning)은 전통적인 기하학적 계획(Geometric Planning)과는 목적이 다르다. 기존 경로 계획(Path Planning)은 충돌 없는 이동 경로를 계산하는 것이 목적이지만, LLM은 무엇을 해야 하는지와 어떤 순서로 수행해야 하는지를 결정한다. 이후 실제 경로 생성은 모션 플래너(Motion Planner)가 담당하여 역할을 분리한다.

LLM은 외부 지식(External Knowledge)도 적극적으로 활용한다. 창고(Warehouse)에서는 깨지기 쉬운 물건은 조심해서 운반해야 하고, 냉장 물품은 냉장 구역으로 이동해야 하며, 위험 물질은 별도의 절차를 따라야 한다. 이러한 산업 지식(Domain Knowledge)은 작업 계획의 품질을 크게 향상시키며, 다양한 응용 분야에 쉽게 적용될 수 있다.

환경 이해(Environment Understanding) 역시 중요한 기능이다. 비전-언어 모델(VLM, Vision-Language Model)이나 비전 인코더가 생성한 장면 설명(Scene Description)을 기반으로 LLM은 "상자는 책상 아래 있다.", "드라이버는 공구함 옆에 있다.", "충전기는 로봇 뒤쪽에 있다."와 같은 공간 관계를 이해한다. 이러한 의미적 정보는 단순 좌표 기반 시스템보다 훨씬 높은 수준의 추론을 가능하게 한다.

작업 계획에서는 다양한 제약 조건(Constraint)도 함께 고려된다. 배터리 용량(Battery Capacity), 적재 하중(Payload), 관절 가동 범위(Joint Limit), 안전 거리(Safety Distance), 작업 우선순위(Priority), 충돌 회피(Collision Avoidance)와 같은 요소는 모두 계획 과정에 반영된다. 따라서 생성되는 작업은 단순히 논리적일 뿐 아니라 실제 실행 가능한 계획이 된다.

시간적 추론(Temporal Reasoning)은 작업 순서를 결정하는 핵심 요소이다. 어떤 작업은 반드시 이전 단계가 완료되어야 수행될 수 있다. 예를 들어 물건을 집기 전에 먼저 접근해야 하며, 닫힌 문을 통과하기 위해서는 문을 먼저 열어야 한다. LLM은 이러한 시간적 의존성을 자연스럽게 이해하고 올바른 실행 순서를 생성한다.

사람의 명령은 종종 매우 간략하다. "식탁을 준비해."라는 명령에는 접시를 놓고, 컵을 배치하고, 수저를 정리하며, 냅킨을 놓는 과정이 포함되어 있지만 사용자는 이를 모두 말하지 않는다. LLM은 이러한 생략된 절차를 자동으로 보완하여 완전한 목표(Task Goal)로 확장한다.

LLM은 작업 계획을 사람이 이해하기 쉬운 의미 표현(Semantic Representation)으로 생성한다. 예를 들어 작업 그래프(Task Graph), 행동 트리(Behavior Tree), JSON 구조(JSON Structure), 심볼릭 계획(Symbolic Plan) 등의 형태로 출력하며, 이후 로봇 미들웨어(Middleware)와 제어 시스템이 이를 실제 실행 코드로 변환한다.

현실 환경에는 항상 불확실성(Uncertainty)이 존재한다. 객체가 가려져 있거나, 동일한 물체가 여러 개 존재하거나, 사용자의 표현이 모호할 수도 있다. 최신 LLM은 이러한 상황에서 임의로 판단하지 않고 "어느 파란 상자를 말씀하시는 건가요?"와 같이 추가 질문(Clarification Question)을 생성하여 오류 가능성을 줄인다.

대화 기반 계획(Interactive Planning)은 인간과 로봇의 협업을 더욱 자연스럽게 만든다. 로봇은 작업을 이해한 내용을 사용자에게 설명하고, 애매한 부분은 확인하며, 작업 도중에도 상황 변화에 따라 추가 질문을 수행할 수 있다. 이러한 상호작용은 실제 서비스 로봇(Service Robot)에서 매우 중요한 기능이다.

최근에는 검색 증강 생성(RAG, Retrieval-Augmented Generation) 기술이 함께 활용된다. LLM은 자체 학습 지식뿐 아니라 작업 매뉴얼(Manual), 공장 절차서(Standard Operating Procedure), 환경 지도(Map), 이전 작업 기록(Task History), 장비 데이터베이스(Database) 등을 검색하여 보다 정확한 계획을 생성한다. 이는 환각(Hallucination)을 줄이고 산업 환경에서의 신뢰성을 향상시킨다.

로봇은 일반적으로 이동(Navigation), 물체 집기(Grasping), 문 열기(Door Opening), 버튼 누르기(Button Pressing), 도킹(Docking), 검사(Inspection) 등 다양한 기술(Skill)을 보유하고 있다. LLM은 현재 작업에 적합한 기술을 선택하고, 가장 효율적인 실행 순서를 결정하여 기술 라이브러리(Skill Library)를 활용한다.

산업용 시스템에서는 행동 트리(Behavior Tree)나 계층형 상태 머신(Hierarchical State Machine)과 LLM을 함께 사용하는 경우가 많다. LLM은 상위 수준의 의미 기반 계획을 생성하고, 행동 트리는 이를 안정적으로 실행한다. 이러한 구조는 LLM의 유연성과 산업용 제어 시스템의 신뢰성을 동시에 확보할 수 있는 장점이 있다.

실제 로봇은 여러 목표를 동시에 최적화해야 한다. 작업 시간(Task Time), 에너지 소비(Energy Consumption), 안전성(Safety), 사용자 편의성(User Comfort), 생산성(Productivity)은 서로 상충될 수 있다. LLM은 이러한 요소를 종합적으로 고려하여 균형 잡힌 계획을 생성한다.

협업 로봇(Collaborative Robot)에서는 사람의 행동도 함께 예측한다. LLM은 사람이 다음에 수행할 행동을 추론하고, 로봇이 언제 도움을 제공해야 하는지, 언제 대기해야 하는지, 어떤 작업을 분담해야 하는지를 계획한다. 이를 통해 사람과 로봇은 자연스럽게 공동 작업을 수행할 수 있다.

실제 환경은 지속적으로 변화하므로 동적 재계획(Dynamic Replanning)이 필수적이다. 사람이 갑자기 이동하거나 장애물이 나타나거나 목표 물체가 다른 위치로 옮겨질 수 있다. LLM은 이러한 변화를 감지하여 전체 작업을 처음부터 다시 만드는 것이 아니라 영향을 받은 부분만 수정하여 새로운 계획을 생성한다.

장기 작업에서는 메모리(Memory) 기능도 매우 중요하다. 에피소드 메모리(Episodic Memory)는 이전 작업 경험을 저장하고, 의미 메모리(Semantic Memory)는 일반 지식을 저장하며, 절차 메모리(Procedural Memory)는 기술 실행 방법을 기억한다. 작업 메모리(Working Memory)는 현재 수행 중인 작업 상태를 유지하여 장시간 임무에서도 일관성을 유지하도록 지원한다.

현대 VLA 시스템은 일반적으로 계층형 구조(Hierarchical Architecture)를 사용한다. 인식 모듈(Perception Module)은 환경을 이해하고, LLM은 의미적 추론을 수행하며, 작업 계획기(Task Planner)는 실행 계획을 생성한다. 이후 모션 계획기(Motion Planner)가 경로를 계산하고, 제어기(Controller)가 모터를 제어하며, 피드백 시스템(Feedback System)이 결과를 지속적으로 모니터링한다.

그러나 LLM에도 한계는 존재한다. 물리적으로 불가능한 계획을 생성하거나, 안전 조건을 간과하거나, 존재하지 않는 객체를 잘못 추론하는 환각(Hallucination)이 발생할 수 있다. 따라서 실제 로봇 시스템에서는 기하학적 검증(Geometric Validation), 어포던스 분석(Affordance Analysis), 시뮬레이션 검증(Simulation Validation), 안전 제어기(Safety Controller)를 통해 생성된 계획을 반드시 검증한 후 실행한다.

추론 속도(Inference Latency) 역시 중요한 과제이다. 대규모 모델은 응답 생성에 수백 밀리초(Millisecond)에서 수 초(Second)가 소요될 수 있으므로, 실시간 제어(Real-Time Control)는 여전히 별도의 로컬 제어기(Local Controller)가 담당한다. LLM은 비교적 느린 고수준 계획을 수행하고, 저지연 제어기는 연속적인 피드백 제어를 수행하는 계층형 구조가 가장 현실적인 접근 방식이다.

향후 로봇 시스템은 언어(Language), 영상(Image), 깊이(Depth), 포인트 클라우드(Point Cloud), 음성(Audio), 촉각(Tactile), 힘(Force), 로봇 상태(Robot State), 장기 메모리(Long-Term Memory)를 동시에 이해하는 멀티모달 기반 모델(Multimodal Foundation Model)로 발전할 것이다. 이러한 모델은 단순한 언어 생성기를 넘어 로봇의 인지 엔진(Cognitive Engine)으로 동작하며, 복잡한 환경을 이해하고 미래를 예측하며 다수의 로봇을 협업시키고 새로운 상황에도 적응하는 범용 물리 AI(Physical AI)의 핵심 기술이 될 것으로 전망된다.

##  

## 3.2 LLM-Based Task Planning and Chain-of-Thought (with Code)

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

Large Language Models (LLMs) have significantly expanded the capabilities of autonomous robotic systems by enabling robots to reason about complex tasks instead of merely reacting to predefined commands. Traditional robotic task planners primarily rely on manually engineered state machines, symbolic planners, or behavior trees in which every possible situation must be explicitly represented by human engineers. While these methods provide deterministic behavior, they often struggle when robots encounter ambiguous instructions, unfamiliar environments, or situations requiring common-sense reasoning. LLM-based task planning introduces a fundamentally different paradigm in which robots perform semantic reasoning before generating executable plans. One of the most influential concepts supporting this capability is Chain of Thought (CoT) reasoning, which enables the model to decompose complicated problems into logically connected intermediate reasoning steps before producing a final decision.

Chain of Thought reasoning does not simply produce an answer directly from an input instruction. Instead, it encourages the model to perform structured internal reasoning that gradually transforms an abstract user request into an executable sequence of robotic actions. Rather than treating task planning as a direct language-to-action mapping, CoT allows the robot to reason about goals, environmental conditions, constraints, object relationships, temporal dependencies, and alternative solutions before deciding which actions should be executed.

Within modern Vision-Language-Action architectures, the LLM serves as the cognitive planning engine operating above perception, mapping, localization, motion planning, and low-level control systems. The perception subsystem continuously produces semantic descriptions of the environment using RGB cameras, depth cameras, LiDAR, force sensors, tactile sensors, proprioceptive measurements, and environmental maps. Vision encoders transform these heterogeneous observations into high-dimensional semantic embeddings, while Vision-Language Models generate natural language scene descriptions. These representations become inputs to the LLM together with user instructions, robot status information, task history, memory, and operational constraints.

The first stage of Chain of Thought planning begins with comprehensive instruction interpretation. Human instructions are rarely complete specifications of robotic behavior. Instead, users naturally omit details that humans normally infer from common sense. A command such as "Clean the kitchen" leaves numerous questions unanswered. Which surfaces should be cleaned first? Which cleaning tools are appropriate? Should fragile objects be moved beforehand? Should dishes be washed before wiping countertops? A conventional robot controller cannot answer these questions without explicit programming, whereas an LLM employing Chain of Thought reasoning systematically fills these gaps using prior knowledge and contextual understanding.

Intent recognition forms the foundation of the reasoning process. The LLM first determines the user\'s primary objective rather than immediately considering possible actions. It identifies whether the task involves navigation, manipulation, inspection, transportation, cleaning, assembly, interaction, or multiple combined objectives. The desired outcome becomes the anchor that guides all subsequent reasoning stages.

Following intent recognition, the model identifies relevant objects mentioned explicitly or implicitly within the instruction. Objects are analyzed not only according to their names but also according to their functional roles, affordances, semantic categories, physical properties, expected locations, and interaction requirements. The LLM understands that a coffee mug is graspable, contains liquid, should remain upright during transportation, and is commonly located in kitchens, offices, or dining rooms. This semantic knowledge dramatically improves planning robustness.

Spatial reasoning constitutes another major component of Chain of Thought planning. Robot tasks almost always involve spatial relationships among objects, humans, and environmental structures. Rather than reasoning only about Cartesian coordinates, the LLM interprets qualitative spatial concepts such as "next to," "inside," "behind," "above," "under," and "near." These semantic spatial relationships complement metric localization produced by mapping systems and greatly simplify high-level planning.

Temporal reasoning naturally emerges during the reasoning sequence. Many robotic tasks involve prerequisite relationships between actions. A robot cannot place an object before grasping it, cannot inspect the contents of a cabinet before opening the cabinet door, and cannot recharge its battery before reaching the charging station. Chain of Thought reasoning explicitly identifies these dependencies and organizes them into coherent execution sequences.

Task decomposition is perhaps the most visible characteristic of Chain of Thought planning. Instead of attempting to solve an entire mission in one step, the LLM recursively divides the problem into manageable subtasks. Each subtask may itself be further decomposed until it becomes executable by existing robot skills. This hierarchical reasoning process closely resembles how experienced human operators naturally approach complicated tasks.

Consider a warehouse inspection robot receiving the instruction, "Inspect all damaged pallets in Zone B and report the results." Rather than producing a single action, the LLM reasons through multiple planning stages. It identifies the inspection objective, determines the destination, plans navigation to Zone B, identifies pallet detection as a prerequisite, filters damaged pallets from normal ones, generates an efficient inspection sequence, performs visual inspection, records observations, aggregates results, and finally generates an inspection report. Each of these stages may itself involve additional reasoning regarding navigation safety, battery capacity, sensor visibility, and environmental conditions.

Constraint reasoning represents another essential element of Chain of Thought planning. Robots operate under numerous physical, computational, operational, and safety constraints. Battery limitations, payload capacity, joint limits, sensor visibility, communication bandwidth, collision avoidance, human safety regulations, and mission deadlines all influence the planning process. Rather than treating constraints as independent filters, the LLM integrates them into the reasoning sequence itself, producing plans that are feasible from the beginning rather than requiring extensive post-processing.

Chain of Thought planning also incorporates commonsense reasoning acquired during large-scale language pretraining. Humans naturally understand that glass objects require careful handling, slippery floors reduce stability, doors should be opened before passing through, and liquids should remain upright during transportation. These facts rarely appear explicitly within robot programming languages but significantly improve practical performance. The LLM leverages this implicit knowledge to generate safer and more realistic plans.

One particularly important capability is ambiguity resolution. Human instructions often contain incomplete references, vague descriptions, or underspecified objectives. When ambiguity cannot be resolved confidently using context, Chain of Thought reasoning allows the LLM to identify uncertainty rather than making unsupported assumptions. The robot may ask clarification questions, request additional visual observations, or consult stored task history before continuing execution. This interactive planning strategy greatly reduces execution failures.

Environmental reasoning extends beyond object recognition into understanding scene dynamics. The LLM reasons about movable obstacles, crowded environments, temporary access restrictions, changing lighting conditions, and dynamic human activities. Rather than assuming a static environment, the planner continuously evaluates whether previously generated plans remain valid as new observations arrive.

Memory plays an increasingly important role throughout the reasoning process. Episodic memory stores previous task executions, semantic memory contains factual knowledge, procedural memory stores robot skills, and working memory maintains the current planning context. Chain of Thought reasoning frequently retrieves relevant memories to improve decision quality. A service robot delivering items in a hospital, for example, remembers previously visited patient rooms, preferred navigation routes, equipment locations, and historical delivery schedules.

Hierarchical planning generated by Chain of Thought naturally integrates with robot skill libraries. Most robots possess reusable action primitives such as navigation, grasping, docking, opening doors, pressing buttons, object inspection, charging, or elevator operation. Rather than inventing entirely new motor behaviors, the LLM selects appropriate skills and determines the sequence in which they should be executed. This separation between semantic reasoning and motor execution greatly improves modularity.

The reasoning process frequently produces symbolic intermediate representations including task graphs, behavior trees, finite-state machines, planning graphs, JSON execution plans, or domain-specific planning languages. These representations allow deterministic execution systems to verify and execute the generated plans while preserving the flexibility of natural language reasoning.

Chain of Thought planning is particularly effective when integrated with Task and Motion Planning frameworks. High-level reasoning determines what objectives should be achieved, while classical optimization algorithms determine how individual motions should be executed. For example, the LLM may determine that a robot should retrieve an object from a shelf, while a motion planner computes collision-free arm trajectories and navigation paths satisfying physical constraints.

Modern robotic systems increasingly employ Retrieval-Augmented Generation to strengthen Chain of Thought reasoning. Rather than relying exclusively on pretrained knowledge, the LLM retrieves relevant documentation, facility maps, equipment specifications, maintenance procedures, previous execution logs, operating manuals, and enterprise databases. Retrieved information becomes part of the reasoning context, enabling more accurate planning for specialized industrial environments.

Multi-step planning often requires evaluating alternative execution strategies. Chain of Thought reasoning naturally supports comparison among competing plans. The LLM may analyze multiple navigation routes, different manipulation sequences, or alternative task allocations before selecting the most appropriate solution according to safety, efficiency, energy consumption, completion time, or operational priorities.

Error anticipation represents another valuable capability. During reasoning, the model predicts potential execution failures before they occur. It may recognize that an object is likely too heavy for the robot, that a corridor is unusually narrow, that battery capacity may become insufficient, or that lighting conditions may reduce perception accuracy. By anticipating these issues during planning rather than after execution begins, robots become substantially more reliable.

Dynamic replanning extends Chain of Thought reasoning beyond initial plan generation. Real-world environments continuously evolve as humans move, objects change locations, equipment becomes unavailable, or weather conditions change. The LLM continuously incorporates new observations into its reasoning process, modifying only affected portions of the execution plan while preserving completed work whenever possible.

Human-robot collaboration benefits greatly from explicit reasoning. The LLM determines not only what the robot should do but also when human intervention is appropriate. Certain tasks requiring fine dexterity, regulatory approval, or subjective judgment may be delegated to humans, while repetitive or physically demanding subtasks remain assigned to the robot. This collaborative reasoning enables efficient human-robot teamwork.

Industrial applications frequently require reasoning over standard operating procedures. Manufacturing robots must follow quality inspection protocols, logistics robots must satisfy warehouse regulations, healthcare robots must comply with clinical workflows, and infrastructure inspection robots must obey maintenance procedures. Chain of Thought reasoning incorporates these procedural constraints while adapting them to the current environment.

Safety remains a primary consideration throughout the reasoning process. Although Chain of Thought improves logical consistency, LLM-generated plans must never be executed without validation. Modern robotic architectures therefore employ multiple verification layers including geometric validation, affordance analysis, collision prediction, rule-based safety checking, simulation testing, formal verification where appropriate, and runtime monitoring. These mechanisms ensure that generated plans satisfy physical and operational safety requirements before execution.

One limitation of Chain of Thought planning is computational latency. Large reasoning models may require substantial inference time, making them unsuitable for high-frequency feedback control. Consequently, robotic systems adopt hierarchical architectures in which Chain of Thought reasoning operates at relatively low frequency for strategic planning, while reactive controllers, model predictive controllers, local planners, and safety monitors operate independently at real-time frequencies.

Another challenge involves reasoning reliability. Although Chain of Thought often improves logical coherence, it does not eliminate hallucinations or incorrect assumptions. Models may occasionally infer nonexistent objects, misunderstand spatial relationships, or generate physically impossible plans. Consequently, perception systems, symbolic validators, geometric planners, and execution monitors must continuously verify every generated decision.

Recent research explores internal reasoning optimization techniques that reduce unnecessary reasoning steps while preserving planning quality. Adaptive reasoning selectively increases reasoning depth only for difficult tasks while using shorter reasoning sequences for routine operations. This approach significantly reduces computational cost and improves real-time responsiveness.

Small Language Models optimized for robotics increasingly adopt distilled forms of Chain of Thought reasoning suitable for edge deployment. Rather than executing extremely large foundation models onboard robots, compressed reasoning models perform localized planning while cloud-based foundation models handle exceptionally complex reasoning tasks. This hierarchical deployment strategy balances computational efficiency, communication latency, and planning capability.

Future Vision-Language-Action systems are expected to extend Chain of Thought into multimodal reasoning that simultaneously considers language, images, videos, depth maps, point clouds, tactile observations, force measurements, robot proprioception, environmental dynamics, world models, and long-term episodic memory. Instead of reasoning only through textual representations, future robotic foundation models will reason directly over multimodal world representations that more accurately reflect physical reality.

As Physical AI continues to evolve, Chain of Thought reasoning will become a foundational cognitive capability enabling robots to transform ambiguous human instructions into structured, verifiable, adaptive, and executable task plans. By combining semantic reasoning, hierarchical decomposition, commonsense knowledge, environmental understanding, memory retrieval, uncertainty estimation, and continuous replanning, LLM-based task planning establishes a bridge between human intentions and autonomous robotic behavior. Rather than replacing traditional robotic planning algorithms, Chain of Thought complements them by providing high-level cognitive intelligence while allowing specialized motion planners, controllers, perception systems, and safety mechanisms to perform the deterministic computations required for reliable operation in complex real-world environments.

대규모 언어 모델 기반 작업 계획(LLM-Based Task Planning)은 로봇이 단순히 미리 정의된 명령을 수행하는 수준을 넘어, 작업의 의미를 이해하고 스스로 계획을 수립할 수 있도록 하는 핵심 기술이다. 기존 로봇은 상태 머신(State Machine), 행동 트리(Behavior Tree), 심볼릭 플래너(Symbolic Planner)에 의존하여 모든 상황을 사람이 미리 설계해야 했다. 그러나 이러한 방식은 새로운 환경이나 모호한 명령에 대한 적응력이 부족하였다. LLM은 언어 이해와 추론 능력을 활용하여 보다 유연한 작업 계획을 생성할 수 있다.

이러한 LLM 기반 계획의 핵심 개념이 바로 사고의 연쇄(CoT, Chain of Thought) 추론이다. CoT는 질문을 입력받자마자 답을 생성하는 것이 아니라, 문제를 여러 개의 논리적인 중간 단계로 분해한 후 최종 결론을 도출하는 방식이다. 로봇은 이를 통해 사용자의 명령을 단계별로 분석하고, 필요한 하위 작업(Subtask)을 생성하며, 실행 가능한 계획으로 변환할 수 있다.

비전-언어-행동(VLA, Vision-Language-Action) 시스템에서 LLM은 인식(Perception)과 제어(Control) 사이의 인지 계층(Cognitive Layer)에 위치한다. 카메라(Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 힘 센서(Force Sensor), 촉각 센서(Tactile Sensor), 관절 센서(Proprioception)가 수집한 데이터는 비전 인코더(Vision Encoder)를 통해 의미적 특징(Semantic Feature)으로 변환된다. 이러한 정보와 사용자의 자연어 명령, 로봇 상태, 메모리 정보가 함께 LLM의 입력으로 사용된다.

CoT 기반 계획의 첫 번째 단계는 명령 해석(Instruction Interpretation)이다. 사람은 일반적으로 "주방을 청소해.", "창고를 점검해."처럼 매우 간단한 문장으로 작업을 지시한다. 그러나 실제 작업에는 수많은 세부 절차가 포함되어 있다. LLM은 이러한 생략된 과정을 상식(Common Sense)과 문맥(Context)을 이용하여 보완하고 전체 작업의 의미를 이해한다.

다음 단계는 의도 인식(Intent Recognition)이다. LLM은 먼저 사용자가 무엇을 원하는지를 파악한다. 작업이 이동(Navigation), 물체 조작(Manipulation), 검사(Inspection), 운반(Transportation), 청소(Cleaning), 조립(Assembly) 중 어떤 유형인지 분석하고 최종 목표(Goal)를 결정한다. 이러한 목표는 이후 생성되는 모든 작업 계획의 기준이 된다.

이후 객체 이해(Object Understanding)가 수행된다. LLM은 단순히 객체 이름만 인식하는 것이 아니라 기능(Function), 물리적 특성(Property), 사용 목적(Affordance), 일반적인 위치(Location) 등을 함께 이해한다. 예를 들어 커피잔(Coffee Mug)은 손으로 잡을 수 있으며, 액체를 담고 있고, 운반 시 기울이면 안 되며, 일반적으로 주방이나 사무실에서 발견된다는 상식을 함께 활용한다.

공간 추론(Spatial Reasoning)은 로봇 작업에서 매우 중요한 요소이다. LLM은 "위(Above)", "아래(Below)", "옆(Next To)", "안쪽(Inside)", "뒤(Behind)"와 같은 공간 관계를 이해하여 환경을 의미적으로 해석한다. 이는 단순한 좌표 계산보다 사람의 사고방식과 유사한 수준에서 작업을 계획할 수 있도록 지원한다.

시간적 추론(Temporal Reasoning)은 작업 순서를 결정하는 과정이다. 로봇은 물건을 놓기 전에 먼저 집어야 하고, 문을 통과하려면 먼저 문을 열어야 하며, 충전을 하려면 충전기로 먼저 이동해야 한다. CoT는 이러한 선행 관계(Prerequisite)를 단계적으로 분석하여 올바른 실행 순서를 생성한다.

CoT의 가장 큰 특징은 계층적 작업 분해(Hierarchical Task Decomposition)이다. 복잡한 작업을 한 번에 해결하지 않고 작은 작업으로 나누어 처리한다. 예를 들어 창고 점검이라는 작업은 이동하기, 구역 탐색하기, 팔레트 탐지하기, 손상 여부 판단하기, 사진 촬영하기, 결과 저장하기, 보고서 작성하기 등의 여러 단계로 분해된다. 각각의 단계는 다시 더 작은 작업으로 나누어질 수 있으며 최종적으로는 로봇이 수행 가능한 기술(Skill) 단위까지 세분화된다.

예를 들어 "B구역의 손상된 팔레트를 모두 검사하고 보고서를 작성하라."는 명령을 받은 경우 LLM은 먼저 목적을 이해하고, B구역으로 이동한 후 팔레트를 탐색하고, 손상 여부를 판단하며, 검사 순서를 결정하고, 사진과 데이터를 수집한 뒤 최종 보고서를 생성하는 순차적인 계획을 수립한다. 이러한 계획은 단순한 행동 목록이 아니라 논리적인 추론 과정을 거쳐 생성된다.

CoT는 다양한 제약 조건(Constraint)도 함께 고려한다. 배터리 잔량(Battery Capacity), 적재 하중(Payload), 관절 제한(Joint Limit), 충돌 회피(Collision Avoidance), 작업 시간(Time Limit), 통신 상태(Network Condition), 안전 규정(Safety Regulation) 등은 계획 생성 과정에서 동시에 평가된다. 따라서 처음부터 실행 가능한 계획이 생성될 가능성이 높아진다.

LLM은 대규모 사전학습을 통해 획득한 상식(Common Sense)을 적극적으로 활용한다. 유리 제품은 조심해서 다루어야 하고, 액체는 흘리지 않도록 운반해야 하며, 미끄러운 바닥에서는 천천히 이동해야 한다는 사실은 일반적인 로봇 제어 프로그램에는 포함되지 않을 수도 있다. 하지만 CoT는 이러한 상식을 추론 과정에 포함시켜 보다 현실적인 계획을 생성한다.

사람의 명령은 종종 모호하다. "파란 상자를 가져와."라고 했을 때 파란 상자가 여러 개 존재할 수 있다. CoT는 이러한 불확실성(Uncertainty)을 인식하고 무리하게 추측하지 않는다. 대신 "어느 파란 상자를 말씀하시는 건가요?"와 같은 확인 질문(Clarification Question)을 생성하여 오류 가능성을 줄인다.

환경 추론(Environment Reasoning)도 중요한 기능이다. 로봇은 사람이 이동하거나, 장애물이 새롭게 생기거나, 조명이 변하거나, 특정 구역이 폐쇄되는 등 동적인 환경을 고려해야 한다. CoT는 이러한 환경 변화를 지속적으로 반영하면서 기존 계획이 여전히 유효한지를 판단한다.

메모리(Memory)는 CoT의 중요한 구성 요소이다. 에피소드 메모리(Episodic Memory)는 이전 작업 경험을 저장하고, 의미 메모리(Semantic Memory)는 일반적인 지식을 저장하며, 절차 메모리(Procedural Memory)는 기술(Skill)의 실행 방법을 저장한다. 작업 메모리(Working Memory)는 현재 수행 중인 계획을 유지하며 지속적인 추론을 가능하게 한다.

CoT는 기술 라이브러리(Skill Library)와도 밀접하게 연동된다. 대부분의 로봇은 이동(Navigation), 물체 집기(Grasping), 문 열기(Door Opening), 버튼 누르기(Button Pressing), 도킹(Docking), 검사(Inspection), 충전(Charging)과 같은 기본 기술을 이미 보유하고 있다. LLM은 새로운 기술을 만드는 것이 아니라 현재 작업에 필요한 기술을 선택하고 가장 적절한 순서로 배치한다.

LLM은 생성한 계획을 작업 그래프(Task Graph), 행동 트리(Behavior Tree), 상태 머신(State Machine), JSON 계획(JSON Plan), 심볼릭 계획(Symbolic Plan) 등 구조화된 표현으로 변환한다. 이러한 표현은 기존 로봇 소프트웨어가 쉽게 실행할 수 있으며, 산업 현장의 안정성과 확장성을 동시에 확보할 수 있다.

CoT는 작업 및 모션 계획(TAMP, Task and Motion Planning)과 결합될 때 더욱 강력한 성능을 발휘한다. LLM은 "무엇을 해야 하는가"를 결정하고, 모션 플래너(Motion Planner)는 "어떻게 움직일 것인가"를 계산한다. 이처럼 의미 기반 계획과 기하학 기반 계획이 분리됨으로써 시스템 전체의 효율성과 신뢰성이 향상된다.

최근에는 검색 증강 생성(RAG, Retrieval-Augmented Generation)이 함께 활용되고 있다. LLM은 내부 지식뿐 아니라 공장 매뉴얼(Manual), 작업 절차(Standard Operating Procedure), 환경 지도(Map), 유지보수 기록(Log), 기업 데이터베이스(Database) 등을 검색하여 보다 정확한 작업 계획을 생성한다. 이는 산업 환경에서 발생할 수 있는 환각(Hallucination)을 크게 감소시키는 효과가 있다.

CoT는 여러 개의 실행 전략을 동시에 비교할 수도 있다. 여러 이동 경로, 다양한 작업 순서, 서로 다른 작업 분담 방식을 평가하여 안전성(Safety), 에너지 소비(Energy Consumption), 작업 시간(Task Time), 생산성(Productivity) 등을 종합적으로 고려한 최적의 계획을 선택한다.

또한 실행 전에 오류를 예측(Error Anticipation)하는 능력도 갖추고 있다. 특정 물체가 너무 무겁거나, 통로가 너무 좁거나, 배터리가 부족하거나, 조명이 어두워 인식 성능이 떨어질 가능성을 사전에 예측하여 작업 실패를 예방한다. 이는 실제 산업용 로봇에서 매우 중요한 기능이다.

현실 환경은 지속적으로 변화하기 때문에 동적 재계획(Dynamic Replanning)이 필요하다. 사람이 지나가거나, 장애물이 발생하거나, 목표 물체가 이동하면 CoT는 새로운 정보를 반영하여 기존 계획의 일부만 수정한다. 이미 완료된 작업은 유지하고 필요한 부분만 다시 계획함으로써 작업 효율을 높인다.

사람과 로봇의 협업(Human-Robot Collaboration)에서도 CoT는 중요한 역할을 한다. 정밀한 판단이 필요한 작업은 사람에게 맡기고, 반복적이거나 위험한 작업은 로봇이 수행하도록 역할을 분담한다. 이를 통해 인간과 로봇은 서로의 장점을 활용하는 협업 체계를 구축할 수 있다.

산업 환경에서는 표준 작업 절차(SOP, Standard Operating Procedure)를 반드시 따라야 한다. 제조, 물류, 의료, 시설 점검과 같은 분야에서는 정해진 절차와 규정을 준수해야 하며, CoT는 이러한 절차를 현재 작업 환경에 맞게 적용하여 계획을 생성한다.

안전(Safety)은 CoT 기반 계획에서도 가장 중요한 요소이다. 아무리 논리적으로 보이는 계획이라도 실제 실행 전에 기하학적 검증(Geometric Validation), 어포던스 검증(Affordance Validation), 충돌 예측(Collision Prediction), 시뮬레이션(Simulation), 규칙 기반 안전 검사(Rule-Based Safety Check)를 반드시 수행해야 한다. 이를 통해 물리적으로 불가능하거나 위험한 계획이 실행되는 것을 방지한다.

대규모 LLM은 추론 시간(Inference Latency)이 길다는 단점이 있다. 따라서 실제 시스템에서는 CoT가 고수준 전략 계획을 수행하고, 저수준 제어(Low-Level Control)는 실시간 제어기(Real-Time Controller)가 담당하는 계층형 구조(Hierarchical Architecture)를 사용한다. 이는 높은 지능과 빠른 제어 성능을 동시에 확보하기 위한 현실적인 설계 방식이다.

LLM은 여전히 존재하지 않는 객체를 추론하거나 공간 관계를 잘못 이해하는 환각(Hallucination)이 발생할 수 있다. 따라서 인식 시스템(Perception System), 기하학 검증기(Geometric Validator), 실행 모니터(Execution Monitor), 안전 모듈(Safety Module)이 생성된 계획을 지속적으로 검증해야 한다.

최근 연구에서는 적응형 사고의 연쇄(Adaptive Chain of Thought)가 활발히 연구되고 있다. 쉬운 작업은 짧은 추론 과정을 사용하고, 복잡한 작업은 깊은 추론을 수행함으로써 계산량을 줄이고 실시간 성능을 향상시키는 방식이다.

또한 소형 언어 모델(SLM, Small Language Model)은 CoT를 경량화하여 엣지 로봇(Edge Robot)에 적용하고 있다. 로봇 내부에서는 압축된 모델이 빠른 추론을 수행하고, 매우 복잡한 문제는 클라우드 기반 대형 모델이 처리하는 계층형 구조가 점차 보편화되고 있다.

향후 비전-언어-행동(VLA) 시스템은 언어(Language), 영상(Image), 비디오(Video), 깊이 정보(Depth), 포인트 클라우드(Point Cloud), 촉각(Tactile), 힘(Force), 로봇 상태(Proprioception), 세계 모델(World Model), 장기 메모리(Long-Term Memory)를 동시에 활용하는 멀티모달 사고의 연쇄(Multimodal Chain of Thought)로 발전할 것으로 전망된다.

결국 CoT는 단순한 언어 생성 기술이 아니라 사람의 추상적인 의도를 실행 가능한 로봇 행동으로 연결하는 핵심 인지 기술이다. 의미 기반 추론(Semantic Reasoning), 계층적 작업 분해(Hierarchical Decomposition), 상식(Common Sense), 메모리(Memory), 환경 이해(Environment Understanding), 불확실성 추론(Uncertainty Reasoning), 지속적인 재계획(Replanning)을 통합함으로써 인간의 의도를 안전하고 신뢰성 높은 자율 로봇 행동으로 변환하는 핵심 기반 기술로 자리매김하고 있다.

##  

## 3.3 Code-as-Policies: Generating Robot Programs Using LLMs (with Code)

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

Code as Policies (CaP) represents one of the most influential advances in modern robot intelligence by transforming Large Language Models (LLMs) from passive language understanding systems into active robotic programming engines. Instead of generating only natural language responses or symbolic task descriptions, an LLM directly produces executable Python code that defines robotic behavior. This paradigm significantly reduces the gap between human instructions and autonomous robot execution by allowing robots to synthesize programs dynamically rather than relying solely on manually engineered behavior trees, state machines, or predefined motion scripts. Within Vision-Language-Action (VLA) systems, Code as Policies enables robots to translate natural language into structured software logic that can interact with perception modules, planning systems, robot middleware, and low-level controllers.

Traditional robot software development requires engineers to manually implement every task as source code before deployment. Navigation routines, manipulation sequences, inspection procedures, recovery behaviors, and exception handling must all be explicitly programmed, tested, and maintained. While this approach provides deterministic execution, it lacks flexibility when robots encounter new instructions, unfamiliar environments, or previously unseen task combinations. Code as Policies fundamentally changes this workflow by allowing robots to generate task-specific programs at runtime using the reasoning capability of LLMs.

The central idea behind Code as Policies is that natural language can be translated into executable policy functions. Instead of interpreting user commands through manually designed intent classifiers, the LLM generates Python functions that directly invoke robot APIs, perception modules, planning algorithms, and control interfaces. The resulting program serves as a policy describing how the robot should perceive, reason, decide, and act throughout task execution.

Within a modern VLA architecture, Code as Policies operates between high-level reasoning and robotic execution. Human instructions, perception results, semantic maps, robot status information, memory, environmental constraints, and available robot capabilities are provided as contextual inputs to the language model. The LLM analyzes this information and generates Python code that orchestrates robot behaviors using existing software components rather than attempting to replace them.

The generated Python code typically interacts with modular robot software libraries rather than implementing low-level algorithms from scratch. Navigation modules, manipulation APIs, perception services, localization systems, motion planners, inverse kinematics solvers, safety monitors, and communication interfaces remain independently developed components. The LLM simply generates the orchestration logic that coordinates these modules according to the current task.

Consider a service robot receiving the instruction, "Bring me the red coffee mug from the kitchen." Instead of generating only a textual task description, the LLM may produce Python code that first invokes object detection to locate red mugs, then calls the navigation system to reach the kitchen, activates visual servoing for approach, invokes grasp planning, executes manipulation routines, verifies successful grasping using force feedback, navigates to the user, and finally performs safe object delivery. Every step is expressed as executable software logic.

One of the most important advantages of Code as Policies is compositionality. Rather than storing separate programs for every possible task, robots possess reusable software primitives that can be combined dynamically into entirely new behaviors. Existing APIs for navigation, object detection, manipulation, docking, charging, inspection, localization, and communication become software building blocks that the LLM assembles into task-specific programs. This greatly improves scalability because the number of achievable behaviors grows combinatorially as the skill library expands.

The generated code generally follows hierarchical software architecture principles. High-level policy functions describe overall task flow, intermediate functions coordinate subtasks, while lower-level functions call existing robot libraries responsible for perception, planning, and control. This layered organization preserves software modularity while allowing flexible task generation.

Perception plays a central role throughout the generated programs. Code as Policies does not assume complete knowledge of the environment before execution. Instead, generated Python code frequently performs perception-driven decision making by repeatedly invoking computer vision systems, object detectors, semantic segmentation networks, depth estimation modules, or Vision-Language Models. Decisions are therefore conditioned on current observations rather than static assumptions.

Closed-loop execution distinguishes Code as Policies from static action sequencing. Rather than producing a fixed sequence of actions, the generated Python program continuously evaluates environmental conditions during execution. Sensor feedback influences conditional branches, loop termination, error recovery, and dynamic replanning decisions. This feedback-driven execution significantly improves robustness in changing environments.

Conditional reasoning naturally emerges within generated code. Python provides expressive control structures including if statements, loops, exception handling, recursion, list comprehensions, asynchronous execution, and modular functions. The LLM exploits these language constructs to express sophisticated robotic decision logic that would otherwise require extensive manual programming.

Loop structures become particularly valuable for autonomous exploration and search tasks. Robots may repeatedly inspect candidate locations until a target object is found, continue cleaning until no dirt remains, repeatedly scan shelves until inventory verification completes, or patrol facilities continuously while monitoring for anomalies. These repetitive behaviors are naturally represented through program loops generated by the LLM.

Exception handling represents another major strength of Code as Policies. Real-world robotic execution inevitably encounters failures such as missing objects, blocked navigation paths, grasp failures, communication interruptions, localization uncertainty, or sensor malfunction. Generated Python programs incorporate try-except blocks, recovery procedures, timeout mechanisms, retry policies, fallback behaviors, and contingency planning. Instead of immediately terminating upon failure, robots adapt their behavior dynamically.

Memory integration significantly enhances generated policies. Programs may query episodic memory to retrieve previous object locations, semantic memory to obtain domain knowledge, procedural memory to identify appropriate skills, or task history databases to avoid redundant actions. This memory-aware programming enables long-term autonomy across extended missions.

Code as Policies also supports Retrieval-Augmented Generation. During code synthesis, the LLM may retrieve API documentation, software examples, operating procedures, environment maps, robot capability descriptions, or developer documentation. Retrieved information supplements the model\'s internal knowledge, enabling more accurate code generation for specialized robotic platforms.

Generated programs frequently invoke ROS 2 interfaces when operating within modern robotic middleware. Publishers, subscribers, services, actions, lifecycle nodes, parameter servers, TF transformations, and DDS communication mechanisms are coordinated through Python APIs generated by the LLM. Rather than replacing ROS 2, Code as Policies automates application development on top of existing middleware infrastructure.

Task decomposition naturally appears within generated programs through modular function definitions. Complex missions become collections of reusable helper functions, each responsible for specific subtasks such as locating objects, planning trajectories, verifying grasp quality, or updating execution status. This modular structure improves readability, debugging, verification, and future reuse.

Code as Policies frequently integrates with Task and Motion Planning systems. High-level generated Python code invokes symbolic planners to determine action sequences while motion planning libraries compute collision-free trajectories. This hybrid architecture combines semantic reasoning provided by the LLM with mathematically rigorous optimization algorithms responsible for physical execution.

Vision-Language Models increasingly complement Code as Policies by providing semantic scene descriptions directly to generated programs. Rather than relying exclusively on numerical perception outputs, generated Python code may query multimodal models using natural language prompts such as "Describe the objects on the table," "Identify empty shelves," or "Locate damaged equipment." This greatly simplifies high-level perception integration.

Industrial robotics particularly benefits from Code as Policies because manufacturing environments frequently require customized workflows. Traditional automation requires extensive software engineering whenever production procedures change. With LLM-based code generation, robots dynamically synthesize updated task logic while continuing to use verified low-level software components. Production flexibility therefore increases without sacrificing execution reliability.

Warehouse automation provides another compelling example. Picking orders, inventory inspection, package sorting, pallet verification, shelf replenishment, and exception handling often require numerous workflow variations. Rather than manually implementing separate software pipelines for every situation, Code as Policies dynamically generates customized execution programs based on current inventory conditions, order requirements, and warehouse status.

Inspection robots similarly benefit from dynamic code generation. Inspection procedures often depend on asset type, detected anomalies, environmental conditions, customer requirements, regulatory standards, or previous maintenance records. Generated Python programs selectively invoke sensing routines, image processing algorithms, anomaly detection models, report generation functions, and cloud synchronization services according to current inspection objectives.

Outdoor autonomous robots also gain substantial flexibility through Code as Policies. Infrastructure inspection, agricultural monitoring, security patrol, construction surveying, environmental sampling, and utility maintenance frequently require adapting mission behavior according to weather, terrain, lighting, accessibility, equipment availability, or operational priorities. Generated programs dynamically integrate these contextual variables into execution logic.

Despite its flexibility, Code as Policies does not eliminate the need for software engineering discipline. Generated code must satisfy software quality requirements including readability, modularity, maintainability, robustness, determinism where appropriate, exception safety, resource management, concurrency control, and interface consistency. Consequently, many robotic systems constrain code generation using predefined software templates, approved APIs, coding standards, and static analysis tools.

Security considerations become increasingly important when robots execute dynamically generated code. Arbitrary code execution introduces significant cybersecurity risks if generated programs can access unrestricted system resources. Modern implementations therefore employ sandboxed execution environments, restricted Python interpreters, permission-controlled APIs, resource limitations, digital signatures, runtime monitoring, and policy enforcement mechanisms. Generated programs typically execute within carefully controlled software environments rather than receiving unrestricted operating system access.

Formal verification and code validation constitute additional safety mechanisms. Before execution, generated Python programs may undergo syntax checking, static code analysis, type verification, API validation, dependency analysis, simulation testing, safety rule verification, and execution tracing. These validation layers reduce the probability of unsafe or logically inconsistent behavior reaching physical hardware.

Simulation-first execution has become common practice for Code as Policies. Newly generated programs are first executed inside high-fidelity simulation environments such as Isaac Sim, Gazebo, MuJoCo, or custom digital twins. Simulation evaluates task correctness, collision risks, execution timing, resource consumption, and behavioral consistency before deployment on physical robots. This sim-to-real workflow significantly improves operational safety.

Runtime monitoring continues throughout execution. Dedicated supervisory systems observe generated programs for abnormal behavior including excessive execution time, unexpected API usage, unsafe trajectories, excessive force application, localization divergence, communication failures, or hardware anomalies. Supervisors retain authority to interrupt or override generated programs whenever safety conditions are violated.

One limitation of Code as Policies involves reasoning reliability. Large Language Models occasionally generate syntactically correct code that nevertheless contains logical errors, invalid API calls, unsupported assumptions, or hallucinated software interfaces. Therefore generated code should never be considered inherently correct without validation against actual robot software libraries and operational constraints.

Computational latency represents another practical challenge. Generating complete software programs using very large foundation models may require substantial inference time. Consequently, robots often combine cloud-based code generation for strategic tasks with lightweight onboard execution engines responsible for low-latency control. Once generated, Python programs execute locally while real-time controllers continue operating independently.

Recent research increasingly combines Chain of Thought reasoning with Code as Policies. Instead of directly generating source code, the LLM first reasons through task decomposition, environmental constraints, object relationships, safety considerations, and execution strategies before translating this reasoning into structured Python functions. This reasoning-first approach improves code quality, modularity, and execution reliability.

Another promising direction involves integrating Code as Policies with world models. Rather than generating code solely from current observations, future systems will reason over predictive simulations of future environmental states. Generated programs may proactively avoid anticipated failures, optimize long-horizon objectives, and coordinate multiple robots through predictive planning.

Multi-robot systems also benefit from code generation. LLMs may synthesize distributed coordination programs assigning complementary subtasks to multiple autonomous robots while managing communication, synchronization, resource allocation, and shared workspace safety. Python code becomes the orchestration layer governing collaborative robotic behavior across entire fleets.

Foundation models specifically trained on robotics software repositories are expected to substantially improve future code generation quality. Exposure to large-scale ROS packages, industrial automation frameworks, simulation environments, manipulation libraries, perception pipelines, and robot control software enables LLMs to produce increasingly reliable, efficient, and maintainable robotic programs.

As Physical AI continues evolving toward increasingly autonomous embodied intelligence, Code as Policies is likely to become one of the fundamental mechanisms connecting language understanding with executable robotic behavior. Instead of viewing programming as a manual engineering activity performed exclusively before deployment, future robotic systems will continuously synthesize, adapt, optimize, verify, and execute software throughout their operational lifetime. By combining semantic reasoning, multimodal perception, reusable software libraries, structured Python generation, formal validation, simulation-based verification, and runtime monitoring, Code as Policies establishes a practical pathway toward highly adaptive robots capable of converting human intentions directly into safe, verifiable, and executable software policies suitable for complex real-world environments.

코드 기반 정책(Code as Policies, CaP)은 대규모 언어 모델(LLM, Large Language Model)이 단순히 자연어를 이해하는 수준을 넘어, 실행 가능한 파이썬(Python) 프로그램을 직접 생성하여 로봇을 제어하는 새로운 패러다임이다. 기존에는 사람이 직접 로봇 프로그램을 작성해야 했지만, CaP에서는 사용자의 자연어 명령을 입력받아 LLM이 실행 가능한 정책(Policy) 코드를 자동으로 생성한다. 이를 통해 사람의 의도와 로봇의 실제 동작 사이의 간극을 크게 줄일 수 있다.

기존 로봇 소프트웨어는 이동(Navigation), 조작(Manipulation), 검사(Inspection), 도킹(Docking), 복구(Recovery) 등의 모든 동작을 사람이 직접 프로그래밍해야 했다. 이러한 방식은 안정성은 높지만 새로운 작업이나 예상하지 못한 환경 변화에는 유연하게 대응하기 어렵다. 반면 Code as Policies는 상황에 따라 필요한 프로그램을 실시간(Runtime)에 생성하므로 새로운 작업에도 빠르게 적응할 수 있다.

Code as Policies의 핵심 개념은 자연어를 실행 가능한 정책 함수(Policy Function)로 변환하는 것이다. LLM은 단순한 작업 설명을 생성하는 것이 아니라 파이썬 함수를 작성하여 로봇 API(Application Programming Interface), 인식 시스템(Perception System), 계획기(Planner), 제어기(Controller)를 호출한다. 이렇게 생성된 코드는 로봇이 작업을 수행하기 위한 실제 실행 프로그램 역할을 한다.

비전-언어-행동(VLA, Vision-Language-Action) 아키텍처에서 Code as Policies는 고수준 추론과 실제 제어 사이에 위치한다. 사용자 명령, 환경 정보, 로봇 상태, 메모리(Memory), 지도(Map), 작업 이력(Task History), 센서 정보 등이 LLM의 입력으로 제공되며, LLM은 이를 분석하여 현재 상황에 적합한 파이썬 코드를 생성한다.

생성된 코드는 기존 로봇 라이브러리를 그대로 활용한다. 이동 알고리즘, 물체 인식(Object Detection), 위치 추정(Localization), 모션 계획(Motion Planning), 역기구학(Inverse Kinematics), 안전 제어(Safety Controller) 등을 새롭게 구현하지 않는다. 대신 이미 검증된 소프트웨어 모듈(Module)을 호출하고 조합하는 오케스트레이션(Orchestration) 역할을 수행한다.

예를 들어 "주방에서 빨간 머그컵을 가져와."라는 명령을 받으면 LLM은 먼저 빨간 컵을 탐지하는 함수를 호출하고, 주방으로 이동한 후, 시각 서보(Visual Servoing)를 이용해 접근하고, 그리퍼(Gripper)를 이용하여 컵을 집고, 힘 센서(Force Sensor)로 성공 여부를 확인한 뒤, 사용자에게 이동하여 안전하게 전달하는 전체 파이썬 프로그램을 자동으로 생성할 수 있다.

Code as Policies의 가장 큰 장점은 조합 가능성(Compositionality)이다. 로봇은 모든 작업을 별도로 저장하는 것이 아니라 이동, 물체 집기, 문 열기, 충전, 검사 등 다양한 기본 기술(Skill)을 보유하고 있으며, LLM은 이러한 기술들을 현재 작업에 맞게 조합하여 새로운 프로그램을 생성한다. 따라서 기술 라이브러리(Skill Library)가 커질수록 수행 가능한 작업도 기하급수적으로 증가한다.

생성되는 코드는 일반적으로 계층형 소프트웨어 구조(Hierarchical Software Architecture)를 따른다. 상위 함수는 전체 작업을 관리하고, 중간 함수는 하위 작업(Subtask)을 수행하며, 최하위 함수는 기존 로봇 라이브러리를 호출한다. 이러한 구조는 유지보수성과 재사용성을 높이는 동시에 복잡한 작업도 체계적으로 관리할 수 있게 한다.

Code as Policies는 환경을 지속적으로 인식하면서 동작하는 폐루프 제어(Closed-Loop Execution)를 지원한다. 실행 중에도 카메라, 라이다(LiDAR), 힘 센서 등의 정보를 지속적으로 확인하며 현재 환경에 맞게 다음 행동을 결정한다. 즉, 미리 정해진 순서를 그대로 실행하는 것이 아니라 센서 피드백(Sensor Feedback)을 이용하여 상황에 맞게 행동을 수정한다.

파이썬은 조건문(If Statement), 반복문(Loop), 예외 처리(Exception Handling), 재귀 함수(Recursion), 비동기 처리(Asynchronous Programming) 등 풍부한 제어 구조(Control Structure)를 제공한다. LLM은 이러한 기능을 활용하여 복잡한 로봇 의사결정 로직을 자연스럽게 생성할 수 있으며, 사람이 직접 작성한 프로그램과 유사한 수준의 구조를 가진다.

반복문은 탐색(Search)이나 순찰(Patrol)과 같은 작업에서 매우 유용하다. 목표 물체를 찾을 때까지 여러 위치를 계속 탐색하거나, 더 이상 오염이 없을 때까지 청소를 반복하거나, 모든 선반을 검사할 때까지 순차적으로 이동하는 등의 작업을 반복문을 이용하여 자연스럽게 구현할 수 있다.

예외 처리(Exception Handling)는 실제 로봇에서 매우 중요한 기능이다. 이동 중 장애물이 나타나거나, 물체를 집지 못하거나, 통신이 끊기거나, 센서 오류가 발생하는 상황은 매우 흔하다. 생성된 코드는 Try-Except 구조를 활용하여 오류를 감지하고 재시도(Retry), 대체 경로(Fallback), 복구 절차(Recovery Procedure)를 수행함으로써 작업 실패를 최소화한다.

메모리(Memory)와의 연동도 중요한 특징이다. 생성된 프로그램은 에피소드 메모리(Episodic Memory)에서 이전 작업 위치를 검색하고, 의미 메모리(Semantic Memory)에서 일반 지식을 활용하며, 절차 메모리(Procedural Memory)를 이용하여 적절한 기술을 선택할 수 있다. 이를 통해 장기간 자율 작업(Long-Term Autonomy)이 가능해진다.

최근에는 검색 증강 생성(RAG, Retrieval-Augmented Generation)과도 결합되고 있다. LLM은 코드를 생성하기 전에 API 문서(API Documentation), 개발 문서(Developer Guide), 공장 매뉴얼(Manual), 작업 절차(Standard Operating Procedure), 환경 지도(Map) 등을 검색하여 보다 정확하고 신뢰성 높은 프로그램을 생성한다.

ROS 2 기반 로봇에서는 생성된 코드가 퍼블리셔(Publisher), 서브스크라이버(Subscriber), 서비스(Service), 액션(Action), 라이프사이클 노드(Lifecycle Node), TF 좌표 변환(TF Transformation) 등을 직접 호출한다. 즉, ROS 2를 대체하는 것이 아니라 ROS 2 위에서 자동으로 애플리케이션(Application)을 개발하는 형태라고 볼 수 있다.

복잡한 작업은 함수(Function) 단위로 분리된다. 예를 들어 물체 탐색(Object Search), 이동(Navigation), 물체 집기(Grasp), 성공 여부 확인(Verification), 결과 보고(Reporting)를 각각 독립적인 함수로 생성한 후 이를 하나의 상위 함수에서 호출하는 구조를 만든다. 이러한 구조는 코드의 가독성과 유지보수성을 크게 향상시킨다.

Code as Policies는 작업 및 모션 계획(TAMP, Task and Motion Planning)과도 긴밀하게 연동된다. LLM은 수행해야 할 작업 순서를 생성하고, 모션 플래너(Motion Planner)는 충돌 없는 이동 경로와 관절 궤적(Trajectory)을 계산한다. 즉, 의미 기반 계획과 기하학 기반 계획을 자연스럽게 연결하는 역할을 수행한다.

비전-언어 모델(VLM, Vision-Language Model)도 함께 활용된다. 생성된 파이썬 코드는 "테이블 위에 있는 물체를 설명해.", "빈 선반을 찾아.", "손상된 장비를 찾아."와 같은 자연어 질의를 VLM에 전달하여 환경 정보를 획득할 수 있다. 이를 통해 인식 모듈과의 통합이 더욱 쉬워진다.

산업용 로봇에서는 Code as Policies의 장점이 더욱 크다. 제조 공정은 자주 변경되며 작업 절차도 지속적으로 수정된다. 기존에는 프로그램을 다시 개발해야 했지만, LLM은 새로운 작업 요구사항에 맞추어 파이썬 코드를 즉시 생성할 수 있으므로 생산 라인의 유연성(Flexibility)이 크게 향상된다.

물류 창고(Warehouse)에서도 다양한 주문(Order), 재고 상태(Inventory), 작업 우선순위(Priority)에 따라 프로그램을 실시간으로 생성할 수 있다. 피킹(Picking), 재고 조사(Inventory Inspection), 선반 보충(Replenishment), 예외 처리(Exception Handling) 등 다양한 작업을 별도의 프로그램 없이 자동으로 구성할 수 있다.

검사 로봇(Inspection Robot) 역시 자산 유형(Asset Type), 이상 상태(Anomaly), 고객 요구사항(Customer Requirement), 유지보수 기록(Maintenance History)에 따라 검사 절차가 달라진다. LLM은 현재 상황에 맞는 검사 프로그램을 생성하고, 영상 처리(Image Processing), 이상 탐지(Anomaly Detection), 보고서 작성(Report Generation), 클라우드 전송(Cloud Synchronization) 등을 자동으로 조합한다.

실외 자율주행 로봇(Outdoor Autonomous Robot)도 날씨(Weather), 지형(Terrain), 조명(Lighting), 접근 가능성(Accessibility), 장비 상태(Equipment Status)를 고려하여 실행 프로그램을 생성할 수 있다. 이는 기존의 고정된 프로그램보다 훨씬 높은 적응성을 제공한다.

그러나 Code as Policies도 일반 소프트웨어 공학(Software Engineering)의 원칙을 반드시 따라야 한다. 생성된 코드는 가독성(Readability), 모듈성(Modularity), 유지보수성(Maintainability), 안정성(Robustness), 예외 처리(Exception Safety), 자원 관리(Resource Management), 인터페이스 일관성(Interface Consistency)을 만족해야 한다. 따라서 많은 시스템에서는 미리 정의된 코드 템플릿(Code Template)과 승인된 API만 사용하도록 제한한다.

보안(Security)은 매우 중요한 문제이다. LLM이 생성한 코드가 운영체제(OS)에 자유롭게 접근할 경우 심각한 보안 문제가 발생할 수 있다. 이를 방지하기 위해 샌드박스(Sandbox), 제한된 파이썬 인터프리터(Restricted Python Interpreter), 권한 기반 API(Permission-Controlled API), 자원 제한(Resource Limitation), 런타임 모니터(Runtime Monitor) 등을 이용하여 실행 환경을 보호한다.

실행 전에 코드 검증(Code Validation)도 수행된다. 생성된 프로그램은 문법 검사(Syntax Check), 정적 분석(Static Analysis), 타입 검사(Type Verification), API 검증(API Validation), 의존성 분석(Dependency Analysis), 시뮬레이션(Simulation), 안전 규칙 검사(Safety Rule Verification)를 거친 후에야 실제 로봇에서 실행된다.

최근에는 먼저 시뮬레이션(Simulation)에서 실행한 후 실제 로봇에 적용하는 방식이 일반적이다. 아이작 심(Isaac Sim), 가제보(Gazebo), 무조코(MuJoCo), 디지털 트윈(Digital Twin) 환경에서 충돌 가능성, 실행 시간, 자원 사용량 등을 충분히 검증한 후 실제 장비에 배포한다. 이러한 Sim-to-Real 접근법은 안전성을 크게 향상시킨다.

실행 중에도 런타임 모니터링(Runtime Monitoring)이 수행된다. 실행 시간이 지나치게 길어지거나, 예상하지 못한 API가 호출되거나, 위험한 이동 경로가 생성되거나, 과도한 힘이 발생하면 감독 시스템(Supervisor)이 즉시 프로그램 실행을 중단하거나 대체 동작을 수행한다.

LLM은 문법적으로는 올바르지만 논리적으로 잘못된 코드나 존재하지 않는 API를 생성하는 환각(Hallucination)이 발생할 수 있다. 따라서 생성된 코드는 실제 라이브러리와 인터페이스를 기반으로 반드시 검증되어야 하며, 무조건 신뢰해서는 안 된다.

대규모 모델은 코드 생성 시간이 길기 때문에 클라우드(Cloud)에서 전략적인 프로그램을 생성하고, 실제 실행은 로봇 내부(Edge)에서 수행하는 계층형 구조가 일반적으로 사용된다. 이렇게 하면 생성 속도와 실시간성을 모두 확보할 수 있다.

최근에는 사고의 연쇄(CoT, Chain of Thought)와 Code as Policies를 결합하는 연구가 활발하다. LLM은 먼저 작업을 단계적으로 추론한 후 그 결과를 파이썬 코드로 변환한다. 이러한 추론 기반 코드 생성은 프로그램의 품질과 안정성을 더욱 향상시킨다.

향후에는 세계 모델(World Model)과 Code as Policies가 결합될 전망이다. 현재 환경뿐 아니라 미래의 환경 변화까지 예측하여 프로그램을 생성하고, 실패를 미리 방지하며, 여러 로봇의 협업까지 고려하는 예측 기반 정책 생성(Predictive Policy Generation)이 가능해질 것으로 기대된다.

다중 로봇(Multi-Robot) 시스템에서도 LLM은 여러 로봇에게 서로 다른 작업을 자동으로 할당하고, 통신(Communication), 동기화(Synchronization), 자원 관리(Resource Allocation), 작업 분배(Task Allocation)를 수행하는 협업 프로그램을 생성할 수 있다.

앞으로는 로봇 전용 소프트웨어 저장소를 학습한 로봇 파운데이션 모델(Robot Foundation Model)이 등장하면서 ROS 패키지(ROS Package), 산업 자동화 프레임워크(Industrial Automation Framework), 시뮬레이션 라이브러리(Simulation Library), 인식 파이프라인(Perception Pipeline), 제어 소프트웨어(Control Software)를 더욱 정확하게 생성할 것으로 기대된다.

궁극적으로 Code as Policies는 프로그래밍을 사람이 미리 작성하는 작업이 아니라 로봇이 필요할 때마다 스스로 생성하고 수정하며 최적화하는 과정으로 변화시키고 있다. 의미 기반 추론(Semantic Reasoning), 멀티모달 인식(Multimodal Perception), 재사용 가능한 소프트웨어 라이브러리(Reusable Software Library), 파이썬 코드 생성(Python Code Generation), 시뮬레이션 검증(Simulation Validation), 런타임 모니터링(Runtime Monitoring)을 통합함으로써 사람의 의도를 안전하고 검증 가능한 실행 프로그램으로 변환하는 차세대 물리 AI(Physical AI)의 핵심 기술로 자리잡아 가고 있다.

##  

## 3.4 Scene Reasoning and Spatial Understanding (with Code)

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

Scene reasoning and spatial understanding constitute two of the most fundamental cognitive capabilities required for intelligent robotic systems operating in real-world environments. While perception systems are responsible for detecting objects, estimating depth, recognizing semantic categories, and constructing environmental maps, these outputs alone are insufficient for autonomous decision making. Robots must also understand how objects relate to one another, how humans interact with their surroundings, how environmental structures constrain possible actions, and how spatial relationships influence task execution. Large Language Models (LLMs) significantly enhance robotic intelligence by providing high-level semantic reasoning over perception results, enabling robots to interpret complex scenes, infer hidden relationships, predict future interactions, and generate context-aware plans. Within modern Vision-Language-Action (VLA) architectures, scene reasoning transforms raw perception into actionable knowledge that supports planning, manipulation, navigation, and human-robot collaboration.

Traditional robotic perception pipelines primarily focus on recognizing individual objects using computer vision algorithms. Object detectors identify chairs, tables, cups, doors, people, and machines, while semantic segmentation assigns labels to image regions and depth estimation predicts three-dimensional geometry. Although these perception modules successfully identify environmental components, they generally provide limited understanding of how these objects relate to one another or how they should influence robot behavior. Scene reasoning extends perception beyond recognition by interpreting the semantic structure of the environment as a coherent whole rather than a collection of isolated objects.

Large Language Models operate as semantic reasoning engines positioned above perception systems. Instead of processing raw pixels directly, the LLM receives structured scene descriptions generated by vision encoders, Vision-Language Models, object detectors, semantic mapping systems, depth estimation algorithms, point cloud processors, simultaneous localization and mapping systems, and robot state estimators. These multimodal representations provide the contextual information required for higher-level reasoning regarding environmental organization, object functionality, task feasibility, and human intent.

Scene understanding begins with semantic scene representation. Rather than describing an environment solely through geometric coordinates, the robot constructs an interpretable semantic model identifying rooms, furniture, tools, obstacles, pathways, workspaces, storage locations, and interaction zones. Each detected object possesses attributes describing category, location, orientation, physical dimensions, affordances, accessibility, ownership, and contextual significance. The LLM integrates these attributes into a comprehensive representation of the environment.

Semantic scene representation also captures relationships among objects. Instead of recognizing only that a cup and a table exist, the robot understands that the cup is located on the table, near a laptop, beside a notebook, within arm\'s reach of a seated person, and partially occluded by another object. These relational descriptions provide significantly richer information than isolated detections because robotic actions frequently depend on object relationships rather than individual object identities.

Spatial understanding requires interpreting both absolute and relative positions. Absolute localization determines where objects exist within global coordinate systems using mapping and localization technologies. Relative spatial reasoning instead considers relationships such as left, right, above, below, behind, inside, outside, adjacent to, between, surrounding, aligned with, attached to, or supporting. Humans naturally communicate using these qualitative spatial concepts, making their interpretation essential for effective human-robot interaction.

Large Language Models naturally excel at reasoning over qualitative spatial relationships because they are extensively represented throughout natural language. Instructions such as "Pick up the wrench next to the toolbox," "Inspect the valve beneath the pipe," or "Move the pallet behind the forklift" require semantic spatial reasoning rather than precise numerical coordinates. The LLM translates these natural language descriptions into structured spatial constraints that guide perception and motion planning.

Three-dimensional reasoning further extends spatial understanding. Modern robotic systems frequently employ RGB-D cameras, stereo vision, LiDAR sensors, structured light cameras, time-of-flight sensors, and multi-view reconstruction techniques to generate dense three-dimensional scene representations. Rather than analyzing only two-dimensional images, the LLM reasons over volumetric spatial structures, free space, occupied regions, reachable workspaces, obstacle geometry, and manipulation feasibility.

Affordance reasoning represents another critical component of scene understanding. Affordances describe the possible interactions between robots and environmental objects. Chairs support sitting, handles enable pulling, buttons allow pressing, containers hold objects, shelves provide storage, doors permit passage after opening, and charging stations replenish batteries. The LLM combines visual observations with commonsense knowledge to infer object affordances even when they are not explicitly represented within perception outputs.

Commonsense reasoning substantially improves environmental interpretation. Human environments contain numerous implicit conventions that are rarely encoded within perception algorithms. Cups are typically found in kitchens or offices, books belong on shelves, tools are stored in toolboxes, dishes should remain upright, fragile objects require careful handling, and emergency exits should remain unobstructed. Large Language Models incorporate this world knowledge into robotic decision making, allowing robots to generate more realistic and contextually appropriate behaviors.

Context plays an essential role throughout scene reasoning. Identical objects may possess entirely different significance depending on the surrounding environment. A chair located in a meeting room supports seating, while a chair positioned beneath a high shelf may serve as a climbing aid, although climbing may violate robot safety policies. A box in a warehouse likely represents inventory, whereas an identical box inside a hospital corridor may obstruct emergency access. Scene reasoning therefore depends on contextual interpretation rather than object recognition alone.

Human presence introduces additional complexity into spatial reasoning. Robots must recognize not only human locations but also predicted human intentions, attention direction, activity context, reachable workspaces, and social interaction zones. A person reaching toward an object may soon occupy nearby space, requiring the robot to modify its navigation plan proactively. Similarly, personal space should remain respected during collaborative interactions even when no physical collision would occur.

Dynamic environments require continuous scene updating. Objects move, doors open and close, humans change locations, lighting conditions vary, temporary obstacles appear, and workspace organization evolves throughout robot operation. The LLM continuously incorporates updated perception results into its semantic representation, allowing planning systems to adapt without reconstructing the entire scene model from scratch.

Object permanence constitutes another important reasoning capability. Humans naturally understand that temporarily occluded objects continue existing even when invisible. Robots employing scene reasoning similarly infer that an object hidden behind a cabinet, under a table, or inside a container likely remains present unless evidence indicates otherwise. This capability significantly improves search efficiency and task continuity.

Occlusion reasoning further supports robust manipulation. Partial visibility frequently complicates robotic perception. Instead of requiring complete object observations, the LLM reasons about partially visible structures using contextual information, object priors, environmental constraints, and commonsense expectations. Such reasoning enables robots to manipulate cluttered environments more effectively than purely appearance-based approaches.

Scene graphs provide an increasingly common intermediate representation supporting LLM reasoning. A scene graph represents environmental entities as nodes while spatial, functional, and semantic relationships become graph edges. Objects, humans, rooms, tools, containers, pathways, and equipment form interconnected semantic networks describing environmental organization. Large Language Models reason naturally over these graph structures to answer spatial queries, generate plans, or explain environmental relationships.

Natural language scene descriptions generated by Vision-Language Models provide another valuable input. Instead of processing only numerical perception outputs, the LLM receives textual descriptions such as "A red toolbox is located beneath the workbench beside a blue storage cabinet." These language-based representations align naturally with the pretrained reasoning capabilities of Large Language Models, improving semantic understanding without requiring specialized symbolic reasoning engines.

Spatial reasoning also supports navigation. Autonomous mobile robots must determine not only collision-free paths but also semantically appropriate routes. Hospital robots should avoid intensive care areas when possible, warehouse robots should respect designated pedestrian crossings, inspection robots should prioritize accessible equipment, and service robots should approach humans from socially acceptable directions. These navigation decisions require semantic scene understanding beyond geometric path planning.

Manipulation planning similarly benefits from spatial reasoning. Before grasping an object, the robot evaluates accessibility, grasp approach direction, neighboring obstacles, object stability, supporting surfaces, human proximity, and task objectives. The LLM integrates these factors into high-level reasoning while motion planners compute physically feasible trajectories satisfying these semantic constraints.

Task execution frequently depends upon spatial hierarchy. Objects belong to containers, containers exist within rooms, rooms belong to buildings, buildings form facilities, and facilities operate within larger organizational contexts. Hierarchical spatial representations enable robots to reason across multiple scales simultaneously. A command to retrieve equipment from Building A requires reasoning first at facility level, then room level, shelf level, container level, and finally object level.

Temporal scene reasoning extends spatial understanding into dynamic prediction. Rather than describing only current environmental states, the LLM anticipates future scene evolution. Humans walking through corridors, forklifts approaching intersections, elevators changing floors, automatic doors opening, and production lines advancing all influence future robot behavior. Predictive scene reasoning therefore supports safer and more efficient planning.

Memory substantially enhances scene reasoning quality. Episodic memory stores previous observations, semantic memory captures environmental knowledge, procedural memory identifies typical object arrangements, and long-term maps preserve persistent spatial structure. Robots remember previously visited locations, commonly used pathways, equipment positions, charging stations, restricted areas, and historical environmental changes. This accumulated experience improves future reasoning efficiency.

Scene reasoning frequently incorporates Retrieval-Augmented Generation. Facility documentation, building maps, operating procedures, maintenance records, digital twins, equipment databases, architectural drawings, and semantic maps provide additional contextual information extending beyond immediate perception. Retrieved knowledge supplements visual observations, allowing more informed reasoning regarding specialized industrial environments.

Industrial applications demonstrate the importance of semantic scene understanding. Manufacturing robots must distinguish workstations, safety zones, storage areas, production lines, maintenance equipment, hazardous materials, and quality inspection stations. Object recognition alone cannot identify appropriate operational procedures because environmental context determines permissible actions.

Warehouse automation similarly depends on spatial reasoning. Pallets, shelves, conveyors, loading docks, charging stations, pedestrian pathways, restricted zones, and inventory locations form complex semantic environments. Robots reason about accessibility, storage organization, traffic flow, package priority, and operational constraints while generating efficient logistics plans.

Inspection robots require detailed understanding of infrastructure layout. Bridges, pipelines, power stations, construction sites, industrial plants, and utility networks contain numerous interconnected components whose spatial organization directly influences inspection procedures. LLM-based scene reasoning interprets these structural relationships to generate systematic inspection strategies.

Outdoor autonomous robots encounter particularly challenging scene reasoning problems because environments remain highly dynamic and weakly structured. Terrain types, vegetation, roads, sidewalks, construction areas, weather conditions, temporary obstacles, and moving vehicles continuously alter spatial relationships. The LLM integrates perception, mapping, and world knowledge to interpret these changing environments.

Human-robot collaboration depends heavily on scene understanding. Collaborative robots reason about shared workspaces, tool ownership, human attention, task allocation, object handover locations, conversational context, and social interaction zones. Scene reasoning enables robots to behave predictably, safely, and naturally during cooperative activities.

Despite remarkable progress, scene reasoning remains fundamentally limited by perception quality. Incorrect object detection, inaccurate depth estimation, localization drift, incomplete maps, sensor noise, or ambiguous observations may propagate into high-level reasoning. Consequently, uncertainty estimation becomes an integral component of semantic reasoning. Rather than assuming perception is perfectly accurate, the LLM explicitly considers confidence estimates while generating plans.

Multi-camera perception significantly improves scene understanding. Head-mounted cameras, wrist cameras, panoramic cameras, overhead cameras, drone imagery, and fixed surveillance systems collectively provide complementary observations reducing occlusions and improving environmental coverage. LLMs integrate these heterogeneous viewpoints into unified semantic scene representations.

World models increasingly complement scene reasoning. Instead of reasoning exclusively about currently observed environments, future robotic systems maintain predictive internal models describing latent environmental dynamics. World models estimate hidden object states, predict future interactions, simulate hypothetical actions, and evaluate long-horizon consequences before physical execution. Large Language Models cooperate with these predictive models to support more sophisticated planning.

Foundation models trained jointly on language, vision, video, depth, three-dimensional geometry, manipulation demonstrations, and environmental interactions continue expanding robotic scene understanding capabilities. Rather than separately processing images, language, and spatial maps, future multimodal foundation models will generate unified world representations integrating appearance, geometry, semantics, temporal dynamics, and physical interactions within a single reasoning framework.

Recent research increasingly combines scene reasoning with Chain of Thought reasoning. Instead of directly answering spatial questions, the LLM incrementally analyzes object relationships, environmental constraints, human intentions, accessibility, safety considerations, and task objectives before producing conclusions. This structured reasoning substantially improves accuracy for complex manipulation and navigation tasks requiring multiple interconnected spatial inferences.

Scene reasoning also supports explainable robotic decision making. Rather than executing actions without explanation, robots may describe their reasoning process using natural language. For example, a robot may explain that it selected a particular grasp because another object blocked the preferred approach or that it avoided a hallway because emergency personnel were currently present. Such explanations improve user trust and facilitate collaborative operation.

Future Vision-Language-Action systems are expected to perform holistic spatial reasoning across multiple abstraction levels simultaneously. Instead of separately interpreting geometry, semantics, temporal dynamics, human activities, object affordances, and environmental constraints, next-generation embodied foundation models will integrate these information sources into unified cognitive world representations. These representations will enable robots to understand environments similarly to humans, supporting flexible task execution, long-horizon planning, natural communication, predictive reasoning, and continual adaptation across diverse industrial, commercial, domestic, healthcare, construction, agricultural, logistics, and service robotics applications. Scene reasoning and spatial understanding will therefore remain foundational cognitive capabilities underpinning the next generation of intelligent Physical AI systems capable of robust autonomous operation in complex real-world environments.

장면 추론(Scene Reasoning)과 공간 이해(Spatial Understanding)는 실제 환경에서 자율적으로 동작하는 지능형 로봇이 반드시 갖추어야 하는 핵심 인지(Cognitive) 능력이다. 인식 시스템(Perception System)은 객체(Object)를 탐지하고 깊이(Depth)를 추정하며 의미 정보를 생성할 수 있지만, 그것만으로는 로봇이 올바른 의사결정을 수행하기 어렵다. 로봇은 객체 간의 관계, 사람과 환경의 상호작용, 공간적 제약, 작업 수행에 필요한 맥락(Context)을 함께 이해해야 한다.

대규모 언어 모델(LLM, Large Language Model)은 인식 시스템보다 상위 계층에서 의미적 추론(Semantic Reasoning)을 수행한다. 카메라(Camera), 라이다(LiDAR), 깊이 카메라(Depth Camera), 포인트 클라우드(Point Cloud), 지도(Map), 로봇 상태(Robot State) 등의 정보를 직접 처리하기보다는, 비전 인코더(Vision Encoder)와 비전-언어 모델(VLM, Vision-Language Model)이 생성한 의미 정보를 입력받아 환경 전체를 종합적으로 해석한다.

장면 이해(Scene Understanding)의 첫 번째 단계는 의미 기반 장면 표현(Semantic Scene Representation)이다. 단순히 좌표(Coordinate)를 저장하는 것이 아니라, 방(Room), 책상(Table), 의자(Chair), 도구(Tool), 통로(Path), 장애물(Obstacle), 작업 공간(Workspace), 보관 장소(Storage Area) 등을 의미적으로 표현한다. 각 객체는 종류(Category), 위치(Location), 방향(Orientation), 크기(Size), 접근 가능성(Accessibility), 기능(Function) 등의 다양한 속성을 함께 가진다.

LLM은 개별 객체를 독립적으로 인식하지 않고 객체 간의 관계(Relationship)를 함께 이해한다. 예를 들어 컵(Cup)이 테이블(Table) 위에 있고, 노트북(Laptop) 옆에 있으며, 사람(Person)의 손이 닿는 위치에 있고, 다른 물체에 일부 가려져 있다는 정보를 하나의 통합된 장면으로 해석한다. 이러한 관계 정보는 단순한 객체 인식보다 훨씬 높은 수준의 작업 계획을 가능하게 한다.

공간 이해는 절대 위치(Absolute Position)와 상대 위치(Relative Position)를 동시에 고려한다. 절대 위치는 지도 좌표를 기반으로 객체의 위치를 나타내며, 상대 위치는 "왼쪽(Left)", "오른쪽(Right)", "위(Above)", "아래(Below)", "뒤(Behind)", "안쪽(Inside)", "옆(Next To)"과 같은 사람 중심의 공간 표현을 의미한다. 사람은 이러한 표현을 사용하여 로봇에게 명령하기 때문에 LLM은 이를 자연스럽게 이해해야 한다.

예를 들어 "공구함 옆의 렌치를 가져와." 또는 "파이프 아래의 밸브를 점검해."와 같은 명령은 정확한 좌표보다 공간적 관계를 이해해야 수행할 수 있다. LLM은 이러한 자연어 표현을 공간 제약(Spatial Constraint)으로 변환하고, 이를 인식 시스템과 모션 계획(Motion Planning)에 전달한다.

최근 로봇은 RGB-D 카메라(RGB-D Camera), 스테레오 카메라(Stereo Camera), 구조광(Structured Light), 라이다(LiDAR), 비행시간 센서(ToF, Time-of-Flight Sensor) 등을 이용하여 3차원 환경을 인식한다. LLM은 이러한 3차원 정보(3D Information)를 활용하여 자유 공간(Free Space), 점유 공간(Occupied Space), 작업 가능 영역(Reachable Workspace), 장애물 구조(Obstacle Geometry)를 이해하고 보다 현실적인 계획을 생성한다.

어포던스(Affordance) 추론은 공간 이해에서 매우 중요한 요소이다. 어포던스란 객체가 어떤 행동을 허용하는지를 의미한다. 예를 들어 의자는 앉을 수 있고, 손잡이는 당길 수 있으며, 버튼은 누를 수 있고, 문은 열어서 통과할 수 있으며, 충전기는 로봇을 충전할 수 있다. LLM은 시각 정보와 상식을 결합하여 이러한 기능을 자동으로 추론한다.

상식(Common Sense)은 장면 이해의 핵심 요소이다. 컵은 일반적으로 주방에 있고, 책은 책장에 있으며, 공구는 공구함에 보관되고, 깨지기 쉬운 물체는 조심해서 다루어야 한다는 지식은 대부분 인식 시스템에는 존재하지 않는다. LLM은 대규모 사전학습을 통해 이러한 상식을 습득하였으며 이를 작업 계획에 적극적으로 활용한다.

동일한 객체라도 주변 환경(Context)에 따라 의미가 달라질 수 있다. 회의실(Meeting Room)의 의자는 사람이 앉기 위한 것이지만, 높은 선반 아래 있는 의자는 발판으로 사용할 수도 있다. 그러나 안전 규칙(Safety Rule)에 따라 로봇은 의자를 이용하여 올라가면 안 될 수도 있다. 따라서 장면 이해는 객체 자체보다 환경 전체를 함께 고려하는 과정이다.

사람(Human)이 존재하는 환경에서는 더욱 복잡한 추론이 필요하다. 로봇은 사람의 위치뿐 아니라 이동 방향(Direction), 시선(Gaze), 작업 의도(Intent), 손이 닿는 영역(Reachable Workspace), 개인 공간(Personal Space)까지 함께 고려해야 한다. 예를 들어 사람이 물건을 집으려는 상황이라면 로봇은 먼저 양보하거나 다른 경로를 선택하는 것이 바람직하다.

현실 세계는 계속 변화하는 동적 환경(Dynamic Environment)이다. 사람이 이동하고, 문이 열리고 닫히며, 장애물이 새롭게 생기고, 조명이 변하고, 물체의 위치도 지속적으로 바뀐다. LLM은 새로운 센서 정보를 지속적으로 반영하여 장면 표현(Scene Representation)을 업데이트하고 기존 작업 계획을 수정한다.

객체 지속성(Object Permanence)은 사람이 자연스럽게 수행하는 추론이다. 물체가 책상 뒤나 상자 안에 잠시 가려졌더라도 계속 존재한다고 판단한다. LLM도 이러한 개념을 이용하여 잠시 보이지 않는 객체가 여전히 존재할 가능성을 고려하므로 탐색(Search) 효율을 크게 향상시킬 수 있다.

가려짐(Occlusion) 추론도 매우 중요하다. 실제 환경에서는 물체 일부만 보이는 경우가 많다. LLM은 보이는 일부 정보와 주변 환경, 상식, 이전 관측 결과를 종합하여 전체 객체를 추론할 수 있다. 이러한 능력은 복잡한 작업 공간이나 창고 환경에서 매우 유용하다.

최근에는 장면 그래프(Scene Graph)가 LLM의 중요한 입력으로 사용된다. 장면 그래프는 객체(Object)를 노드(Node)로 표현하고 객체 간의 관계를 엣지(Edge)로 표현하는 그래프 구조이다. 사람, 로봇, 도구, 작업 공간, 통로 등이 서로 연결된 의미 네트워크를 구성하며, LLM은 이러한 구조를 기반으로 공간 추론을 수행한다.

비전-언어 모델(VLM)은 장면을 자연어로 설명할 수도 있다. 예를 들어 "파란 공구함이 작업대 아래에 있으며, 그 옆에 빨간 상자가 있다."와 같은 문장을 생성한다. LLM은 이러한 자연어 기반 장면 설명을 매우 효과적으로 이해할 수 있으므로 복잡한 기호(Symbol) 기반 추론보다 더욱 자연스러운 의미 해석이 가능하다.

공간 이해는 자율주행(Navigation)에도 매우 중요한 역할을 한다. 병원 로봇은 중환자실을 피해서 이동해야 하고, 창고 로봇은 사람 전용 통로를 우선적으로 고려해야 하며, 서비스 로봇은 사람에게 정면에서 접근하기보다 자연스러운 방향에서 접근하는 것이 바람직하다. 이러한 판단은 단순한 경로 생성(Path Planning)만으로는 수행하기 어렵다.

조작(Manipulation) 작업에서도 공간 추론은 필수적이다. 물체를 집기 전에 접근 가능한지, 다른 물체가 방해하는지, 손을 어느 방향으로 넣어야 하는지, 주변 사람과 충돌하지 않는지 등을 먼저 판단해야 한다. LLM은 이러한 의미적 조건을 분석하고, 이후 모션 플래너(Motion Planner)가 실제 관절 궤적(Trajectory)을 생성한다.

작업 수행은 계층적 공간 구조(Hierarchical Spatial Structure)를 기반으로 이루어진다. 객체는 상자(Container)에 들어 있고, 상자는 방(Room)에 있으며, 방은 건물(Building)에 포함된다. "A동에서 장비를 가져와."라는 명령을 수행하려면 건물 → 방 → 선반 → 상자 → 장비 순서로 공간을 단계적으로 이해해야 한다.

시간적 장면 추론(Temporal Scene Reasoning)은 미래 상황까지 예측한다. 사람이 이동하는 방향, 지게차(Forklift)의 이동, 엘리베이터(Elevator)의 위치 변화, 자동문(Automatic Door)의 개폐 등을 예측하여 현재 계획을 수정한다. 이는 보다 안전하고 효율적인 자율주행을 가능하게 한다.

메모리(Memory)는 공간 이해의 품질을 크게 향상시킨다. 에피소드 메모리(Episodic Memory)는 이전 작업 경험을 저장하고, 의미 메모리(Semantic Memory)는 일반 지식을 저장하며, 장기 지도(Long-Term Map)는 건물 구조와 충전기 위치 등을 기억한다. 이러한 정보는 반복 작업에서 매우 높은 효율을 제공한다.

최근에는 검색 증강 생성(RAG, Retrieval-Augmented Generation)을 함께 사용한다. 시설 도면(Building Map), 유지보수 문서(Maintenance Manual), 설비 데이터베이스(Database), 디지털 트윈(Digital Twin), 운영 절차(Standard Operating Procedure) 등을 검색하여 현재 인식 정보와 함께 활용함으로써 보다 정확한 장면 이해가 가능해진다.

산업용 로봇은 작업 공간(Workspace), 안전 구역(Safety Zone), 위험 물질(Hazardous Material), 생산 설비(Production Equipment), 검사 구역(Inspection Area) 등을 구분해야 한다. 단순히 객체를 인식하는 것만으로는 적절한 작업 절차를 결정할 수 없으며, 환경 전체의 의미를 이해해야 한다.

물류 창고(Warehouse)에서는 선반(Shelf), 팔레트(Pallet), 컨베이어(Conveyor), 충전기(Charging Station), 사람 통로(Pedestrian Lane), 출입 제한 구역(Restricted Area)을 함께 고려해야 한다. LLM은 이러한 공간 구조를 분석하여 이동 경로와 작업 순서를 최적화한다.

시설 점검(Inspection) 로봇은 배관(Pipeline), 발전소(Power Plant), 교량(Bridge), 건설 현장(Construction Site)과 같은 복잡한 구조를 이해해야 한다. 장비 간의 연결 관계와 위치를 분석하여 체계적인 검사 순서를 생성하는 것이 장면 추론의 중요한 역할이다.

실외 자율주행(Outdoor Autonomous Driving)에서는 지형(Terrain), 식생(Vegetation), 도로(Road), 보도(Sidewalk), 공사 구역(Construction Area), 날씨(Weather), 차량(Vehicle) 등 매우 다양한 요소가 동시에 변화한다. LLM은 지도(Map), 인식 정보, 상식을 통합하여 이러한 복잡한 환경을 해석한다.

사람-로봇 협업(Human-Robot Collaboration)에서도 장면 이해는 핵심 기술이다. 사람과 로봇이 공유하는 작업 공간, 공구의 소유권, 물건 전달 위치(Handover Position), 대화 상황(Conversation Context), 사회적 거리(Social Distance)를 함께 고려하여 자연스럽고 안전한 협업을 수행한다.

그러나 장면 추론의 성능은 인식 시스템의 품질에 크게 의존한다. 객체 인식 오류(Object Detection Error), 깊이 추정 오차(Depth Error), 지도 오차(Localization Drift), 센서 잡음(Sensor Noise)은 모두 잘못된 추론으로 이어질 수 있다. 따라서 불확실성 추정(Uncertainty Estimation)을 함께 수행하여 신뢰도가 낮은 정보는 보수적으로 처리해야 한다.

다중 카메라(Multi-Camera) 시스템은 장면 이해를 크게 향상시킨다. 헤드 카메라(Head Camera), 손목 카메라(Wrist Camera), 전방 카메라(Front Camera), 드론(Drone), 고정 카메라(Fixed Camera)의 다양한 시점을 통합하면 가려짐(Occlusion)을 줄이고 더욱 완전한 환경 모델을 구축할 수 있다.

최근에는 세계 모델(World Model)과 장면 추론을 결합하는 연구가 활발하다. 현재 환경뿐 아니라 앞으로 발생할 상황까지 예측하여 작업 계획을 생성하고, 여러 행동의 결과를 시뮬레이션한 후 가장 적합한 행동을 선택하는 예측 기반 추론(Predictive Reasoning)이 가능해지고 있다.

향후에는 언어(Language), 영상(Image), 비디오(Video), 깊이 정보(Depth), 3차원 기하 정보(3D Geometry), 조작 데이터(Manipulation), 환경 변화(Environment Dynamics)를 함께 학습하는 멀티모달 파운데이션 모델(Multimodal Foundation Model)이 등장할 것으로 예상된다. 이러한 모델은 장면을 사람처럼 이해하고 장기 계획(Long-Horizon Planning), 자연스러운 대화(Natural Communication), 예측 기반 의사결정(Predictive Decision Making), 지속적인 적응(Continual Adaptation)을 수행하는 차세대 물리 AI(Physical AI)의 핵심 기술이 될 것으로 전망된다.

##  

## 3.5 SayCan: Grounding Language Models with Affordances (with Code)

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

SayCan represents one of the most influential milestones in the evolution of intelligent robotic systems by demonstrating how Large Language Models (LLMs) can be effectively grounded in the physical capabilities of real robots. Before SayCan, LLMs exhibited remarkable language understanding and reasoning abilities but lacked awareness of whether the actions they proposed could actually be executed by a robot operating in the physical world. Conversely, traditional robotic control systems possessed precise knowledge of robot capabilities but lacked semantic reasoning and natural language understanding. SayCan bridges these complementary strengths by combining high-level language reasoning with low-level robot affordance estimation, enabling robots to generate plans that are both semantically meaningful and physically executable.

The name "SayCan" originates from the integration of two complementary concepts. The "Say" component represents the reasoning capability of a Large Language Model, which determines what actions should ideally be performed to satisfy a user\'s request. The "Can" component represents the robot\'s estimation of whether each candidate action can actually be executed successfully given the current environmental conditions, sensor observations, robot state, and available skills. Rather than selecting actions solely because they appear linguistically appropriate, SayCan chooses actions that maximize both semantic relevance and physical feasibility.

Traditional language-driven robotic systems frequently suffer from a disconnect between reasoning and execution. An LLM may generate an excellent high-level plan describing how to complete a task, yet individual actions may be impossible because the target object cannot be reached, the robot lacks the necessary manipulation skill, the object does not exist in the environment, or environmental constraints prevent successful execution. SayCan addresses this limitation by explicitly incorporating affordance estimation into every planning decision.

Within modern Vision-Language-Action architectures, SayCan occupies the interface between cognitive reasoning and robotic execution. Perception systems continuously provide semantic information describing the current environment through RGB cameras, depth sensors, LiDAR, force sensors, tactile sensors, localization systems, and semantic maps. Vision encoders transform these observations into high-level representations, while robot controllers estimate the likelihood of successfully executing each available skill. Simultaneously, the Large Language Model reasons about the user\'s instruction, decomposes the task into intermediate objectives, and predicts which robot skills are most semantically appropriate. SayCan combines these two information sources to determine the next action.

The central mathematical concept underlying SayCan is action scoring. Instead of selecting actions according to language probability alone, the system computes a combined score incorporating both semantic likelihood and affordance probability. The language model estimates how useful a particular skill would be for accomplishing the user\'s objective, while the affordance model estimates the probability that the robot can successfully execute that skill under current conditions. Actions receiving high scores in both dimensions become preferred candidates for execution.

Affordances represent the possible interactions available between a robot and its environment. A robot may be capable of grasping objects, opening drawers, pressing buttons, navigating between locations, operating elevators, placing objects onto shelves, docking with charging stations, or inspecting equipment. However, the success of these actions depends upon current environmental conditions. A grasp may fail because an object is outside the robot\'s workspace. A navigation action may become impossible because a corridor is blocked. A button cannot be pressed if another object obstructs access. SayCan explicitly evaluates these physical constraints before selecting actions.

One of the major innovations introduced by SayCan is separating semantic planning from physical execution while maintaining continuous interaction between the two. The Large Language Model remains responsible for understanding user intent, generating high-level task decomposition, and predicting useful skills. The robot skill execution system independently estimates execution feasibility. Rather than allowing either component to dominate planning decisions, SayCan combines both perspectives into a unified decision-making framework.

Task execution under SayCan proceeds iteratively rather than generating a complete plan at once. At each decision step, the robot evaluates all available skills within its library. The language model predicts which skills best advance the overall task objective, while affordance estimators evaluate current execution probability. The highest-scoring skill is selected and executed. Following execution, perception updates the environmental representation, affordance probabilities are recomputed, and the planning process repeats until task completion.

Skill libraries form an essential component of SayCan. Rather than generating arbitrary motor commands, robots possess predefined reusable skills representing atomic behaviors. These skills may include navigation, object detection, grasping, opening containers, closing doors, placing objects, activating switches, inspecting equipment, charging batteries, communicating with humans, or waiting for external events. The LLM reasons about which skill should be executed next without modifying the internal implementation of individual skills.

Semantic reasoning performed by the language model extends beyond simple keyword matching. The model interprets abstract human goals, understands commonsense knowledge, infers missing procedural details, and predicts intermediate objectives required for successful task completion. For example, if instructed to prepare a cup of tea, the language model recognizes that water must first be obtained, heated, poured, and combined with tea before serving, even though many of these steps remain unstated by the user.

Affordance estimation, however, grounds this reasoning within physical reality. Although preparing tea may require locating a kettle, the robot cannot perform this step if no kettle exists in the environment. Similarly, opening a cabinet becomes impossible if the cabinet door is locked or blocked. The affordance model therefore evaluates actual environmental conditions rather than relying solely upon abstract reasoning.

Perception continuously supports affordance estimation throughout execution. Object detection identifies candidate manipulation targets. Semantic segmentation estimates environmental structure. Depth estimation predicts object accessibility. Point cloud processing computes reachable workspaces. Force sensing monitors grasp success. Localization systems estimate robot position relative to navigation goals. Collectively, these perception modules provide the information necessary for accurate affordance prediction.

SayCan naturally supports long-horizon planning because tasks are decomposed into sequences of executable skills. Rather than attempting to predict every future action simultaneously, the robot repeatedly selects the most appropriate next skill given updated environmental observations. This incremental planning strategy substantially improves robustness within dynamic environments where future conditions remain uncertain.

Human instructions frequently contain ambiguity that requires contextual interpretation. Commands such as "Bring me something to drink" permit multiple valid solutions. The language model reasons about likely user intentions, while affordance estimation determines which candidate objects are currently available and accessible. If water is nearby while coffee supplies are unavailable, the combined reasoning process naturally selects water as the preferred solution.

Context awareness significantly improves planning quality. The robot considers user location, environmental state, object availability, battery capacity, task history, and current robot configuration when evaluating candidate actions. A robot carrying another object cannot immediately grasp a new object without first freeing its gripper. Similarly, low battery capacity may prioritize charging before initiating lengthy navigation tasks. Affordance estimation captures these contextual constraints naturally.

Memory further strengthens SayCan by preserving previous execution experience. Episodic memory records successful and failed task executions, semantic memory stores knowledge regarding object functions and environmental organization, procedural memory captures robot skills, and long-term environmental memory records frequently visited locations. These memories influence both language reasoning and affordance estimation during future planning episodes.

SayCan also supports adaptive replanning. Real-world environments frequently change unexpectedly due to moving humans, relocated objects, blocked pathways, equipment failures, or sensor uncertainty. After every executed skill, the robot updates its environmental understanding and reevaluates all remaining candidate actions. This continuous replanning capability allows the robot to recover gracefully from unexpected situations without abandoning the entire mission.

The framework naturally integrates with Vision-Language Models that generate semantic scene descriptions. Rather than reasoning exclusively over symbolic object labels, the language model may analyze textual descriptions such as "The refrigerator door is open, a milk carton is visible on the middle shelf, and a chair partially blocks the approach." These rich semantic descriptions improve affordance estimation by providing contextual information unavailable through object detection alone.

Navigation tasks particularly benefit from affordance-aware planning. Multiple routes may connect identical start and destination locations, yet temporary obstacles, pedestrian traffic, restricted areas, or hazardous conditions influence which route remains executable. SayCan selects navigation behaviors balancing semantic task objectives with real-time environmental accessibility.

Manipulation planning similarly benefits from affordance reasoning. A requested object may exist but remain unreachable because another object blocks access. Rather than repeatedly attempting impossible grasps, SayCan first selects prerequisite actions such as moving obstructing objects, opening cabinets, repositioning the robot, or adjusting camera viewpoints before attempting the primary manipulation task.

Industrial robotics provides compelling applications for SayCan. Manufacturing environments contain numerous specialized tools, safety procedures, production workflows, and equipment configurations. The language model interprets high-level production goals while affordance estimation evaluates machine availability, workspace accessibility, robot tooling, material presence, and safety constraints. Consequently, generated plans remain grounded within actual factory conditions rather than idealized assumptions.

Warehouse automation similarly benefits from affordance-aware planning. Inventory availability, shelf accessibility, pallet occupancy, aisle congestion, charging status, and robot workload continuously influence operational decisions. SayCan enables warehouse robots to reason semantically about fulfillment objectives while respecting real-time logistical constraints.

Healthcare robotics presents another important application domain. Service robots assisting patients must interpret natural language requests while respecting patient safety, equipment availability, medical protocols, accessibility constraints, and environmental organization. Affordance-aware reasoning substantially reduces the probability of generating unsafe or infeasible actions within these highly sensitive environments.

Inspection robots operating in industrial facilities similarly require grounded reasoning. Inspection procedures frequently depend upon equipment accessibility, environmental safety conditions, operational status, lighting quality, sensor visibility, and regulatory requirements. SayCan dynamically selects inspection skills that remain executable under current conditions while satisfying overall inspection objectives.

One of SayCan\'s greatest strengths lies in its modularity. The language model, perception system, affordance estimator, skill library, navigation stack, manipulation planner, and robot controller remain independently developed software components communicating through well-defined interfaces. Improvements to one subsystem therefore benefit the overall architecture without requiring complete redesign.

Despite its advantages, SayCan also faces important limitations. Affordance estimation quality depends heavily upon perception accuracy. Incorrect object detection, localization errors, incomplete maps, sensor noise, or environmental uncertainty may cause inaccurate feasibility predictions. Consequently, robust perception remains essential for reliable grounded planning.

Language models themselves remain susceptible to hallucinations and incorrect commonsense reasoning. Although affordance estimation filters physically impossible actions, semantically incorrect planning decisions may still occur if the language model misunderstands user intent or generates inappropriate task decompositions. Additional validation layers therefore remain necessary for safety-critical applications.

Scalability presents another research challenge. As robot skill libraries continue expanding, evaluating affordance probabilities for hundreds or thousands of candidate actions becomes increasingly computationally expensive. Efficient retrieval mechanisms, hierarchical skill organization, learned embeddings, and approximate search algorithms therefore become important for maintaining real-time planning performance.

Recent research increasingly extends SayCan beyond single-robot operation. Multi-robot systems require affordance reasoning not only regarding individual robot capabilities but also regarding collaborative execution, resource sharing, task allocation, communication, synchronization, and collective workspace management. Language models coordinate high-level cooperation while affordance estimation evaluates each robot\'s individual ability to contribute toward shared objectives.

Another important extension incorporates world models into affordance reasoning. Instead of evaluating only the current environment, predictive world models estimate future environmental states resulting from candidate actions. Affordance estimation therefore becomes predictive rather than reactive, allowing robots to anticipate future opportunities and avoid impending failures before they occur.

Retrieval-Augmented Generation further enhances SayCan by providing access to external documentation, operating manuals, digital twins, maintenance databases, facility maps, enterprise knowledge bases, and previous execution logs. Retrieved information supplements language reasoning while improving affordance estimation within specialized industrial environments.

Chain of Thought reasoning also complements SayCan effectively. Rather than selecting skills immediately, the language model first performs structured reasoning regarding task decomposition, object relationships, temporal dependencies, environmental constraints, safety considerations, and intermediate goals. Affordance estimation then grounds each reasoning step within executable robot capabilities. This combination significantly improves planning quality for complex long-horizon tasks.

Future Vision-Language-Action architectures are expected to integrate SayCan with multimodal foundation models capable of jointly reasoning over language, vision, depth, video, tactile sensing, force measurements, proprioception, semantic maps, world models, and long-term episodic memory. Rather than combining independent language and affordance modules, future embodied foundation models will learn unified representations simultaneously encoding semantic understanding, physical interaction, environmental dynamics, and robot capabilities.

As Physical AI progresses toward increasingly autonomous embodied intelligence, SayCan remains one of the foundational architectural principles demonstrating how language reasoning can be successfully grounded within the realities of physical robot execution. By combining semantic planning, commonsense reasoning, perception, affordance estimation, reusable skill libraries, continuous replanning, and environmental feedback, SayCan establishes a practical framework for transforming natural language instructions into safe, feasible, adaptive, and executable robotic behaviors suitable for complex real-world environments.

세이캔(SayCan)은 대규모 언어 모델(LLM, Large Language Model)의 언어 이해 능력과 실제 로봇의 물리적 실행 능력을 연결한 대표적인 로봇 인공지능 아키텍처이다. 기존 LLM은 사람의 명령을 이해하고 논리적인 계획을 세우는 데 매우 뛰어났지만, 실제 로봇이 해당 작업을 수행할 수 있는지는 고려하지 못했다. 반대로 기존 로봇 제어 시스템은 자신이 수행 가능한 동작은 정확히 알고 있었지만 사람의 언어를 이해하거나 복잡한 추론을 수행하는 능력이 부족했다. SayCan은 이 두 가지 장점을 결합하여 의미적으로 올바르면서도 실제 실행 가능한 작업 계획을 생성한다.

SayCan이라는 이름은 "Say"와 "Can"이라는 두 요소에서 유래하였다. "Say"는 LLM이 사용자의 명령을 이해하고 어떤 작업을 수행해야 하는지를 추론하는 역할을 의미한다. 반면 "Can"은 현재 로봇이 실제 환경에서 해당 작업을 수행할 수 있는지를 판단하는 능력을 의미한다. 즉, 언어적으로 적절한 행동과 물리적으로 실행 가능한 행동을 동시에 고려하여 최종 작업을 결정하는 것이 SayCan의 핵심 개념이다.

기존의 언어 기반 로봇 시스템에서는 LLM이 매우 논리적인 작업 계획을 생성하더라도 실제 환경에서는 실행이 불가능한 경우가 자주 발생하였다. 예를 들어 물체가 너무 멀리 있거나, 로봇의 팔이 닿지 않거나, 해당 물체가 존재하지 않거나, 장애물 때문에 접근할 수 없는 경우에도 LLM은 이를 고려하지 못했다. SayCan은 이러한 문제를 해결하기 위해 행동마다 실행 가능성(Affordance Probability)을 함께 평가한다.

비전-언어-행동(VLA, Vision-Language-Action) 시스템에서 SayCan은 고수준 추론과 실제 제어 사이에 위치한다. 카메라(Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 힘 센서(Force Sensor), 촉각 센서(Tactile Sensor), 위치 추정(Localization), 의미 지도(Semantic Map) 등의 정보를 이용하여 현재 환경을 이해하고, 비전 인코더(Vision Encoder)는 이를 의미 정보로 변환한다. 동시에 LLM은 사용자의 명령을 분석하여 수행해야 할 작업을 추론하고, 로봇은 현재 자신이 수행 가능한 기술(Skill)을 평가한다.

SayCan의 핵심은 행동 점수(Action Scoring) 계산이다. LLM은 특정 기술(Skill)이 목표를 달성하는 데 얼마나 적합한지를 언어 확률(Language Probability)로 평가한다. 동시에 로봇은 현재 환경에서 해당 기술을 성공적으로 수행할 확률을 실행 가능성(Affordance Probability)으로 계산한다. SayCan은 이 두 값을 결합하여 최종 점수를 계산하며, 가장 높은 점수를 가진 행동을 다음 실행 단계로 선택한다.

어포던스(Affordance)는 로봇과 환경 사이에서 가능한 상호작용을 의미한다. 로봇은 물체를 집고, 문을 열고, 버튼을 누르고, 이동하고, 충전기에 도킹하고, 설비를 검사하는 등의 다양한 기술을 보유하고 있다. 그러나 이러한 기술은 항상 성공하는 것이 아니라 현재 환경에 따라 성공 가능성이 달라진다. 예를 들어 물체가 작업 범위를 벗어나 있거나 통로가 막혀 있으면 해당 행동은 실행할 수 없다. SayCan은 이러한 물리적 제약을 계획 과정에서 함께 고려한다.

SayCan의 가장 큰 특징은 의미 기반 계획과 물리적 실행을 분리하면서도 긴밀하게 연결한다는 점이다. LLM은 사용자의 의도를 이해하고 작업을 여러 단계로 분해하며 어떤 기술이 필요한지를 추론한다. 반면 실행 시스템은 각 기술이 실제로 성공할 가능성을 계산한다. 두 결과를 결합함으로써 현실적으로 수행 가능한 최적의 행동을 선택할 수 있다.

SayCan은 전체 계획을 한 번에 생성하지 않고 반복적인 의사결정(Iterative Decision Making)을 수행한다. 매 단계마다 현재 사용할 수 있는 기술 라이브러리(Skill Library)의 모든 기술을 평가하고, LLM은 의미적으로 가장 적합한 기술을 선택하며, 어포던스 모델은 성공 가능성을 계산한다. 가장 높은 점수를 얻은 기술을 실행한 후 새로운 센서 정보를 반영하여 다시 다음 행동을 선택하는 과정을 반복한다.

기술 라이브러리(Skill Library)는 SayCan의 핵심 구성 요소이다. 로봇은 이동(Navigation), 물체 탐지(Object Detection), 물체 집기(Grasping), 문 열기(Open Door), 물체 놓기(Place Object), 버튼 누르기(Button Pressing), 검사(Inspection), 충전(Charging), 사람과 대화하기(Human Interaction) 등의 기본 기술을 이미 보유하고 있다. LLM은 이러한 기술을 새롭게 만드는 것이 아니라 적절한 순서로 선택하여 작업을 수행한다.

LLM은 단순히 단어를 연결하는 것이 아니라 사용자의 의도(Intent)를 이해하고 상식(Common Sense)을 이용하여 작업을 추론한다. 예를 들어 "차를 준비해."라는 명령을 받으면 물을 준비하고, 물을 끓이고, 찻잔을 가져오고, 차를 우려내는 여러 단계를 자동으로 생성한다. 이러한 절차는 명령에 명시되어 있지 않더라도 LLM이 상식을 이용하여 추론한다.

그러나 LLM이 아무리 좋은 계획을 생성하더라도 실제 환경에 찻주전자(Kettle)가 존재하지 않는다면 작업은 수행할 수 없다. SayCan의 어포던스 모델은 현재 환경을 분석하여 해당 물체가 존재하는지, 접근 가능한지, 로봇이 실제로 수행할 수 있는지를 판단한다. 즉, 의미 기반 추론을 현실 세계에 연결하는 역할을 수행한다.

어포던스 추론은 인식 시스템(Perception System)에 크게 의존한다. 객체 탐지(Object Detection)는 물체를 찾고, 의미 분할(Semantic Segmentation)은 환경 구조를 이해하며, 깊이 추정(Depth Estimation)은 접근 가능성을 계산한다. 포인트 클라우드(Point Cloud)는 작업 공간을 분석하고, 힘 센서(Force Sensor)는 물체를 성공적으로 집었는지를 확인하며, 위치 추정(Localization)은 현재 로봇의 위치를 제공한다.

SayCan은 장기 작업(Long-Horizon Task)에도 적합하다. 모든 행동을 미리 계획하지 않고 현재 상황에서 가장 적합한 기술만 선택하기 때문에 환경이 계속 변화하는 실제 공간에서도 높은 안정성을 유지할 수 있다. 작업이 진행될수록 새로운 센서 정보를 반영하여 계획을 지속적으로 수정한다.

사람의 명령은 종종 모호하다. 예를 들어 "마실 것을 가져와."라는 명령은 여러 가지 해석이 가능하다. LLM은 물, 커피, 음료수 등 다양한 후보를 추론하지만, SayCan은 현재 주변에 실제로 존재하고 접근 가능한 물체를 고려하여 가장 적합한 선택을 수행한다. 따라서 의미와 현실을 동시에 반영한 의사결정이 가능하다.

SayCan은 작업 당시의 문맥(Context)도 함께 고려한다. 사용자의 위치, 현재 로봇 상태, 배터리 잔량(Battery Capacity), 이전 작업(Task History), 현재 손에 들고 있는 물체, 환경 상태 등을 함께 분석한다. 예를 들어 그리퍼(Gripper)가 이미 다른 물체를 잡고 있다면 새로운 물체를 집기 전에 먼저 기존 물체를 내려놓는 행동을 선택한다.

메모리(Memory)는 SayCan의 성능을 더욱 향상시킨다. 에피소드 메모리(Episodic Memory)는 이전 성공 및 실패 사례를 저장하고, 의미 메모리(Semantic Memory)는 일반 지식을 저장하며, 절차 메모리(Procedural Memory)는 기술 실행 방법을 기억한다. 장기 환경 메모리(Long-Term Memory)는 자주 방문하는 장소나 물체 위치를 기억하여 향후 작업 계획에 활용된다.

SayCan은 환경 변화에 따라 동적 재계획(Dynamic Replanning)도 수행한다. 사람이 이동하거나, 물체 위치가 바뀌거나, 통로가 막히거나, 장비가 고장 나면 현재 작업을 중단하지 않고 새로운 환경을 반영하여 남은 작업 계획만 다시 생성한다. 이를 통해 예기치 않은 상황에도 안정적으로 대응할 수 있다.

비전-언어 모델(VLM, Vision-Language Model)과도 자연스럽게 결합된다. 예를 들어 "냉장고 문이 열려 있고 중간 선반에 우유가 있으며 앞쪽에 의자가 있다."와 같은 장면 설명(Scene Description)을 생성하면 LLM은 이를 이용하여 더욱 정확한 의미 추론과 어포던스 평가를 수행할 수 있다.

자율주행(Navigation)에서도 SayCan은 매우 유용하다. 동일한 목적지라도 사람이 많거나, 공사 중이거나, 위험 구역이 포함된 경로는 실행 가능성이 낮다. SayCan은 의미적으로 적합한 목적과 실제 이동 가능한 경로를 동시에 고려하여 최적의 이동 전략을 선택한다.

조작(Manipulation) 작업에서도 동일한 원리가 적용된다. 목표 물체가 존재하더라도 다른 물체가 가로막고 있다면 직접 집는 것이 아니라 먼저 방해물을 치우거나, 문을 열거나, 로봇의 위치를 변경한 후 다시 물체를 집는 계획을 선택한다. 이러한 단계적 계획은 실제 환경에서 성공률을 크게 높여준다.

산업용 로봇에서는 SayCan의 효과가 더욱 크다. 생산 목표는 LLM이 이해하고, 실제 장비 사용 가능 여부, 작업 공간 접근성, 공구 상태, 안전 규칙은 어포던스 모델이 평가한다. 따라서 공장의 실제 상태를 반영한 현실적인 작업 계획을 생성할 수 있다.

물류 창고(Warehouse)에서는 재고 상태(Inventory), 선반 접근성(Shelf Accessibility), 통로 혼잡도(Traffic Congestion), 충전 상태(Charging Status), 작업 우선순위(Priority)를 함께 고려한다. SayCan은 이러한 다양한 요소를 종합하여 가장 효율적인 물류 작업을 계획한다.

의료 로봇(Healthcare Robot)은 환자의 요청을 자연어로 이해하는 동시에 환자 안전(Safety), 의료 장비 접근성, 병원 절차(Medical Protocol), 의료 규정을 함께 고려해야 한다. SayCan은 이러한 현실적인 제약을 계획에 반영함으로써 보다 안전한 의료 서비스를 지원할 수 있다.

시설 점검(Inspection) 로봇도 설비 접근성, 조명 상태, 안전 조건, 작업 허가 등을 고려해야 한다. SayCan은 현재 환경에서 실제 수행 가능한 검사 기술만 선택하여 효율적인 점검 계획을 생성한다.

SayCan의 가장 큰 장점은 모듈성(Modularity)이다. LLM, 인식 시스템, 어포던스 모델, 기술 라이브러리, 모션 플래너(Motion Planner), 제어기(Controller)는 서로 독립적으로 개발될 수 있으며, 하나의 모듈이 개선되더라도 전체 시스템을 다시 설계할 필요가 없다.

그러나 한계도 존재한다. 어포던스 추론은 인식 정확도에 크게 의존하므로 객체 탐지 오류, 위치 오차, 센서 잡음이 발생하면 실행 가능성 평가도 부정확해질 수 있다. 따라서 높은 수준의 인식 기술이 반드시 필요하다.

LLM 역시 환각(Hallucination) 문제를 완전히 해결하지 못한다. 사용자 의도를 잘못 이해하거나 부적절한 작업 계획을 생성할 수 있다. SayCan은 실행 가능성을 평가하여 일부 오류를 제거할 수 있지만, 안전이 중요한 분야에서는 추가적인 검증(Validation)이 반드시 필요하다.

기술 라이브러리가 수백 개 이상으로 증가하면 모든 기술의 실행 가능성을 계산하는 비용도 증가한다. 이를 해결하기 위해 계층형 기술 구조(Hierarchical Skill Organization), 임베딩 검색(Embedding Retrieval), 근사 탐색(Approximate Search) 등의 연구가 활발히 진행되고 있다.

최근에는 다중 로봇(Multi-Robot) 환경으로 SayCan을 확장하는 연구도 진행되고 있다. 여러 로봇이 동시에 작업할 경우 각 로봇의 능력, 자원 공유(Resource Sharing), 작업 분배(Task Allocation), 통신(Communication), 협업(Collaboration)을 함께 고려하여 최적의 작업 계획을 생성한다.

세계 모델(World Model)과 SayCan을 결합하는 연구도 활발하다. 현재 환경뿐 아니라 미래의 환경 변화까지 예측하여 실행 가능성을 평가하므로 보다 선제적인 계획(Predictive Planning)이 가능해지고 있다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)과도 자연스럽게 통합된다. 운영 매뉴얼(Manual), 설비 데이터베이스(Database), 디지털 트윈(Digital Twin), 시설 지도(Map), 유지보수 기록(Log) 등을 검색하여 LLM의 추론과 어포던스 평가를 더욱 정확하게 만든다.

사고의 연쇄(CoT, Chain of Thought)와 SayCan을 함께 사용하는 연구도 증가하고 있다. LLM은 먼저 작업을 단계적으로 추론하고, 이후 각 단계마다 실제 실행 가능성을 평가한다. 이러한 방식은 복잡한 장기 작업에서 더욱 높은 성공률을 제공한다.

향후 VLA 시스템은 언어(Language), 영상(Image), 깊이(Depth), 비디오(Video), 촉각(Tactile), 힘(Force), 자기 상태(Proprioception), 의미 지도(Semantic Map), 세계 모델(World Model), 장기 메모리(Long-Term Memory)를 동시에 학습하는 멀티모달 파운데이션 모델(Multimodal Foundation Model)로 발전할 것으로 예상된다. 미래의 모델은 별도의 언어 모델과 어포던스 모델을 결합하는 방식이 아니라 의미 이해와 물리적 실행 가능성을 하나의 통합 표현(Unified Representation)으로 학습하게 될 것이다.

결국 SayCan은 LLM의 의미 기반 추론과 실제 로봇의 물리적 실행 능력을 연결한 최초의 실용적인 프레임워크 중 하나이다. 상식(Common Sense), 작업 계획(Task Planning), 인식(Perception), 어포던스 추론(Affordance Reasoning), 기술 라이브러리(Skill Library), 지속적인 재계획(Continuous Replanning), 환경 피드백(Environment Feedback)을 통합함으로써 자연어 명령을 실제 실행 가능한 로봇 행동으로 변환하는 차세대 물리 AI(Physical AI)의 핵심 기반 기술로 평가받고 있다.

##  

## 3.6 Fine-Tuning LLMs for Robot Instruction Following (with Code)

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding, reasoning, planning, and code generation. However, foundation models trained on general internet-scale text are not inherently optimized for robotics. Although these models possess broad world knowledge, they often lack the specialized understanding required for robot instruction following, embodied interaction, physical constraints, robot APIs, safety policies, environmental dynamics, and real-time decision making. Fine-tuning has therefore become one of the most important techniques for adapting general-purpose language models into reliable robotic cognitive systems. Within modern Vision-Language-Action (VLA) architectures, fine-tuning enables LLMs to understand robotic tasks more accurately, generate physically feasible plans, follow operator instructions consistently, and interact naturally with both robots and human users.

Robot instruction following differs fundamentally from conventional conversational AI. Human conversations frequently tolerate ambiguity, multiple valid interpretations, and partially correct responses. In contrast, robotic systems require precise, deterministic, safe, and executable outputs because generated instructions ultimately influence physical actions. A misunderstanding that is acceptable in text generation may become dangerous when controlling autonomous machines. Consequently, robotic language models require significantly stronger alignment between language understanding and physical execution.

The primary objective of fine-tuning is to modify the pretrained language model so that it learns the linguistic patterns, operational procedures, safety constraints, task structures, and environmental knowledge specific to robotics. Rather than replacing the broad knowledge already contained within the foundation model, fine-tuning specializes that knowledge for embodied intelligence while preserving general reasoning capability.

Instruction following begins with understanding human intent rather than merely interpreting individual words. Human operators naturally issue commands such as "Bring the inspection camera to the maintenance room," "Check whether any pallets are damaged," or "Return to the charging station after completing today\'s inspection." These instructions frequently contain implicit assumptions regarding environmental context, robot capabilities, temporal ordering, safety procedures, and operational priorities. Fine-tuned language models learn to infer these unstated requirements from examples collected during robotic operation.

The fine-tuning process generally begins with collecting high-quality robotic instruction datasets. These datasets consist of natural language commands paired with desired robot responses, task decompositions, action sequences, symbolic plans, Python policies, behavior trees, ROS command structures, or executable robot skills. Unlike general instruction tuning datasets, robotic datasets emphasize physical interactions, environmental reasoning, manipulation sequences, navigation decisions, perception requests, and safety-critical behaviors.

Dataset diversity plays a crucial role in successful adaptation. Robots operate across industrial manufacturing, logistics, healthcare, agriculture, construction, warehouse automation, infrastructure inspection, domestic assistance, service robotics, and collaborative workspaces. Each domain introduces unique terminology, workflows, equipment, operational constraints, and interaction styles. Fine-tuning datasets therefore benefit from broad coverage across multiple robotic applications while maintaining consistent annotation quality.

Task diversity is equally important. A practical robotic assistant must understand navigation, manipulation, object retrieval, inspection, assembly, charging, monitoring, cleaning, delivery, reporting, emergency response, fault recovery, maintenance procedures, and collaborative interactions. Fine-tuning exposes the language model to these varied task categories so that it learns generalized robotic reasoning rather than memorizing narrow procedural templates.

Multimodal instruction datasets increasingly replace purely textual datasets. Modern robotic systems integrate RGB images, depth maps, LiDAR observations, semantic maps, point clouds, tactile measurements, force sensing, joint positions, battery status, localization estimates, environmental metadata, and task history together with natural language instructions. Fine-tuning on multimodal representations enables the LLM to associate language with physical perception and robot state.

Supervised Fine-Tuning (SFT) remains the most widely used adaptation technique. During SFT, the model learns from manually curated instruction-response pairs demonstrating correct robotic behavior. Human experts create high-quality examples illustrating appropriate task decomposition, safety-conscious planning, correct API usage, error handling, and execution logic. The model gradually adjusts its parameters to reproduce these desired responses for similar instructions.

Instruction tuning extends supervised learning by emphasizing generalizable task understanding rather than memorization. Instead of learning individual robot commands independently, the model learns abstract relationships among goals, subtasks, environmental constraints, robot skills, and execution sequences. This abstraction significantly improves generalization toward previously unseen tasks.

Preference optimization has become increasingly important for robotic instruction following. Human evaluators compare alternative model responses according to correctness, safety, efficiency, interpretability, and execution quality. Methods such as Reinforcement Learning from Human Feedback, Direct Preference Optimization, and related preference-learning algorithms enable language models to align their behavior with expert robotic expectations while reducing undesirable outputs.

Safety alignment represents a particularly critical aspect of robotic fine-tuning. Unlike conversational assistants, robots interact directly with physical environments where unsafe decisions may damage equipment, injure people, or violate operational procedures. Training datasets therefore include numerous examples demonstrating safe navigation, collision avoidance, emergency stopping, hazardous material handling, human proximity management, equipment protection, and regulatory compliance.

Negative examples further improve robustness. Instead of learning exclusively from correct demonstrations, robotic language models benefit from observing incorrect plans, unsafe actions, infeasible manipulations, inappropriate API usage, impossible trajectories, and policy violations together with explanations describing why these outputs should be rejected. This contrastive learning significantly improves decision quality.

Chain of Thought supervision increasingly enhances robotic reasoning. Rather than providing only final answers, fine-tuning datasets include intermediate reasoning demonstrating how complex instructions are decomposed into executable subtasks. These structured reasoning examples teach the model to analyze goals, identify constraints, evaluate alternatives, estimate affordances, and generate logically coherent execution plans before selecting actions.

Fine-tuning also improves spatial reasoning. General-purpose language models possess broad spatial knowledge but frequently lack precision regarding robotic workspaces, object accessibility, manipulation feasibility, navigation constraints, and three-dimensional environmental organization. Exposure to robotics-specific spatial examples substantially improves reasoning about relative positions, reachability, obstacle avoidance, grasp planning, and workspace geometry.

Robot skill grounding constitutes another important adaptation objective. Rather than producing unrestricted natural language, fine-tuned models learn to map user instructions onto predefined robot skill libraries. Navigation, grasping, docking, charging, inspection, scanning, manipulation, perception, communication, and reporting become grounded semantic actions directly connected to executable robot capabilities.

API grounding similarly improves practical deployment. Many robotic systems expose Python APIs, ROS 2 services, action servers, middleware interfaces, hardware drivers, perception services, and planning libraries. Fine-tuning teaches the LLM to invoke these interfaces correctly while respecting parameter constraints, execution order, dependency management, and software architecture principles.

Memory utilization becomes more effective after robotics-specific adaptation. Fine-tuned models learn when to retrieve episodic memory describing previous task executions, semantic memory containing environmental knowledge, procedural memory representing robot skills, and long-term maps describing facility layouts. Effective memory retrieval significantly improves long-duration autonomous operation.

Retrieval-Augmented Generation integrates naturally with fine-tuned models. Rather than relying exclusively upon internal parameters, robotic LLMs retrieve facility documentation, maintenance manuals, operating procedures, digital twins, equipment specifications, inspection histories, software documentation, and enterprise knowledge bases. Fine-tuning teaches the model when retrieval is necessary and how retrieved information should influence subsequent reasoning.

Simulation-generated datasets increasingly supplement human demonstrations. High-fidelity simulators such as Isaac Sim, Gazebo, MuJoCo, Habitat, and digital twin environments enable large-scale generation of robot interactions under diverse environmental conditions. Millions of simulated instruction-following episodes provide valuable supervision while avoiding the cost and safety risks associated with physical robot experimentation.

Sim-to-real transfer becomes substantially more reliable when language models are fine-tuned using both simulated and real-world demonstrations. Simulation provides diversity and scale, while real robotic experience captures sensing noise, actuator uncertainty, environmental variability, and hardware limitations absent from virtual environments. Combining both data sources improves generalization across deployment conditions.

Foundation models specialized for robotics increasingly employ continual fine-tuning throughout operational deployment. Instead of remaining static after initial training, language models continually incorporate new demonstrations, operator corrections, successful task executions, environmental observations, and evolving robot capabilities. Continual adaptation enables robotic intelligence to improve over time without complete retraining.

Parameter-efficient fine-tuning techniques have become especially valuable because modern foundation models often contain tens or hundreds of billions of parameters. Methods such as Low-Rank Adaptation, adapters, prompt tuning, prefix tuning, and related lightweight adaptation strategies enable organizations to customize large language models for specific robotic platforms without retraining entire networks. These approaches significantly reduce computational requirements while preserving most pretrained knowledge.

Small Language Models similarly benefit from targeted fine-tuning. Rather than deploying extremely large foundation models onboard resource-constrained robots, lightweight models adapted specifically for navigation, manipulation, inspection, or warehouse operation achieve competitive performance while satisfying real-time computational requirements. Edge deployment therefore becomes practical for many robotic applications.

Industrial robotics particularly benefits from specialized instruction tuning. Manufacturing facilities possess unique terminology, production workflows, machine interfaces, quality inspection procedures, maintenance schedules, safety regulations, and operational standards. Fine-tuning allows language models to understand domain-specific vocabulary and generate instructions consistent with established industrial practices.

Warehouse automation introduces specialized linguistic patterns involving inventory management, pallet handling, order fulfillment, storage allocation, barcode scanning, shelf organization, autonomous transportation, and logistics optimization. Fine-tuned language models become significantly more accurate when interpreting these operational instructions compared with general-purpose foundation models.

Healthcare robotics requires another level of specialization. Clinical environments involve medical terminology, patient interaction protocols, accessibility considerations, hygiene requirements, medication handling procedures, emergency workflows, and privacy regulations. Robotics-specific fine-tuning enables language models to follow these domain-specific instructions more reliably while maintaining safe behavior.

Inspection robotics similarly requires specialized adaptation. Infrastructure inspection, utility maintenance, pipeline monitoring, bridge assessment, power station inspection, agricultural surveying, and environmental monitoring each involve specialized terminology, sensor usage, reporting formats, anomaly descriptions, and operational procedures. Fine-tuning aligns language understanding with these technical workflows.

Evaluation represents a critical component of robotic fine-tuning. Unlike conventional language benchmarks measuring factual accuracy or conversational quality, robotic evaluation emphasizes task success, instruction following accuracy, planning consistency, API correctness, safety compliance, affordance awareness, execution feasibility, robustness under uncertainty, and recovery from execution failures. Comprehensive evaluation therefore combines language metrics with embodied performance metrics collected during simulation and real-world deployment.

Safety verification remains essential before deployment. Even highly fine-tuned language models may occasionally generate unsafe, infeasible, or logically inconsistent outputs. Consequently, robotic architectures continue employing execution validators, geometric reasoning modules, collision prediction systems, affordance estimators, rule-based safety monitors, simulation verification, and runtime supervision. Fine-tuning complements rather than replaces these deterministic safety mechanisms.

One important challenge involves catastrophic forgetting. Excessive specialization may cause a language model to lose valuable general reasoning capabilities acquired during large-scale pretraining. Successful fine-tuning therefore balances domain adaptation with preservation of broad world knowledge. Parameter-efficient adaptation techniques often reduce catastrophic forgetting compared with full-model retraining.

Data quality remains more important than data quantity. Thousands of carefully curated demonstrations frequently outperform millions of noisy examples containing inconsistent annotations or unsafe behaviors. Expert robotic supervision, consistent labeling, standardized task representations, and accurate execution records therefore play central roles in successful model adaptation.

Future research increasingly explores multimodal fine-tuning integrating language, vision, depth, tactile sensing, force measurements, proprioception, semantic maps, world models, manipulation demonstrations, and long-term memory into unified embodied foundation models. Rather than adapting language alone, future robotic systems will jointly optimize perception, reasoning, planning, and action generation within integrated multimodal learning frameworks.

Another promising direction involves collaborative fine-tuning across robot fleets. Multiple autonomous robots operating in different environments continuously collect instruction-following experiences. Aggregating these experiences enables foundation models to learn broader operational knowledge while preserving adaptation to individual robot platforms through parameter-efficient personalization.

Federated learning also offers significant potential for industrial robotics. Instead of transmitting sensitive operational data to centralized servers, robots locally fine-tune shared foundation models while exchanging only learned parameter updates. This approach improves privacy, reduces communication bandwidth, and enables collaborative learning across geographically distributed robotic deployments.

World models will increasingly influence instruction fine-tuning by allowing language models to reason over predicted future environmental states rather than current observations alone. Fine-tuned LLMs will learn not only how to respond to instructions but also how current decisions influence future task success, energy consumption, safety, productivity, and collaborative efficiency.

As Physical AI evolves toward increasingly autonomous embodied intelligence, fine-tuning will remain one of the foundational technologies enabling reliable robot instruction following. By adapting general-purpose language models to understand robotic tasks, physical environments, perception outputs, safety constraints, reusable skills, software interfaces, multimodal observations, and execution feedback, fine-tuned LLMs establish a practical bridge between natural language communication and dependable robotic behavior. Rather than replacing classical robotics algorithms, fine-tuned language models augment perception, planning, manipulation, navigation, and control systems with high-level semantic intelligence, enabling robots to understand human intentions accurately and transform them into safe, efficient, and executable actions across diverse real-world environments.

대규모 언어 모델(LLM, Large Language Model)은 자연어 이해(Natural Language Understanding), 추론(Reasoning), 계획(Planning), 코드 생성(Code Generation) 등에서 뛰어난 성능을 보이지만, 인터넷 기반 일반 데이터로 학습된 파운데이션 모델(Foundation Model)은 로봇공학(Robotics)에 최적화되어 있지 않다. 로봇은 물리 환경(Physical Environment), 안전(Safety), 센서 정보(Sensor Data), 실시간 제어(Real-Time Control), 작업 절차(Task Procedure)를 이해해야 하므로, 이를 위해 미세조정(Fine-Tuning)이 필수적으로 수행된다.

로봇 명령 수행(Robot Instruction Following)은 일반 대화형 인공지능(Chatbot)과는 근본적으로 다르다. 대화에서는 다소 모호하거나 여러 해석이 가능한 답변도 허용되지만, 로봇은 생성된 명령이 실제 물리적 행동으로 이어지므로 정확하고 안전하며 실행 가능한 결과를 생성해야 한다. 따라서 로봇용 LLM은 자연어 이해뿐 아니라 실제 행동(Action)과 연결되는 높은 수준의 정렬(Alignment)이 요구된다.

미세조정의 목적은 일반 LLM이 보유한 광범위한 지식을 유지하면서 로봇 환경에 필요한 언어 패턴(Language Pattern), 작업 절차(Workflow), 안전 규칙(Safety Policy), 환경 지식(Environment Knowledge), 제어 방식(Control Strategy)을 학습시키는 것이다. 이는 기존 지식을 대체하는 것이 아니라 로봇 분야에 특화된 능력을 추가하는 과정이라고 볼 수 있다.

로봇 명령 수행은 단순히 문장을 해석하는 것이 아니라 사람의 의도(Intent)를 이해하는 것에서 시작된다. 예를 들어 "검사 카메라를 정비실로 가져가.", "손상된 팔레트를 검사해.", "오늘 점검을 마치면 충전기로 돌아가."와 같은 명령에는 환경(Context), 시간 순서(Temporal Order), 안전 규칙, 작업 우선순위 등이 명시되어 있지 않은 경우가 많다. 미세조정된 LLM은 이러한 숨겨진 조건을 자동으로 추론할 수 있다.

미세조정은 우선 고품질 로봇 명령 데이터셋(Robot Instruction Dataset)을 구축하는 것부터 시작된다. 데이터셋에는 자연어 명령, 작업 분해(Task Decomposition), 행동 순서(Action Sequence), 심볼릭 계획(Symbolic Plan), 파이썬 정책(Python Policy), 행동 트리(Behavior Tree), ROS 명령 구조(ROS Command Structure) 등이 포함된다. 일반적인 언어 데이터와 달리 실제 로봇 행동과 직접 연결되는 정보가 중심이 된다.

데이터셋은 다양한 산업 분야를 포함해야 한다. 제조(Manufacturing), 물류(Logistics), 의료(Healthcare), 농업(Agriculture), 건설(Construction), 창고 자동화(Warehouse Automation), 시설 점검(Infrastructure Inspection), 서비스 로봇(Service Robotics), 협동 로봇(Collaborative Robotics) 등은 서로 다른 용어와 절차를 사용한다. 다양한 환경을 포함한 데이터는 모델의 일반화 성능을 크게 향상시킨다.

작업(Task)의 다양성도 매우 중요하다. 이동(Navigation), 조작(Manipulation), 물체 운반(Object Transport), 검사(Inspection), 조립(Assembly), 충전(Charging), 청소(Cleaning), 감시(Monitoring), 배송(Delivery), 유지보수(Maintenance), 복구(Recovery), 사람과의 협업(Human Collaboration) 등 다양한 작업을 학습해야 실제 환경에서 폭넓게 활용할 수 있다.

최근에는 텍스트(Text)뿐 아니라 멀티모달(Multimodal) 데이터가 함께 사용된다. RGB 영상(RGB Image), 깊이 영상(Depth Map), 라이다(LiDAR), 포인트 클라우드(Point Cloud), 촉각(Tactile), 힘 센서(Force Sensor), 배터리 상태(Battery Status), 위치 정보(Localization), 환경 메타데이터(Environment Metadata), 작업 이력(Task History) 등을 자연어와 함께 학습함으로써 언어와 실제 환경을 연결한다.

가장 널리 사용되는 방법은 지도 미세조정(SFT, Supervised Fine-Tuning)이다. 사람이 직접 작성한 고품질 명령과 정답(Response)을 이용하여 모델을 학습시킨다. 이러한 데이터에는 올바른 작업 분해, 안전한 행동 계획, 정확한 API 사용, 예외 처리(Exception Handling), 실행 절차 등이 포함되어 있으며, 모델은 이를 반복 학습하면서 로봇 환경에 적응한다.

명령 미세조정(Instruction Tuning)은 단순히 특정 명령을 외우는 것이 아니라 작업 목표(Goal), 하위 작업(Subtask), 환경 제약(Constraint), 로봇 기술(Skill), 실행 순서(Execution Sequence) 사이의 관계를 이해하도록 학습시킨다. 따라서 처음 보는 명령이라도 기존 지식을 활용하여 적절한 작업 계획을 생성할 수 있다.

최근에는 선호도 학습(Preference Optimization)이 중요해지고 있다. 여러 개의 후보 응답 중 전문가가 가장 적절한 답변을 선택하면 모델은 이를 학습한다. 강화학습 기반 인간 피드백(RLHF, Reinforcement Learning from Human Feedback), 직접 선호도 최적화(DPO, Direct Preference Optimization) 등의 기법을 활용하여 더욱 안전하고 효율적인 작업 계획을 생성하도록 학습시킨다.

안전 정렬(Safety Alignment)은 로봇 미세조정에서 가장 중요한 요소이다. 로봇은 실제 환경에서 사람과 함께 작업하므로 충돌 회피(Collision Avoidance), 비상 정지(Emergency Stop), 위험 물질(Hazardous Material) 처리, 사람과의 거리 유지(Human Proximity), 장비 보호(Equipment Protection), 산업 안전 규정(Regulation) 등을 반드시 준수해야 한다. 따라서 학습 데이터에도 이러한 사례가 충분히 포함되어야 한다.

올바른 사례뿐 아니라 잘못된 사례(Negative Example)도 함께 학습한다. 잘못된 작업 계획, 위험한 행동, 실행 불가능한 조작, 잘못된 API 사용, 물리적으로 불가능한 경로 등을 함께 제시하고 왜 잘못되었는지를 설명함으로써 모델의 판단 능력을 향상시킨다.

사고의 연쇄(CoT, Chain of Thought)를 이용한 학습도 점차 중요해지고 있다. 단순히 정답만 제공하는 것이 아니라 작업을 여러 단계로 분해하는 추론 과정을 함께 학습한다. 이를 통해 목표 분석, 제약 조건 파악, 대안 비교, 어포던스(Affordance) 평가, 작업 순서 결정 등의 과정을 스스로 수행할 수 있게 된다.

공간 추론(Spatial Reasoning) 능력도 미세조정을 통해 크게 향상된다. 일반 LLM은 공간 개념은 이해하지만 작업 가능 영역(Workspace), 접근 가능성(Reachability), 장애물 회피(Obstacle Avoidance), 그리퍼 접근 방향(Grasp Planning)과 같은 로봇 특화 공간 개념은 부족하다. 로봇 데이터셋으로 학습하면 이러한 능력이 크게 개선된다.

미세조정은 로봇 기술(Skill)과 자연어를 연결하는 역할도 수행한다. 사용자의 명령을 이동(Navigation), 물체 집기(Grasping), 충전(Charging), 검사(Inspection), 스캔(Scanning), 보고(Reporting) 등의 기술 라이브러리(Skill Library)와 직접 연결하여 실제 실행 가능한 작업으로 변환한다.

API(Application Programming Interface)와의 연동도 중요한 학습 대상이다. ROS 2 서비스(Service), 액션(Action), 파이썬(Python) API, 하드웨어 드라이버(Hardware Driver), 인식 서비스(Perception Service), 계획 라이브러리(Planning Library)를 올바르게 호출하는 방법을 학습하여 실제 로봇 소프트웨어와 자연스럽게 연결된다.

메모리(Memory) 활용 능력도 향상된다. 에피소드 메모리(Episodic Memory), 의미 메모리(Semantic Memory), 절차 메모리(Procedural Memory), 장기 지도(Long-Term Map)를 언제 활용해야 하는지를 학습하여 장시간 자율 작업(Long-Term Autonomy)의 성능을 높인다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)도 미세조정과 함께 사용된다. 설비 매뉴얼(Manual), 유지보수 문서(Maintenance Guide), 디지털 트윈(Digital Twin), 운영 절차(Standard Operating Procedure), 장비 데이터베이스(Database)를 언제 검색해야 하는지를 학습하여 내부 지식과 외부 지식을 효율적으로 결합한다.

최근에는 시뮬레이션(Simulation)을 이용한 데이터 생성이 활발하다. 아이작 심(Isaac Sim), 가제보(Gazebo), 무조코(MuJoCo), 해비타트(Habitat), 디지털 트윈(Digital Twin) 환경에서 수백만 개의 명령 수행 데이터를 생성할 수 있다. 실제 로봇보다 비용이 저렴하고 안전하며 다양한 환경을 손쉽게 생성할 수 있다는 장점이 있다.

Sim-to-Real 학습은 시뮬레이션과 실제 로봇 데이터를 함께 활용한다. 시뮬레이션은 다양한 상황을 제공하고, 실제 데이터는 센서 잡음(Sensor Noise), 구동 오차(Actuator Error), 환경 변화(Environment Variation)를 포함한다. 두 데이터를 함께 학습하면 실제 환경에서의 일반화 성능이 크게 향상된다.

최근에는 지속적 미세조정(Continual Fine-Tuning)도 연구되고 있다. 초기 학습 후에도 새로운 작업 성공 사례, 운영자 수정, 새로운 장비, 환경 변화 등을 지속적으로 학습하여 시간이 지날수록 성능이 향상되는 적응형 로봇 시스템을 구축할 수 있다.

파라미터 효율적 미세조정(PEFT, Parameter-Efficient Fine-Tuning)은 매우 중요한 기술이다. 수십억 개 이상의 파라미터를 가진 모델 전체를 다시 학습하는 대신 로라(LoRA, Low-Rank Adaptation), 어댑터(Adapter), 프롬프트 튜닝(Prompt Tuning), 프리픽스 튜닝(Prefix Tuning) 등을 이용하여 일부 파라미터만 학습한다. 이를 통해 학습 비용과 메모리 사용량을 크게 줄일 수 있다.

소형 언어 모델(SLM, Small Language Model)도 로봇 분야에서 많이 활용된다. 엣지 컴퓨터(Edge Computer)에서는 대규모 모델을 실행하기 어렵기 때문에 이동, 검사, 조작 등 특정 작업에 맞추어 경량 모델을 미세조정하여 사용한다. 이를 통해 실시간성(Real-Time)과 성능을 동시에 확보할 수 있다.

산업용 로봇은 공장 고유의 용어(Terminology), 생산 절차(Production Workflow), 품질 검사(Quality Inspection), 유지보수 일정(Maintenance Schedule), 안전 규정(Safety Regulation)을 이해해야 한다. 미세조정을 통해 이러한 전문 지식을 학습하면 실제 산업 현장에서 훨씬 높은 성능을 발휘할 수 있다.

창고 자동화(Warehouse Automation)는 재고 관리(Inventory Management), 주문 처리(Order Fulfillment), 바코드 스캔(Barcode Scanning), 선반 관리(Shelf Organization), 물류 최적화(Logistics Optimization) 등 특수한 용어와 작업 절차를 사용한다. 이에 특화된 미세조정은 창고 로봇의 명령 이해 능력을 크게 향상시킨다.

의료 로봇(Healthcare Robot)은 의료 용어(Medical Terminology), 환자 응대(Patient Interaction), 위생 규정(Hygiene Policy), 약품 관리(Medication Handling), 응급 절차(Emergency Workflow), 개인정보 보호(Privacy Regulation)를 이해해야 한다. 의료 환경에 특화된 미세조정을 통해 더욱 안전하고 신뢰성 있는 서비스를 제공할 수 있다.

시설 점검(Inspection Robot) 역시 설비 구조, 이상 탐지(Anomaly Detection), 점검 절차(Inspection Procedure), 보고 형식(Report Format), 센서 운용(Sensor Operation)에 대한 전문 지식을 학습해야 한다. 이러한 데이터는 점검 로봇의 성능을 크게 향상시킨다.

평가(Evaluation)는 일반 언어 모델과 다른 기준을 사용한다. 단순한 문장 품질이 아니라 작업 성공률(Task Success Rate), 명령 수행 정확도(Instruction Following Accuracy), 계획 일관성(Planning Consistency), API 정확도(API Correctness), 안전성(Safety Compliance), 실행 가능성(Execution Feasibility), 오류 복구(Recovery Capability) 등을 종합적으로 평가한다.

미세조정이 완료되더라도 안전 검증(Safety Verification)은 반드시 수행되어야 한다. 실행 검증기(Execution Validator), 기하학적 검증(Geometric Validation), 충돌 예측(Collision Prediction), 어포던스 분석(Affordance Estimation), 규칙 기반 안전 검사(Rule-Based Safety Check), 시뮬레이션 검증(Simulation Verification), 런타임 모니터(Runtime Monitoring)가 함께 동작하여 위험한 행동을 차단한다.

미세조정 과정에서는 재앙적 망각(Catastrophic Forgetting)도 중요한 문제이다. 지나치게 로봇 데이터만 학습하면 기존의 일반 상식과 추론 능력이 감소할 수 있다. 따라서 일반 지식과 로봇 전문 지식 사이의 균형을 유지하는 것이 매우 중요하다.

데이터의 양보다 품질(Quality)이 더욱 중요하다. 수백만 개의 부정확한 데이터보다 전문가가 작성한 수천 개의 고품질 데이터가 더 좋은 성능을 제공하는 경우가 많다. 정확한 라벨(Label), 일관된 표현, 표준화된 작업 구조가 성공적인 미세조정의 핵심 요소이다.

향후에는 언어(Language), 영상(Image), 깊이 정보(Depth), 촉각(Tactile), 힘 센서(Force), 자기 상태(Proprioception), 의미 지도(Semantic Map), 세계 모델(World Model), 조작 시연(Manipulation Demonstration), 장기 메모리(Long-Term Memory)를 함께 학습하는 멀티모달 미세조정(Multimodal Fine-Tuning)이 중심 기술로 발전할 것으로 예상된다.

또한 여러 대의 로봇이 서로의 경험을 공유하는 협업 미세조정(Collaborative Fine-Tuning)도 활발히 연구되고 있다. 서로 다른 환경에서 수행한 명령 수행 경험을 공유함으로써 보다 강력한 로봇 파운데이션 모델(Robot Foundation Model)을 구축할 수 있다.

연합 학습(Federated Learning)은 산업 현장에서 매우 유망한 기술이다. 각 공장에서 데이터를 외부로 보내지 않고 로컬(Local)에서 미세조정을 수행한 후 모델의 파라미터만 공유함으로써 개인정보와 기업 기밀을 보호하면서도 공동 학습이 가능하다.

앞으로는 세계 모델(World Model)과 미세조정이 결합되어 현재 환경뿐 아니라 미래 상황까지 예측하는 명령 수행이 가능해질 것이다. 현재 행동이 미래의 안전성(Safety), 에너지 소비(Energy Consumption), 생산성(Productivity), 협업 효율(Collaboration Efficiency)에 어떤 영향을 미치는지까지 고려하는 예측 기반 로봇 인공지능(Predictive Robot AI)으로 발전할 것으로 기대된다.

결국 로봇 명령 수행을 위한 LLM 미세조정은 일반 언어 모델을 실제 물리 세계에서 동작하는 로봇 인공지능으로 변화시키는 핵심 기술이다. 자연어 이해(Natural Language Understanding), 물리 환경(Physical Environment), 센서 정보(Perception), 안전 규칙(Safety Policy), 기술 라이브러리(Skill Library), 소프트웨어 인터페이스(Software Interface), 실행 피드백(Execution Feedback)을 하나의 통합 인지 시스템으로 연결함으로써 사람의 의도를 안전하고 신뢰성 있게 실제 로봇 행동으로 변환하는 차세대 물리 AI(Physical AI)의 핵심 기반 기술로 자리매김하고 있다.

##  

## 3.7 Small Language Models (SLMs) for Edge Robots (with Code)

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

Small Language Models (SLMs) are becoming one of the most important enabling technologies for the next generation of intelligent edge robotics. While Large Language Models (LLMs) provide remarkable capabilities in natural language understanding, reasoning, planning, and code generation, their computational requirements often exceed the processing, memory, power, and latency constraints of autonomous mobile robots. Edge robots must continuously perceive their surroundings, perform localization, execute motion planning, maintain safety monitoring, and communicate with sensors while operating under limited onboard computational resources. Consequently, deploying extremely large language models directly on mobile robotic platforms is often impractical. Small Language Models address this challenge by providing efficient language reasoning specifically optimized for embedded robotic systems.

Unlike cloud-hosted foundation models containing hundreds of billions of parameters, Small Language Models generally contain hundreds of millions to several billion parameters. This dramatic reduction in model size enables deployment on embedded GPUs, AI accelerators, edge computers, and mobile robotic computing platforms while maintaining useful reasoning capability for domain-specific tasks. Rather than attempting to replicate every capability of extremely large models, SLMs prioritize efficiency, determinism, responsiveness, and specialization.

The emergence of SLMs represents a shift from universal intelligence toward task-oriented embodied intelligence. Most robots do not require comprehensive knowledge covering every domain represented on the internet. Instead, warehouse robots primarily require logistics knowledge, inspection robots require maintenance procedures, hospital robots require healthcare workflows, agricultural robots require crop management expertise, and autonomous mobile robots require navigation and manipulation reasoning. Small Language Models exploit this observation by specializing their capabilities for well-defined operational environments.

Edge robotics introduces several computational constraints rarely encountered by cloud-based artificial intelligence systems. Autonomous robots operate with limited electrical power supplied by batteries, restricted thermal dissipation due to compact mechanical designs, finite onboard memory capacity, and real-time response requirements measured in milliseconds rather than seconds. These constraints strongly influence language model architecture and deployment strategy.

Power consumption remains one of the most critical design considerations for edge robotics. Every watt allocated to language model inference reduces the power available for perception sensors, motor controllers, communication systems, onboard computers, environmental monitoring, and mission endurance. Large cloud-scale models requiring multiple high-performance GPUs cannot satisfy the energy budgets of battery-powered autonomous robots. Small Language Models significantly reduce computational demand while extending operational lifetime.

Latency presents another fundamental challenge. Robots interact with continuously changing physical environments where delayed decisions may compromise safety or operational performance. Cloud inference introduces unpredictable communication delays resulting from wireless networks, internet connectivity, congestion, packet loss, or temporary service interruptions. Edge-deployed Small Language Models eliminate network dependence by executing inference directly on onboard hardware, enabling deterministic response times suitable for safety-critical robotic applications.

Privacy and security further motivate edge deployment. Industrial robots frequently operate within factories containing confidential manufacturing processes, proprietary equipment configurations, production schedules, customer information, or intellectual property. Healthcare robots similarly encounter sensitive patient information protected by regulatory requirements. Running Small Language Models locally ensures that operational data remains within the robot rather than being transmitted to external cloud services.

Reliability also improves through onboard inference. Robots operating in underground facilities, remote construction sites, agricultural fields, military environments, offshore platforms, disaster zones, tunnels, mines, forests, or isolated infrastructure often experience intermittent or nonexistent network connectivity. Small Language Models enable continued autonomous operation regardless of communication availability because language understanding remains entirely local.

Modern edge robotic architectures typically separate computational responsibilities across multiple processing tiers. High-frequency perception, localization, control, obstacle avoidance, and safety monitoring execute continuously on real-time processors. Small Language Models operate at an intermediate cognitive layer responsible for language understanding, semantic reasoning, task interpretation, dialogue management, and high-level planning. Computationally intensive long-horizon reasoning or model retraining may optionally execute within cloud infrastructure when communication becomes available.

Hardware acceleration plays a central role in practical SLM deployment. Modern edge computers incorporate GPUs, NPUs, TPUs, AI accelerators, tensor processors, and specialized inference hardware optimized for transformer execution. Platforms such as NVIDIA Jetson Orin, Jetson Thor, Qualcomm Robotics RB series, Intel Core Ultra AI processors, AMD Ryzen AI processors, and dedicated neural accelerators provide sufficient computational performance to execute compressed language models while maintaining acceptable power consumption.

Model compression constitutes one of the most important technologies enabling Small Language Models. Compression techniques reduce memory requirements and computational complexity while preserving reasoning quality. Quantization represents the most widely adopted approach by reducing numerical precision from floating-point representations to lower-bit integer formats. Four-bit, six-bit, and eight-bit quantization substantially reduce model size and inference latency while maintaining acceptable language performance for robotic applications.

Knowledge distillation further improves deployment efficiency. Large teacher models transfer their reasoning capability into smaller student models trained to imitate high-quality outputs. Rather than learning directly from massive text corpora, distilled Small Language Models inherit specialized reasoning behavior while requiring only a fraction of the computational resources associated with the original foundation models.

Pruning provides another effective optimization strategy. Transformer networks frequently contain redundant neurons, attention heads, intermediate activations, and feed-forward parameters contributing minimally to final prediction quality. Structured pruning removes these unnecessary components while preserving essential reasoning pathways, resulting in smaller and faster inference models.

Parameter-efficient adaptation enables organizations to customize Small Language Models without modifying all pretrained parameters. Techniques such as Low-Rank Adaptation, adapters, prefix tuning, prompt tuning, and related methods allow robotics developers to specialize models for navigation, inspection, manipulation, warehouse automation, healthcare assistance, or industrial maintenance using relatively small task-specific datasets.

Fine-tuning plays an equally important role in SLM deployment. General-purpose compressed language models may possess broad linguistic capability but insufficient understanding of robotic terminology, sensor interfaces, environmental constraints, safety procedures, middleware APIs, or robot skill libraries. Fine-tuning aligns the model with operational requirements specific to particular robotic platforms while preserving computational efficiency.

Instruction following becomes significantly more reliable after robotics-specific adaptation. Small Language Models learn to interpret natural language commands according to available robot capabilities rather than generating unrestricted conversational responses. User requests become grounded within executable robot skills such as navigation, manipulation, docking, charging, inspection, communication, perception, and reporting.

Robot skill grounding represents a defining characteristic of successful SLM deployment. Rather than reasoning over arbitrary internet knowledge, edge models learn direct associations between linguistic expressions and robot software interfaces. Navigation controllers, manipulation libraries, perception pipelines, ROS 2 services, Python APIs, behavior trees, task planners, and execution managers become semantic primitives understood by the language model.

Memory utilization substantially enhances Small Language Model performance despite limited parameter capacity. Episodic memory stores previous robot experiences, semantic memory captures environmental knowledge, procedural memory records robot skills, and working memory maintains current task context. External memory systems compensate for reduced internal model capacity while improving long-duration autonomous behavior.

Retrieval-Augmented Generation becomes particularly valuable for Small Language Models. Instead of storing extensive factual knowledge internally, edge robots retrieve facility documentation, maintenance manuals, standard operating procedures, digital twins, equipment specifications, inspection histories, semantic maps, software documentation, and enterprise databases only when required. This retrieval mechanism significantly expands effective knowledge capacity without increasing model size.

Multimodal integration allows Small Language Models to reason beyond language alone. RGB images, depth maps, LiDAR observations, point clouds, tactile sensing, force measurements, robot state variables, battery condition, localization estimates, environmental maps, and semantic scene descriptions provide additional context supporting language reasoning. Multimodal inputs compensate for reduced language model capacity by providing explicit environmental information.

Vision-Language integration has become especially important for edge robotics. Instead of requiring the language model to infer visual context from textual descriptions alone, lightweight vision encoders generate semantic embeddings describing observed environments. Small Language Models combine these visual representations with natural language instructions to perform task planning, scene reasoning, object retrieval, and navigation decisions.

Chain of Thought reasoning can also be adapted for Small Language Models despite limited computational resources. Rather than performing unrestricted long-form reasoning, edge systems employ compact structured reasoning focusing on task decomposition, safety verification, environmental constraints, object relationships, and robot capability selection. Adaptive reasoning strategies dynamically adjust reasoning depth according to task complexity.

World models increasingly complement Small Language Models. Instead of relying exclusively upon language reasoning, predictive world models estimate future environmental evolution, object dynamics, robot motion outcomes, and interaction consequences. The language model reasons over these predictive representations while maintaining relatively small internal parameter counts.

Edge robots frequently employ hierarchical cognitive architectures combining multiple language models. Lightweight Small Language Models execute continuously onboard for routine interaction, task interpretation, and local planning. More capable cloud-hosted Large Language Models provide strategic reasoning, complex troubleshooting, software development, or long-horizon planning when communication becomes available. This hierarchical architecture balances responsiveness, capability, and computational efficiency.

Industrial manufacturing provides numerous applications well suited for Small Language Models. Production robots receive natural language work instructions, interpret maintenance requests, generate inspection reports, retrieve operating procedures, explain equipment status, coordinate material handling, and interact with factory personnel while executing entirely on local industrial edge computers.

Warehouse automation similarly benefits from compact language reasoning. Autonomous mobile robots interpret picking instructions, explain navigation decisions, summarize inventory status, coordinate fleet activities, retrieve storage information, generate logistics reports, and communicate with warehouse operators without requiring continuous cloud connectivity.

Inspection robots operating within power plants, factories, pipelines, construction sites, bridges, transportation infrastructure, or utility networks similarly benefit from edge language models. Local inference allows robots to interpret inspection procedures, explain detected anomalies, generate maintenance reports, retrieve historical inspection data, and recommend follow-up actions despite operating within communication-limited environments.

Healthcare robotics represents another important application domain. Service robots assisting patients within hospitals or elderly care facilities frequently process confidential medical information. Running Small Language Models entirely on local hardware preserves patient privacy while enabling natural language interaction, medication reminders, navigation assistance, patient monitoring, and healthcare workflow support.

Agricultural robotics increasingly adopts edge language reasoning. Autonomous field robots interpret farmer instructions, identify crop conditions, retrieve cultivation procedures, explain environmental observations, generate harvesting reports, and coordinate multiple field operations while operating far beyond reliable communication infrastructure.

Construction robotics similarly benefits from localized language intelligence. Autonomous inspection robots, surveying systems, material transport platforms, and construction assistants interpret project documentation, explain site observations, retrieve engineering procedures, generate progress reports, and coordinate worker interactions without depending upon continuous internet connectivity.

Safety remains a primary consideration throughout Small Language Model deployment. Although compact models provide efficient reasoning, they must never directly control low-level actuators without independent verification. Safety controllers, collision prediction systems, affordance estimators, geometric reasoning modules, emergency stop mechanisms, runtime monitors, and deterministic motion planners continue providing execution guarantees independent of language model outputs.

Evaluation methodologies for Small Language Models differ substantially from traditional natural language benchmarks. Robotics evaluation emphasizes instruction following accuracy, task completion success, planning consistency, response latency, energy consumption, memory utilization, execution feasibility, robustness under environmental uncertainty, recovery from failures, and long-duration operational stability. These embodied performance metrics better reflect practical deployment quality than purely linguistic evaluations.

One significant challenge involves balancing model size against reasoning capability. Excessive compression may degrade commonsense reasoning, instruction following accuracy, planning quality, and contextual understanding. Conversely, unnecessarily large models exceed edge hardware limitations. Successful deployment therefore requires careful optimization considering computational resources, task complexity, operational constraints, and acceptable performance tradeoffs.

Continual learning represents another important research direction. Rather than remaining static following deployment, future Small Language Models will incrementally learn from successful task executions, operator corrections, environmental observations, maintenance logs, and evolving operational procedures while avoiding catastrophic forgetting. Continuous adaptation allows robotic intelligence to improve throughout operational lifetime.

Federated learning further extends this concept across robot fleets. Multiple autonomous robots operating within different environments locally adapt shared Small Language Models while exchanging only compressed parameter updates rather than raw operational data. This collaborative learning strategy preserves privacy while accelerating collective improvement across distributed robotic systems.

Future developments will likely produce robotics-specific Small Language Models jointly trained on multimodal perception, manipulation demonstrations, navigation trajectories, robot software repositories, industrial documentation, simulation environments, world models, semantic maps, and long-term memory systems. Rather than compressing general-purpose conversational models, future architectures will be designed specifically for embodied intelligence from the beginning.

As Physical AI continues evolving toward increasingly autonomous edge intelligence, Small Language Models will become foundational cognitive components enabling efficient, private, reliable, and responsive robot behavior. By combining efficient transformer architectures, multimodal perception, robotics-specific fine-tuning, retrieval-augmented knowledge, compressed reasoning, onboard execution, external memory, and deterministic safety systems, Small Language Models establish a practical pathway toward intelligent autonomous robots capable of operating independently within complex real-world environments while satisfying the computational constraints of embedded robotic platforms.

소형 언어 모델(SLM, Small Language Model)은 차세대 엣지 로봇(Edge Robot)을 위한 핵심 인공지능 기술로 주목받고 있다. 대규모 언어 모델(LLM, Large Language Model)은 자연어 이해(Natural Language Understanding), 추론(Reasoning), 계획(Planning), 코드 생성(Code Generation)에서 뛰어난 성능을 제공하지만, 막대한 연산량과 메모리 사용량으로 인해 대부분의 자율주행 로봇에 직접 탑재하기는 어렵다. 반면 SLM은 수억 개에서 수십억 개 수준의 파라미터(Parameter)를 사용하여 제한된 하드웨어에서도 실시간 추론이 가능하도록 설계되었다.

엣지 로봇은 이동하면서 주변 환경을 인식하고, 위치를 추정하며, 경로를 계획하고, 센서를 처리하고, 안전을 유지해야 한다. 이러한 모든 기능은 제한된 전력(Power), 메모리(Memory), 발열(Thermal Budget), 연산 성능(Compute Performance) 안에서 수행되어야 한다. 따라서 클라우드에서 사용하는 초대형 모델보다 경량화된 SLM이 실제 로봇에서는 훨씬 현실적인 선택이 된다.

SLM은 범용 인공지능(General Intelligence)보다 목적 중심 인공지능(Task-Oriented Intelligence)을 지향한다. 대부분의 로봇은 인터넷 전체의 지식을 이해할 필요가 없다. 창고 로봇은 물류(Logistics)를, 검사 로봇은 유지보수(Maintenance)를, 병원 로봇은 의료 절차(Healthcare Workflow)를, 농업 로봇은 작물 관리(Crop Management)를 이해하면 충분하다. 따라서 특정 업무에 특화된 SLM이 실제 성능과 효율성 모두에서 유리하다.

엣지 로봇은 클라우드 AI와 다른 제약 조건을 가진다. 배터리(Battery) 기반으로 동작하므로 소비 전력이 매우 중요하며, 작은 기구 구조 때문에 냉각 능력도 제한적이다. 또한 메모리 용량이 제한되어 있으며, 사람과 함께 작업하는 경우 수 밀리초(Millisecond) 수준의 빠른 응답 속도가 요구된다. 이러한 조건은 SLM 설계의 중요한 기준이 된다.

전력 소비(Power Consumption)는 가장 중요한 요소 중 하나이다. 언어 모델 추론에 많은 전력을 사용하면 센서, 모터, 통신 장치, 제어기 등이 사용할 수 있는 에너지가 감소한다. 배터리 기반 자율주행 로봇에서는 여러 개의 고성능 GPU가 필요한 초대형 LLM을 사용하는 것이 사실상 불가능하다. SLM은 연산량을 줄여 배터리 사용 시간을 크게 증가시킨다.

지연 시간(Latency)도 매우 중요하다. 로봇은 계속 변화하는 환경에서 즉시 의사결정을 내려야 한다. 클라우드 기반 AI는 네트워크(Network) 상태에 따라 응답 시간이 달라질 수 있으며, 인터넷 연결이 끊기면 작업이 불가능해질 수 있다. 반면 SLM은 로봇 내부에서 직접 추론을 수행하므로 일정한 응답 속도를 유지할 수 있으며 안전성이 크게 향상된다.

보안(Security)과 개인정보 보호(Privacy) 역시 엣지 AI가 필요한 이유이다. 공장에서는 생산 기술과 설비 정보가 외부로 유출되어서는 안 되며, 병원에서는 환자의 개인정보를 외부 서버로 전송할 수 없다. SLM을 로봇 내부에서 실행하면 모든 데이터가 로컬(Local)에 유지되므로 보안과 개인정보 보호 수준이 크게 향상된다.

신뢰성(Reliability)도 중요한 장점이다. 광산(Mine), 터널(Tunnel), 농장(Farm), 건설 현장(Construction Site), 산악 지역(Remote Area)처럼 통신이 불안정한 환경에서는 클라우드 AI에 의존할 수 없다. SLM은 네트워크 없이도 자연어 이해와 작업 계획을 수행할 수 있으므로 독립적인 자율 운용이 가능하다.

현대의 엣지 로봇은 계층형 컴퓨팅(Hierarchical Computing Architecture)을 사용한다. 저수준 제어(Low-Level Control), 위치 추정(Localization), 충돌 회피(Collision Avoidance), 안전 제어(Safety Monitoring)는 실시간 제어기에서 수행한다. SLM은 자연어 이해, 작업 계획(Task Planning), 대화(Dialogue), 의미 기반 추론(Semantic Reasoning)을 담당하며, 매우 복잡한 장기 추론(Long-Horizon Reasoning)은 필요할 경우 클라우드 LLM이 처리하는 구조가 일반적이다.

하드웨어 가속(Hardware Acceleration)은 SLM의 핵심 기술이다. 엔비디아(NVIDIA) Jetson Orin, Jetson Thor, 퀄컴(Qualcomm) Robotics RB 시리즈, 인텔(Intel) Core Ultra AI, AMD Ryzen AI, 전용 신경망 처리 장치(NPU, Neural Processing Unit) 등은 SLM을 효율적으로 실행하도록 설계되어 있다. 이러한 플랫폼은 낮은 소비 전력으로도 충분한 추론 성능을 제공한다.

모델 압축(Model Compression)은 SLM을 가능하게 하는 핵심 기술이다. 양자화(Quantization)는 부동소수점(Floating Point) 연산을 8비트, 6비트, 4비트 정수(Integer) 연산으로 변환하여 메모리 사용량과 연산량을 크게 줄인다. 이 과정에서도 대부분의 언어 이해 성능을 유지할 수 있어 로봇 분야에서 널리 사용된다.

지식 증류(Knowledge Distillation)는 대형 모델(Teacher Model)의 지식을 소형 모델(Student Model)로 전달하는 기술이다. 학생 모델은 인터넷 전체를 다시 학습하는 대신 교사 모델의 추론 결과를 학습하여 훨씬 적은 파라미터로도 높은 성능을 얻을 수 있다. 이는 SLM 성능 향상의 핵심 방법 중 하나이다.

가지치기(Pruning)는 중요하지 않은 뉴런(Neuron), 어텐션 헤드(Attention Head), 피드포워드 계층(Feed-Forward Layer)을 제거하여 모델을 더욱 가볍게 만든다. 이를 통해 연산량과 메모리 사용량을 줄이면서도 핵심 추론 능력을 유지할 수 있다.

파라미터 효율적 미세조정(PEFT, Parameter-Efficient Fine-Tuning)은 SLM에 매우 적합하다. 로라(LoRA, Low-Rank Adaptation), 어댑터(Adapter), 프롬프트 튜닝(Prompt Tuning), 프리픽스 튜닝(Prefix Tuning) 등을 이용하여 전체 모델을 다시 학습하지 않고도 특정 로봇 환경에 빠르게 적응시킬 수 있다.

미세조정(Fine-Tuning)은 일반 SLM을 로봇 전용 모델로 변환하는 과정이다. 일반 언어 모델은 로봇 용어, 센서 구조, 안전 규칙, ROS API, 작업 절차 등을 충분히 이해하지 못한다. 로봇 데이터로 미세조정을 수행하면 특정 플랫폼에 최적화된 명령 수행 능력을 갖추게 된다.

SLM은 자연어 명령을 로봇 기술(Skill)과 직접 연결한다. 이동(Navigation), 물체 집기(Grasping), 검사(Inspection), 충전(Charging), 도킹(Docking), 보고(Reporting) 등의 기술을 자연어와 연결함으로써 사용자의 명령을 즉시 실행 가능한 작업으로 변환할 수 있다.

ROS 2 API, 파이썬(Python) 인터페이스, 행동 트리(Behavior Tree), 작업 계획기(Task Planner), 실행 관리자(Execution Manager)도 SLM과 연결된다. 따라서 사용자의 명령을 단순한 텍스트가 아니라 실제 로봇 소프트웨어가 실행 가능한 명령으로 변환할 수 있다.

외부 메모리(External Memory)는 SLM의 부족한 파라미터를 보완한다. 에피소드 메모리(Episodic Memory), 의미 메모리(Semantic Memory), 절차 메모리(Procedural Memory), 작업 메모리(Working Memory)를 활용하여 장시간 작업에서도 높은 성능을 유지할 수 있다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 SLM에서 특히 중요하다. 모델 내부에 모든 지식을 저장하는 대신 필요할 때 설비 매뉴얼(Manual), 작업 절차(Standard Operating Procedure), 디지털 트윈(Digital Twin), 장비 데이터베이스(Database), 의미 지도(Semantic Map)를 검색하여 활용한다. 작은 모델도 방대한 지식을 활용할 수 있게 된다.

멀티모달(Multimodal) 입력은 SLM의 성능을 크게 향상시킨다. RGB 영상(RGB Image), 깊이 정보(Depth Map), 라이다(LiDAR), 포인트 클라우드(Point Cloud), 촉각(Tactile), 힘 센서(Force Sensor), 배터리 상태(Battery Status), 지도(Map) 등을 함께 입력받아 언어 추론과 결합함으로써 모델의 한계를 보완한다.

비전-언어 통합(Vision-Language Integration)은 엣지 로봇에서 매우 중요하다. 비전 인코더(Vision Encoder)는 장면을 의미 벡터(Semantic Embedding)로 변환하고, SLM은 이를 자연어 명령과 함께 이해하여 작업 계획, 장면 이해(Scene Understanding), 물체 탐색(Object Retrieval), 자율주행(Navigation)을 수행한다.

사고의 연쇄(CoT, Chain of Thought)도 SLM에 적용할 수 있다. 다만 대형 모델처럼 긴 추론을 수행하는 대신 작업 분해(Task Decomposition), 안전 확인(Safety Verification), 환경 분석(Environment Analysis), 기술 선택(Skill Selection) 등 필요한 부분만 간결하게 수행하여 연산량을 줄인다.

세계 모델(World Model)은 SLM의 성능을 크게 향상시킨다. 미래 환경 변화를 예측하는 세계 모델과 결합하면 작은 모델도 단순한 현재 상태뿐 아니라 앞으로 발생할 상황까지 고려하여 의사결정을 수행할 수 있다.

실제 로봇에서는 계층형 AI 구조(Hierarchical AI Architecture)가 많이 사용된다. SLM은 항상 로봇 내부에서 실행되어 빠른 응답을 제공하고, 복잡한 설계, 소프트웨어 개발, 장기 전략 수립은 필요할 때만 클라우드 LLM이 수행한다. 이러한 구조는 성능과 비용을 동시에 만족시키는 현실적인 접근 방식이다.

제조 공장에서는 SLM이 작업 지시를 이해하고, 설비 상태를 설명하며, 유지보수 절차를 검색하고, 검사 결과를 보고하며, 작업자와 자연스럽게 대화할 수 있다. 모든 처리가 공장 내부에서 이루어지므로 보안과 실시간성이 보장된다.

물류 창고에서는 피킹(Picking), 재고 확인(Inventory Check), 물류 보고(Logistics Report), 차량 협업(Fleet Coordination), 선반 위치 검색(Storage Retrieval) 등을 네트워크 연결 없이 수행할 수 있다. 이는 대규모 창고 자동화에서 매우 중요한 장점이다.

시설 점검 로봇은 발전소(Power Plant), 공장(Factory), 배관(Pipeline), 교량(Bridge), 철도(Railway), 건설 현장 등에서 점검 절차를 이해하고, 이상 상태를 설명하며, 유지보수 기록을 조회하고, 점검 보고서를 생성할 수 있다.

의료 로봇은 환자 정보가 외부로 유출되지 않도록 로컬에서 모든 추론을 수행한다. 자연어 대화, 약 복용 알림, 병실 안내, 환자 모니터링 등을 개인정보를 보호하면서 처리할 수 있다.

농업 로봇은 농부의 명령을 이해하고 작물 상태를 설명하며, 재배 절차를 검색하고, 수확 보고서를 생성한다. 인터넷 연결이 어려운 농장에서도 독립적으로 작업을 수행할 수 있다.

건설 로봇은 현장 문서를 이해하고, 공사 진행 상황을 설명하며, 설계 절차를 검색하고, 작업 보고서를 생성한다. 네트워크가 불안정한 공사 현장에서도 안정적으로 동작할 수 있다.

그러나 SLM도 저수준 모터 제어를 직접 수행해서는 안 된다. 충돌 예측(Collision Prediction), 기하학적 검증(Geometric Validation), 어포던스 분석(Affordance Estimation), 비상 정지(Emergency Stop), 런타임 모니터(Runtime Monitor)는 반드시 독립적인 안전 시스템이 담당해야 한다.

SLM의 평가는 일반 언어 모델과 다르다. 명령 수행 정확도(Instruction Following Accuracy), 작업 성공률(Task Success Rate), 응답 지연(Response Latency), 에너지 소비(Energy Consumption), 메모리 사용량(Memory Utilization), 실행 가능성(Execution Feasibility), 오류 복구(Recovery) 등을 중심으로 평가한다.

SLM 설계에서는 모델 크기와 추론 성능 사이의 균형이 매우 중요하다. 지나친 압축은 상식(Common Sense)과 추론 능력을 저하시킬 수 있으며, 너무 큰 모델은 엣지 하드웨어에서 실행하기 어렵다. 따라서 하드웨어 성능과 작업 난이도에 맞는 적절한 크기를 선택해야 한다.

지속적 학습(Continual Learning)은 향후 중요한 연구 분야이다. 로봇은 작업 성공 사례, 운영자의 수정, 새로운 환경 정보를 지속적으로 학습하여 성능을 개선할 수 있다. 이를 통해 시간이 지날수록 더욱 똑똑한 로봇으로 발전할 수 있다.

연합 학습(Federated Learning)은 여러 대의 로봇이 개인정보를 공유하지 않고도 함께 학습하는 기술이다. 각 로봇은 로컬에서 모델을 업데이트한 후 파라미터만 공유하므로 보안을 유지하면서도 전체 로봇 시스템의 성능을 향상시킬 수 있다.

향후에는 로봇 전용 SLM이 언어(Language), 영상(Image), 조작 시연(Manipulation Demonstration), 자율주행 데이터(Navigation Trajectory), ROS 소프트웨어 저장소, 산업 문서, 세계 모델(World Model), 장기 메모리(Long-Term Memory)를 함께 학습하는 형태로 발전할 것으로 예상된다. 단순히 LLM을 축소하는 것이 아니라 처음부터 물리 AI(Physical AI)를 목표로 설계된 전용 모델이 등장할 가능성이 높다.

결국 SLM은 엣지 로봇 시대의 핵심 인지 엔진(Cognitive Engine)이다. 경량 트랜스포머(Lightweight Transformer), 로봇 특화 미세조정(Robot-Specific Fine-Tuning), 멀티모달 인식(Multimodal Perception), 검색 증강 생성(RAG), 외부 메모리(External Memory), 온디바이스 추론(On-Device Inference), 계층형 AI(Hierarchical AI), 독립적인 안전 시스템(Safety System)을 통합함으로써 제한된 하드웨어에서도 빠르고 안전하며 신뢰성 높은 자율 로봇을 구현하는 차세대 물리 AI의 핵심 기술로 자리잡고 있다.

##  

## 3.8 Error Recovery and Replanning Using LLMs (with Code)

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

Error recovery and replanning have become fundamental capabilities for intelligent robotic systems operating in dynamic real-world environments. While perception, planning, and control enable robots to perform complex tasks autonomously, no robotic system can avoid unexpected failures entirely. Environmental uncertainty, sensor noise, hardware limitations, communication failures, localization drift, object movement, human intervention, actuator errors, and unforeseen obstacles inevitably introduce deviations from the original execution plan. Traditional robotic systems typically respond to these failures using manually engineered recovery procedures, predefined state machines, or fixed exception handlers. Although such methods provide reliable recovery for anticipated situations, they struggle to generalize when robots encounter novel failure modes. Large Language Models (LLMs) introduce a fundamentally different approach by enabling robots to understand failures semantically, analyze their underlying causes, reason about alternative solutions, and dynamically generate new execution plans. Within modern Vision-Language-Action (VLA) architectures, LLM-based error recovery transforms robotic autonomy from rigid execution into adaptive problem solving.

Traditional robotic control architectures generally assume that tasks progress according to predefined workflows. Navigation systems expect planned paths to remain accessible, manipulation systems assume target objects remain reachable, localization systems assume maps remain consistent, and communication systems assume stable network connectivity. Real-world environments rarely satisfy these assumptions continuously. Doors become unexpectedly closed, humans relocate target objects, temporary obstacles block navigation routes, equipment malfunctions occur, lighting conditions change, batteries discharge faster than expected, and perception systems encounter ambiguous observations. Autonomous robots must therefore treat failures as ordinary operational events rather than exceptional conditions.

The central principle of LLM-based recovery is semantic understanding of failure. Rather than simply detecting that an operation has failed, the robot attempts to understand why the failure occurred, how the surrounding environment has changed, what constraints now exist, and which alternative strategies remain feasible. This semantic interpretation enables adaptive recovery that extends beyond predefined exception handling rules.

Error recovery begins with failure detection. Multiple subsystems continuously monitor execution status throughout robotic operation. Motion controllers report trajectory deviations, navigation systems identify blocked paths, manipulation controllers detect unsuccessful grasps, perception systems estimate confidence levels, localization modules evaluate pose uncertainty, battery management systems monitor energy reserves, communication modules detect network interruptions, and safety systems identify hazardous conditions. These low-level monitoring components provide the raw information required for higher-level reasoning.

Failure classification constitutes the next processing stage. Instead of treating every error identically, the robotic cognitive system categorizes failures according to their underlying characteristics. Navigation failures may result from blocked corridors, localization uncertainty, moving obstacles, inaccessible destinations, or environmental changes. Manipulation failures may originate from inaccurate object detection, unstable grasps, object deformation, insufficient reachability, or unexpected object motion. Communication failures, hardware malfunctions, perception ambiguity, software exceptions, and human intervention similarly represent distinct categories requiring different recovery strategies.

Large Language Models significantly improve failure interpretation because they integrate environmental context, commonsense knowledge, previous task history, semantic scene understanding, and user intent into the diagnostic process. Instead of recognizing only that grasp execution failed, the LLM may infer that another object partially occluded the target, the object shifted during approach, lighting conditions degraded perception quality, or human interaction altered the environment. Such semantic diagnosis enables more intelligent recovery planning.

Context awareness plays a central role throughout recovery reasoning. Identical execution failures may require entirely different responses depending on environmental conditions. Failure to grasp a fragile glass container demands cautious manipulation adjustments, whereas failure to grasp a heavy industrial component may require changing grasp points, requesting human assistance, or selecting alternative tooling. LLMs naturally incorporate contextual reasoning into recovery planning because they possess broad world knowledge regarding object properties, environmental semantics, and task objectives.

Error recovery also requires distinguishing temporary failures from permanent failures. Temporary obstacles, moving humans, transient communication interruptions, or brief sensor occlusions frequently disappear after short delays. Permanent failures such as broken equipment, depleted batteries, inaccessible locations, missing objects, or damaged manipulators require substantial replanning rather than repeated retries. LLMs analyze environmental evidence to estimate whether observed failures likely represent temporary or persistent conditions.

Replanning differs fundamentally from simple retry mechanisms. Traditional systems frequently repeat identical actions following failure, assuming success may occur during subsequent attempts. LLM-based replanning instead modifies execution strategy according to newly acquired environmental knowledge. If a navigation route becomes blocked, the robot searches for alternative paths. If an object cannot be reached directly, intermediate actions reposition obstacles or alter robot configuration. If required tools are unavailable, substitute tools or modified procedures become considered.

Task decomposition substantially enhances recovery capability. Rather than viewing tasks as indivisible sequences, the LLM represents missions hierarchically using interconnected subtasks. Failures affecting individual subtasks need not invalidate entire missions. Instead, replanning focuses only upon affected task components while preserving successfully completed work. This localized adaptation improves efficiency while reducing unnecessary repetition.

Memory integration significantly strengthens error recovery. Episodic memory stores previous recovery experiences describing successful strategies applied under similar circumstances. Semantic memory provides general knowledge regarding object properties, environmental organization, operational procedures, and physical constraints. Procedural memory records robot skills together with historical execution performance. Long-term environmental memory captures persistent facility layouts, equipment locations, charging stations, restricted areas, and frequently encountered obstacles. LLMs retrieve relevant memories when constructing recovery strategies.

Retrieval-Augmented Generation further improves recovery quality. Rather than relying exclusively upon internal model knowledge, robots retrieve operating manuals, maintenance procedures, troubleshooting guides, equipment documentation, digital twins, facility maps, software logs, and previous execution records. Retrieved information supplements semantic reasoning, enabling recovery decisions tailored to specialized industrial environments.

Perception continuously updates environmental understanding throughout recovery. Vision systems identify newly appeared obstacles, relocated objects, changing lighting conditions, human activities, damaged equipment, or altered workspace organization. Depth sensing estimates accessibility, semantic segmentation identifies environmental structure, LiDAR updates occupancy maps, and force sensing monitors manipulation outcomes. These observations ensure recovery planning reflects current reality rather than outdated assumptions.

Scene reasoning plays an essential role during replanning. Rather than considering isolated objects, the LLM interprets the semantic organization of the environment. If a requested object disappears from its expected location, nearby containers, storage shelves, workbenches, transport carts, or human operators become plausible search locations according to contextual reasoning. Such scene-level understanding substantially improves recovery efficiency.

Affordance reasoning similarly supports adaptive behavior. Recovery planning evaluates not only desired actions but also which actions remain physically executable under current conditions. A blocked corridor eliminates navigation affordances. A damaged gripper limits manipulation affordances. Low battery reserves restrict long-distance travel. Affordance estimation grounds semantic reasoning within actual robot capabilities, ensuring generated recovery plans remain executable.

Chain of Thought reasoning increasingly complements recovery planning. Rather than immediately proposing corrective actions, the LLM first performs structured reasoning analyzing task objectives, completed subtasks, current failures, environmental constraints, available robot skills, safety requirements, alternative strategies, and expected consequences. This explicit reasoning process substantially improves recovery quality for complex long-horizon missions.

Closed-loop execution distinguishes intelligent recovery from static replanning. Following every corrective action, perception updates the environmental representation, execution status is reevaluated, and remaining recovery steps become adjusted according to new observations. Continuous feedback enables robots to adapt dynamically throughout extended recovery episodes rather than committing prematurely to fixed alternative plans.

Human interaction frequently contributes to recovery. Robots operating within collaborative environments may request clarification, confirmation, or assistance whenever autonomous recovery remains uncertain. Natural language communication allows operators to provide missing information, authorize modified procedures, identify relocated objects, or intervene directly when necessary. LLMs naturally facilitate these conversational recovery interactions.

Industrial manufacturing illustrates the importance of adaptive recovery. Production environments contain moving personnel, changing work orders, temporary equipment outages, maintenance activities, material shortages, and evolving production schedules. Rather than halting production whenever unexpected events occur, robots employing LLM-based replanning dynamically adjust task sequences while respecting operational priorities and safety constraints.

Warehouse automation similarly benefits from semantic recovery. Inventory locations frequently change, aisles become temporarily congested, pallets shift unexpectedly, forklifts interrupt transportation routes, charging stations become occupied, and urgent fulfillment requests modify operational priorities. LLMs continuously reinterpret these changing conditions while generating updated execution plans maximizing overall productivity.

Inspection robotics presents additional recovery challenges. Weather conditions, lighting variations, sensor contamination, restricted access, equipment vibration, environmental hazards, and changing operational status frequently disrupt inspection procedures. Rather than abandoning inspections after individual failures, semantic recovery enables robots to modify inspection order, substitute sensing modalities, revisit inaccessible assets later, or generate partial inspection reports documenting remaining limitations.

Healthcare robotics requires particularly cautious recovery behavior. Patient safety always overrides task completion. If requested medications become unavailable, pathways become crowded, elevators malfunction, or patients unexpectedly relocate, robots must generate safe alternative plans while clearly communicating status updates to healthcare personnel. LLMs support these context-sensitive recovery strategies using natural language reasoning.

Construction robotics similarly encounters highly dynamic environments characterized by evolving site layouts, temporary barriers, equipment relocation, weather changes, incomplete structures, and continuous human activity. Fixed planning rapidly becomes obsolete under such conditions. LLM-based replanning continuously adapts execution strategies according to current construction progress.

Agricultural robotics also benefits from adaptive recovery. Weather conditions, crop growth, terrain variability, equipment availability, animal movement, irrigation changes, and seasonal operations introduce substantial environmental uncertainty. Semantic reasoning allows robots to modify harvesting routes, inspection priorities, spraying procedures, or navigation strategies according to changing field conditions.

Software failures require semantic interpretation alongside physical failures. Middleware interruptions, unavailable services, API exceptions, synchronization problems, memory limitations, scheduling conflicts, or corrupted data may disrupt execution despite unchanged physical environments. LLMs analyze software logs, execution traces, and diagnostic messages to recommend corrective actions extending beyond conventional exception handling.

Multi-robot systems introduce additional recovery complexity. Failures affecting one robot influence task allocation, resource availability, communication patterns, and mission coordination throughout the entire fleet. LLMs support collaborative replanning by redistributing subtasks, reallocating resources, coordinating replacement robots, and updating collective execution schedules while minimizing overall mission disruption.

Cloud-edge collaboration enhances recovery capability without sacrificing responsiveness. Lightweight onboard language models provide immediate local diagnosis and short-term recovery planning. More computationally intensive cloud-based LLMs optionally perform comprehensive root-cause analysis, long-horizon optimization, software debugging, or fleet-wide coordination whenever communication infrastructure permits. Hierarchical reasoning balances computational efficiency with planning sophistication.

Simulation significantly improves recovery policy development. Digital twins and high-fidelity simulators expose robots to diverse failure scenarios impossible or unsafe to reproduce physically. Blocked pathways, sensor failures, communication outages, hardware degradation, adverse weather, unexpected human interactions, and equipment malfunctions become systematically incorporated into training datasets. Exposure to diverse simulated failures substantially improves real-world robustness.

Fine-tuning further specializes language models for recovery reasoning. General-purpose foundation models possess broad commonsense knowledge but limited understanding of robot diagnostics, industrial procedures, middleware architectures, manipulation failures, localization uncertainty, or hardware constraints. Robotics-specific fine-tuning teaches models to interpret execution failures according to actual robotic operational principles.

Safety remains the highest priority throughout recovery. LLM-generated recovery plans never execute directly without independent validation. Motion planners verify collision-free trajectories, affordance estimators evaluate physical feasibility, safety controllers enforce operational constraints, runtime monitors supervise execution, emergency stop mechanisms retain override authority, and deterministic control systems guarantee safe low-level behavior. LLM reasoning complements rather than replaces classical robotic safety mechanisms.

Evaluation of recovery performance differs substantially from conventional language benchmarks. Success metrics include recovery success rate, mission completion after failure, replanning efficiency, execution latency, resource consumption, human intervention frequency, safety compliance, task continuity, robustness under uncertainty, and operational productivity. These embodied performance measures better reflect practical robotic capability than conversational quality alone.

One significant challenge involves uncertainty estimation. Recovery decisions frequently rely upon incomplete or ambiguous perception data. Incorrect diagnosis may produce ineffective or unsafe corrective actions. Future robotic architectures increasingly integrate probabilistic reasoning, confidence estimation, Bayesian inference, and uncertainty-aware planning alongside language reasoning to improve decision reliability.

Another research direction explores predictive recovery through world models. Rather than reacting only after failures occur, predictive models estimate future environmental evolution, identify emerging risks, anticipate equipment degradation, forecast obstacle movement, and recommend preventive actions before failures manifest. LLMs reason over these predictive simulations to generate proactive rather than reactive recovery strategies.

Continual learning also enhances long-term recovery performance. Robots accumulate operational experience throughout deployment, recording successful recovery episodes, unsuccessful interventions, environmental changes, maintenance records, operator corrections, and evolving workflows. Incremental learning gradually expands recovery capability while adapting to organization-specific operational practices.

Future Vision-Language-Action systems are expected to integrate multimodal perception, semantic scene understanding, affordance reasoning, Chain of Thought planning, world models, external memory, retrieval-augmented knowledge, simulation experience, and continual learning into unified embodied cognitive architectures. Rather than treating failure recovery as an isolated software module, future Physical AI systems will view every unexpected event as an opportunity for reasoning, adaptation, and learning.

As autonomous robots increasingly operate across manufacturing, logistics, healthcare, agriculture, construction, infrastructure inspection, domestic service, and collaborative workplaces, LLM-based error recovery and replanning will become one of the defining capabilities distinguishing truly intelligent embodied systems from conventional automation. By combining semantic understanding, contextual reasoning, environmental awareness, reusable robot skills, memory retrieval, multimodal perception, adaptive planning, and deterministic safety verification, LLM-based recovery establishes a practical framework through which robots can continue operating safely and productively despite the uncertainty and unpredictability that characterize real-world environments.

LLM 기반 오류 복구(Error Recovery)와 재계획(Replanning)은 실제 환경에서 동작하는 지능형 로봇의 핵심 기술이다. 자율주행, 물체 조작, 환경 인식, 작업 계획이 아무리 정교하더라도 실제 환경에서는 예상하지 못한 다양한 오류가 발생한다. 센서 잡음(Sensor Noise), 위치 추정 오차(Localization Drift), 장애물(Obstacle), 사람의 개입(Human Intervention), 통신 장애(Communication Failure), 하드웨어 이상(Hardware Failure), 환경 변화(Environment Change) 등은 기존 계획을 무력화할 수 있다. 따라서 로봇은 오류를 감지하고 스스로 새로운 계획을 생성하는 능력이 필요하다.

기존 로봇 시스템은 대부분 상태 기계(State Machine)나 예외 처리(Exception Handler)를 이용하여 오류를 처리하였다. 이러한 방식은 미리 정의된 상황에서는 안정적으로 동작하지만, 새로운 유형의 오류나 복합적인 문제에는 유연하게 대응하지 못한다. 반면 대규모 언어 모델(LLM, Large Language Model)은 오류의 원인을 의미적으로 이해하고, 환경을 분석하며, 새로운 해결 전략을 생성할 수 있으므로 기존 방식보다 훨씬 높은 적응성을 제공한다.

LLM 기반 오류 복구의 핵심은 단순히 "실패했다"는 사실을 인식하는 것이 아니라 "왜 실패했는가"를 이해하는 것이다. 로봇은 현재 환경이 어떻게 변화했는지, 어떤 제약 조건이 새롭게 발생했는지, 기존 계획이 왜 더 이상 유효하지 않은지를 분석하고, 이를 바탕으로 새로운 작업 계획을 생성한다. 즉, 실패를 단순한 예외가 아니라 새로운 상황(Context)으로 해석한다.

오류 복구의 첫 단계는 실패 감지(Failure Detection)이다. 이동 제어기(Motion Controller)는 경로 이탈을 감지하고, 자율주행(Navigation)은 경로 차단을 확인하며, 조작기(Manipulator)는 물체를 제대로 잡지 못했는지를 판단한다. 또한 인식 시스템(Perception System)은 객체 탐지 신뢰도(Object Detection Confidence)를 계산하고, 위치 추정(Localization)은 위치 오차를 측정하며, 배터리 관리 시스템(BMS)은 전력 부족을 감지한다.

오류가 감지되면 다음 단계는 오류 분류(Failure Classification)이다. 모든 오류를 동일하게 처리하지 않고 원인에 따라 구분한다. 이동 실패는 장애물, 지도 오류, 위치 오차, 목적지 접근 불가 등으로 나뉘며, 조작 실패는 물체 인식 오류, 그리퍼(Gripper) 미끄러짐, 물체 이동, 작업 공간 부족 등으로 구분된다. 통신 장애, 센서 이상, 소프트웨어 오류도 각각 다른 복구 절차를 요구한다.

LLM은 오류를 의미적으로 해석한다. 예를 들어 단순히 "물체를 잡지 못했다."가 아니라 "다른 물체가 목표를 가리고 있었으며, 조명이 어두워 인식 정확도가 떨어졌고, 접근 중 사람이 물체를 이동시켰다."와 같이 원인을 종합적으로 분석한다. 이러한 의미 기반 진단(Semantic Diagnosis)은 이후의 재계획 품질을 크게 향상시킨다.

문맥(Context)은 오류 복구에서 매우 중요한 요소이다. 같은 집기 실패라도 깨지기 쉬운 유리컵(Glass Cup)과 무거운 산업용 부품(Industrial Component)은 서로 다른 복구 전략이 필요하다. LLM은 물체의 특성(Property), 작업 목적(Task Goal), 주변 환경(Environment)을 함께 고려하여 가장 적합한 해결 방법을 선택한다.

LLM은 일시적인 오류와 영구적인 오류를 구분한다. 잠시 사람이 지나가거나 통신이 끊긴 경우에는 기다리거나 재시도(Retry)하면 해결될 가능성이 높다. 반면 장비 고장, 배터리 부족, 작업 공간 접근 불가와 같은 문제는 단순 재시도가 아니라 새로운 계획(Replanning)이 필요하다.

재계획은 단순 반복과 다르다. 기존 시스템은 실패한 행동을 그대로 다시 수행하는 경우가 많았지만, LLM은 현재 환경을 다시 분석하여 새로운 경로를 찾거나, 다른 접근 방향을 선택하거나, 방해물을 먼저 이동시키는 등 전략 자체를 변경한다. 따라서 동일한 오류가 반복될 가능성이 크게 줄어든다.

LLM은 작업을 계층적으로 분해(Task Decomposition)한다. 전체 작업을 여러 개의 하위 작업(Subtask)으로 나누기 때문에 특정 단계에서 오류가 발생하더라도 전체 작업을 처음부터 다시 시작하지 않는다. 문제가 발생한 하위 작업만 수정하고 나머지 작업은 그대로 유지하므로 효율성이 크게 향상된다.

메모리(Memory)는 오류 복구 성능을 크게 향상시킨다. 에피소드 메모리(Episodic Memory)는 과거의 성공 및 실패 사례를 저장하며, 의미 메모리(Semantic Memory)는 물체 특성과 일반 지식을 제공한다. 절차 메모리(Procedural Memory)는 로봇 기술(Skill)을 저장하고, 장기 환경 메모리(Long-Term Environmental Memory)는 건물 구조, 충전기 위치, 제한 구역 등을 기억하여 복구 계획에 활용된다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 오류 복구에서도 매우 유용하다. 유지보수 매뉴얼(Maintenance Manual), 운영 절차(Standard Operating Procedure), 디지털 트윈(Digital Twin), 설비 문서(Document), 과거 작업 로그(Log)를 검색하여 현재 오류와 가장 유사한 사례를 찾아 새로운 해결 방법을 생성할 수 있다.

인식 시스템은 복구 과정에서도 계속 환경을 갱신한다. 카메라(Camera)는 새로운 장애물과 사람을 인식하고, 깊이 센서(Depth Sensor)는 접근 가능성을 분석하며, 라이다(LiDAR)는 새로운 지도(Map)를 생성한다. 힘 센서(Force Sensor)는 집기 성공 여부를 확인하며, 이러한 모든 정보는 재계획 과정에 지속적으로 반영된다.

장면 추론(Scene Reasoning)은 오류 복구에서 중요한 역할을 한다. 목표 물체가 원래 위치에 없으면 주변 선반(Shelf), 상자(Container), 작업대(Workbench), 운반 카트(Cart), 사람 근처 등을 후보 위치로 추론한다. 단순한 객체 탐색보다 훨씬 높은 성공률을 제공하는 이유가 여기에 있다.

어포던스(Affordance) 추론도 함께 사용된다. 이동 가능한 경로가 막혔는지, 현재 그리퍼가 물체를 잡을 수 있는지, 배터리 상태로 장거리 이동이 가능한지 등을 평가하여 실제 수행 가능한 행동만 선택한다. 의미 기반 계획과 물리적 실행 가능성을 동시에 고려하는 것이 특징이다.

최근에는 사고의 연쇄(CoT, Chain of Thought)를 오류 복구에 적용하는 연구가 활발하다. LLM은 즉시 해결책을 생성하지 않고 먼저 작업 목표, 현재 상태, 실패 원인, 환경 변화, 사용 가능한 기술(Skill), 안전 규칙(Safety Rule), 대안(Alternative)을 단계적으로 추론한 후 최적의 복구 계획을 생성한다.

오류 복구는 폐루프 제어(Closed-Loop Execution)로 수행된다. 새로운 행동을 수행한 후 센서를 이용하여 결과를 확인하고, 다시 환경을 분석한 후 다음 행동을 결정한다. 즉, 복구 과정 자체도 지속적인 피드백(Feedback)을 이용하여 동적으로 수정된다.

사람과 협업(Human-Robot Collaboration)하는 환경에서는 사람의 도움도 적극적으로 활용한다. 로봇은 "물체 위치를 알려주세요.", "문을 열어 주시겠습니까?", "현재 작업을 계속 진행해도 될까요?"와 같이 자연어로 질문하여 필요한 정보를 얻거나 작업 허가를 받을 수 있다. LLM은 이러한 대화 기반 복구(Dialogue-Based Recovery)를 자연스럽게 수행한다.

산업용 제조 환경에서는 설비 점검, 생산 계획 변경, 자재 부족(Material Shortage), 작업 순서 변경 등으로 인해 지속적인 재계획이 필요하다. LLM은 생산 목표를 유지하면서 작업 순서를 변경하거나 대체 설비를 선택하여 생산성을 최대한 유지하도록 계획을 수정한다.

물류 창고(Warehouse)에서는 재고 위치 변경, 통로 혼잡, 팔레트 이동, 충전기 사용 중 등의 상황이 자주 발생한다. LLM은 이러한 환경 변화를 분석하여 새로운 경로를 생성하고 작업 우선순위를 변경하며, 여러 작업을 동시에 최적화할 수 있다.

시설 점검(Inspection Robot)에서는 날씨(Weather), 조명(Lighting), 센서 오염, 접근 제한, 장비 진동 등이 검사에 영향을 준다. LLM은 점검 순서를 변경하거나 다른 센서를 활용하거나, 접근 가능한 설비부터 먼저 검사하는 방식으로 작업을 계속 수행한다.

의료 로봇(Healthcare Robot)은 환자의 안전이 가장 중요하다. 약품 부족, 엘리베이터 고장, 병실 이동, 응급 상황이 발생하면 기존 작업보다 환자의 안전을 우선하여 새로운 계획을 생성하고 의료진에게 현재 상황을 설명한다.

건설 현장(Construction Site)은 작업 환경이 지속적으로 변한다. 새로운 구조물이 설치되고 장비가 이동하며 공사 구역이 변경된다. LLM은 이러한 변화를 반영하여 검사 경로, 운반 계획, 자재 이동 순서를 계속 수정한다.

농업 로봇(Agricultural Robot)은 날씨, 작물 성장, 지형 변화, 동물 이동, 관개 상태에 따라 작업 계획을 변경해야 한다. LLM은 현재 환경을 분석하여 수확 순서, 방제 구역, 이동 경로 등을 실시간으로 재계획한다.

소프트웨어 오류(Software Failure)도 의미 기반으로 분석할 수 있다. ROS 서비스 오류(Service Failure), API 예외(Exception), 동기화 문제(Synchronization Issue), 메모리 부족(Memory Shortage) 등의 로그(Log)를 분석하여 적절한 해결 방법을 제안할 수 있다.

다중 로봇(Multi-Robot) 시스템에서는 한 대의 로봇 오류가 전체 작업에 영향을 미친다. LLM은 다른 로봇에게 작업을 재할당(Task Reallocation)하거나 자원을 공유(Resource Sharing)하고 전체 작업 일정을 다시 조정하여 시스템 전체의 효율성을 유지한다.

최근에는 엣지(Edge)와 클라우드(Cloud)를 함께 사용하는 계층형 복구 구조도 많이 사용된다. 로컬 SLM(Small Language Model)은 즉각적인 오류 분석과 단기 복구를 수행하고, 클라우드 LLM은 근본 원인 분석(Root Cause Analysis), 장기 계획(Long-Horizon Planning), 다중 로봇 협업 등을 수행한다.

시뮬레이션(Simulation)은 오류 복구 학습에서 매우 중요한 역할을 한다. 아이작 심(Isaac Sim), 디지털 트윈(Digital Twin), 가제보(Gazebo) 등을 이용하여 장애물, 센서 고장, 통신 장애, 악천후, 장비 고장 등의 다양한 실패 상황을 대량으로 생성하고 학습할 수 있다.

로봇 전용 미세조정(Fine-Tuning)은 오류 복구 성능을 더욱 향상시킨다. 일반 LLM은 로봇의 오류 코드(Error Code), 센서 상태, ROS 로그, 산업 장비 구조를 충분히 이해하지 못하기 때문에 로봇 데이터로 추가 학습하면 훨씬 정확한 오류 분석이 가능해진다.

그러나 안전(Safety)은 항상 최우선이다. LLM이 생성한 복구 계획은 직접 실행되지 않는다. 모션 플래너(Motion Planner), 충돌 예측(Collision Prediction), 어포던스 분석(Affordance Estimation), 런타임 모니터(Runtime Monitor), 비상 정지(Emergency Stop), 규칙 기반 안전 검사(Rule-Based Safety Check)가 먼저 검증한 후 실제 로봇에서 실행된다.

오류 복구 성능 평가는 일반 언어 모델과 다르다. 복구 성공률(Recovery Success Rate), 작업 완료율(Task Completion Rate), 재계획 시간(Replanning Time), 자원 사용량(Resource Consumption), 사람의 개입 횟수(Human Intervention), 안전성(Safety Compliance), 작업 연속성(Task Continuity) 등을 종합적으로 평가한다.

불확실성(Uncertainty) 처리도 중요한 연구 분야이다. 센서 정보가 불완전하거나 모호한 경우 잘못된 오류 진단이 이루어질 수 있다. 최근에는 베이지안 추론(Bayesian Inference), 확률 기반 계획(Probabilistic Planning), 신뢰도 추정(Confidence Estimation)을 함께 사용하여 더욱 안정적인 오류 복구를 수행한다.

향후에는 세계 모델(World Model)을 이용한 예측 기반 오류 복구가 발전할 것으로 예상된다. 오류가 발생한 후 대응하는 것이 아니라 미래의 장비 고장, 장애물 이동, 배터리 부족 등을 미리 예측하여 사전에 작업 계획을 수정하는 예방적 복구(Proactive Recovery)가 가능해질 것이다.

지속적 학습(Continual Learning)도 중요한 발전 방향이다. 로봇은 운영 과정에서 발생한 다양한 오류와 복구 사례를 지속적으로 학습하여 시간이 지날수록 더욱 높은 복구 성공률을 갖게 된다.

결국 LLM 기반 오류 복구와 재계획은 기존의 고정된 예외 처리 방식을 넘어 의미 기반 추론(Semantic Reasoning), 환경 이해(Scene Understanding), 어포던스 분석(Affordance Reasoning), 메모리(Memory), 검색 증강 생성(RAG), 멀티모달 인식(Multimodal Perception), 세계 모델(World Model), 지속적 학습(Continual Learning)을 통합하여 실제 환경에서 발생하는 다양한 실패를 스스로 이해하고 해결하는 차세대 물리 AI(Physical AI)의 핵심 기술로 자리잡고 있다.

##  

## 3.9 Multi-Turn Human-Robot Dialogue (with Code)

![](images/image9.png){width="7.268055555555556in" height="7.268055555555556in"}

Multi-turn dialogue has become one of the most important capabilities for intelligent robotic systems as robots transition from executing isolated commands to participating in continuous human-robot collaboration. Traditional robot interfaces generally assume that users provide complete and unambiguous instructions before execution begins. However, real-world tasks rarely follow such simplified interaction patterns. Human operators naturally communicate through incremental conversations, refining goals, correcting misunderstandings, providing additional information, asking questions, changing priorities, and reacting to evolving environmental conditions. Large Language Models (LLMs) enable robots to participate in these extended conversational interactions while maintaining awareness of task context, dialogue history, environmental knowledge, robot capabilities, and execution status. Within modern Vision-Language-Action (VLA) architectures, multi-turn dialogue transforms robotic systems from command-driven automation into collaborative cognitive partners capable of long-term interactive task execution.

Human communication is inherently contextual. People rarely express every detail of a task in a single sentence. Instead, instructions evolve through dialogue. A user may initially request, "Inspect the production line," followed by additional statements such as "Start with the packaging area," "Skip the damaged conveyor," "Focus on temperature sensors," and "Generate a maintenance report afterward." Each new utterance modifies the original task rather than replacing it entirely. A robot must therefore maintain conversational memory capable of integrating multiple instructions into a coherent evolving mission plan.

Unlike single-turn command processing, multi-turn dialogue requires persistent context management. Every conversational exchange influences future interpretation. Pronouns, implicit references, abbreviated expressions, and previously discussed objects become meaningful only when interpreted relative to accumulated dialogue history. If a user says, "Bring the inspection camera," followed later by "Place it on the charging station," the robot must recognize that "it" refers to the inspection camera rather than another recently mentioned object. Maintaining this contextual continuity represents one of the primary strengths of LLM-based dialogue systems.

Dialogue memory extends beyond linguistic history into environmental context. During execution, the robot continuously updates its understanding of object locations, navigation progress, task completion status, sensor observations, battery condition, human positions, environmental changes, and available resources. Conversational reasoning therefore combines spoken interaction with continuously evolving physical knowledge. Responses become grounded not only in previous dialogue but also in the robot\'s current understanding of the world.

Task-oriented dialogue differs substantially from general conversational AI. In ordinary conversation, the objective is information exchange. In robotics, dialogue directly influences physical behavior. Every conversational decision may modify navigation goals, manipulation plans, inspection procedures, scheduling priorities, or collaborative workflows. Consequently, robotic dialogue systems must produce responses that are semantically meaningful, operationally consistent, physically executable, and compliant with safety constraints.

Intent understanding represents the foundation of multi-turn interaction. Human instructions frequently contain ambiguity, incomplete specifications, or implicit assumptions. Instead of immediately executing uncertain commands, LLMs identify missing information and initiate clarification dialogues. If a user requests, "Move the equipment," the robot may ask, "Which equipment would you like me to move?" or "Where should I place it?" Such clarification significantly reduces execution errors while improving overall task success.

Clarification dialogue demonstrates one of the most valuable characteristics of intelligent robotic communication. Traditional command systems often require perfectly formatted instructions before execution. LLMs instead engage in cooperative conversation, identifying ambiguities naturally and resolving them through incremental questioning. This interaction more closely resembles collaboration between human coworkers than interaction with conventional automation systems.

Dialogue management also supports dynamic task modification. Users frequently alter objectives during execution. For example, while the robot transports inspection equipment, the user may request that another urgent task receive higher priority. Rather than abandoning previous work entirely, the LLM analyzes existing task progress, completed subtasks, remaining objectives, environmental constraints, and scheduling implications before generating a revised execution plan. Multi-turn dialogue therefore becomes tightly integrated with dynamic task replanning.

Long-horizon tasks particularly benefit from conversational interaction. Industrial inspections, warehouse inventory management, hospital delivery services, agricultural monitoring, infrastructure maintenance, and construction site operations often require hours of continuous activity. Throughout these missions, human supervisors periodically request progress updates, modify priorities, provide additional instructions, answer robot questions, or respond to unexpected events. Multi-turn dialogue maintains collaborative continuity throughout these extended operational periods.

Dialogue memory generally consists of multiple complementary layers. Short-term conversational memory preserves recent exchanges necessary for interpreting ongoing discussion. Working memory stores active task variables, temporary decisions, unresolved questions, and current execution context. Episodic memory records completed conversations together with corresponding robot experiences. Semantic memory stores persistent knowledge regarding facilities, equipment, procedures, terminology, and object relationships. Collectively, these memory systems enable coherent long-term interaction.

Retrieval-Augmented Generation further enhances dialogue quality by providing access to external knowledge during conversation. Rather than relying exclusively upon internal model parameters, the robot retrieves maintenance manuals, operating procedures, equipment specifications, facility documentation, digital twins, software documentation, inspection histories, semantic maps, and enterprise databases whenever users ask domain-specific questions. Retrieved information becomes integrated naturally into ongoing dialogue while remaining grounded in authoritative documentation.

Multimodal perception significantly enriches conversational interaction. Rather than discussing abstract concepts alone, robots perceive surrounding environments through RGB cameras, depth sensors, LiDAR, tactile sensing, force measurements, semantic segmentation, localization systems, and object detection pipelines. When users ask questions such as "What is blocking the corridor?" or "Is the valve still leaking?" the robot combines visual perception with language reasoning to generate accurate responses describing current environmental conditions.

Vision-Language Models frequently complement multi-turn dialogue systems by generating semantic descriptions of observed scenes. These descriptions become part of the conversational context interpreted by the LLM. Rather than reasoning exclusively over raw sensor data, the language model receives structured semantic information describing objects, relationships, spatial organization, human activities, environmental changes, and potential hazards. This integration substantially improves dialogue quality within complex physical environments.

Reference resolution represents another critical capability. Human dialogue frequently includes expressions such as "that box," "the previous pallet," "the same room," "the left shelf," "our last inspection," or "the damaged machine." LLMs resolve these references using dialogue history, environmental observations, spatial reasoning, semantic memory, and previous execution records. Accurate reference resolution prevents misunderstandings during collaborative robotic operation.

Grounding language in robot capabilities remains essential throughout conversation. Not every requested action can be executed successfully. If users request physically impossible manipulations, inaccessible navigation goals, unavailable equipment, or unsupported robot functions, the LLM explains these limitations while suggesting feasible alternatives. Such grounded dialogue maintains realistic expectations regarding robot capabilities while supporting productive collaboration.

Dialogue also facilitates transparency and explainability. Rather than executing actions silently, robots communicate reasoning processes, execution status, environmental observations, detected anomalies, expected completion times, and recovery strategies. Users therefore remain informed regarding robot intentions, increasing trust, situational awareness, and operational confidence. Explainable dialogue becomes especially valuable within industrial, healthcare, and safety-critical environments.

Error recovery frequently depends upon conversational interaction. When autonomous recovery remains uncertain, robots initiate dialogue requesting clarification or assistance. Questions such as "I cannot locate the requested inspection tool. Has it been moved?" or "The planned route is blocked. Would you like me to take an alternative path?" enable collaborative decision making rather than unilateral autonomous behavior. LLMs naturally generate contextually appropriate recovery conversations.

Human preferences also emerge gradually through repeated dialogue. Some operators prefer concise status updates, whereas others request detailed explanations. Some prioritize speed, while others emphasize safety or inspection accuracy. Multi-turn dialogue enables LLMs to infer these interaction preferences and adapt communication style accordingly without requiring explicit configuration.

Task delegation becomes more effective through conversational planning. Rather than assigning rigid command sequences, supervisors describe objectives at varying levels of abstraction. The LLM decomposes these objectives into executable subtasks while periodically confirming major planning decisions through dialogue. This collaborative planning process combines human strategic judgment with robotic execution capability.

Industrial manufacturing provides numerous examples illustrating the importance of dialogue-based interaction. Production supervisors issue changing work orders, respond to equipment failures, adjust inspection priorities, coordinate maintenance schedules, authorize recovery actions, and monitor production progress throughout each shift. Multi-turn dialogue allows robots to participate naturally within these continuously evolving operational workflows.

Warehouse automation similarly benefits from conversational coordination. Supervisors redirect inventory transportation, prioritize urgent shipments, modify storage assignments, explain unexpected inventory discrepancies, coordinate multiple autonomous mobile robots, and resolve operational conflicts through continuous dialogue rather than isolated command execution.

Healthcare robotics requires particularly sophisticated conversational capability. Hospital personnel issue patient-specific instructions, update treatment priorities, modify delivery schedules, report changing patient conditions, clarify medication requirements, and coordinate emergency responses. LLM-based dialogue supports these sensitive interactions while maintaining awareness of patient safety, privacy requirements, medical terminology, and clinical workflows.

Inspection robotics also benefits substantially from multi-turn communication. Engineers discuss detected anomalies, request additional measurements, modify inspection procedures, review historical maintenance records, compare observations with previous inspections, authorize follow-up actions, and generate collaborative maintenance reports through ongoing dialogue with robotic inspection systems.

Construction robotics operates within rapidly changing environments where site layouts, work priorities, equipment availability, weather conditions, and human activities continuously evolve. Conversational interaction enables construction supervisors to communicate these changes naturally without repeatedly specifying complete task definitions from the beginning.

Agricultural robotics similarly benefits from long-term conversational collaboration. Farmers discuss crop conditions, weather forecasts, irrigation schedules, harvesting priorities, disease observations, fertilizer recommendations, equipment status, and seasonal planning while robots continuously adapt operational behavior according to evolving agricultural requirements.

Emotionally intelligent communication also contributes to effective collaboration. Although industrial robots need not simulate human emotions, they benefit from recognizing operator frustration, urgency, uncertainty, satisfaction, or confusion through linguistic patterns. Appropriate communication style adaptation improves usability, reduces misunderstandings, and strengthens long-term human-robot relationships.

Dialogue systems must also manage interruptions effectively. Human conversations naturally include topic changes, temporary suspensions, emergency requests, side questions, and resumed discussions. Multi-turn dialogue enables robots to pause ongoing conversational threads, address urgent issues, and later resume previous discussions without losing contextual continuity. This flexibility significantly improves operational realism.

Planning and dialogue increasingly operate as mutually supportive processes. Conversations influence planning decisions, while planning status influences subsequent conversations. Every completed task, detected obstacle, recovered failure, or environmental observation updates both execution state and dialogue context. The cognitive architecture therefore integrates language reasoning, memory, perception, planning, and execution within a continuously synchronized information model.

Small Language Models deployed on edge robots increasingly support local conversational interaction without requiring constant cloud connectivity. Lightweight onboard dialogue enables immediate responses, offline operation, privacy preservation, and deterministic latency. More computationally intensive cloud-based Large Language Models optionally provide deeper reasoning, extended planning, complex troubleshooting, multilingual translation, or comprehensive report generation whenever communication becomes available.

Retrieval systems further personalize dialogue by incorporating organization-specific terminology, operating procedures, technical documentation, maintenance records, facility layouts, robot software repositories, and enterprise knowledge bases. Rather than generating generic conversational responses, LLMs communicate using terminology and workflows familiar to local operators.

Safety remains the highest priority throughout conversational interaction. Language models never directly execute physical commands without independent verification. Motion planning, affordance estimation, collision prediction, runtime monitoring, emergency stopping, geometric validation, and deterministic control systems independently evaluate every proposed action before execution. Dialogue influences planning but does not bypass established robotic safety mechanisms.

Evaluation of multi-turn robotic dialogue differs significantly from conventional conversational benchmarks. Practical evaluation emphasizes instruction following accuracy, contextual consistency, dialogue coherence, task completion rate, clarification effectiveness, reference resolution accuracy, user satisfaction, execution success, planning adaptability, latency, robustness under interruptions, and collaborative efficiency. These embodied interaction metrics better reflect operational performance than purely linguistic measures.

Future developments increasingly integrate world models, multimodal perception, external memory, continual learning, retrieval-augmented reasoning, semantic scene understanding, affective communication, predictive planning, and collaborative multi-agent dialogue into unified cognitive architectures. Rather than functioning merely as conversational interfaces, future robotic dialogue systems will become persistent collaborative partners capable of maintaining months or years of accumulated interaction history while continuously adapting to evolving users, environments, workflows, and operational objectives.

As Physical AI advances toward fully embodied cognitive intelligence, LLM-based multi-turn dialogue will become a defining capability distinguishing intelligent collaborative robots from conventional automation. By integrating contextual memory, semantic reasoning, multimodal perception, grounded planning, explainable communication, adaptive interaction, task management, external knowledge retrieval, continual learning, and deterministic safety verification, multi-turn dialogue establishes a practical framework through which robots can collaborate naturally with humans across complex long-duration real-world tasks while maintaining both operational reliability and conversational fluency.

LLM 기반 다중 턴 대화(Multi-Turn Dialogue)는 지능형 로봇이 단순한 명령 수행 장치를 넘어 사람과 지속적으로 협업하는 인지 시스템(Cognitive System)으로 발전하기 위한 핵심 기술이다. 기존 로봇은 사용자가 한 번에 완전한 명령을 입력한다고 가정했지만, 실제 환경에서는 작업 목표가 대화를 통해 점진적으로 구체화된다. 사람은 작업 중간에 새로운 요구사항을 추가하고, 우선순위를 변경하며, 질문을 하고, 환경 변화에 따라 계획을 수정한다. LLM은 이러한 연속적인 대화를 이해하고 작업을 지속적으로 조정할 수 있다.

사람의 대화는 항상 문맥(Context)에 의존한다. 예를 들어 "생산 라인을 점검해."라는 명령 이후 "포장 구역부터 시작하고.", "고장 난 컨베이어는 제외해.", "온도 센서를 집중적으로 확인해.", "점검 후 보고서를 작성해."와 같은 추가 명령이 이어질 수 있다. 로봇은 각각의 명령을 독립적으로 처리하는 것이 아니라 이전 대화와 연결하여 하나의 작업(Task)으로 통합해야 한다.

단일 명령 처리(Single-Turn Command)와 달리 다중 턴 대화는 지속적인 문맥 관리(Context Management)가 필요하다. 예를 들어 사용자가 "검사 카메라를 가져와."라고 말한 후 "그것을 충전기에 올려놔."라고 말하면 "그것(It)"이 검사 카메라를 의미한다는 것을 이해해야 한다. 이러한 대명사(Pronoun) 해석과 참조 해결(Reference Resolution)은 LLM의 중요한 장점 중 하나이다.

대화 메모리(Dialogue Memory)는 단순한 대화 기록만 저장하는 것이 아니다. 로봇은 작업 진행 상태(Task Progress), 물체 위치(Object Location), 센서 정보(Sensor Data), 배터리 상태(Battery Status), 사람의 위치(Human Position), 환경 변화(Environment Change) 등을 함께 기억한다. 따라서 대화는 언어 정보와 실제 환경 정보를 동시에 이용하여 진행된다.

작업 중심 대화(Task-Oriented Dialogue)는 일반 챗봇(Chatbot)과 다르다. 일반 대화에서는 정보 전달이 목적이지만, 로봇 대화에서는 사용자의 대화가 실제 이동(Navigation), 물체 조작(Manipulation), 검사(Inspection), 작업 순서(Task Scheduling)에 직접 영향을 준다. 따라서 생성되는 모든 응답은 실제 실행 가능성과 안전성(Safety)을 반드시 고려해야 한다.

다중 턴 대화의 첫 번째 단계는 사용자 의도(Intent Understanding)를 정확히 이해하는 것이다. 사람은 종종 불완전하거나 모호한 명령을 사용한다. 예를 들어 "장비를 옮겨."라는 명령만으로는 어떤 장비인지, 어디로 이동해야 하는지 알 수 없다. LLM은 즉시 작업을 수행하지 않고 "어떤 장비를 말씀하시나요?", "어디로 옮길까요?"와 같은 질문을 통해 필요한 정보를 확인한다.

이러한 명확화 대화(Clarification Dialogue)는 기존 로봇과 가장 큰 차이점이다. 기존 시스템은 정확한 명령이 입력되지 않으면 오류를 발생시키지만, LLM은 사람과 자연스럽게 대화하여 부족한 정보를 스스로 수집한다. 이는 사람과 사람 사이의 협업 방식과 매우 유사하다.

다중 턴 대화는 작업 변경(Task Modification)에도 매우 유용하다. 예를 들어 로봇이 검사 장비를 운반하는 중 사용자가 긴급 작업을 먼저 수행하도록 요청하면 LLM은 현재까지 완료된 작업, 남은 작업, 우선순위, 이동 거리 등을 종합적으로 분석하여 새로운 계획을 생성한다. 기존 작업을 단순히 취소하는 것이 아니라 가장 효율적인 방식으로 수정한다.

장시간 작업(Long-Horizon Task)에서는 다중 턴 대화가 더욱 중요하다. 생산 설비 점검, 창고 재고 관리, 병원 배송, 농업 모니터링, 건설 현장 관리 등은 수 시간 이상 지속될 수 있다. 작업 도중 관리자는 진행 상황을 확인하고 새로운 작업을 추가하거나 우선순위를 변경한다. LLM은 이러한 긴 작업 동안 전체 문맥을 유지하면서 자연스럽게 협업한다.

대화 메모리는 여러 계층으로 구성된다. 단기 메모리(Short-Term Memory)는 최근 대화를 기억하고, 작업 메모리(Working Memory)는 현재 수행 중인 작업과 미해결 질문을 저장한다. 에피소드 메모리(Episodic Memory)는 이전 작업 경험을 저장하며, 의미 메모리(Semantic Memory)는 시설 구조, 장비 정보, 작업 절차 등을 장기적으로 기억한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 대화 품질을 크게 향상시킨다. 사용자가 설비 매뉴얼, 유지보수 절차, 검사 기록, 운영 규정 등에 대해 질문하면 로봇은 내부 지식뿐 아니라 유지보수 문서(Maintenance Manual), 디지털 트윈(Digital Twin), 설비 데이터베이스(Database), 작업 이력(History)을 검색하여 보다 정확한 답변을 생성한다.

멀티모달 인식(Multimodal Perception)은 대화를 더욱 풍부하게 만든다. RGB 카메라(RGB Camera), 깊이 센서(Depth Sensor), 라이다(LiDAR), 촉각 센서(Tactile Sensor), 힘 센서(Force Sensor), 위치 추정(Localization), 객체 탐지(Object Detection)를 이용하여 현재 환경을 이해한 후 "통로를 막고 있는 것이 무엇인가?", "밸브에서 아직 누수가 발생하는가?"와 같은 질문에 실제 환경을 반영한 답변을 제공할 수 있다.

비전-언어 모델(VLM, Vision-Language Model)은 장면(Scene)을 자연어로 설명한다. 예를 들어 "작업대 아래에 공구함이 있으며, 오른쪽에 사람이 서 있다."와 같은 정보를 생성하면 LLM은 이를 현재 대화의 일부로 활용하여 더욱 정확한 작업 계획과 응답을 생성한다.

참조 해결(Reference Resolution)은 매우 중요한 기능이다. 사람은 "저 상자", "지난번 팔레트", "왼쪽 선반", "고장 난 기계"와 같은 표현을 자주 사용한다. LLM은 이전 대화와 현재 환경, 공간 정보(Spatial Information), 작업 기록(Task History)을 종합하여 정확한 대상을 찾아낸다.

모든 대화는 실제 로봇의 능력(Capability)에 기반해야 한다. 사용자가 현재 로봇이 수행할 수 없는 작업을 요청하면 LLM은 불가능한 이유를 설명하고 가능한 대안을 제시한다. 예를 들어 작업 범위를 벗어난 물체를 요청하면 접근 가능한 위치로 이동해 달라고 요청하거나 사람의 도움을 제안할 수 있다.

다중 턴 대화는 설명 가능성(Explainability)도 향상시킨다. 로봇은 현재 어떤 작업을 수행하고 있는지, 왜 이러한 경로를 선택했는지, 얼마나 시간이 걸리는지, 어떤 문제가 발생했는지를 사람에게 자연스럽게 설명할 수 있다. 이는 사람의 신뢰(Trust)를 높이고 협업 효율을 향상시킨다.

오류 복구(Error Recovery) 과정에서도 대화는 중요한 역할을 한다. 목표 물체를 찾지 못하거나 경로가 차단된 경우 로봇은 "검사 장비가 현재 위치에 없습니다. 다른 곳으로 이동했습니까?", "현재 통로가 막혀 있습니다. 우회 경로를 사용할까요?"와 같이 사람과 협의하여 새로운 작업 계획을 생성한다.

LLM은 반복적인 대화를 통해 사용자의 선호도(User Preference)도 학습할 수 있다. 어떤 사용자는 간단한 보고를 선호하고, 어떤 사용자는 상세한 설명을 요구한다. 어떤 사용자는 속도를 우선하고, 어떤 사용자는 안전을 더 중요하게 생각한다. LLM은 이러한 대화 패턴을 학습하여 개인 맞춤형 인터페이스를 제공할 수 있다.

작업 위임(Task Delegation)도 자연스럽게 이루어진다. 사용자는 세부 명령을 일일이 입력하지 않고 목표만 제시하면 된다. LLM은 이를 여러 개의 하위 작업(Subtask)으로 분해하고 중요한 단계마다 사용자와 대화하여 계획을 확인하면서 작업을 수행한다.

산업용 제조 환경에서는 생산 계획 변경, 설비 고장, 유지보수 일정, 검사 우선순위 등이 계속 바뀐다. LLM은 관리자의 새로운 지시를 대화를 통해 이해하고 기존 작업 계획을 수정하면서 생산성을 유지한다.

물류 창고(Warehouse)에서는 긴급 배송(Urgent Delivery), 재고 위치 변경(Inventory Relocation), 선반 변경(Storage Assignment), 다수의 AMR 협업(Fleet Coordination) 등이 자주 발생한다. 다중 턴 대화는 이러한 지속적인 운영 변경을 자연스럽게 지원한다.

의료 로봇(Healthcare Robot)은 환자 상태 변화, 약품 배송, 치료 우선순위 변경, 의료진의 추가 지시 등을 대화를 통해 처리한다. 의료 용어(Medical Terminology), 개인정보 보호(Privacy), 환자 안전(Safety)을 함께 고려하여 신뢰성 높은 협업을 수행한다.

시설 점검 로봇(Inspection Robot)은 이상 현상(Anomaly), 추가 측정 요청, 과거 점검 기록 비교, 유지보수 계획 등을 엔지니어와 지속적으로 대화하면서 점검 품질을 높일 수 있다.

건설 현장(Construction Site)은 구조물 설치, 장비 이동, 작업 순서 변경, 날씨 변화 등이 매우 빈번하다. 다중 턴 대화는 이러한 환경 변화에 맞추어 작업 계획을 지속적으로 수정하는 데 큰 장점을 가진다.

농업 로봇(Agricultural Robot)은 작물 상태(Crop Condition), 날씨(Weather), 관개(Irrigation), 병충해(Disease), 수확 계획(Harvesting Plan)을 농부와 지속적으로 대화하면서 작업을 수행할 수 있다.

감정 인식(Affective Communication)도 향후 중요한 기술이다. 산업용 로봇이 사람의 감정을 흉내 낼 필요는 없지만, 사용자의 긴급함(Urgency), 혼란(Confusion), 불만(Frustration), 만족(Satisfaction)을 언어적으로 이해하고 적절한 방식으로 응답하면 협업 품질이 크게 향상된다.

대화는 중간에 끊기거나 다른 주제로 전환될 수도 있다. LLM은 긴급 질문에 먼저 답변한 후 이전 작업 대화로 자연스럽게 돌아올 수 있으며, 여러 개의 대화 흐름을 동시에 관리할 수 있다. 이러한 능력은 실제 업무 환경에서 매우 중요하다.

작업 계획(Task Planning)과 대화(Dialogue)는 서로 긴밀하게 연결된다. 새로운 대화는 작업 계획을 수정하며, 작업 진행 상황은 다시 대화의 문맥이 된다. 따라서 언어(Language), 메모리(Memory), 인식(Perception), 계획(Planning), 실행(Execution)은 하나의 통합 인지 시스템으로 동작한다.

최근에는 엣지 로봇(Edge Robot)에 탑재되는 소형 언어 모델(SLM, Small Language Model)이 다중 턴 대화를 수행하기 시작하였다. 로컬에서 즉시 응답하고, 개인정보를 보호하며, 네트워크가 없는 환경에서도 대화를 유지할 수 있다. 복잡한 전략 수립이나 긴 문서 생성은 필요할 때만 클라우드 LLM이 담당하는 계층형 AI 구조가 일반화되고 있다.

검색 시스템(RAG)은 기업별 용어(Terminology), 운영 절차(Standard Operating Procedure), 설비 문서(Document), 소프트웨어 저장소(Repository), 시설 지도(Map)를 대화에 통합하여 조직 맞춤형 응답을 생성한다. 따라서 일반적인 답변이 아니라 실제 기업 환경에 맞는 전문적인 대화를 제공할 수 있다.

그러나 안전(Safety)은 항상 최우선이다. LLM의 대화가 직접 로봇을 제어하지는 않는다. 모션 플래너(Motion Planner), 충돌 예측(Collision Prediction), 어포던스 분석(Affordance Estimation), 런타임 모니터(Runtime Monitor), 비상 정지(Emergency Stop), 안전 제어기(Safety Controller)가 모든 행동을 검증한 후 실제 실행을 허가한다.

다중 턴 대화의 평가는 일반 챗봇과 다르다. 문맥 유지(Context Consistency), 참조 해결(Reference Resolution), 명령 수행 성공률(Task Success Rate), 명확화 대화(Clarification Effectiveness), 사용자 만족도(User Satisfaction), 응답 지연(Response Latency), 작업 연속성(Task Continuity), 협업 효율(Collaboration Efficiency) 등을 종합적으로 평가한다.

향후에는 세계 모델(World Model), 멀티모달 인식(Multimodal Perception), 외부 메모리(External Memory), 지속적 학습(Continual Learning), 검색 증강 생성(RAG), 장면 이해(Scene Understanding), 감정 기반 상호작용(Affective Interaction), 예측 계획(Predictive Planning), 다중 에이전트 협업(Multi-Agent Collaboration)이 모두 통합된 차세대 대화형 물리 AI(Physical AI)가 등장할 것으로 예상된다.

결국 LLM 기반 다중 턴 대화는 단순한 음성 인터페이스를 넘어 사람과 로봇이 장기간 함께 협업하기 위한 핵심 기술이다. 문맥(Context), 메모리(Memory), 의미 추론(Semantic Reasoning), 멀티모달 인식(Multimodal Perception), 작업 계획(Task Planning), 설명 가능성(Explainability), 검색 증강 생성(RAG), 지속적 학습(Continual Learning), 안전 검증(Safety Verification)을 하나의 통합 인지 시스템으로 결합함으로써, 실제 산업·물류·의료·건설·농업 환경에서 사람과 자연스럽게 협력하는 차세대 물리 AI의 핵심 기반 기술로 발전하고 있다.

##  

## 3.10 Real-Time LLM Inference Optimization (with Code)

![](images/image10.png){width="7.268055555555556in" height="7.268055555555556in"}

![](images/image11.png){width="7.268055555555556in" height="7.268055555555556in"}

Real-time inference optimization has become one of the most critical research topics in modern Physical AI because the practical deployment of Large Language Models (LLMs) in robotic systems depends not only on reasoning quality but also on deterministic execution speed, predictable latency, efficient resource utilization, and continuous responsiveness. While recent foundation models demonstrate extraordinary capabilities in natural language understanding, planning, reasoning, and multimodal cognition, their computational complexity presents significant challenges for autonomous robots operating under strict power, thermal, memory, and timing constraints. Unlike cloud-based conversational systems that may tolerate response delays of several seconds, autonomous robots continuously interact with dynamic physical environments where delayed reasoning can directly influence safety, navigation accuracy, manipulation success, and human-robot collaboration. Consequently, inference optimization has become a foundational technology enabling the integration of advanced language intelligence into real-time robotic platforms.

Real-time robotics differs fundamentally from traditional cloud artificial intelligence. In cloud environments, language models typically process isolated requests independently, often utilizing large GPU clusters capable of accommodating substantial computational workloads. Robotic systems instead execute multiple concurrent workloads continuously throughout operation. Localization, perception, obstacle avoidance, motion planning, sensor fusion, control loops, navigation, mapping, object recognition, manipulation planning, communication, diagnostics, safety monitoring, and language reasoning all compete for limited onboard computational resources. Efficient inference therefore requires careful coordination between language processing and every other subsystem operating within the robotic architecture.

Latency represents one of the most important optimization objectives. In robotics, latency is not merely an inconvenience but frequently determines operational safety and task success. A robot receiving delayed reasoning output may continue following obsolete plans after environmental conditions have changed. Humans may enter the robot\'s workspace, obstacles may appear unexpectedly, target objects may move, communication may change task priorities, or emergency conditions may require immediate response. Minimizing end-to-end inference latency therefore directly improves robot adaptability and safety.

Inference latency consists of multiple sequential stages. Natural language instructions must first be tokenized before entering the transformer architecture. Embedding generation converts discrete tokens into continuous representations processed through attention mechanisms and feed-forward layers. Decoder operations generate output tokens autoregressively, followed by detokenization and response interpretation by downstream planning modules. Each stage contributes to overall response delay. Effective optimization therefore targets the complete inference pipeline rather than isolated computational kernels.

Throughput represents another critical performance metric. Modern robotic systems frequently process multiple simultaneous language interactions originating from human operators, autonomous planning modules, perception summaries, monitoring systems, maintenance diagnostics, and fleet coordination services. High-throughput inference enables the robotic cognitive architecture to handle concurrent reasoning tasks without introducing unacceptable scheduling delays or computational bottlenecks.

Deterministic execution distinguishes robotic inference from conventional language processing. Human users may tolerate variable response times during casual conversation, whereas robotic control systems require predictable execution timing. Motion planners, safety controllers, sensor fusion algorithms, and scheduling systems depend upon consistent computational behavior. Consequently, inference optimization emphasizes bounded latency rather than merely maximizing average computational performance.

Model architecture strongly influences inference efficiency. Transformer-based language models perform extensive matrix multiplications, attention computations, normalization operations, activation functions, and memory transfers throughout each decoding step. While these architectures achieve remarkable reasoning performance, they require careful optimization for deployment within embedded robotic computing platforms. Architectural modifications reducing computational complexity while preserving reasoning capability therefore represent active research areas.

Model compression constitutes one of the most widely adopted optimization strategies. Quantization reduces numerical precision from floating-point representations to lower-bit integer formats such as INT8, INT6, INT4, or mixed-precision representations. Quantized models consume significantly less memory, reduce bandwidth requirements, improve cache utilization, and accelerate tensor computations while preserving acceptable reasoning quality for robotic applications. Modern hardware accelerators increasingly provide native support for low-precision inference, making quantization particularly effective for edge deployment.

Knowledge distillation further enhances inference efficiency by transferring reasoning capability from large teacher models into compact student models. Rather than deploying extremely large foundation models directly onboard robots, smaller distilled models reproduce much of the teacher\'s reasoning behavior while requiring substantially fewer computational resources. This approach significantly improves inference speed without requiring complete redesign of downstream robotic software.

Structured pruning removes redundant parameters from transformer networks. Many attention heads, neurons, feed-forward dimensions, and intermediate activations contribute minimally to final prediction quality. Eliminating these unnecessary components reduces computational complexity, memory consumption, and execution latency while maintaining robust reasoning performance. Structured pruning generally produces more efficient hardware execution than unstructured sparsity because optimized kernels exploit regular computational patterns.

Attention optimization represents another major research direction. Conventional self-attention exhibits quadratic computational complexity with respect to sequence length, making long-context reasoning increasingly expensive. Numerous optimized attention mechanisms reduce computational complexity using sparse attention, local attention, grouped attention, sliding windows, linear approximations, or memory compression. These methods significantly improve inference performance for robotic dialogue, planning, and contextual reasoning tasks involving extended interaction histories.

Caching mechanisms substantially reduce repeated computation during autoregressive decoding. Key-value caches preserve intermediate attention representations generated during previous decoding steps, allowing subsequent tokens to reuse existing computations rather than recomputing entire attention matrices repeatedly. Efficient cache management significantly accelerates sequential response generation while reducing processor utilization.

Prompt optimization also contributes to inference efficiency. Rather than repeatedly providing lengthy contextual descriptions, robotic systems organize prompts hierarchically, separating persistent system instructions, environmental context, task objectives, safety constraints, and temporary conversational information. Carefully structured prompts minimize unnecessary token processing while preserving essential contextual knowledge.

Retrieval-Augmented Generation further reduces computational burden. Instead of embedding extensive factual knowledge directly within model parameters or repeatedly including large contextual prompts, robots retrieve maintenance manuals, semantic maps, operating procedures, equipment documentation, digital twins, inspection histories, software documentation, and enterprise knowledge bases only when necessary. Retrieval therefore decreases prompt length while simultaneously improving factual accuracy.

Memory hierarchy optimization plays an equally important role. Modern GPUs contain multiple memory levels including registers, shared memory, cache hierarchies, high-bandwidth memory, and host memory. Efficient placement of model parameters, activation tensors, attention caches, and intermediate computations minimizes costly memory transfers while maximizing arithmetic throughput. Memory bandwidth frequently becomes a greater performance limitation than raw computational capability during transformer inference.

Batch scheduling further improves hardware utilization. Robotic systems frequently execute multiple language requests simultaneously originating from dialogue management, task planning, perception reasoning, diagnostics, reporting, and fleet communication. Intelligent scheduling groups compatible inference requests into efficient execution batches while respecting individual latency requirements. Adaptive schedulers balance throughput against response time according to current operational priorities.

Pipeline parallelism enables different hardware components to process distinct inference stages concurrently. Tokenization, embedding generation, transformer execution, decoding, and output interpretation overlap through carefully coordinated execution pipelines. Such parallel processing significantly increases overall system responsiveness without increasing hardware resources proportionally.

Hardware acceleration remains indispensable for real-time robotic inference. Modern robotic computing platforms increasingly integrate GPUs, Neural Processing Units (NPUs), Tensor Processing Units (TPUs), dedicated AI accelerators, FPGA-based inference engines, tensor cores, and specialized transformer processors. These hardware architectures accelerate matrix operations, attention computation, low-precision arithmetic, and memory movement while reducing power consumption relative to general-purpose processors.

Edge robotic platforms such as NVIDIA Jetson Orin, Jetson Thor, Intel Core Ultra AI processors, AMD Ryzen AI processors, Qualcomm Robotics platforms, and custom embedded AI modules increasingly provide heterogeneous computing architectures combining CPUs, GPUs, NPUs, and dedicated inference engines. Effective inference optimization therefore includes intelligent workload distribution across heterogeneous computational resources according to each processor\'s strengths.

Power optimization directly influences robotic operational endurance. Autonomous mobile robots typically operate from battery-powered electrical systems where computational energy competes with locomotion, sensing, communication, lighting, environmental monitoring, and manipulation hardware. Efficient inference minimizes energy consumption per generated token, extending mission duration without sacrificing cognitive capability.

Thermal management similarly affects long-duration deployment. Continuous transformer inference generates substantial heat within embedded computing platforms. Excessive thermal accumulation triggers frequency throttling, reducing computational performance precisely when sustained reasoning becomes necessary. Thermal-aware scheduling, dynamic frequency scaling, workload balancing, and adaptive inference strategies therefore maintain stable long-term performance.

Adaptive inference has emerged as a particularly promising optimization approach. Rather than executing identical computational workloads for every request, robots dynamically adjust reasoning depth according to task complexity. Simple confirmation responses require minimal inference, whereas complex mission planning, failure recovery, or long-horizon reasoning activate more sophisticated computational pathways. Adaptive computation significantly reduces average inference cost while preserving high reasoning quality when necessary.

Hierarchical language architectures further improve efficiency. Lightweight Small Language Models continuously execute onboard, handling routine dialogue, task interpretation, local planning, and rapid decision making. Larger cloud-hosted models perform strategic reasoning, software development, complex diagnostics, and extensive report generation whenever communication infrastructure permits. This hierarchical approach balances responsiveness, capability, computational efficiency, and operational reliability.

Task-specific specialization reduces unnecessary computation. Rather than deploying a single universal language model responsible for every cognitive function, robotic architectures increasingly employ specialized reasoning modules optimized for navigation, manipulation, inspection, maintenance, human interaction, diagnostics, reporting, or software assistance. Specialized models frequently achieve superior latency while maintaining domain-specific reasoning quality.

Multimodal integration also benefits inference optimization. Vision encoders, semantic mapping systems, object detectors, localization modules, force sensing, tactile perception, and environmental monitoring preprocess raw sensory information before language reasoning begins. Providing structured semantic representations rather than raw sensor data significantly reduces language model computational requirements while improving reasoning quality.

Planning architectures increasingly separate symbolic reasoning from language reasoning. Deterministic planners efficiently compute geometric trajectories, collision-free paths, inverse kinematics, and motion optimization, while LLMs focus upon semantic interpretation, task decomposition, dialogue management, contextual reasoning, and high-level planning. This division of responsibility prevents expensive language inference from performing computations better suited to conventional robotics algorithms.

Continuous inference scheduling becomes essential within multitasking robotic systems. Language reasoning cannot monopolize computational resources because perception, localization, control, and safety monitoring operate under strict real-time deadlines. Modern robotic operating systems therefore employ priority-aware schedulers coordinating language inference alongside deterministic control tasks while guaranteeing timing requirements for safety-critical processes.

Cloud-edge collaboration provides another optimization strategy. Immediate local reasoning executes entirely onboard using compressed language models, whereas computationally intensive planning, retraining, fleet coordination, digital twin simulation, and large-scale knowledge retrieval execute within cloud infrastructure whenever communication becomes available. Local autonomy remains preserved during network outages while cloud resources augment cognitive capability opportunistically.

Fine-tuning substantially improves inference efficiency indirectly. Domain-specific language models require fewer reasoning steps because they understand robotic terminology, operational procedures, software interfaces, and environmental semantics more naturally than general-purpose models. Specialized knowledge therefore reduces prompt complexity, reasoning depth, and generated output length while improving overall response quality.

Continual learning further enhances efficiency over time. Robots accumulate operational experience describing frequently encountered tasks, environmental conditions, maintenance procedures, dialogue patterns, recovery strategies, and user preferences. Personalized adaptation gradually reduces computational effort required for routine interactions because previously learned patterns become increasingly familiar to the optimized model.

Simulation environments provide valuable platforms for optimization research. High-fidelity digital twins enable evaluation of latency, throughput, energy consumption, scheduling efficiency, thermal behavior, and recovery performance under diverse operational scenarios without risking physical hardware. Simulation supports systematic optimization before deployment within safety-critical industrial environments.

Industrial manufacturing particularly benefits from optimized inference. Production lines require deterministic cycle times, continuous inspection, maintenance coordination, quality assurance, and operator interaction. Efficient language inference enables robots to generate reports, interpret work orders, answer technical questions, coordinate production activities, and recover from failures without disrupting manufacturing throughput.

Warehouse automation similarly depends upon responsive cognitive processing. Inventory updates, transportation requests, fleet coordination, charging management, obstacle handling, and order fulfillment continuously generate language reasoning tasks. Optimized inference maintains low latency despite heavy operational workloads while preserving energy efficiency throughout extended autonomous operation.

Healthcare robotics introduces additional timing constraints because patient interaction frequently requires immediate conversational response while simultaneously maintaining navigation, medication delivery, monitoring, and safety supervision. Efficient inference ensures natural communication without compromising real-time robotic behavior.

Inspection robotics operating across industrial plants, infrastructure, utilities, construction sites, and agricultural environments similarly require rapid reasoning under challenging computational conditions. Robots continuously analyze anomalies, interpret sensor observations, retrieve maintenance documentation, generate reports, and respond to operator questions while maintaining autonomous navigation and environmental awareness.

Safety remains paramount throughout inference optimization. Faster reasoning must never compromise correctness, reliability, or deterministic verification. Motion planning, collision prediction, affordance estimation, runtime monitoring, emergency stopping, geometric validation, and rule-based safety controllers continue independently verifying every proposed robotic action regardless of language model inference speed. Optimization therefore complements rather than replaces classical safety architectures.

Evaluation of inference optimization extends beyond conventional language benchmarks. Practical robotic metrics include end-to-end latency, first-token latency, tokens per second, throughput, energy consumption, memory utilization, thermal stability, deterministic response timing, task completion rate, safety compliance, hardware utilization, recovery responsiveness, and long-duration operational reliability. These embodied performance measures more accurately reflect deployment readiness than isolated computational benchmarks.

Future Physical AI systems are expected to integrate adaptive transformer architectures, multimodal perception, hierarchical reasoning, retrieval-augmented knowledge, external memory, world models, predictive scheduling, heterogeneous hardware acceleration, continual learning, and dynamic computational scaling into unified cognitive frameworks. Rather than executing fixed computational pipelines, future robotic language systems will continuously optimize inference according to environmental complexity, operational urgency, available computational resources, battery status, thermal conditions, and mission objectives.

As intelligent autonomous robots become increasingly prevalent across manufacturing, logistics, healthcare, agriculture, infrastructure inspection, construction, domestic assistance, and collaborative industrial environments, LLM inference optimization will remain one of the defining technologies enabling practical real-time embodied intelligence. By combining model compression, hardware acceleration, adaptive computation, hierarchical reasoning, multimodal integration, efficient memory management, retrieval-augmented knowledge, cloud-edge collaboration, deterministic scheduling, and rigorous safety verification, optimized language inference establishes the computational foundation necessary for deploying advanced cognitive capabilities within resource-constrained robotic platforms operating safely and efficiently in complex real-world environments.

실시간 로봇(Real-Time Robot)을 위한 대규모 언어 모델(LLM, Large Language Model) 추론 최적화(Inference Optimization)는 현대 물리 AI(Physical AI)의 핵심 기술이다. 최신 LLM은 뛰어난 자연어 이해(Natural Language Understanding), 추론(Reasoning), 계획(Planning) 능력을 제공하지만, 연산량과 메모리 요구량이 매우 크기 때문에 배터리 기반 자율주행 로봇에 그대로 적용하기는 어렵다. 따라서 제한된 전력(Power), 메모리(Memory), 발열(Thermal Budget), 실시간성(Real-Time)을 만족하면서도 높은 추론 성능을 유지하는 최적화 기술이 반드시 필요하다.

실시간 로봇은 클라우드 AI와 근본적으로 다른 요구사항을 가진다. 클라우드에서는 하나의 요청을 수 초 동안 처리해도 문제가 없지만, 로봇은 자율주행(Navigation), 위치 추정(Localization), 센서 융합(Sensor Fusion), 충돌 회피(Collision Avoidance), 작업 계획(Task Planning), 제어(Control), 안전 모니터링(Safety Monitoring)을 동시에 수행해야 한다. 따라서 언어 모델은 다른 모든 시스템과 연산 자원을 공유하면서도 일정한 응답 시간을 유지해야 한다.

지연 시간(Latency)은 가장 중요한 성능 지표이다. 로봇은 움직이는 사람, 갑작스러운 장애물, 이동하는 물체, 긴급 작업 요청 등 변화하는 환경에 즉시 대응해야 한다. 언어 모델의 응답이 늦어지면 이미 환경이 바뀐 후 잘못된 계획을 실행할 가능성이 높아진다. 따라서 추론 속도는 단순한 성능 문제가 아니라 안전(Safety)과 직접 연결된다.

LLM 추론은 여러 단계로 이루어진다. 먼저 사용자의 문장을 토큰(Token)으로 분리하고(Tokenization), 이를 임베딩(Embedding) 벡터로 변환한 후, 트랜스포머(Transformer)의 어텐션(Attention)과 피드포워드 네트워크(Feed-Forward Network)를 통과한다. 이후 생성된 토큰을 다시 문장으로 변환(Detokenization)하여 작업 계획기로 전달한다. 따라서 전체 파이프라인(Pipeline)을 최적화해야 실질적인 응답 속도를 높일 수 있다.

처리량(Throughput)도 중요한 요소이다. 실제 로봇은 사람과의 대화뿐 아니라 작업 계획, 센서 분석, 상태 보고, 자율주행, 진단 시스템 등 여러 모듈이 동시에 언어 모델을 사용할 수 있다. 따라서 여러 요청을 동시에 빠르게 처리할 수 있는 높은 처리량이 요구된다.

실시간 로봇에서는 결정적 실행(Deterministic Execution)이 중요하다. 평균 응답 속도가 빠른 것보다 항상 일정한 시간 안에 응답하는 것이 더 중요하다. 제어기(Control Loop), 모션 플래너(Motion Planner), 안전 시스템(Safety System)은 예측 가능한 실행 시간을 요구하므로 추론 시간의 변동성을 최소화해야 한다.

트랜스포머(Transformer) 구조는 대량의 행렬 곱(Matrix Multiplication), 어텐션 계산, 활성화 함수(Activation Function), 메모리 접근을 반복 수행한다. 이러한 구조는 뛰어난 성능을 제공하지만 연산량이 매우 크므로 엣지 로봇(Edge Robot)에 맞게 최적화하는 연구가 활발히 진행되고 있다.

모델 압축(Model Compression)은 가장 널리 사용되는 최적화 방법이다. 양자화(Quantization)는 FP16이나 FP32 대신 INT8, INT6, INT4와 같은 저정밀도 정수(Integer)를 사용하여 메모리 사용량과 연산량을 크게 줄인다. 최신 GPU와 AI 가속기(AI Accelerator)는 이러한 저정밀 연산을 하드웨어 수준에서 지원하여 추론 속도를 크게 향상시킨다.

지식 증류(Knowledge Distillation)는 대형 모델(Teacher Model)의 지식을 작은 학생 모델(Student Model)에게 전달하는 기술이다. 학생 모델은 훨씬 적은 파라미터(Parameter)만 사용하면서도 대부분의 추론 능력을 유지할 수 있어 실시간 로봇에 매우 적합하다.

가지치기(Pruning)는 중요하지 않은 뉴런(Neuron), 어텐션 헤드(Attention Head), 피드포워드 계층을 제거하여 모델을 경량화한다. 불필요한 계산을 줄이면서도 핵심 추론 능력을 유지할 수 있어 연산 효율이 크게 향상된다.

어텐션 최적화(Attention Optimization)도 중요한 연구 분야이다. 일반적인 셀프 어텐션(Self-Attention)은 입력 길이가 길어질수록 연산량이 제곱으로 증가한다. 이를 해결하기 위해 희소 어텐션(Sparse Attention), 로컬 어텐션(Local Attention), 슬라이딩 윈도우(Sliding Window), 선형 어텐션(Linear Attention) 등이 개발되어 긴 문맥(Long Context)에서도 빠른 추론이 가능해지고 있다.

키-값 캐시(KV Cache, Key-Value Cache)는 자동회귀(Auto-Regressive) 생성 과정에서 이전 계산 결과를 저장하여 동일한 계산을 반복하지 않도록 한다. 이를 통해 응답 생성 속도를 크게 높이고 GPU 사용률도 감소시킬 수 있다.

프롬프트 최적화(Prompt Optimization)는 불필요한 토큰(Token)을 줄이는 기술이다. 시스템 프롬프트(System Prompt), 환경 정보(Environment Context), 작업 목표(Task Goal), 안전 규칙(Safety Rule)을 계층적으로 구성하여 반복되는 정보를 최소화하면 전체 추론 시간이 감소한다.

검색 증강 생성(RAG, Retrieval-Augmented Generation)은 모델 내부에 모든 정보를 저장하지 않고 필요할 때만 유지보수 문서(Maintenance Manual), 디지털 트윈(Digital Twin), 작업 절차(Standard Operating Procedure), 설비 데이터베이스(Database)를 검색하여 사용한다. 이는 프롬프트 길이를 줄이고 추론 정확도도 동시에 향상시킨다.

메모리 계층(Memory Hierarchy) 최적화도 매우 중요하다. GPU 레지스터(Register), 공유 메모리(Shared Memory), 캐시(Cache), 고대역폭 메모리(HBM), 시스템 메모리(System Memory)를 효율적으로 사용하여 메모리 이동을 최소화하면 GPU 계산 성능을 극대화할 수 있다.

배치 스케줄링(Batch Scheduling)은 여러 개의 언어 요청을 동시에 처리하여 GPU 활용도를 높인다. 다만 로봇에서는 응답 속도가 중요하므로 배치 크기(Batch Size)와 응답 지연 사이의 균형을 유지해야 한다.

파이프라인 병렬화(Pipeline Parallelism)는 토큰화(Tokenization), 임베딩(Embedding), 트랜스포머 계산, 출력 생성 등을 동시에 수행하여 전체 응답 시간을 단축한다. 여러 하드웨어가 각 단계를 동시에 수행함으로써 효율을 크게 높일 수 있다.

하드웨어 가속(Hardware Acceleration)은 실시간 추론의 핵심이다. GPU, NPU(Neural Processing Unit), TPU(Tensor Processing Unit), FPGA(Field Programmable Gate Array), Tensor Core와 같은 AI 전용 하드웨어는 트랜스포머 연산을 일반 CPU보다 훨씬 빠르고 전력 효율적으로 수행한다.

현대의 엣지 AI 플랫폼은 CPU, GPU, NPU를 함께 사용하는 이기종 컴퓨팅(Heterogeneous Computing)을 채택하고 있다. 각 프로세서의 특성에 맞게 작업을 분배하면 전체 시스템 성능을 극대화할 수 있다.

전력 최적화(Power Optimization)는 이동 로봇에서 매우 중요하다. AI 추론이 소비하는 전력이 증가하면 모터(Motor), 센서(Sensor), 통신 장치(Communication), 조명(Lighting)에 사용할 수 있는 전력이 감소한다. 따라서 토큰당 에너지 소비(Energy per Token)를 최소화하는 것이 장시간 운용(Long Mission)의 핵심이다.

발열 관리(Thermal Management)도 중요한 문제이다. 지속적인 LLM 추론은 많은 열을 발생시키며, 일정 온도를 초과하면 GPU는 자동으로 클럭(Clock)을 낮추는 스로틀링(Thermal Throttling)을 수행한다. 따라서 작업 스케줄링과 동적 주파수 조절(Dynamic Frequency Scaling)을 이용하여 장시간 안정적인 성능을 유지해야 한다.

적응형 추론(Adaptive Inference)은 최근 가장 주목받는 기술이다. 단순한 질문에는 간단한 추론만 수행하고, 복잡한 작업 계획이나 오류 복구가 필요한 경우에만 깊은 추론을 수행한다. 이를 통해 평균 연산량을 크게 줄일 수 있다.

계층형 언어 모델(Hierarchical Language Architecture)은 실시간 로봇에 매우 적합하다. 로컬의 소형 언어 모델(SLM, Small Language Model)은 즉각적인 대화와 작업 계획을 수행하고, 클라우드의 대형 언어 모델(LLM)은 장기 전략(Long-Horizon Planning), 소프트웨어 개발, 복잡한 진단을 담당한다. 이 구조는 성능과 응답 속도를 동시에 만족시킨다.

작업별 특화 모델(Task-Specific Model)도 효율성을 높인다. 하나의 범용 모델 대신 자율주행, 조작, 검사, 유지보수, 사람과의 대화 등 기능별로 특화된 모델을 사용하면 추론 시간이 감소하고 정확도도 향상된다.

멀티모달(Multimodal) 처리 역시 최적화에 기여한다. 비전 인코더(Vision Encoder), 객체 탐지(Object Detection), 의미 지도(Semantic Map), 위치 추정(Localization) 등이 먼저 환경을 분석하고, LLM은 이미 구조화된 정보를 이용하여 추론하기 때문에 전체 계산량이 감소한다.

작업 계획(Task Planning)은 기하학적 경로 계산과 의미 추론을 분리한다. 경로 생성(Path Planning), 충돌 검사(Collision Checking), 역기구학(Inverse Kinematics)은 기존 로봇 알고리즘이 수행하고, LLM은 작업 분해(Task Decomposition), 의미 이해(Semantic Reasoning), 사용자 대화(Dialogue)를 담당한다. 이러한 역할 분리는 연산 효율을 크게 향상시킨다.

실시간 운영체제(RTOS)나 ROS 2 기반 시스템에서는 언어 모델이 다른 실시간 제어 작업을 방해하지 않도록 우선순위 기반 스케줄링(Priority Scheduling)을 수행한다. 안전 제어와 모터 제어는 항상 가장 높은 우선순위를 유지하며, 언어 모델은 남는 연산 자원을 효율적으로 활용한다.

클라우드-엣지 협업(Cloud-Edge Collaboration)은 현실적인 최적화 방법이다. 즉각적인 판단은 로컬에서 수행하고, 복잡한 분석이나 디지털 트윈(Digital Twin) 기반 시뮬레이션, 플릿(Fleet) 관리, 장기 계획은 클라우드에서 처리한다. 네트워크가 끊겨도 로컬 SLM만으로 기본적인 자율성을 유지할 수 있다.

도메인 특화 미세조정(Fine-Tuning)은 추론 효율도 향상시킨다. 산업용 로봇 전용 모델은 공장 용어, ROS API, 설비 구조를 이미 이해하고 있으므로 불필요한 추론 과정이 줄어들고 응답도 더욱 정확해진다.

지속적 학습(Continual Learning)은 시간이 지날수록 추론 효율을 높인다. 반복되는 작업, 작업자의 선호도, 자주 사용하는 명령, 시설 구조를 학습함으로써 동일한 작업에 필요한 계산량이 점차 감소한다.

시뮬레이션(Simulation)은 추론 최적화 연구에서 매우 중요한 역할을 한다. 디지털 트윈(Digital Twin)과 아이작 심(Isaac Sim)을 이용하면 지연 시간(Latency), 처리량(Throughput), GPU 사용률(Utilization), 전력 소비(Power Consumption), 발열(Thermal Behavior)을 실제 환경과 유사하게 측정하고 최적화할 수 있다.

제조 공장에서는 실시간 추론이 생산성(Productivity)에 직접 연결된다. 작업 지시 해석, 품질 검사(Quality Inspection), 유지보수 지원, 보고서 생성 등을 빠르게 수행하면서도 생산 라인의 주기 시간(Cycle Time)을 방해하지 않아야 한다.

물류 창고에서는 배송 요청, 재고 관리, 다중 AMR 협업, 충전 관리, 장애물 회피 등이 동시에 발생한다. 최적화된 추론은 이러한 다양한 요청을 낮은 지연 시간으로 처리하여 전체 물류 효율을 향상시킨다.

의료 로봇은 환자와의 자연스러운 대화와 실시간 이동을 동시에 수행해야 한다. 약품 배송, 환자 안내, 상태 모니터링 등을 지연 없이 처리하기 위해서는 매우 효율적인 추론 구조가 필요하다.

시설 점검 로봇은 점검 결과 분석, 이상 탐지(Anomaly Detection), 유지보수 문서 검색, 작업자 질문 응답을 수행하면서도 동시에 자율주행과 센서 처리를 계속해야 한다. 따라서 추론 최적화는 점검 효율을 결정하는 핵심 요소가 된다.

그러나 추론 속도가 빨라진다고 해서 안전 검증(Safety Verification)을 생략해서는 안 된다. 모션 플래너(Motion Planner), 충돌 예측(Collision Prediction), 어포던스 분석(Affordance Estimation), 런타임 모니터(Runtime Monitor), 비상 정지(Emergency Stop)는 LLM과 독립적으로 동작하며 모든 행동을 검증한 후 실행을 허가한다.

실시간 추론 성능은 단순한 토큰 생성 속도(Tokens per Second)만으로 평가하지 않는다. 첫 토큰 생성 시간(First Token Latency), 전체 응답 시간(End-to-End Latency), 처리량(Throughput), 메모리 사용량(Memory Utilization), GPU 활용률(GPU Utilization), 전력 소비(Power Consumption), 발열 안정성(Thermal Stability), 작업 성공률(Task Success Rate), 장시간 운용 안정성(Long-Term Reliability)을 함께 평가해야 한다.

향후에는 적응형 트랜스포머(Adaptive Transformer), 멀티모달 인식(Multimodal Perception), 검색 증강 생성(RAG), 세계 모델(World Model), 외부 메모리(External Memory), 이기종 하드웨어(Heterogeneous Hardware), 예측 스케줄링(Predictive Scheduling), 지속적 학습(Continual Learning)이 통합된 차세대 실시간 추론 구조가 등장할 것으로 예상된다. 환경 변화, 배터리 상태, 발열, 작업 중요도에 따라 추론 구조가 스스로 최적화되는 지능형 시스템이 핵심 기술이 될 것이다.

결국 LLM 추론 최적화는 모델 압축(Model Compression), 하드웨어 가속(Hardware Acceleration), 적응형 추론(Adaptive Inference), 계층형 AI(Hierarchical AI), 멀티모달 처리(Multimodal Processing), 효율적인 메모리 관리(Memory Management), 검색 증강 생성(RAG), 클라우드-엣지 협업(Cloud-Edge Collaboration), 결정적 실행(Deterministic Execution), 안전 검증(Safety Verification)을 하나의 통합 아키텍처로 결합하여 제한된 하드웨어에서도 실시간으로 동작하는 차세대 물리 AI(Physical AI)의 핵심 기반 기술로 자리잡고 있다.
