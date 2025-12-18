---
categories:
- Literature Review
- U-Net
date: 2025-12-17
draft: false
params:
  arxiv_id: 2512.15422v1
  arxiv_link: http://arxiv.org/abs/2512.15422v1
  pdf_path: //172.22.138.185/Research_pdf/2512.15422v1.pdf
tags:
- Automated Driving Systems (ADS)
- Scenario-Based Testing (SBT)
- Multimodal AI
- Scenario Generation
- Evaluation Metrics (AII, RAS, OCS)
title: Can AI Generate more Comprehensive Test Scenarios? Review on Automated Driving
  Systems Test Scenario Generation Methods
---

## Abstract
Ensuring the safety and reliability of Automated Driving Systems (ADS) remains a critical challenge, as traditional verification methods such as large-scale on-road testing are prohibitively costly and time-consuming.To address this,scenario-based testing has emerged as a scalable and efficient alternative,yet existing surveys provide only partial coverage of recent methodological and technological advances.This review systematically analyzes 31 primary studies,and 10 surveys identified through a comprehensive search spanning 2015~2025;however,the in-depth methodological synthesis and comparative evaluation focus primarily on recent frameworks(2023~2025),reflecting the surge of Artificial Intelligent(AI)-assisted and multimodal approaches in this period.Traditional approaches rely on expert knowledge,ontologies,and naturalistic driving or accident data,while recent developments leverage generative models,including large language models,generative adversarial networks,diffusion models,and reinforcement learning frameworks,to synthesize diverse and safety-critical scenarios.Our synthesis identifies three persistent gaps:the absence of standardized evaluation metrics,limited integration of ethical and human factors,and insufficient coverage of multimodal and Operational Design Domain (ODD)-specific scenarios.To address these challenges,this review contributes a refined taxonomy that incorporates multimodal extensions,an ethical and safety checklist for responsible scenario design,and an ODD coverage map with a scenario-difficulty schema to enable transparent benchmarking.Collectively,these contributions provide methodological clarity for researchers and practical guidance for industry,supporting reproducible evaluation and accelerating the safe deployment of higher-level ADS.

## PDF Download
[Local PDF View](//172.22.138.185/Research_pdf/2512.15422v1.pdf) | [Arxiv Original](http://arxiv.org/abs/2512.15422v1)

이 보고서는 'Can AI Generate more Comprehensive Test Scenarios? – Review on Automated Driving Systems Test Scenario Generation Methods' 논문을 분석하여 작성되었습니다.

---

## 1. 요약 (Executive Summary)

이 논문은 자율주행 시스템(ADS)의 안전성 및 신뢰성을 검증하기 위한 시나리오 기반 테스트(SBT) 방법론에 대한 체계적인 리뷰입니다. 기존의 온로드 테스트의 비효율성을 극복하기 위해 AI 기반의 생성 모델이 급증하는 2023년~2025년의 방법론적 발전에 초점을 맞춥니다.

*   **연구 배경:** ADS의 안전성 검증을 위한 기존의 대규모 온로드 테스트는 비용과 시간이 많이 소요되어 비현실적이며, SBT가 확장 가능하고 효율적인 대안으로 부상했습니다.
*   **방법론적 전환:** 전통적인 접근 방식(전문가 지식, 온톨로지, 자연 운전 데이터)에서 LLMs, GANs, Diffusion Models, RL 프레임워크를 활용하는 AI 기반 생성 모델로 전환되고 있습니다.
*   **주요 한계점 식별:** 기존 연구에서 표준화된 평가 지표의 부재, 윤리적 및 인간적 요소의 통합 부족, 멀티모달 및 ODD(Operational Design Domain) 특정 시나리오 커버리지 부족이라는 세 가지 지속적인 격차를 확인했습니다.
*   **주요 기여:**
    1.  멀티모달 확장을 통합한 정제된 분류 체계(Taxonomy)를 제안합니다.
    2.  투명한 벤치마킹을 위한 3축 평가 시스템(AII, RAS, OCS)을 도입합니다.
    3.  책임 있는 시나리오 설계를 위한 윤리 및 안전 체크리스트를 제공합니다.
    4.  시나리오의 복잡도와 위험도를 분류하는 ODD 커버리지 맵 및 시나리오 난이도 스키마를 제안합니다.
*   **목표:** 연구자들에게 방법론적 명확성을 제공하고, 산업계에 실질적인 지침을 제공하여 ADS의 안전한 배포를 가속화하는 것입니다.

---

## 2. 7가지 핵심 질문 분석 (Key Analysis)

### 1) What is new in the work? (기존 연구와의 차별점)

이 리뷰는 2015년부터 2025년까지의 시나리오 생성 방법론을 포괄적으로 분석하되, 특히 AI 기반 및 멀티모달 접근 방식이 지배적이 된 **2023년~2025년의 변곡점**에 초점을 맞춘 것이 차별점입니다. 기존 설문조사가 부분적인 커버리지에 그쳤던 것과 달리, 이 논문은 방법론적 합성 및 비교 평가를 심층적으로 수행합니다. 또한, 학술적 영향(AII), 재현성(RAS), ODD 커버리지(OCS)를 포괄하는 **재현 가능한 3축 평가 시스템**과 윤리적 원칙을 구체적인 평가 기준으로 변환한 **윤리 및 안전 체크리스트**를 제안하여, 기존 연구의 평가 투명성 및 표준화 부재라는 한계를 직접적으로 해결합니다.

### 2) Why is the work important? (연구의 중요성)

이 연구는 ADS의 안전하고 윤리적인 배포를 가속화하는 데 필수적입니다. ADS의 복잡성이 증가함에 따라, 실제 도로 주행만으로는 통계적 안전성을 입증하는 것이 불가능해졌습니다. 이 리뷰는 LLMs, GANs, Diffusion Models과 같은 **생성형 AI**를 활용하여 다양하고 안전에 중요한(safety-critical) 시나리오를 합성하는 최신 기술 동향을 체계적으로 정리합니다. 제안된 분류 체계와 평가 프레임워크는 연구자들이 방법론을 명확히 이해하고, 산업계가 투명하고 윤리적으로 정렬된 방식으로 시나리오 생성 프레임워크를 벤치마킹하고 채택할 수 있는 구조적 기반을 제공합니다.

### 3) What is the literature gap? (기존 연구의 한계점)

논문은 시나리오 생성 연구 분야의 세 가지 주요 한계점을 지적합니다. 첫째, **표준화된 평가 지표의 부재**로 인해 시나리오의 품질과 관련성을 객관적으로 검증하기 어렵습니다. 둘째, **윤리적 및 인간적 요소의 통합 부족**입니다. 대부분의 프레임워크가 기술적 효율성에 집중하여 취약한 도로 사용자(VRU)의 다양성이나 개인 정보 보호와 같은 윤리적 고려 사항을 간과할 위험이 있습니다. 셋째, **멀티모달 및 ODD 특정 시나리오의 불충분한 커버리지**입니다. 대부분의 프레임워크가 구조화된 환경에 집중하며, 악천후, 복잡한 다중 에이전트 상호작용과 같은 롱테일(long-tail) 또는 안전에 중요한 조건에 대한 데이터가 부족합니다.

### 4) How is the gap filled? (해결 방안)

이 논문은 세 가지 주요 도구를 통해 한계를 해결합니다. 첫째, **정제된 분류 체계**를 통해 AI 기반 및 멀티모달 접근 방식을 포함하여 방법론을 알고리즘 계열과 모달리티 커버리지별로 구조화합니다. 둘째, **3축 평가 시스템(AII, RAS, OCS)**을 도입하여 재현성, 학술적 영향, ODD 커버리지를 정량적으로 측정할 수 있는 표준화된 벤치마킹 기준을 마련합니다. 셋째, **윤리 및 안전 체크리스트**를 통해 편향 완화, 개인 정보 보호, 표준(ISO 21448, UN R157) 준수와 같은 추상적인 윤리적 원칙을 감사 가능한 기준으로 변환하여 윤리적 공백을 메웁니다.

### 5) What is achieved with the new method? (달성한 성과 - Table의 수치를 인용할 것)

이 리뷰는 제안된 3축 평가 시스템을 사용하여 2023년~2025년의 주요 시나리오 생성 프레임워크 14개를 비교 분석했습니다 (Table 4 및 Figure 6 참조).

| 프레임워크 | AII (학술적 영향) | RAS (재현성/접근성) | OCS (ODD 커버리지) |
| :--- | :--- | :--- | :--- |
| **TrafficComposer [59]** | 7.2% (★★★★) | **95% (★★★★★)** | **62.0% (★★★★)** |
| **GAIA-1 [10]** | **78.2% (★★★★★)** | 41.0% (★★★) | 30.0% (★★★) |
| **ChatScene [57]** | **78.2% (★★★★★)** | 90% (★★★★★) | 41.0% (★★★) |
| **Genesis [14]** | 35.6% (★) | 25% (★★★) | 31.0% (★★) |

*   **재현성 및 접근성 (RAS):** **TrafficComposer**는 **95% (★★★★★)**로 가장 높은 점수를 기록하여, 공개된 코드, 실행 가능한 파이프라인, 공개 데이터셋 보너스를 통해 높은 재현성과 투명성을 입증했습니다.
*   **학술적 영향 (AII):** **GAIA-1**과 **ChatScene**이 **78.2% (★★★★★)**로 가장 높은 AII를 기록하며, 해당 분야에서 가장 큰 학술적 영향력을 보였습니다.
*   **ODD 커버리지 (OCS):** **TrafficComposer**가 **62.0% (★★★★)**로 가장 넓은 ODD 커버리지를 달성했습니다. 이는 시뮬레이터 중심의 광범위한 ODD 커버리지를 강조하는 목표와 일치합니다. 반면, **Genesis**는 31.0% (★★)로 상대적으로 낮은 ODD 커버리지를 보였는데, 이는 특정 데이터셋(nuScenes)의 비디오-LiDAR 생성 충실도에 중점을 둔 목표를 반영합니다.

### 6) What data are used? (사용 데이터셋 - 도메인 특성 포함)

시나리오 생성에 사용되는 소스 데이터는 크게 두 가지 방법론으로 분류됩니다 (Section 4.3).

1.  **지식 기반 (Knowledge-based):**
    *   **소스:** 전문가 권고, 교통 규정, 온톨로지(Ontologies).
    *   **특성:** 구조화된 환경(고속도로, 도시 중심)에서 관련성 있고 현실적인 테스트 시나리오를 생성하는 데 중요하며, 기계가 이해할 수 있고 인간이 해석 가능한 형식으로 지식을 구조화합니다.

2.  **데이터 기반 (Data-based):**
    *   **자연 운전 데이터 (NDD):** 실제 차량 또는 고정 센서(LiDAR, 카메라)에서 수집된 데이터 (예: Waymo Open Motion Dataset, nuScenes, highD).
    *   **특성:** 실제 교통 상황을 자연스럽게 추출하지만, 희귀하거나 안전에 중요한 이벤트 데이터는 양과 다양성이 제한적이며 수집 비용이 높습니다.
    *   **사고 데이터 (Accident Data):** NHTSA의 Pre-Crash Scenario Typology, CIREN, NMVCCS 등에서 얻은 사고 보고서나 스케치.
    *   **특성:** 실제 사고 정보를 바탕으로 안전에 중요한 시나리오를 생성하는 데 사용되며, NDD에 비해 비용이 저렴하고 노동력이 적게 듭니다.

### 7) What are the limitations? (저자가 언급한 한계점)

저자는 리뷰의 한계점과 방법론적 격차를 다음과 같이 언급합니다 (Section 7.6).

*   **재현성 점수(RAS)의 제약:** 프레임워크 간의 불균일한 문서화와 데이터셋 접근성(사유 또는 자체 수집 데이터)으로 인해 RAS 계산의 투명성과 완전성이 제한됩니다.
*   **개념적 도구의 검증 필요성:** ODD 커버리지 맵 및 시나리오 난이도 스키마는 현재 개념적 도구이며, 표준화된 보고가 아닌 추론된 증거에 기반하여 적용되었으므로, 향후 연구에서 벤치마크 데이터셋 및 시뮬레이션 플랫폼에 통합하여 **경험적 검증**이 필요합니다.
*   **방법론의 예비적 성격:** 제안된 분류 체계, 체크리스트, 스키마는 예비적이며, 그 효과가 대규모 벤치마크에서 체계적으로 검증되지 않았습니다. 적용 가능성은 데이터셋 접근성 및 도메인별 요구 사항에 따라 달라질 수 있습니다.

---

## 3. 아키텍처 및 방법론 (Architecture & Methodology)

이 논문은 특정 U-Net 기반 모델을 제안하는 것이 아니라, ADS 시나리오 생성 방법론의 전체적인 분류 체계와 워크플로우를 분석하고 새로운 평가 프레임워크를 제안하는 리뷰 논문입니다. 따라서 아키텍처 분석은 논문에서 제시한 **시나리오 생성 워크플로우(Figure 3)**와 **연구 조직도(Figure 1)**를 중심으로 이루어집니다.

### Figure 분석: 시나리오 생성 워크플로우

**Figure 3: Generic Test Scenario Generation Process**는 시나리오 생성의 일반적인 파이프라인을 보여주며, 전통적인 접근 방식과 AI 기반 접근 방식의 통합을 강조합니다.

1.  **Source Data Collection:**
    *   **Raw Data** (Knowledge, Multimodal Pre-processed Data)를 수집하여 **Scenario Database**를 구축합니다.

2.  **Scenario Generation / Extraction:**
    *   **Traditional Approach:**
        *   **Knowledge-based:** 온톨로지 활용(Ontology Utilization) 및 AST(Abstract Syntax Trees)를 통해 시나리오를 정의하고 추출합니다.
        *   **Data-driven:** 데이터 기반 매개변수화(Parameterization) 및 분류(Classification)를 통해 시나리오를 추출합니다.
        *   이후 **Scenario Variation & Generation** 단계를 거쳐 구체적인 시나리오를 생성합니다.
    *   **AI-assisted Approach:**
        *   **Generative Models:** Diffusion Model, LLM (Transformer-based), GAN, VAE/Autoregressive와 같은 주요 생성 모델을 사용합니다.
        *   **Aux Technics:** RNN, RL과 같은 보조 기술을 통합하여 생성 모델의 성능을 향상시킵니다.
        *   이 모델들은 제공된 데이터를 활용하여 직접적으로 **Concrete scenarios**를 생성합니다.

3.  **Scenario Evaluation & Validation (V&V):**
    *   생성된 구체적인 시나리오들은 최종적으로 검증 및 평가 단계를 거칩니다.

### 수식 상세

이 논문은 새로운 모델을 제안하지 않았으므로, Loss Function이나 Tensor Shape 대신, 논문에서 제안한 핵심적인 **3축 평가 지표**의 수식을 LaTeX로 작성합니다.

#### 1. Academic Influence Index (AII)

AII는 학술적 영향력을 평가하며, 특히 프리프린트(preprint)의 역할을 반영하기 위해 고안되었습니다.

$$
\text{AII} = 0.4 \times C_{\text{norm}} + 0.3 \times R_{\text{early}} + 0.3 \times H_{\text{mean}} \quad (1)
$$

여기서:
*   $C_{\text{norm}}$: 정규화된 인용 횟수 (비교 대상 중 최대 인용 횟수로 나눈 값, $[0, 1]$ 범위).
*   $R_{\text{early}}$: 출판 후 첫 해에 받은 인용의 비율.
*   $H_{\text{mean}}$: 주 저자들의 정규화된 평균 H-인덱스.

#### 2. Resource Accessibility Score (RAS)

RAS는 프레임워크의 재현성 및 책임 있는 사용을 지원하는 정도를 평가합니다.

$$
\text{RAS} = \min \left(100, 100 \times \sum_{i=1}^{5} w_i s_i + B_{\text{dataset}}\right) \quad (2)
$$

여기서:
*   $w_i$: $i$-번째 구성 요소의 가중치 (Table 2 참조: Code Availability 30%, Minimal Reproducible Pipeline 25%, Environment & Build 20%, Model/Asset Accessibility 15%, Documentation Quality 10%).
*   $s_i$: $i$-번째 구성 요소의 점수 ($s_i \in \{0, 0.5, 1\}$).
*   $B_{\text{dataset}}$: 데이터셋 접근성 보너스 ($B_{\text{dataset}} \in \{0, 2, 5\}$).

#### 3. ODD Coverage Score (OCS)

OCS는 시나리오 생성 프레임워크가 다루는 운영 설계 도메인(ODD) 조건의 폭을 정량화합니다.

$$
\text{OCS} = 100 \times \frac{1}{5} \sum_{k=1}^{5} E a_k \quad (3)
$$

여기서:
*   $a_k$: $k$-번째 차원(Road Type, VRU Presence, Topological Complexity, Interaction Complexity, Scenario Controllability)에 할당된 점수 ($a_k \in [0, 1]$).
*   $E$: $k$-번째 차원에 할당된 추가 가중치 (Appendix C.1 참조: VRU Presence 및 Topological Complexity에 높은 가중치 0.25 부여).

### Vanilla U-Net 비교

이 논문은 U-Net과 같은 특정 신경망 아키텍처를 다루지 않으므로, 대신 **전통적인 시나리오 생성 방법론**과 **AI 기반 시나리오 생성 방법론**을 비교하여 정리합니다.

| 구분 | 전통적인 접근 방식 (Traditional Approaches) | AI 기반 접근 방식 (AI-assisted Approaches) |
| :--- | :--- | :--- |
| **핵심 방법론** | 규칙 기반(Ontologies, Petri nets), 데이터 기반(Statistical extraction, Monte Carlo, Clustering) | LLMs, GANs, Diffusion Models (DMs), VAE/Autoregressive, Reinforced Learning (RL) |
| **주요 목표** | ODD 정의 내에서 대표성 및 통계적 분포 보장. | 다양성, 현실성, 안전에 중요한 희귀 시나리오 합성. |
| **데이터 소스** | 전문가 지식, 교통 규정, 자연 운전 데이터, 사고 보고서. | 대규모 멀티모달 데이터 (텍스트, 시각, LiDAR, 궤적). |
| **주요 특징** | 잘 이해되고 규제된 환경(고속도로)에 효과적. 확장성 및 복잡한 상호작용 모델링에 한계. | **멀티모달 통합** (텍스트-비디오-LiDAR 일관성), **제어 가능성** (Controllability) 강조. |
| **최근 동향** | 2023년 이후 AI 기반 생성 모델에 의해 주도권이 넘어감. | LLM 기반의 자연어 인터페이스 및 Diffusion Model 기반의 고충실도(High-fidelity) 장면 합성이 주류. |

---

## 4. 태그 제안 (Tags Suggestion)

1.  Automated Driving Systems (ADS)
2.  Scenario-Based Testing (SBT)
3.  Multimodal AI
4.  Scenario Generation
5.  Evaluation Metrics (AII, RAS, OCS)