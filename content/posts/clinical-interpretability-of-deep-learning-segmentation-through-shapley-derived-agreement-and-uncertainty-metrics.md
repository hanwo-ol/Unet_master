---
categories:
- Literature Review
- U-Net
date: 2025-12-08
draft: false
params:
  arxiv_id: 2512.07224v1
  arxiv_link: http://arxiv.org/abs/2512.07224v1
  pdf_path: //172.22.138.185/Research_pdf/2512.07224v1.pdf
tags:
- Deep learning
- Explainable AI (XAI)
- Shapley value
- Medical Segmentation
- Uncertainty estimation
title: Clinical Interpretability of Deep Learning Segmentation Through Shapley-Derived
  Agreement and Uncertainty Metrics
---

## Abstract
Segmentation is the identification of anatomical regions of interest, such as organs, tissue, and lesions, serving as a fundamental task in computer-aided diagnosis in medical imaging. Although deep learning models have achieved remarkable performance in medical image segmentation, the need for explainability remains critical for ensuring their acceptance and integration in clinical practice, despite the growing research attention in this area. Our approach explored the use of contrast-level Shapley values, a systematic perturbation of model inputs to assess feature importance. While other studies have investigated gradient-based techniques through identifying influential regions in imaging inputs, Shapley values offer a broader, clinically aligned approach, explaining how model performance is fairly attributed to certain imaging contrasts over others. Using the BraTS 2024 dataset, we generated rankings for Shapley values for four MRI contrasts across four model architectures. Two metrics were proposed from the Shapley ranking: agreement between model and ``clinician" imaging ranking, and uncertainty quantified through Shapley ranking variance across cross-validation folds. Higher-performing cases (Dice \textgreater0.6) showed significantly greater agreement with clinical rankings. Increased Shapley ranking variance correlated with decreased performance (U-Net: $r=-0.581$). These metrics provide clinically interpretable proxies for model reliability, helping clinicians better understand state-of-the-art segmentation models.

## PDF Download
[Local PDF View](//172.22.138.185/Research_pdf/2512.07224v1.pdf) | [Arxiv Original](http://arxiv.org/abs/2512.07224v1)

## Research Agent - Draft Refiner Module 리포트

**논문 제목:** CLINICAL INTERPRETABILITY OF DEEP LEARNING SEGMENTATION THROUGH SHAPLEY-DERIVED AGREEMENT AND UNCERTAINTY METRICS
(Shapley 기반 합의 및 불확실성 메트릭을 통한 딥러닝 분할의 임상적 해석 가능성)

---

### 1. 요약 (Executive Summary)

*   **연구 목표:** 의료 영상 분할 분야에서 딥러닝 모델의 임상적 수용을 높이기 위해 모델 결정의 해석 가능성(Explainability)을 향상시키는 데 중점을 둡니다.
*   **핵심 방법론:** 모델 입력(MRI 대비)의 체계적인 교란을 통해 특징 중요도를 평가하는 **대비 수준(Contrast-level) Shapley 값**을 활용합니다. 이는 기존의 픽셀 기반 방법보다 임상적으로 정렬된 설명을 제공합니다.
*   **제안된 메트릭:** Shapley 값 순위(Ranking)를 기반으로 두 가지 새로운 임상 해석 가능 메트릭을 제안했습니다.
    1.  **합의(Agreement):** 모델의 대비 순위와 임상 프로토콜 순위 간의 일치도.
    2.  **불확실성(Uncertainty):** 교차 검증 폴드(cross-validation folds) 간 Shapley 순위 분산.
*   **주요 결과 (합의):** 성능이 높은 사례(Dice > 0.6)는 임상 순위와 유의미하게 더 큰 합의를 보였습니다.
*   **주요 결과 (불확실성):** Shapley 순위 분산이 증가할수록 성능이 감소하는 경향이 나타났습니다 (예: U-Net에서 $r = -0.581$).
*   **결론:** 제안된 메트릭은 모델 신뢰도에 대한 임상적으로 해석 가능한 대리 지표를 제공하여, 최신 분할 모델에 대한 임상의의 이해를 돕습니다.

---

### 2. 7가지 핵심 질문 분석 (Key Analysis)

#### 1) What is new in the work? (기존 연구와의 차별점)

이 연구는 기존의 Grad-CAM과 같은 그래디언트 기반 시각화 기법이나 픽셀 수준의 Shapley 분석에서 벗어나, **대비 수준(Contrast-level) Shapley 값**을 사용하여 다중 대비 MRI 분할 모델의 결정을 설명합니다. 가장 큰 차별점은 이 Shapley 값을 단순히 설명하는 데 그치지 않고, 임상적 의사 결정 과정에 직접적으로 연결되는 두 가지 정량적 메트릭(합의 및 불확실성)을 도출하여 모델의 신뢰도를 평가했다는 점입니다.

#### 2) Why is the work important? (연구의 중요성)

이 연구는 딥러닝 모델이 의료 영상 분할에서 높은 성능을 달성했음에도 불구하고 임상 현장에서 '블랙 박스'로 남아있는 문제를 해결하는 데 중요합니다. 모델이 특정 MRI 대비(T1c, T2f 등)를 어떻게 우선순위로 두는지에 대한 정량적이고 해석 가능한 설명을 제공함으로써, 임상의가 모델의 추론 과정을 이해하고 예측의 일관성을 신뢰할 수 있도록 돕습니다. 이는 의료 AI의 임상 통합 및 수용을 촉진하는 데 필수적입니다.

#### 3) What is the literature gap? (기존 연구의 한계점)

기존의 설명 가능 AI(XAI) 연구들은 주로 픽셀 수준의 중요도를 시각화하는 데 초점을 맞추었으며, 이는 임상적 해석을 위해 임계값 설정이 필요하여 해석이 복잡해지는 한계가 있었습니다. 또한, 기존 Shapley 연구들은 모델의 성능 기여도를 정량화했지만, 그 결과가 임상 프로토콜이나 진단 불확실성(예: 임상의 간 의견 불일치)과 어떻게 연관되는지에 대한 직접적인 연결 고리가 부족했습니다.

#### 4) How is the gap filled? (해결 방안)

연구진은 Shapley 값을 1부터 4까지의 순위로 변환하여 이 간극을 메웠습니다.
1.  **합의:** 모델이 생성한 Shapley 순위를 임상 표준 프로토콜 순위($R_K$)와 비교하기 위해 **정규화된 스피어만 풋룰 거리(Normalized Spearman Footrule Distance, NSF)**를 사용했습니다. NSF 값이 높을수록 모델의 대비 우선순위가 임상적 지침과 일치함을 의미합니다.
2.  **불확실성:** 5-겹 교차 검증(five-fold cross-validation) 폴드 전반에 걸쳐 Shapley 순위의 **분산($V$)**을 계산하여 모델의 예측 일관성 및 불확실성을 정량화했습니다.

#### 5) What is achieved with the new method? (달성한 성과)

제안된 메트릭은 모델 성능과 신뢰도 간의 강력한 상관관계를 입증했습니다.

*   **합의 성과 (Figure 2 분석):** U-Net 모델의 경우, Dice 점수 0.6 이상 그룹은 Dice 점수 0.5 미만 그룹에 비해 임상 표준 순위와의 합의(NSF)가 **유의미하게 더 높았습니다** ($p < 0.001$). 이는 성능이 좋은 모델일수록 임상적 상식과 일치하는 방식으로 대비를 활용함을 보여줍니다.
*   **불확실성 성과 (Figure 3 분석):** Shapley 순위 분산($V$)이 증가할수록 평균 Dice 점수가 감소하는 **음의 상관관계**가 모든 모델에서 관찰되었습니다.
    *   U-Net 모델에서 가장 강한 상관관계: $r = -0.581$ (95% CI: $[-0.607, -0.553]$).
    *   Seg-Resnet 모델에서 가장 강한 상관관계: $r = -0.604$ (95% CI: $[-0.636, -0.569]$).
    *   이는 모델의 불확실성이 높을수록 분할 성능이 저하될 가능성이 높다는 것을 정량적으로 입증합니다.

#### 6) What data are used? (사용 데이터셋 - 도메인 특성 포함)

*   **데이터셋:** Brain Tumor Segmentation (BraTS) Challenge 2024 GOAT challenge.
*   **도메인 특성:** 뇌종양 분할(Glioma segmentation)을 위한 의료 영상 데이터.
*   **구성:** 총 1351명의 환자 데이터. 각 환자는 네 가지 MRI 대비(T1c, T1n, T2f, T2w)를 포함하는 다중 채널 입력으로 구성됩니다.
*   **Ground Truth:** 괴사성 코어(necrotic core), 부종(edema), 조영 증강 종양(enhancing tumor)의 세 가지 종양 하위 영역 주석을 포함합니다.

#### 7) What are the limitations? (저자가 언급한 한계점)

저자들은 다음과 같은 한계점을 언급했습니다.
1.  **임상의 합의 부족:** Shapley 순위 합의 분석을 위해 숙련된 방사선 전문의로부터 직접적인 합의를 얻지 못하고, 대신 의대생 주석자 합의와 논문 기반의 임상 표준을 사용했습니다. 향후 더 강력한 임상 비교가 필요합니다.
2.  **불확실성의 주관성:** 모델 결정의 불확실성 개념은 주관적이며 다양한 방식으로 접근될 수 있습니다 (예: Monte Carlo 드롭아웃을 사용하여 예측 불확실성을 정량화하는 방법).

---

### 3. 아키텍처 및 방법론 (Architecture & Methodology)

#### Figure 분석: 메인 아키텍처 및 흐름 (Figure 1)

Figure 1은 새로운 아키텍처 구조가 아닌, 제안된 **설명 가능성 메트릭 도출 과정**을 시각적으로 요약합니다.

1.  **입력 (Multi-contrast MRI):** T1c, T1n, T2f, T2w 네 가지 MRI 대비 영상이 딥러닝 모델의 입력으로 사용됩니다.
2.  **Shapley 값 계산:** 각 대비 영상에 대해 Shapley 값이 계산됩니다. 이 값은 해당 대비가 모델의 성능(Dice Score)에 기여하는 정도를 정량화합니다.
3.  **Shapley 순위 변환:** 계산된 Shapley 값은 순위(1: 가장 중요, 4: 가장 덜 중요)로 변환됩니다 (예: T1n=1, T2f=2).
4.  **메트릭 도출:** 이 Shapley 순위를 기반으로 두 가지 임상 해석 가능 메트릭이 도출됩니다.
    *   **합의 (Agreement):** 모델 순위와 임상 프로토콜 순위 간의 일치도.
    *   **불확실성 (Uncertainty):** 교차 검증 폴드 간 모델 순위의 변동성.
5.  **최종 목표:** 임상의가 모델의 의사 결정 및 성능을 이해하도록 돕습니다.

#### 수식 상세 (Loss Function, Input/Output Tensor Shape, 주요 모듈의 수식)

**Input/Output Tensor Shape:**

*   **입력 ($I$):** 4채널 3D-MRI 영상.
    $$I \in \mathbb{R}^{N \times D \times W \times H}$$
    *여기서 $N=4$는 MRI 대비 채널 수(T1c, T1n, T2f, T2w)이며, $D, W, H$는 깊이, 너비, 높이입니다.*
*   **출력 ($\hat{Y}$):** 종양 레이블 예측.
    $$\hat{Y} = \omega(I)$$
    *여기서 $\omega$는 딥러닝 모델(U-Net, Seg-Resnet 등)을 나타냅니다.*
*   **평가 메트릭:** Dice 유사도 계수 ($D$).

**1. 대비 수준 Shapley 값 ($\Phi_i(D)$)**

특정 대비 $i$가 Dice 점수 $D$에 기여하는 정도를 계산합니다. $N$은 전체 대비 집합, $n_i$는 $i$번째 대비, $S$는 $n_i$를 포함하지 않는 $N$의 부분 집합입니다.

$$\Phi_i(D) = \sum_{S \subseteq N \setminus \{n_i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (D(S \cup \{n_i\}) - D(S))$$

**2. 정규화된 스피어만 풋룰 거리 (Normalized Spearman Footrule Distance, NSF)**

모델의 Shapley 순위 $R_{\Phi_i}(D)$와 임상 표준 순위 $R_{K_i}$ 간의 합의를 정량화합니다. $|N|=4$이며, $D_{\text{max}}=8$입니다. NSF는 0 (완전 불일치)부터 1 (완전 일치) 사이의 값을 가집니다.

$$NSF = 1 - \left(1/D_{\text{max}}\right) \sum_{i=1}^{|N|} |R_{\Phi_i}(D) - R_{K_i}|$$

**3. Shapley 순위 폴드 간 분산 ($v$)**

모델의 불확실성을 정량화하기 위해 5개 폴드($K=5$)에 걸친 Shapley 순위의 분산을 계산합니다.

$$v = (1/|N|) \sum_{i=1}^{|N|} \text{Var}(R_i)$$

**4. 표준 분산 공식 ($\text{Var}(R_i)$)**

$R_i$는 $i$번째 대비의 5개 폴드 순위 벡터이며, $\bar{r}_i$는 평균 순위입니다.

$$\text{Var}(R_i) = \frac{1}{(K-1)} \sum_{k=1}^{K} (r_{\Phi_i}(D)_k - \bar{r}_i)^2$$

#### Vanilla U-Net 비교: 추가/수정된 모듈

이 논문의 핵심은 새로운 아키텍처를 제안하는 것이 아니라, 기존의 딥러닝 분할 모델(U-Net, Seg-Resnet, UNETR, Swin-UNETR)에 **새로운 설명 가능성 프레임워크**를 적용하는 것입니다.

*   **U-Net 구조 자체의 변경:** 논문에서 U-Net의 인코더/디코더 블록이나 스킵 연결(skip connection) 등 내부 구조에 대한 구체적인 수정 사항은 언급되지 않았습니다. U-Net은 4가지 MRI 대비를 입력으로 받는 표준적인 다중 채널 분할 모델로 사용되었습니다.
*   **추가된 모듈/프레임워크:** 모델의 출력(Dice Score)을 분석하기 위해 **대비 수준 Shapley 값 계산 모듈**이 후처리 단계에 추가되었습니다. 이 모듈은 모델의 성능을 설명하고, 그 결과를 기반으로 NSF (합의) 및 분산 $V$ (불확실성) 메트릭을 도출합니다.

---

### 4. 태그 제안 (Tags Suggestion)

1.  Deep learning
2.  Explainable AI (XAI)
3.  Shapley value
4.  Medical Segmentation
5.  Uncertainty estimation