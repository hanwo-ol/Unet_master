---
categories:
- Literature Review
- U-Net
date: 2025-05-03
draft: false
params:
  arxiv_id: 2505.01639v2
  arxiv_link: http://arxiv.org/abs/2505.01639v2
  pdf_path: //172.22.138.185/Research_pdf/2505.01639v2.pdf
tags:
- Auto-Generated
- Draft
title: Fast Likelihood-Free Parameter Estimation for Lévy Processes
---

## Abstract
Lévy processes are widely used in financial modeling due to their ability to capture discontinuities and heavy tails, which are common in high-frequency asset return data. However, parameter estimation remains a challenge when associated likelihoods are unavailable or costly to compute. We propose a fast and accurate method for Lévy parameter estimation using the neural Bayes estimation (NBE) framework -- a simulation-based, likelihood-free approach that leverages permutation-invariant neural networks to approximate Bayes estimators. We contribute new theoretical results, showing that NBE results in consistent estimators whose risk converges to the Bayes estimator under mild conditions. Moreover, through extensive simulations across several Lévy models, we show that NBE outperforms traditional methods in both accuracy and runtime, while also enabling two complementary approaches to uncertainty quantification. We illustrate our approach on a challenging high-frequency cryptocurrency return dataset, where the method captures evolving parameter dynamics and delivers reliable and interpretable inference at a fraction of the computational cost of traditional methods. NBE provides a scalable and practical solution for inference in complex financial models, enabling parameter estimation and uncertainty quantification over an entire year of data in just seconds. We additionally investigate nearly a decade of high-frequency Bitcoin returns, requiring less than one minute to estimate parameters under the proposed approach.

## PDF Download
## PDF Download
[Local PDF View](//172.22.138.185/Research_pdf/2505.01639v2.pdf) | [Arxiv Original](http://arxiv.org/abs/2505.01639v2)

이 논문은 Lévy 프로세스의 매개변수 추정을 위한 빠르고 정확한 우도-자유(Likelihood-Free) 방법인 **신경망 베이즈 추정(Neural Bayes Estimation, NBE)** 프레임워크를 제안합니다.

---

## 1. 요약 (Executive Summary)

*   **문제 정의:** 금융 모델링에 널리 사용되는 Lévy 프로세스는 불연속성과 두꺼운 꼬리(heavy tails)를 포착하는 데 유용하지만, 복잡한 모델(예: Deep Variance Gamma, DVG)의 경우 우도(likelihood) 함수를 계산하기 어렵거나 불가능하여 매개변수 추정에 어려움이 있습니다.
*   **해결책 제안:** 시뮬레이션 기반의 우도-자유 접근 방식인 NBE를 사용하여 베이즈 추정량을 근사합니다. 특히, Lévy 프로세스 증분(increments)의 순열 불변성(permutation invariance)을 활용하기 위해 DeepSets 기반 신경망 아키텍처를 채택했습니다.
*   **이론적 기여:** NBE 추정량이 일관성(consistent)을 가지며, 온화한 조건 하에서 베이즈 추정량으로 위험(risk)이 수렴한다는 새로운 이론적 결과를 제공하여 방법론의 대규모 표본 행동에 대한 형식적인 보장을 확립했습니다.
*   **성능 우위:** Compound Poisson, Merton, Variance Gamma 등 여러 표준 Lévy 모델에 대한 광범위한 시뮬레이션에서, NBE는 기존의 고전적인 방법(LSQ, MELE) 대비 **정확도(RMSE)**와 **계산 효율성(Runtime)** 모두에서 일관되게 우수한 성능을 보였습니다.
*   **실제 적용 및 속도:** 고빈도 암호화폐 수익률 데이터(9년간의 비트코인 1분 데이터 포함)에 적용한 결과, NBE는 전체 데이터에 대한 일일 매개변수 추정 및 불확실성 정량화를 단 몇 초 만에 완료하여, 기존 방법 대비 수백 배에서 수천 배 빠른 속도(Amortized Inference)를 입증했습니다.

---

## 2. 7가지 핵심 질문 분석 (Key Analysis)

### 1) What is new in the work? (기존 연구와의 차별점)

본 연구는 Lévy 프로세스 매개변수 추정을 위해 DeepSets 아키텍처를 활용하는 유연한 NBE 프레임워크를 최초로 도입했습니다. DeepSets는 Lévy 프로세스 증분의 **순열 불변성**을 활용하도록 설계되어, 데이터의 순서에 관계없이 효율적인 특징 학습을 가능하게 합니다. 또한, NBE 추정량이 일관성을 가지며 베이즈 추정량으로 위험이 수렴한다는 새로운 **이론적 보장(Theorem 2)**을 제공하여, 방법론의 통계적 건전성을 확립했습니다. 마지막으로, 불확실성 정량화(UQ)를 위해 부트스트랩 기반 신뢰 구간과 후방 분위수 추정이라는 두 가지 보완적인 접근 방식을 통합하여 제공합니다.

### 2) Why is the work important? (연구의 중요성)

Lévy 프로세스는 금융 시장의 불연속적인 움직임(점프)과 두꺼운 꼬리를 모델링하는 데 필수적입니다. 그러나 Deep Variance Gamma (DVG)와 같은 복잡하고 표현력이 풍부한 모델은 폐쇄형 우도를 가지지 않아 고전적인 추론 방법 적용이 어렵습니다. NBE는 우도 계산 없이 시뮬레이션만으로 추론을 수행하는 **우도-자유(likelihood-free)** 접근 방식을 제공하며, 특히 훈련 후 추론 시간이 극도로 짧은 **상각된 추론(amortized inference)** 특성을 통해 고빈도 금융 데이터의 실시간 분석 및 대규모 시뮬레이션 연구에 필요한 확장성과 실용성을 제공합니다.

### 3) What is the literature gap? (기존 연구의 한계점)

기존의 고전적인 Lévy 프로세스 추정 방법(예: LSQ, MELE)은 복잡한 모델에서 우도가 다루기 어렵거나, 매개변수 추정을 위해 매번 새로운 최적화 문제를 해결해야 하므로 계산 비용이 매우 높습니다. 특히 고빈도 금융 데이터처럼 대규모 데이터셋에 반복적으로 적용할 경우, 이러한 계산 부담은 실시간 분석을 불가능하게 만듭니다. 또한, 기존 우도-자유 방법들은 통계적 효율성이 떨어지는 경향이 있었습니다.

### 4) How is the gap filled? (해결 방안)

본 연구는 NBE 프레임워크를 통해 이 문제를 해결합니다. NBE는 베이즈 위험($R_{\Omega}$)을 최소화하도록 신경망을 훈련시키며, 이 훈련 과정은 시뮬레이션된 데이터셋을 기반으로 Monte Carlo 근사화를 통해 수행됩니다. 훈련 비용은 초기에 한 번 발생하지만, 일단 훈련된 신경망은 새로운 데이터셋이 들어올 때마다 매개변수 추정치를 거의 즉각적으로 출력할 수 있습니다. 이는 계산 비용을 '상각(amortize)'하여 추론 단계의 속도를 극적으로 향상시킵니다.

### 5) What is achieved with the new method? (달성한 성과)

NBE는 기존 방법 대비 압도적인 계산 효율성과 우수한 정확도를 달성했습니다.

**정확도 (RMSE 비교):**
표준 Lévy 모델 3가지(Compound Poisson, Merton, Variance Gamma) 모두에서 NBE는 모든 매개변수에 대해 가장 낮은 RMSE를 기록했습니다.

| 모델 | 매개변수 | NBE RMSE | LSQ RMSE | MELE RMSE |
| :--- | :--- | :--- | :--- | :--- |
| Compound Poisson | $\lambda$ (점프 강도) | **0.051** | 0.058 | 0.70 |
| Merton | $\mu$ (확산 평균) | **0.11** | 0.17 | 0.37 |
| Variance Gamma | $\gamma$ (드리프트) | **0.048** | 0.16 | 0.73 |

**계산 효율성 (추론 시간 비교):**
1,000개의 시뮬레이션 데이터셋에 대한 매개변수 추정 시간(초)을 비교했을 때, NBE의 추론 속도는 기존 방법을 압도합니다.

| 모델 | NBE 추론 시간 [s] | LSQ 추론 시간 [s] | MELE 추론 시간 [h] |
| :--- | :--- | :--- | :--- |
| Compound Poisson | **2.8** | 833 | 777.7 |
| Merton | **4** | 695 | 1155 |
| Variance Gamma | **3.5** | 398 | 1024 |

실제 암호화폐 데이터 적용 시, 2022년 전체 기간에 대한 일일 매개변수 추정 시간(Table 4)은 NBE가 **5초**인 반면, LSQ는 656초, MELE는 45,552초(약 12.6시간)가 소요되어, NBE가 LSQ보다 약 130배, MELE보다 약 9,000배 빠릅니다.

### 6) What data are used? (사용 데이터셋)

*   **시뮬레이션 데이터:** Compound Poisson, Merton Jump-Diffusion, Variance Gamma, Level-2 Deep Variance Gamma (DVG) 프로세스.
*   **실제 데이터:** 고빈도 암호화폐 수익률 데이터.
    *   **출처:** Klein [2022]의 1분 해상도 가격 데이터.
    *   **대상:** 비트코인(BTC), 이더리움(ETH), 리플(XRP)의 2022년 1년치 데이터.
    *   **장기 분석:** 비트코인(BTC)의 2014년부터 2022년까지 9년치 데이터.
*   **도메인 특성:** 이 데이터는 고빈도 금융 시계열로서, Lévy 프로세스가 모델링에 적합한 점프(jumps) 및 두꺼운 꼬리(heavy-tailed) 특성을 강하게 나타냅니다.

### 7) What are the limitations? (저자가 언급한 한계점)

저자들은 다음과 같은 한계점을 언급했습니다 (Page 36):

1.  **저신호 영역에서의 추정 난이도:** Level-2 DVG 모델과 같이 깊은 종속성(deep subordination)과 관련된 매개변수($\alpha_1, \alpha_2$)는 값이 매우 작거나 신호가 낮은 영역에서 정확한 추정이 더 어려웠습니다.
2.  **입력 차원 고정:** 현재 설정에서 신경망은 고정된 입력 차원($n_t$)에 맞게 훈련되므로, 다른 차원의 데이터에 적용하려면 네트워크를 재훈련해야 합니다.
3.  **미래 확장 방향:** 방법론을 다변량 Lévy 프로세스로 일반화하거나, 훈련 시 사용된 사전 분포 범위를 벗어난 매개변수 값에 대한 외삽(extrapolation) 능력을 탐색하는 것이 필요합니다.

---

## 3. 아키텍처 및 방법론 (Architecture & Methodology)

### Figure 분석: DeepSets 아키텍처

이 논문에서 사용된 신경망 아키텍처는 DeepSets 프레임워크 [Zaheer et al., 2017]를 기반으로 하며, Lévy 프로세스 증분 데이터의 **순열 불변성**을 활용하도록 설계되었습니다. 이는 이미지 분할에 사용되는 U-Net과는 구조적 목적이 다릅니다.

**아키텍처 구성 요소 및 흐름:**

1.  **입력 데이터 ($\mathbf{X}$):** Lévy 프로세스의 이산적으로 관측된 로그 수익률 증분 벡터 $\mathbf{X} = (\tilde{X}_1, \tilde{X}_2, \ldots, \tilde{X}_{n-1})^T$가 입력됩니다. 여기서 $\tilde{X}_j = X_{j+1} - X_j$입니다.
2.  **Summary Network ($\Lambda$):** 각 데이터 포인트 $\tilde{X}_i$는 Summary Network $\Lambda$ (매개변수 $\zeta_{\Lambda}$)를 통해 독립적으로 처리되어 학습된 특징 공간으로 매핑됩니다.
3.  **집계 함수 ($a(\cdot)$):** 변환된 특징들 $\left\{ \Lambda(\tilde{X}_i; \zeta_{\Lambda}) \right\}_{i=1}^{n-1}$은 순열 불변 함수 $a(\cdot)$를 통해 단일 특징 벡터 $T(\mathbf{X}; \zeta_{\Lambda})$로 집계됩니다. 본 연구에서는 **산술 평균(arithmetic mean)**을 집계 함수로 사용했습니다.
4.  **Inference Network ($\phi$):** 집계된 특징 벡터 $T$는 Inference Network $\phi$ (매개변수 $\zeta_{\phi}$)를 통과하여 최종 매개변수 추정치 $\hat{\theta}$를 출력합니다.

**구체적인 네트워크 설정 (Page 20, 47):**
*   **네트워크 유형:** $\Lambda$와 $\phi$ 모두 완전 연결 심층 신경망(DNN)을 사용했습니다.
*   **레이어:** 3개의 은닉층, 각 레이어당 32개의 뉴런.
*   **활성화 함수:** Leaky ReLU.
*   **입력 차원 ($n_t$):** 최종 실험에서는 $n_t = 1,440$ (하루 동안의 1분 로그 수익률 수)으로 설정되었습니다.

### 수식 상세

#### 1. Bayes Risk (베이즈 위험)
베이즈 추정량 $\hat{\theta}^*$는 손실 함수 $L(\theta, \hat{\theta}(\mathbf{X}))$에 대해 베이즈 위험 $R_{\Omega}(\hat{\theta}(\cdot))$을 최소화하는 함수입니다.

$$R_{\Omega}(\hat{\theta}(\cdot)) = \int_{\Theta} \left( \int_{\mathcal{S}} L(\theta, \hat{\theta}(\mathbf{X})) f(\mathbf{x}|\theta) d\mathbf{x} \right) d\Omega(\theta) \quad (2)$$

여기서:
*   $\Theta$: 매개변수 공간.
*   $\mathcal{S}$: 데이터 실현 공간 ($\mathcal{S} = \mathbb{R}^n$).
*   $L(\cdot, \cdot)$: 비음수 손실 함수.
*   $f(\mathbf{x}|\theta)$: $\theta$가 주어졌을 때 데이터 $\mathbf{X}$의 밀도 함수.
*   $\Omega(\cdot)$: $\theta$에 대한 사전 분포.

#### 2. Neural Bayes Estimator (NBE) 최적화 문제
신경망 매개변수 $\zeta$를 포함하는 신경망 추정량 $\hat{\theta}(\mathbf{X}, \zeta)$를 훈련하여 베이즈 위험을 근사적으로 최소화합니다.

$$\zeta^* = \arg \min_{\zeta} R_{\Omega}(\hat{\theta}(\cdot, \zeta))$$

Monte Carlo 근사화를 통한 경험적 위험(Empirical Risk)은 다음과 같습니다 (Eq. 3):

$$R_{\Omega}(\hat{\theta}(\cdot, \zeta)) \approx \frac{1}{K} \sum_{\theta \in \Theta} \frac{1}{J} \sum_{\mathbf{X} \in \mathcal{X}_{\theta}} L(\theta, \hat{\theta}(\mathbf{X}, \zeta)) \quad (3)$$

여기서 $K$는 사전 분포 $\Omega(\cdot)$에서 샘플링된 매개변수 수, $J$는 각 $\theta$에 대해 시뮬레이션된 데이터셋 수, $\mathcal{X}_{\theta}$는 $\theta$에서 생성된 $J$개의 데이터셋 집합입니다.

#### 3. DeepSets 아키텍처 수식
순열 불변성을 강제하는 신경망 추정량 $\hat{\theta}(\mathbf{X}; \zeta)$는 다음과 같이 정의됩니다 (Eq. 4):

$$\hat{\theta}(\mathbf{X}; \zeta) = \phi \left( T(\mathbf{X}; \zeta_{\Lambda}); \zeta_{\phi} \right)$$
$$T(\mathbf{X}; \zeta_{\Lambda}) = a \left( \left\{ \Lambda(\tilde{X}_i; \zeta_{\Lambda}) \right\}_{i=1}^{n-1} \right) \quad (4)$$

여기서:
*   $\phi$와 $\Lambda$: 각각 Inference Network와 Summary Network (심층 신경망).
*   $a(\cdot)$: 순열 불변 함수 (본 연구에서는 평균).
*   $\tilde{X}_i$: Lévy 프로세스의 $i$번째 증분.

#### 4. 손실 함수 (Loss Function)
본 연구에서 최종적으로 선택된 손실 함수는 **평균 제곱 로그 오차(Mean Squared Logarithmic Error, MSLE)**입니다 (Page 20).

$$L(\theta, \hat{\theta}) = \frac{1}{P} \sum_{p=1}^{P} (\log(1 + \theta_p) - \log(1 + \hat{\theta}_p))^2$$
(여기서 $P$는 매개변수의 총 개수이며, $\theta_p$와 $\hat{\theta}_p$는 $p$번째 참값과 추정값입니다. MSLE는 특히 양수 값 매개변수(예: 분산, 강도)에 대해 0 근처의 작은 값에 민감하게 반응하도록 합니다.)

### Vanilla U-Net 비교

이 논문에서 사용된 DeepSets 기반 NBE 아키텍처는 이미지 분할에 사용되는 **Vanilla U-Net**과는 근본적으로 다릅니다.

| 특징 | DeepSets 기반 NBE | Vanilla U-Net |
| :--- | :--- | :--- |
| **주요 목적** | 시계열 데이터의 매개변수 점 추정 | 이미지 분할 (픽셀 단위 분류) |
| **핵심 구조** | Summary Network ($\Lambda$) + 집계 함수 ($a$) + Inference Network ($\phi$) | 인코더(Downsampling) + 디코더(Upsampling) + 스킵 연결 |
| **불변성** | **순열 불변성(Permutation Invariance)** 활용 (입력 순서 무관) | 공간적 계층 구조 및 지역적 특징 추출 |
| **입력/출력** | 입력: Lévy 증분 벡터 $\mathbf{X}$. 출력: 매개변수 벡터 $\hat{\theta}$ (점 추정치) | 입력: 이미지. 출력: 분할 마스크 (입력과 동일한 해상도) |
| **추가/수정 모듈** | Lévy 증분의 순열 불변성을 활용하는 **Summary Network ($\Lambda$)와 평균 집계 함수 ($a$)**가 핵심적인 추가 모듈임. | 해당 없음. DeepSets는 U-Net의 인코더-디코더 구조를 사용하지 않음. |

---

## 4. 태그 제안 (Tags Suggestion)

1.  Lévy Processes (Lévy 프로세스)
2.  Neural Bayes Estimation (신경망 베이즈 추정)
3.  Likelihood-Free Inference (우도-자유 추론)
4.  Amortized Inference (상각된 추론)
5.  DeepSets Architecture (DeepSets 아키텍처)