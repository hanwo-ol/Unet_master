---
categories:
- Literature Review
- U-Net
date: 2025-07-06
draft: false
params:
  arxiv_id: 2507.04417v2
  arxiv_link: http://arxiv.org/abs/2507.04417v2
  pdf_path: //172.22.138.185/Research_pdf/2507.04417v2.pdf
tags:
- Auto-Generated
- Draft
title: Neural Networks for Tamed Milstein Approximation of SDEs with Additive Symmetric
  Jump Noise Driven by a Poisson Random Measure
---

## Abstract
This work aims to estimate the drift and diffusion functions in stochastic differential equations (SDEs) driven by a particular class of Lévy processes with finite jump intensity, using neural networks. We propose a framework that integrates the Tamed-Milstein scheme with neural networks employed as non-parametric function approximators. Estimation is carried out in a non-parametric fashion for the drift function $f: \mathbb{Z} \to \mathbb{R}$, the diffusion coefficient $g: \mathbb{Z} \to \mathbb{R}$. The model of interest is given by \[ dX(t) = ξ+ f(X(t))\, dt + g(X(t))\, dW_t + γ\int_{\mathbb{Z}} z\, N(dt,dz), \] where $W_t$ is a standard Brownian motion, and $N(dt,dz)$ is a Poisson random measure on $(\mathbb{R}_{+} \times \mathbb{Z}$, $\mathcal{B} (\mathbb{R}_{+}) \otimes \mathcal{Z}$, $λ( Λ\otimes v))$, with $λ, γ> 0$, $Λ$ being the Lebesgue measure on $\mathbb{R}_{+}$, and $v$ a finite measure on the measurable space $(\mathbb{Z}, \mathcal{Z})$. Neural networks are used as non-parametric function approximators, enabling the modeling of complex nonlinear dynamics without assuming restrictive functional forms. The proposed methodology constitutes a flexible alternative for inference in systems with state-dependent noise and discontinuities driven by Lévy processes.

## PDF Download
## PDF Download
[Local PDF View](//172.22.138.185/Research_pdf/2507.04417v2.pdf) | [Arxiv Original](http://arxiv.org/abs/2507.04417v2)

이 논문은 Lévy 과정에 의해 구동되는 가산 대칭 점프 노이즈(Additive Symmetric Jump Noise)를 포함하는 확률 미분 방정식(SDEs)의 표류(drift) 및 확산(diffusion) 함수를 신경망을 사용하여 비모수적으로 추정하는 방법론을 제안합니다.

---

## 1. 요약 (Executive Summary)

*   **연구 목표:** 유한 점프 강도(finite jump intensity)를 가진 Lévy 과정에 의해 구동되는 SDE의 표류 함수 $f(X(t))$와 확산 함수 $g(X(t))$를 비모수적으로 추정하는 프레임워크를 개발합니다.
*   **핵심 방법론:** 신경망(Neural Networks)을 비모수적 함수 근사기로 활용하고, 이를 **Tamed-Milstein 수치 기법**과 통합합니다.
*   **모델 형태:** 관심 SDE는 다음과 같습니다.
    $$dX(t) = \xi + f(X(t))dt + g(X(t))dW + \int_Z \gamma z N(dt, dz)$$
    여기서 $W$는 표준 브라운 운동, $N(dt, dz)$는 푸아송 랜덤 측도(Poisson Random Measure)입니다.
*   **훈련 전략:** 증분(increments)의 조건부 1차 및 2차 모멘트 최소화에 의존하는 손실 함수를 사용하여 신경망을 훈련합니다.
*   **주요 성과:** 제안된 방법론은 상태 의존적 노이즈와 불연속성(점프)을 가진 시스템에 대해 유연한 추론 대안을 제공하며, 수치 실험을 통해 연속 및 점프 구동 환경 모두에서 표류 및 확산 계수를 정확하게 추정함을 입증했습니다.
*   **수치적 우수성:** 제안된 알고리즘을 사용했을 때, 표류 및 확산 계수 추정의 최대 평균 제곱 오차(MSE)가 $10^{-3}$ 수준으로 나타나, 높은 정확도를 달성했습니다.

---

## 2. 7가지 핵심 질문 분석 (Key Analysis)

### 1) What is new in the work? (기존 연구와의 차별점)

이 연구는 신경망을 사용하여 점프를 포함하는 SDE의 계수를 추정하는 비모수적 프레임워크를 제안합니다. 기존 연구(예: [11])가 1차 수렴 순서(strong convergence order)를 갖는 Euler-Maruyama 근사를 사용했던 것과 달리, 본 논문은 **Tamed-Milstein 수치 기법**을 통합하여 더 높은 수렴 순서(점프가 없을 때 $\kappa=1$)를 달성합니다. 또한, 확산 계수 $g$가 수렴 영역 근처에서 과적합되는 문제를 해결하기 위해, 조건부 분산에 기반한 **선택적 교대 랜덤 훈련 전략(selective alternating random training strategy)**을 도입하여 훈련 데이터셋의 균형 잡힌 표현을 보장합니다.

### 2) Why is the work important? (연구의 중요성)

SDE는 금융, 생태학, 신경과학 등 불확실성과 노이즈에 의해 구동되는 시스템을 모델링하는 데 필수적입니다. 특히 점프-확산 모델은 자산 가격의 급격한 변화와 같은 불연속적인 현상을 포착할 수 있어 현실적인 모델링에 중요합니다. 전통적인 모수적 접근 방식은 함수 형태에 제한적인 가정을 부과하여 모델 오지정(misspecification)을 초래할 수 있지만, 이 연구에서 제안된 신경망 기반의 비모수적 접근 방식은 복잡한 비선형 동역학을 유연하게 모델링할 수 있게 합니다. Tamed-Milstein의 사용은 수치 근사의 견고성과 정확도를 높여 신뢰할 수 있는 추론을 가능하게 합니다.

### 3) What is the literature gap? (기존 연구의 한계점)

기존 연구의 주요 한계점은 두 가지입니다. 첫째, SDE 계수 추정을 위한 전통적인 방법은 함수 형태에 엄격한 구조적 제약을 부과했습니다. 둘째, 신경망을 사용한 기존의 비모수적 추정 프레임워크(예: [11])는 낮은 수렴 순서의 Euler-Maruyama 근사를 사용했습니다. 또한, 특히 궤적이 0으로 수렴하여 변동성이 사라지는 영역에서, 기존의 우도(likelihood) 기반 2단계 추정 절차는 확산 계수 $g$를 과소평가하는 경향이 있었습니다 (Figure 2 분석).

### 4) How is the gap filled? (해결 방안)

이 연구는 Tamed-Milstein 수치 기법을 채택하여 SDE 해의 2차 근사를 제공함으로써 수렴 순서 문제를 해결했습니다. 확산 계수 $g$의 과소평가 문제를 해결하기 위해 **3단계 알고리즘**을 제안했습니다. 이 알고리즘은 조건부 분산(conditional variance)을 기반으로 손실 함수를 정의하고, 높은 손실 값을 가진 브랜치(branch)를 선호하는 **선택적 랜덤 훈련**을 수행합니다. 특히, 훈련 샘플 선택 시 조건부 분산을 표준화하여(Equation 19), 궤적이 수렴하는 영역 근처에서 확산 계수가 과적합되는 것을 방지하고 훈련 세트의 균질성을 높입니다.

### 5) What is achieved with the new method? (달성한 성과)

제안된 방법론은 다양한 시나리오에서 높은 정확도로 $f$와 $g$를 추정했습니다.

| 시나리오 (Table I: $\lambda=0$ 또는 $\gamma=0$) | 표류 계수 $f(X_t)$ | 확산 계수 $g(X_t)$ | $L_{2,f}$ 오차 | $L_{2,g}$ 오차 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | $-0.25X_t^3$ | $0.57X_t$ | 0.00492 | 0.00347 |
| 2 | $0.15(X_t-X_t^5)$ | $0.32 \sin(X_t)$ | 0.00292 | 0.00013 |
| 3 | $1-X_t$ | 1 | 0.01600 | 0.00009 |

| 시나리오 (Table II: $\lambda \neq 0$ 및 $\gamma \neq 0$) | $(\gamma, \lambda)$ | 점프 $z_i$ | $f(X_t)$ | $g(X_t)$ | $L_{2,f}$ 오차 | $L_{2,g}$ 오차 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | (0.8, 1.2) | $U(-0.1, 0.1)$ | $1-X_t$ | $0.31X_t$ | 0.00401 | 0.00046 |
| 2 | (0.31, 1.7) | $N(0, 0.12)$ | $0.28(X_t-X_t^3)$ | 1 | 0.05564 | 0.00094 |
| 3 | (1.47, 0.5) | $Laplace(0, 0.1)$ | $\cos(X_t)$ | 1 | 0.00675 | 0.00008 |

전반적으로, 참 함수와 신경망 추정치 사이의 최대 MSE는 $10^{-2}$ 수준이며, 대부분의 경우 $10^{-3}$ 이하의 높은 정확도를 보였습니다. 특히, Phase 3 알고리즘을 사용했을 때 (Figure 4, Panel B), $f$와 $g$ 모두에 대해 MSE가 $0.0049$와 $0.0034$로 크게 개선되었습니다.

### 6) What data are used? (사용 데이터셋)

이 연구는 모두 **시뮬레이션된 합성 데이터**를 사용합니다.
*   **데이터 생성:** SDE (Equation 7)의 $K=10$개 독립 궤적을 $[0, 5]$ 구간에서 $N=1000$ 시간 단계로 균일하게 이산화하여 시뮬레이션했습니다. 초기 조건은 $X(0)=1.5$로 고정되었습니다.
*   **도메인 특성:** 점프 노이즈는 푸아송 랜덤 측도에 의해 구동되는 가산 대칭 점프 노이즈입니다. 점프 크기 $z_i$는 0에 대해 대칭이며 유한 2차 모멘트를 갖는다고 가정합니다.
*   **사용된 점프 분포:** 수치 실험에서는 균일 분포 $U(-0.1, 0.1)$, 정규 분포 $N(0, 0.12)$, 라플라스 분포 $Laplace(0, 1)$ 등이 사용되었습니다.

### 7) What are the limitations? (저자가 언급한 한계점)

저자가 언급한 주요 한계점은 다음과 같습니다.
1.  **비식별성(Non-identifiability):** 점프 크기 $z_i$가 스케일링 매개변수 $\gamma$에 의해 조정될 때, $\gamma$를 1로 설정하고 점프 값을 $\gamma z_i$로 대체하는 과정과 구별할 수 없어 $\gamma$가 비식별적일 수 있습니다. 이를 해결하기 위해 $z_i$를 표준화(평균 0, 단위 분산)해야 합니다.
2.  **일반 함수 $\gamma(x, z)$ 추정의 어려움:** 일반적인 상태 의존적 점프 함수 $\gamma(x, z)$를 추정하려면 점프 분포의 고차 모멘트를 평가해야 할 수 있으며, 이는 모델 가정에 따라 존재하지 않을 수 있습니다.
3.  **다차원 시스템 확장:** 제안된 방법론을 다차원 SDE로 확장하려면 수치 기법과 신경망 아키텍처 모두에 대한 신중한 조정이 필요합니다.
4.  **하이퍼파라미터 선택:** 훈련 하이퍼파라미터($R_1, R_2, R_3, R_4, \text{train}_f$)의 선택은 여전히 개방된 문제이며, 현재는 경험적으로 선택되고 있습니다.

---

## 3. 아키텍처 및 방법론 (Architecture & Methodology)

### Figure 분석: 아키텍처 구조

논문은 이미지 분할에 주로 사용되는 U-Net 구조를 사용하지 않고, 표류 함수 $f$와 확산 함수 $g$를 근사하기 위해 두 개의 독립적인 **다층 퍼셉트론(MLP) 또는 심층 신경망(DNN)**을 사용합니다.

*   **표류 함수 $f$를 위한 신경망:**
    *   입력층: 선형 입력층.
    *   은닉층: 4개의 은닉층, 각 층은 32개의 뉴런을 가지며 **ELU (Exponential Linear Unit)** 활성화 함수를 사용합니다.
    *   출력층: 선형 출력층.
*   **확산 함수 $g$를 위한 신경망:**
    *   입력층: 선형 입력층.
    *   은닉층: 3개의 은닉층, 각 층은 32개의 뉴런을 가지며 **ELU** 활성화 함수를 사용합니다.
    *   출력층: **Softplus** 활성화 함수를 사용하는 출력층. (확산 계수 $g$가 양수 값을 갖도록 보장하기 위함.)

### 수식 상세

#### SDE 모델 (Equation 1)
이 연구에서 다루는 SDE는 다음과 같습니다.
$$dX(t) = \xi + f(X(t))dt + g(X(t))dW + \int_Z \gamma z N(dt, dz)$$
여기서 $f: \mathbb{R} \to \mathbb{R}$는 표류 함수, $g: \mathbb{R} \to \mathbb{R}$는 확산 계수, $\gamma > 0$는 점프 크기 스케일링 매개변수입니다.

#### Tamed-Milstein 근사 (Equation 7)
시간 간격 $h$에 대한 이산화 근사 $X_{t+\Delta t}$는 다음과 같습니다.
$$\begin{aligned} X_{t+\Delta t} = X_t &+ f^{\Delta t}(X_t) \Delta t + g(X_t) \Delta W_t \\ &+ \frac{1}{2} g(X_t) g'(X_t) ((\Delta W_t)^2 - \Delta t) \\ &+ \sum_{i=1}^{N((t, t+\Delta t], Z)} \gamma z_i \\ &+ \sum_{i=1}^{N((t, t+\Delta t], Z)} (g(X_t + \gamma z_i) - g(X_t)) (\Delta W_{t+\Delta t} - \Delta W_{t_i}) \end{aligned}$$
여기서 $f^{\Delta t}(x)$는 테이밍된(tamed) 표류 항으로, 다음과 같이 정의됩니다.
$$f^{\Delta t}(x) = \frac{f(x)}{1+\Delta t f^2(x)}$$

#### 조건부 1차 모멘트 (Conditional Expectation, Equation 8)
$$E (X_{t+\Delta t} | \mathcal{F}(X_t)) = X_t + f^{\Delta t}(X_t) \Delta t$$

#### 조건부 2차 모멘트 (Conditional Second Moment, Equation 15)
증분의 조건부 분산은 $E(M_1^2 | \mathcal{F}(X_t)) + E(M_2^2 | \mathcal{F}(X_t))$로 주어지며, $\lambda \neq 0$ 및 $\gamma \neq 0$일 때 다음과 같습니다.
$$\begin{aligned} E \left( (X_{t+\Delta t} - E(X_{t+\Delta t} | \mathcal{F}(X_t)))^2 | \mathcal{F}(X_t) \right) &= g^2(X_t)\Delta t + \frac{1}{4} (g(X_t)g'(X_t)\Delta t)^2 \\ &+ \gamma^2 \mu_2 \lambda \Delta t + \text{higher order jump terms} \end{aligned}$$
여기서 $\mu_2 = E[z^2]$는 점프 크기의 2차 모멘트입니다.

#### 표류 함수 손실 $D_1$ (Loss Function for Drift, Equation 11)
Phase 1에서 표류 함수 $\hat{f}$를 추정하기 위해 사용되는 평균 제곱 오차(MSE) 손실 함수입니다.
$$D_1(\hat{f}, \hat{g}, B_k, j) := \frac{1}{|B_k|-1} \sum_{t_i \in B_k \setminus \{t_{k, |B_k|}\}} \left[ X_{j, t_{i+1}} - X_{j, t_i} - \hat{f}^{\Delta t}(X_{j, t_i}) \Delta t_{i+1} \right]^2$$

#### 확산 함수 손실 $D_2$ (Loss Function for Diffusion, Equation 12)
Phase 1에서 확산 함수 $\hat{g}$를 추정하기 위해 사용되는 근사 우도(likelihood) 손실 함수입니다.
$$D_2(\hat{f}, \hat{g}, B_k, j) := - \sum_{t_i \in B_k \setminus \{t_{k, |B_k|}\}} \log f_{t_i, \Delta t_i}^{M, h, a} (X_{j, t_{i+1}} | X_{j, t_i})$$
여기서 $f^{M, h, a}$는 특성 함수(Characteristic Function)를 이용해 근사된 조건부 밀도 함수입니다.

#### 표준화된 조건부 증분 $Y_{t, \Delta t}^*$ (Equation 19, $\gamma=0$일 때)
Phase 3에서 선택적 훈련을 위해 사용되는 표준화된 조건부 증분입니다.
$$Y_{t, \Delta t}^* := \frac{X_{t+\Delta t} - (X_t + \hat{f}^{\Delta t}(X_t)\Delta t)}{\sqrt{g^2(X_t)\Delta t + \frac{1}{4} (g(X_t)g'(X_t)\Delta t)^2}}$$

### Vanilla U-Net 비교

이 논문에서 사용된 아키텍처는 **U-Net 구조가 아닙니다.**

| 특징 | Vanilla U-Net | 본 논문의 아키텍처 (MLP/DNN) |
| :--- | :--- | :--- |
| **목적** | 이미지 분할, 시퀀스-투-시퀀스 매핑 | SDE 계수 $f(X)$ 및 $g(X)$의 비모수적 함수 근사 |
| **구조** | 인코더-디코더 구조, 스킵 커넥션 사용 | 표준 다층 퍼셉트론 (MLP) |
| **입력/출력** | 이미지/텐서 (고차원) | SDE 상태 $X_t$ (1차원 실수) / $f(X_t)$ 또는 $g(X_t)$ (1차원 실수) |
| **주요 모듈** | 컨볼루션, 풀링, 업샘플링 | 선형층, ELU/Softplus 활성화 함수 |
| **변경/추가된 모듈** | 해당 없음 | 확산 계수 $g$의 출력층에 **Softplus**를 사용하여 양수 값 보장 |

---

## 4. 태그 제안 (Tags Suggestion)

1.  Stochastic Differential Equations (SDEs)
2.  Lévy Processes / Jump Diffusion Models
3.  Tamed Milstein Scheme
4.  Neural Networks (NN) / Non-parametric Estimation
5.  Conditional Moments Inference