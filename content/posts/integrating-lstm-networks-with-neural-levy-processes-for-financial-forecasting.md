---
categories:
- Literature Review
- U-Net
date: 2025-11-26
draft: false
params:
  arxiv_id: 2512.07860v1
  arxiv_link: http://arxiv.org/abs/2512.07860v1
  pdf_path: //172.22.138.185/Research_pdf/2512.07860v1.pdf
tags:
- Financial Forecasting (금융 예측)
- LSTM Networks
- Lévy Processes
- Jump Diffusion Models (점프 확산 모델)
- Neural Network Calibration (신경망 보정)
title: Integrating LSTM Networks with Neural Levy Processes for Financial Forecasting
---

## Abstract
This paper investigates an optimal integration of deep learning with financial models for robust asset price forecasting. Specifically, we developed a hybrid framework combining a Long Short-Term Memory (LSTM) network with the Merton-Lévy jump-diffusion model. To optimise this framework, we employed the Grey Wolf Optimizer (GWO) for the LSTM hyperparameter tuning, and we explored three calibration methods for the Merton-Levy model parameters: Artificial Neural Networks (ANNs), the Marine Predators Algorithm (MPA), and the PyTorch-based TorchSDE library. To evaluate the predictive performance of our hybrid model, we compared it against several benchmark models, including a standard LSTM and an LSTM combined with the Fractional Heston model. This evaluation used three real-world financial datasets: Brent oil prices, the STOXX 600 index, and the IT40 index. Performance was assessed using standard metrics, including Mean Squared Error (MSE), Mean Absolute Error(MAE), Mean Squared Percentage Error (MSPE), and the coefficient of determination (R2). Our experimental results demonstrate that the hybrid model, combining a GWO-optimized LSTM network with the Levy-Merton Jump-Diffusion model calibrated using an ANN, outperformed the base LSTM model and all other models developed in this study.

## PDF Download
[Local PDF View](//172.22.138.185/Research_pdf/2512.07860v1.pdf) | [Arxiv Original](http://arxiv.org/abs/2512.07860v1)

이 논문은 **"Integrating LSTM Networks with Neural Lévy Processes for Financial Forecasting"**에 대한 상세 분석 리포트입니다.

---

## 1. 요약 (Executive Summary)

이 논문은 금융 자산 가격 예측의 정확성과 견고성을 높이기 위해 딥러닝 모델과 금융 확률 모델을 통합한 하이브리드 프레임워크를 제안합니다.

*   **하이브리드 모델 제안:** Long Short-Term Memory (LSTM) 네트워크와 Merton-Lévy 점프-확산(Jump-Diffusion) 모델을 결합하여 자산 가격의 연속적인 변동과 불연속적인 급등(Jumps) 현상을 동시에 포착합니다.
*   **최적화 기법 적용:** LSTM 네트워크의 하이퍼파라미터를 최적화하기 위해 메타휴리스틱 알고리즘인 Grey Wolf Optimizer (GWO)를 사용했습니다.
*   **모델 보정(Calibration) 방법 비교:** Merton-Lévy 모델의 파라미터($\mu, \sigma, \lambda, m, \delta$)를 추정하기 위해 세 가지 방법론(Artificial Neural Networks, Marine Predators Algorithm, TorchSDE)을 비교 분석했습니다.
*   **실험 데이터셋:** Brent 유가, STOXX 600 지수, IT40 지수 등 세 가지 실세계 금융 시계열 데이터를 사용했습니다.
*   **주요 성과:** GWO로 최적화된 LSTM과 **NN(신경망)으로 보정된 Lévy-Merton Jump-Diffusion 모델**의 조합이 표준 LSTM 모델 및 다른 벤치마크 모델 대비 가장 우수한 예측 성능(최저 MSE, 최고 $R^2$)을 달성했습니다. 특히 NN 기반 보정 방법이 다른 방법론(MPA, TorchSDE) 대비 계산 효율성(run time) 측면에서도 월등히 뛰어났습니다.

---

## 2. 7가지 핵심 질문 분석 (Key Analysis)

### 1) What is new in the work? (기존 연구와의 차별점)

이 연구의 가장 큰 차별점은 **GWO로 최적화된 LSTM** 예측과 **신경망(NN)으로 보정된 Merton-Lévy Jump-Diffusion 모델**을 결합한 앙상블 하이브리드 프레임워크를 제시했다는 점입니다. 기존 연구들이 딥러닝 모델이나 확률 모델 중 하나에 집중하거나, 하이브리드 모델을 사용하더라도 전통적인 수치적 보정 방법을 사용했던 것과 달리, 이 논문은 딥러닝(LSTM)과 확률 모델(Lévy-Merton)의 장점을 결합하고, 특히 **NN을 사용하여 복잡한 확률 모델의 파라미터를 빠르고 정확하게 보정**하는 새로운 접근 방식을 탐구했습니다.

### 2) Why is the work important? (연구의 중요성)

이 연구는 금융 시장의 핵심적인 특징인 **'점프(Jumps)'와 '확률적 변동성(Stochastic Volatility)'**을 효과적으로 모델링하여 자산 가격 예측의 정확도를 높였다는 점에서 중요합니다. 순수 딥러닝 모델은 복잡한 비선형 관계를 학습하지만, 금융 시장의 급격한 변화(예: 거시경제 이벤트)를 설명하는 데 한계가 있습니다. 반면, Merton-Lévy 모델은 점프 현상을 포착하지만, 파라미터 보정이 어렵습니다. 이 하이브리드 접근법은 딥러닝의 예측 능력과 금융 모델의 해석 가능성 및 견고성을 결합하여, 특히 극단적인 가격 움직임의 위험을 식별하는 데 더 정확한 확률적 예측을 제공합니다.

### 3) What is the literature gap? (기존 연구의 한계점)

기존 연구의 주요 한계점은 다음과 같습니다. 첫째, Black-Scholes와 같은 전통적인 확률 모델은 상수 변동성(constant volatility)을 가정하여 실제 금융 시장의 경험적 특성(점프, 왜도, 첨도)을 반영하지 못합니다. 둘째, Heston이나 Merton 모델과 같은 고급 확률 모델은 파라미터가 많아 **적절한 보정(calibration)이 필수적이지만, 이는 시간이 오래 걸리고 정확도가 떨어질 수 있습니다.** 셋째, 순수 딥러닝 모델은 해석 가능성(interpretability)이 부족하고 과적합(overfitting)에 취약하여 시장 상황 변화 시 일반화 능력이 제한됩니다.

### 4) How is the gap filled? (해결 방안)

이 연구는 세 가지 주요 구성 요소를 통해 한계를 해결합니다.
1. **LSTM:** 대규모 데이터 내의 복잡한 비선형 관계와 시계열 의존성을 학습하여 예측 성능을 향상시킵니다.
2. **Merton-Lévy Jump-Diffusion Model:** 자산 가격 동역학에 점프 요소를 통합하여 시장의 불연속적인 움직임을 현실적으로 표현합니다.
3. **Neural Networks (NN) Calibration:** 복잡한 Merton-Lévy 모델의 파라미터를 시장 가격을 입력으로 받아 실시간으로 추정함으로써, 기존의 느리고 불안정한 수치적 보정 방법의 문제를 해결하고 계산 효율성을 높입니다.

### 5) What is achieved with the new method? (달성한 성과)

제안된 하이브리드 모델(LSTM-Lévy with NN calibration)은 모든 데이터셋에서 벤치마크 모델들을 능가했습니다.

| 데이터셋 | 모델 | MAE | MSE | $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Brent Oil** | **LSTM-Lévy (NN)** | $0.0004500$ | $\mathbf{0.0000045}$ | $\mathbf{0.996976}$ |
| Brent Oil | LSTM (Standalone) | $0.0021000$ | $0.0000210$ | $0.9858887$ |
| **STOXX 600** | **LSTM-Lévy (NN)** | $0.0009000$ | $\mathbf{0.0000090}$ | $\mathbf{0.9766209}$ |
| STOXX 600 | LSTM (Standalone) | $0.0012000$ | $0.0000120$ | $0.9688279$ |
| **IT 40** | **LSTM-Lévy (NN)** | $0.0006500$ | $\mathbf{0.0000065}$ | $\mathbf{0.9981446}$ |
| IT 40 | LSTM (Standalone) | $0.0011500$ | $0.0000115$ | $0.996717$ |

**Table 분석:** Brent Oil 데이터셋에서 LSTM-Lévy (NN) 모델은 MSE $0.0000045$를 기록하여, 단독 LSTM 모델의 MSE $0.0000210$ 대비 약 4.6배 낮은 오류를 보였습니다. $R^2$ 값 역시 $0.996976$로 거의 완벽에 가까운 설명력을 보여주었습니다. 또한, **Table 2, 4, 6**의 Run Time 분석 결과, NN 기반 보정은 TorchSDE나 MPA 대비 수십 배 빠른 계산 효율성을 입증했습니다 (예: Brent Oil 데이터셋에서 NN은 240초, TorchSDE는 3158초 소요).

### 6) What data are used? (사용 데이터셋)

세 가지 실세계 금융 시계열 데이터셋이 사용되었으며, 모두 2010년 1월부터 2024년 7월까지의 일별 가격 데이터를 포함합니다.
1.  **Brent Oil Price Dataset:** 글로벌 유가 벤치마크인 Brent 원유의 일일 가격. (상품 시장 특성)
2.  **STOXX 600 Dataset:** 17개 유럽 국가의 대형, 중형, 소형주 600개 기업을 포함하는 유럽 주식 시장 지수. (광범위한 유럽 주식 시장 특성)
3.  **IT40 Dataset (FTSE MIB):** 이탈리아 주식 시장의 주요 벤치마크 지수. (단일 국가 주식 시장 특성)

### 7) What are the limitations? (저자가 언급한 한계점)

저자는 다음과 같은 한계점과 개선 기회를 언급했습니다.
1.  **확산 및 점프 구성 요소 분리 어려움:** Merton-Lévy 모델에서 확산(Diffusion, $\sigma$)과 점프(Jump, $\lambda, \delta$) 구성 요소의 기여도를 명확히 분리하는 것이 어렵습니다. 특히 데이터셋이 작을 경우 파라미터 추정의 비식별성(non-identifiability) 문제가 발생할 수 있습니다.
2.  **파라미터 해석 가능성 및 과적합:** NN 기반 보정은 효율적이지만, 파라미터의 경제적 합리성을 보장하기 위해 베이지안 정규화(Bayesian regularization)와 같은 고급 정규화 기법이 필요합니다.
3.  **비정상성(Non-stationarity) 처리:** 14년(2010~2024)에 걸친 데이터는 지정학적 위기나 통화 정책 변화로 인한 구조적 변화(structural breaks)를 포함할 수 있으며, 기존의 선형 정규화(StandardScaler, MinMaxScaler)만으로는 이러한 비정상성을 완전히 해결하기 어렵습니다.

---

## 3. 아키텍처 및 방법론 (Architecture & Methodology)

### Figure 분석: 메인 아키텍처 (Figure 1)

Figure 1은 제안된 하이브리드 프레임워크의 전체 구조를 보여줍니다.

1.  **입력 (Stock prices):** 과거 주가 시계열 데이터($X_{t-2}, X_{t-1}, X_t$)가 입력됩니다.
2.  **LSTM 블록:** 입력된 시계열 데이터를 처리하여 다음 시점의 **가격(price)** 예측($X_{t+1}$)에 기여합니다.
3.  **확률 모델 블록 (Stochastic Model):** Fractional Heston 또는 Merton Jump-Diffusion 모델이 사용됩니다. 이 모델은 입력 데이터를 바탕으로 **변동성(volatility)** 정보를 생성합니다.
4.  **보정 방법 (Calibration Methods):** 확률 모델의 파라미터는 NN, MPA, 또는 TorchSDE를 통해 시장 데이터에 맞춰 최적화됩니다.
5.  **앙상블 예측:** 최종 예측 가격($X_{t+1}$)은 GWO로 튜닝된 LSTM의 예측과 NN으로 보정된 Merton Jump-Diffusion 모델의 예측을 결합한 앙상블 형태로 도출됩니다 (텍스트 설명 기반). 이 구조는 딥러닝의 예측력과 금융 모델의 변동성/점프 동역학을 통합하는 핵심입니다.

### 수식 상세

#### 1. Lévy Merton Jump-Diffusion Model (SDE)
자산 가격 $S_t$의 동역학을 나타내는 확률 미분 방정식(Stochastic Differential Equation, SDE)은 다음과 같습니다 (p. 5):

$$d S_t = S_t \left( (\mu - \lambda k) dt + \sigma d W_t + d Q_t \right)$$

여기서 각 항은 다음과 같습니다:
*   $S_t$: 시간 $t$에서의 자산 가격.
*   $\mu$: 자산의 표류율(drift rate, 예상 수익률).
*   $\lambda$: 포아송 과정(Poisson process)의 강도(intensity), 즉 점프의 빈도.
*   $k$: 예상 상대적 점프 크기($E[Y_i - 1]$).
*   $\sigma$: 위너 과정(Wiener process)의 연간 표준 편차(변동성).
*   $d W_t$: 표준 위너 과정(브라운 운동)의 미분. 연속적인 가격 변화(확산 성분)를 포착합니다.
*   $d Q_t$: 복합 포아송 과정(Compound Poisson process)으로 모델링된 점프 성분. 불연속적인 가격 변화를 포착합니다.

점프 성분 $d Q_t$는 다음과 같이 정의됩니다 (p. 5):

$$d Q_t = \sum_{i=1}^{N_t} (Y_i - 1)$$

여기서 $N_t$는 강도 $\lambda$를 갖는 포아송 과정이며, $Y_i$는 점프 크기를 나타내는 독립적이고 동일하게 분포된(i.i.d.) 로그 정규 분포 $\ln(Y_i) \sim \mathcal{N}(m, \delta^2)$를 따르는 확률 변수입니다.

#### 2. Calibration Loss Function (NN 기반 보정)
신경망을 사용하여 모델 파라미터 $\theta$를 보정할 때 사용되는 손실 함수는 시장 가격과 모델 가격 간의 평균 제곱 오차(MSE)입니다 (p. 6):

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} (P_{\text{market}, i} - P_{\text{model}, i}(\theta))^2$$

여기서:
*   $N$: 데이터 포인트의 수.
*   $P_{\text{market}, i}$: $i$번째 관측된 시장 가격.
*   $P_{\text{model}, i}(\theta)$: 파라미터 $\theta$를 사용하여 모델이 계산한 가격.

#### 3. 신경망 파라미터 업데이트 (Gradient Descent)
신경망의 가중치 $W^{(l)}$와 편향 $b^{(l)}$는 경사 하강법(Adam)을 사용하여 다음과 같이 업데이트됩니다 (p. 7):

$$W_{\text{new}}^{(l)} = W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}$$

$$b_{\text{new}}^{(l)} = b^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial b^{(l)}}$$

여기서 $\eta$는 학습률(learning rate)입니다.

### Vanilla U-Net 비교 (LSTM/Hybrid 구조 비교로 대체)

이 논문은 U-Net 구조를 사용하지 않고 LSTM을 사용하므로, 비교 대상을 **기존의 단독 LSTM(Vanilla LSTM)**과 **제안된 하이브리드 LSTM-Lévy 모델**로 대체하여 설명합니다.

| 특징 | Vanilla LSTM | 제안된 Hybrid LSTM-Lévy (NN) |
| :--- | :--- | :--- |
| **기본 구조** | 시계열 데이터의 시간적 의존성 학습. | LSTM 예측과 Merton-Lévy 모델 예측의 앙상블. |
| **변동성 모델링** | 내재적으로 학습된 비선형 패턴에 의존. | Merton-Lévy 모델을 통해 명시적으로 점프 및 확률적 변동성($\sigma, \lambda, \delta$)을 모델링. |
| **점프 처리** | 급격한 변화를 이상치(outlier)로 처리하거나 학습하기 어려움. | $d Q_t$ 성분을 통해 불연속적인 가격 점프를 명시적으로 포착. |
| **최적화** | 표준 경사 하강법(Adam) 사용. | GWO를 사용하여 LSTM 하이퍼파라미터 튜닝. |
| **파라미터 보정** | 해당 없음. | NN을 사용하여 확률 모델 파라미터를 빠르고 정확하게 보정. |
| **주요 장점** | 비선형 관계 학습 능력. | 높은 예측 정확도, 금융 시장 특성(점프) 반영, 빠른 보정 속도. |

---

## 4. 태그 제안 (Tags Suggestion)

1.  Financial Forecasting (금융 예측)
2.  LSTM Networks
3.  Lévy Processes
4.  Jump Diffusion Models (점프 확산 모델)
5.  Neural Network Calibration (신경망 보정)