---
categories:
- Literature Review
- U-Net
date: 2025-12-09
draft: false
params:
  arxiv_id: 2512.08888v2
  arxiv_link: http://arxiv.org/abs/2512.08888v2
  pdf_path: //172.22.138.185/Research_pdf/2512.08888v2.pdf
tags:
- Rotation Invariance
- UAV Image Segmentation
- Scatter Convolution
- GPU Optimization
- Rotation Equivariance
title: Accelerated Rotation-Invariant Convolution for UAV Image Segmentation
---

## Abstract
Rotation invariance is essential for precise, object-level segmentation in UAV aerial imagery, where targets can have arbitrary orientations and exhibit fine-scale details. Conventional segmentation architectures like U-Net rely on convolution operators that are not rotation-invariant, leading to degraded segmentation accuracy across varying viewpoints. Rotation invariance can be achieved by expanding the filter bank across multiple orientations; however, this will significantly increase computational cost and memory traffic. In this paper, we introduce a GPU-optimized rotation-invariant convolution framework that eliminates the traditional data-lowering (im2col) step required for matrix-multiplication-based convolution. By exploiting structured data sharing among symmetrically rotated filters, our method achieves multi-orientation convolution with greatly reduced memory traffic and computational redundancy. We further generalize the approach to accelerate convolution with arbitrary (non-symmetric) rotation angles.   Across extensive benchmarks, the proposed convolution achieves 20--55% faster training and 15--45% lower energy consumption than CUDNN, while maintaining accuracy comparable to state-of-the-art rotation-invariant methods. In the eight-orientation setting, our approach achieves up to 45% speedup and 41% energy savings on 256\(\times\)256 inputs, and 32% speedup and 23% lower energy usage on 1024\(\times\)1024 inputs. Integrated into a U-Net segmentation model, the framework yields up to 6% improvement in accuracy over the non-rotation-aware baseline. These results demonstrate that the proposed method provides an effective and highly efficient alternative to existing rotation-invariant CNN frameworks.

## PDF Download
[Local PDF View](//172.22.138.185/Research_pdf/2512.08888v2.pdf) | [Arxiv Original](http://arxiv.org/abs/2512.08888v2)

이 논문은 UAV(무인 항공기) 이미지 분할을 위한 회전 불변 컨볼루션(Rotation-Invariant Convolution)의 효율성을 GPU 환경에서 극대화하는 새로운 프레임워크를 제안합니다.

---
## 1. 요약 (Executive Summary)

본 논문은 UAV 항공 이미지 분할에서 객체의 임의의 방향성(arbitrary orientation)으로 인해 발생하는 정확도 저하 문제를 해결하기 위해 회전 불변 컨볼루션을 가속화하는 방법을 제시합니다.

*   **문제 정의:** 기존의 U-Net과 같은 심층 학습 분할 아키텍처는 회전 불변성이 부족하여 다양한 시점에서 캡처된 UAV 이미지의 분할 정확도가 저하됩니다.
*   **기존 방법의 한계:** 회전 불변성을 달성하기 위해 필터 뱅크를 여러 방향으로 확장하는 기존 방법(예: G-convolution)은 계산 비용과 메모리 요구 사항을 크게 증가시킵니다. 특히 행렬 곱셈 기반 컨볼루션에 필요한 데이터 복제 단계(im2col)가 비효율성을 심화시킵니다.
*   **제안된 해결책 (Scatter-based Convolution):** 전통적인 데이터 로어링(im2col) 단계를 제거하고 대칭적으로 회전된 필터 간의 구조화된 데이터 공유를 활용하는 GPU 최적화된 스캐터(scatter) 기반 컨볼루션 프레임워크를 도입합니다.
*   **주요 성과:**
    *   비회전 인식(non rotation aware) 기준선 대비 분할 정확도를 최대 **5.7%** 향상시켰습니다.
    *   cuDNN 기반 구현 대비 **20~57%** 더 빠른 학습 속도와 **15~45%** 낮은 에너지 소비를 달성했습니다.
    *   스캐터 기반 설계의 효율성 덕분에 기존 방법으로는 불가능했던 **16개 방향** 컨볼루션 및 풀링을 실용적으로 구현하여 추가적인 정확도 향상을 얻었습니다.

---
## 2. 7가지 핵심 질문 분석 (Key Analysis)

### 1) What is new in the work? (기존 연구와의 차별점)

본 연구는 회전 불변 컨볼루션을 구현하는 데 있어 기존의 행렬 곱셈 기반 접근 방식(gather-style, im2col 사용) 대신 **스캐터(scatter) 기반** 컨볼루션 프레임워크를 GPU에 최적화하여 도입했다는 점이 새롭습니다. 이 스캐터 매핑은 데이터 로어링(im2col) 단계를 완전히 제거하여 메모리 트래픽과 계산 중복성을 줄입니다. 특히, 대칭적으로 회전된 필터들($0^\circ, 90^\circ, 180^\circ, 270^\circ$) 간에 중간 곱셈 결과를 재사용(reuse)함으로써, 각 대칭 그룹당 한 번의 곱셈만 수행하고 그 결과를 여러 출력 위치에 분산(scatter)시킵니다. 이는 기존 G-convolution이 각 회전 필터마다 독립적인 계산을 수행해야 했던 한계를 극복합니다.

### 2) Why is the work important? (연구의 중요성)

이 연구는 UAV 및 원격 감지 이미지 분할 분야에서 회전 불변성을 달성하는 데 필수적인 계산 효율성을 제공합니다. 기존 회전 불변 방법들은 방향성 해상도(orientation resolution)를 높일수록 기하급수적으로 증가하는 계산량과 메모리 오버헤드 때문에 고해상도 이미지에 적용하기 어려웠습니다. 본 논문에서 제안된 스캐터 기반 접근 방식은 계산 중복성과 메모리 오버헤드를 줄여, **실제 GPU 메모리 제약 조건 내에서** 8개 또는 16개와 같은 미세한 방향 샘플링을 실용적으로 가능하게 합니다. 이는 정확도와 효율성 사이의 유리한 균형점을 제시하며, 현대 CNN을 위한 실용적인 드롭인(drop-in) 업그레이드를 제공합니다.

### 3) What is the literature gap? (기존 연구의 한계점)

기존 연구의 주요 한계는 **계산 효율성**과 **확장성**입니다. G-convolution과 같은 회전 등변(equivariant) 컨볼루션은 회전된 필터마다 독립적인 계산을 수행해야 하므로, 2D 공간에서 $4$배, 3D 공간에서 최대 $24$배까지 계산 오버헤드가 증가합니다. 또한, GPU에서 컨볼루션을 행렬 곱셈으로 구현하기 위해 필수적인 `im2col` 방식은 입력 데이터를 $K_h \times K_w$ 배만큼 복제하여 메모리 트래픽과 계산량을 증가시킵니다. 이러한 비효율성 때문에, 방향성 해상도를 높여 정확도를 향상시키려는 시도는 GPU 메모리 제약으로 인해 실현 불가능했습니다.

### 4) How is the gap filled? (해결 방안)

이러한 한계는 **스캐터 기반 컨볼루션**을 통해 해결됩니다. 스캐터 컨볼루션은 입력 픽셀이 필터 가중치와 곱해진 후 결과가 해당 출력 위치로 분산되는 방식으로, `im2col`과 같은 데이터 로어링 단계를 우회합니다. 특히 대칭 회전($p4$ 그룹)의 경우, 동일한 입력-가중치 곱셈 결과가 네 개의 회전된 컨볼루션 모두에서 다른 출력 위치에 나타난다는 점을 활용하여, 곱셈을 한 번만 수행하고 그 결과를 네 방향의 출력 맵에 재사용합니다. 이로써 곱셈 횟수는 방향의 수에 관계없이 일정하게 유지되며, 계산은 주로 덧셈 연산에 의해 지배됩니다. 또한, 임의의 회전 각도에 대해서는 **Steerable Filter**와 대칭 매핑을 결합하여 보간(interpolation) 없이 회전 등변성을 달성하고, 필요한 고유 곱셈 횟수를 75%까지 줄입니다.

### 5) What is achieved with the new method? (달성한 성과 - *여기서 Table의 수치를 인용할 것*)

제안된 스캐터 기반 컨볼루션은 기존 방법 대비 압도적인 효율성을 보였습니다.

| 데이터셋 | 해상도 | 방향 수 | 방법 | Training Time (s) | Energy (kWh) | Test Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Plant | 256x256 | 4 | cuDNN | 20234.6 | 0.943 | 75.64 |
| Plant | 256x256 | 4 | **Scatter** | **10962.6** | **0.581** | 75.61 |
| Drone | 256x256 | 4 | cuDNN | 7713.1 | 0.3570 | 91.88 |
| Drone | 256x256 | 4 | **Scatter** | **4109.6** | **0.2175** | 91.88 |
| Plant | 1024x1024 | 8 | cuDNN | 100269.79 | 6.8162 | 75.86 |
| Plant | 1024x1024 | 8 | **Scatter** | **65703.51** | **5.3363** | **75.82** |

**주요 성과 요약:**

*   **4-방향 대칭 회전 (Table III, IV):** 256 해상도에서 Plant 데이터셋의 학습 시간을 cuDNN 대비 약 **45.9%** (20234.6s $\to$ 10962.6s) 단축했으며, Drone 데이터셋에서는 약 **46.7%** (7713.1s $\to$ 4109.6s) 단축했습니다. 정확도는 기존 SOTA 방법(E2CNN, ORN)과 동등하거나 우수했습니다.
*   **8-방향 대칭 회전 (Table V, VI):** 1024 해상도에서 Plant 데이터셋의 학습 시간을 cuDNN 대비 약 **34.5%** (100269.79s $\to$ 65703.51s) 단축했으며, 에너지 소비를 약 **21.7%** 절감했습니다.
*   **16-방향 대칭 회전 (Table VII, VIII):** 기존 방법으로는 비실용적이었던 16-방향 컨볼루션을 구현하여, 256 해상도 Drone 데이터셋에서 94.39%의 높은 정확도를 달성했습니다.

### 6) What data are used? (사용 데이터셋 - 도메인 특성 포함)

1.  **In-farm Plant Segmentation Dataset:**
    *   **도메인 특성:** 고해상도 UAV 이미지(GSD 0.15 cm/pixel)로 획득한 혼합 종 목초지(mixed-species pastures) 데이터셋입니다. 식물 종(클로버 잎, 잔디, 잡초 등)을 분할하는 것이 목표이며, 잎과 캐노피 구조가 임의의 방향으로 나타나기 때문에 회전 불변성이 필수적입니다.
    *   **규모:** 1000개 샘플 (학습 800, 검증 100, 테스트 100).

2.  **Semantic Segmentation Drone Dataset:**
    *   **도메인 특성:** 공개된 데이터셋으로, 도시 주거 지역의 고해상도 나디르 뷰(nadir-view) 이미지(고도 5m~30m)를 포함합니다. 24개 객체 범주에 대한 픽셀 수준의 의미론적 레이블이 지정되어 있습니다.
    *   **규모:** 400개 이미지 (학습 340, 검증 32, 테스트 28).

### 7) What are the limitations? (저자가 언급한 한계점)

저자는 제안된 스캐터 기반 컨볼루션이 모든 시나리오에서 cuDNN보다 우수하지는 않다고 언급합니다 (Appendix A 및 Conclusion).

*   **고해상도 초기 레이어에서의 성능:** 공간 해상도가 매우 높고 채널 깊이가 얕은 네트워크의 **첫 번째 또는 두 번째 인코더 레이어**에서는 cuDNN이 여전히 선호될 수 있습니다.
*   **최적 성능 조건:** 스캐터 커널의 성능 이점은 공간 풋프린트가 적당하거나($\le 32 \times 32$) 채널-필터 곱이 작은($\sim 16K$ 미만) 중간-깊은 레이어에서 가장 두드러집니다. 이는 커널 실행 지연 시간(kernel-launch latency)과 전역 메모리 트래픽이 컨볼루션 비용을 지배하는 영역입니다.

---
## 3. 아키텍처 및 방법론 (Architecture & Methodology)

### Figure 분석: U-Net 아키텍처 (Figure 9)

제안된 방법은 기존 U-Net 아키텍처를 기반으로 하며, 특히 인코더 경로의 기본 컨볼루션 블록을 회전 불변 컨볼루션 블록으로 대체했습니다.

**U-Net 구조의 흐름:**

1.  **입력:** $256 \times 256 \times 3$ (RGB 이미지).
2.  **인코더 경로 (Downsampling):**
    *   각 인코더 단계는 **회전 불변 컨볼루션 $3 \times 3$ + 표준 컨볼루션 $3 \times 3$ + BN + ReLU**로 구성된 블록으로 시작합니다.
    *   이후 **Maxpool $2 \times 2$**를 통해 공간 해상도를 절반으로 줄이고 채널 깊이를 두 배로 늘립니다 (예: $256 \times 256 \times 64 \to 128 \times 128 \times 128$).
3.  **바틀넥 (Bottleneck):** 가장 깊은 레이어($16 \times 16 \times 1024$)에서 동일한 회전 불변 컨볼루션 블록이 적용됩니다.
4.  **디코더 경로 (Upsampling):**
    *   **Up Sample**을 통해 해상도를 두 배로 늘립니다.
    *   인코더의 해당 해상도 피처 맵을 **Skip Connection**을 통해 연결(Concatenation)합니다.
    *   이후 **Conv $1 \times 1$**을 포함한 표준 컨볼루션 블록이 적용되어 채널 수를 줄이고 피처를 융합합니다.
5.  **출력:** 최종적으로 분할 마스크($256 \times 256 \times C_{out}$)를 생성합니다.

**핵심 변경 사항:** U-Net의 첫 번째 컨볼루션 블록을 제안된 **회전 불변 컨볼루션**으로 대체하여 회전 인식 피처 추출을 가능하게 합니다. 이 회전 불변 블록은 성능과 차원 폭발을 균형 있게 맞추기 위해, 첫 번째 컨볼루션에만 회전 불변 컨볼루션을 적용하고 두 번째 컨볼루션에는 표준 컨볼루션을 사용합니다.

### 수식 상세

#### 1. 표준 컨볼루션 출력 (Standard Convolution Output, Eq. 1)

입력 텐서 $X \in \mathbb{R}^{C_{in} \times H \times W}$와 필터 $W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$에 대한 유효(valid) 컨볼루션의 출력 $Y_{c_o, h, w}$는 다음과 같습니다.

$$Y_{c_o, h, w} = \sum_{c_i=0}^{C_{in}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} W_{c_o, c_i, i, j} X_{c_i, h+i, w+j}$$

여기서 출력 공간 크기는 $H' = H - K_h + 1$, $W' = W - K_w + 1$ 입니다.

#### 2. 스캐터 기반 컨볼루션 (Convolution with Scatter Operation, Eq. 4)

$C_{in}$ 입력 채널과 $C_{out}$ 출력 채널에 대한 스캐터 기반 컨볼루션은 다음과 같이 표현됩니다.

$$Y_{c_o, i-m+[K_h/2], j-n+[K_w/2]} += \sum_{c_i=0}^{C_{in}-1} X_{c_i, h, w} W_{c_o, c_i, m, n}$$

이 수식에서 $\sum_{c_i=0}^{C_{in}-1} X_{c_i, h, w} W_{c_o, c_i, m, n}$ 항은 채널별 곱셈 및 누적(accumulation)을 나타내며, 이 결과는 특정 출력 위치 $(i-m+[K_h/2], j-n+[K_w/2])$에 더해집니다. 이 방식은 데이터 복제 없이 행렬 곱셈을 사용하여 채널별 곱셈 및 합산을 수행합니다.

#### 3. 대칭 회전 등변 컨볼루션 (Symmetric Rotation Equivariant Convolution, Eq. 8)

$p4$ 그룹(4개의 대칭 회전 $r \in \{0, 1, 2, 3\}$)에 대해 스캐터 기반 컨볼루션을 적용할 때, 회전된 커널 좌표 $(m', n') = R_r(m, n)$를 사용하여 다음과 같이 표현됩니다.

$$Y_{c_o, i-m'+[K_h/2], j-n'+[K_w/2]} += \sum_{c_i=0}^{C_{in}-1} X_{c_i, h, w} W_{c_o, c_i, m, n}$$

핵심은 채널별 곱셈 및 누적 항 $\sum X W$이 **한 번만 계산**되고, 각 회전 $r$에 대해 커널 좌표 $(m, n)$가 회전된 좌표 $(m', n')$로 변환됨에 따라 **다른 출력 위치**에 분산된다는 점입니다.

#### 4. 임의 회전 스티어러블 필터 (Arbitrary Rotation Steerable Filter, Eq. 17)

임의의 회전 각도 $\theta$에 대한 스티어러블 필터 $\psi_\theta(x, y)$는 두 개의 학습된 기저 필터 $f_x$와 $f_y$의 선형 조합으로 구성됩니다.

$$\psi_\theta(x, y) = \sin(\theta) \cdot f_x(x, y) + \cos(\theta) \cdot f_y(x, y)$$

#### 5. 전체 손실 함수 (Total Loss Function, Eq. 18)

학습된 기저 필터가 스티어러블 동작을 보이도록 장려하기 위해 표준 교차 엔트로피 손실 $\mathcal{L}_{CE}$에 두 가지 정규화 손실이 추가됩니다.

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{mag} \mathcal{L}_{mag} + \lambda_{orth} \mathcal{L}_{orth}$$

*   **크기 일치 손실 (Magnitude-matching Loss, Eq. 19):** 두 기저 필터 $f_x$와 $f_y$의 평탄화된 커널 가중치 $w_b^{(x)}$와 $w_b^{(y)}$가 유사한 에너지 레벨을 유지하도록 장려합니다.
    $$\mathcal{L}_{mag} = \frac{1}{B} \sum_{b=1}^{B} \left( \left\| w_b^{(x)} \right\|_2^2 - \left\| w_b^{(y)} \right\|_2^2 \right)^2$$
*   **직교성 손실 (Orthogonality Loss, Eq. 20):** 두 방향 기저 간의 상관관계를 억제하여 회전 독립성을 촉진합니다.
    $$\mathcal{L}_{orth} = \frac{1}{B} \sum_{b=1}^{B} \left( \frac{\langle w_b^{(x)}, w_b^{(y)} \rangle}{\| w_b^{(x)} \|_2 \| w_b^{(y)} \|_2 + \epsilon} \right)^2$$

### Vanilla U-Net 비교

| 특징 | Vanilla U-Net | 제안된 U-Net (Scatter-based) |
| :--- | :--- | :--- |
| **핵심 컨볼루션** | 표준 컨볼루션 (Gather-style) | 스캐터 기반 회전 불변 컨볼루션 |
| **회전 불변성** | 없음 (회전 민감) | 회전 등변성 + 방향 풀링(Max/Avg)을 통해 달성 |
| **데이터 처리** | im2col을 통한 데이터 로어링 및 복제 | 스캐터 연산을 통해 데이터 로어링 제거 |
| **계산 효율성 (대칭 회전)** | 각 회전 필터마다 독립적인 계산 수행 | 중간 곱셈 결과를 재사용하여 곱셈 횟수 4배 감소 |
| **주요 추가 모듈** | 없음 | **Rotation-Invariant Conv Block** (Steerable Filter, Scatter Operation, Orientation Pooling) |
| **손실 함수** | $\mathcal{L}_{CE}$ | $\mathcal{L}_{CE} + \mathcal{L}_{mag} + \mathcal{L}_{orth}$ (정규화 항 추가) |

---
## 4. 태그 제안 (Tags Suggestion)

1.  Rotation Invariance
2.  UAV Image Segmentation
3.  Scatter Convolution
4.  GPU Optimization
5.  Rotation Equivariance