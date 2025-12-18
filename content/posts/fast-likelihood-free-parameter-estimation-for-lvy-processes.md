---
title: "Fast Likelihood-Free Parameter Estimation for Lévy Processes"
date: 2025-05-03
categories: ["Literature Review", "U-Net"]
tags: ["Auto-Generated", "Draft"]
draft: true
params:
  arxiv_id: "2505.01639v2"
  pdf_path: "//172.22.138.185/Research_pdf/2505.01639v2.pdf"
  arxiv_link: "http://arxiv.org/abs/2505.01639v2"
---

## Abstract
Lévy processes are widely used in financial modeling due to their ability to capture discontinuities and heavy tails, which are common in high-frequency asset return data. However, parameter estimation remains a challenge when associated likelihoods are unavailable or costly to compute. We propose a fast and accurate method for Lévy parameter estimation using the neural Bayes estimation (NBE) framework -- a simulation-based, likelihood-free approach that leverages permutation-invariant neural networks to approximate Bayes estimators. We contribute new theoretical results, showing that NBE results in consistent estimators whose risk converges to the Bayes estimator under mild conditions. Moreover, through extensive simulations across several Lévy models, we show that NBE outperforms traditional methods in both accuracy and runtime, while also enabling two complementary approaches to uncertainty quantification. We illustrate our approach on a challenging high-frequency cryptocurrency return dataset, where the method captures evolving parameter dynamics and delivers reliable and interpretable inference at a fraction of the computational cost of traditional methods. NBE provides a scalable and practical solution for inference in complex financial models, enabling parameter estimation and uncertainty quantification over an entire year of data in just seconds. We additionally investigate nearly a decade of high-frequency Bitcoin returns, requiring less than one minute to estimate parameters under the proposed approach.

## PDF Download
## PDF Download
[Local PDF View](//172.22.138.185/Research_pdf/2505.01639v2.pdf) | [Arxiv Original](http://arxiv.org/abs/2505.01639v2)

## 1. Visual Architecture Analysis (To be filled)
> Placeholder: Analyze the main architecture diagram (Figure 1/2).

## 2. Performance & Tables (To be filled)
> Placeholder: Extract SOTA metrics from tables.

## 3. Critical Review
> Placeholder: Strengths and limitations based on visual & text analysis.

