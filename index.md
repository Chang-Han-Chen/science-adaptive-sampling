---
title: "Variance reduction, adaptive sampling, and a mystery"
# date: 2026-11-30         # set on the day we publish
affiliation: UC Berkeley
authors:
  - name: Chang-Han Chen
    url: https://chang-han-chen.github.io/
excerpt: "When is adaptive sampling about variance reduction? Can we solve it as a stochastic optimization under (training+inference) compute constraint?"
---


<div class="notice--info" markdown="1">
**TL;DR**

We investigate adaptive sampling in LLM RL with binary rewards.

1. We survey the past literature and identify two distinct focuses: **signal seeking**, which aims to get the first success on a hard prompt, and **variance reduction**, which aims to reduce the noise once the mean gradient is non-zero.
2. We pose variance reduction as a stochastic optimization problem: how many more rollouts $$n_i$$ to sample for each prompt $$i$$, and how many successes $$S_i$$ and failures $$F_i$$ to train on. Schematically,

   $$
   \begin{aligned}
   \{n_i,S_i,F_i\}={}&\arg\min\ \operatorname{Var}(\text{batch gradient})\\
   &\text{s.t.}\ \mathbb E\big[\text{inference tokens} + 2 \times \text{training tokens}\big]\le C.
   \end{aligned}
   $$

   Of course, the numbers of successes and failures at our disposal after sampling vary from run to run, so our algorithm also adapts to all possible scenarios. We solve the optimization problem exactly with an integer program solver.
3. We empirically check our toy model of $$\operatorname{Var}(\text{batch gradient})$$ on competition-math rollouts from Qwen3-4B.
4. For Qwen3-8B on competition math, our algorithm's gradient has **17% less variance** and reaches **higher training and held-out accuracy than both GRPO-8 and Seq-Balance** (matched by optimizer step, total compute or training compute).
5. **A mystery.** When we turn off the sampler-learner KL, the gain almost goes away, while GRPO-8 barely changes. We suspect that adaptive sampling looks promising but also more brittle to off-policiness, and hope to understand how in the future ([Section 5](#mystery)).
</div>

Train accuracy for Qwen3-8B on competition math. Our algorithm wins by every standard ([Section 4.2](#training)).

<figure class="three-views">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}"><picture>
    <source media="(max-width: 600px)" srcset="{{ '/assets/images/train-accuracy-three-views-stacked.png' | relative_url }}">
    <img src="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}" alt="Accuracy on the training prompts for Seq-Balance, GRPO-8 and our algorithm, against optimizer updates, total compute and training compute. Our algorithm ends highest in all three panels.">
  </picture></a>
</figure>

## 1. Background: adaptive sampling for LLM RLVR {#background}

Reinforcement learning with verifiable rewards (RLVR) ([Lambert et al., 2024](#ref-lambert2024)) unlocks the reasoning capabilities of large language models (LLMs) ([OpenAI, 2024](#ref-openai2024); [DeepSeek-AI, 2025](#ref-deepseek2025)). The algorithm is remarkably simple and intuitive: sample a group of answers to each question, mark each one right or wrong, and make the right answers more likely and the wrong ones less likely. This is essentially the simplest RL algorithm, REINFORCE ([Williams, 1992](#ref-williams1992)),

$$
\nabla J(\theta)=\mathbb E_{y\sim\pi_\theta}\big[R(y)\,\nabla\log\pi_\theta(y)\big],
$$

with a binary reward $$R\in\{0,1\}$$. For binary reward, $$J$$ is just the average success rate (or accuracy) $$p$$.

It is kind of mind-blowing that, if you implement this simple algorithm with _extreme_ attention to detail, it scales! The top priority is to remove as much potential **bias** as possible, such as from numerical rounding or length normalization.

Then, you need to worry about variance reduction. Gradients in LLM RL are extremely noisy. In our measurements, the signal is more than two orders of magnitude smaller than the noise, in squared norm ([Section 3](#assumptions)). The simplest solution is to sample a lot of rollouts. As in classic statistics, the variance of the training signal goes down like the inverse of the number of samples $$B$$,

$$
\operatorname{Var}\sim\frac{1}{B}.
$$

Frontier-scale RL runs use very large batches: the main RL run of Inkling 2 Small took 1,075 steps of 16K rollouts each, over 17 million rollouts in total on 2,048 GPUs ([Thinking Machines Lab, 2026](#ref-tml2026)).

In addition, a common technique is a _reward baseline_ $$b$$. The reward in REINFORCE is then replaced by the _advantage_ $$A(y)=R(y)-b$$. A simple and effective baseline is the group's mean reward,

$$
b=\mathbb E[R]=p,
$$

and it is widely used in practice. When all the answers in a group get the same reward, however, every advantage is zero and the prompt teaches the model nothing. There is no learning signal in this case.

From this perspective, any RLVR sampling algorithm has two phases:

1. **Signal seeking:** sample enough to get at least one success on as many prompts of interest as possible.
2. **Variance reduction:** among the prompts with at least one success, how should we allocate compute to reduce the variance of the batch gradient?

The figure below follows one prompt through both phases. Each sketch shows how its gradient estimate varies from batch to batch, projected onto the direction of the true gradient, the direction that actually improves the model. The dashed line marks the true value.

<figure class="phase-fig" aria-label="Three stages of a prompt's gradient. With zero successes in eight rollouts, the gradient estimate is exactly zero: zero mean and zero variance. Signal seeking leads to one success in eight, where the estimate has a tiny mean and a huge variance. Variance reduction leads to more rollouts at the same success rate, which keeps the mean and shrinks the variance.">
<div class="pf-card">
<div class="pf-title">0 of 8 succeed</div>
<div class="pf-dots"><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span></div>
<svg viewBox="0 0 200 72" aria-hidden="true"><line x1="50" y1="58" x2="50" y2="10" class="pf-curve" stroke-width="3" stroke-linecap="round"/><line x1="8" y1="58" x2="192" y2="58" class="pf-axis"/><line x1="50" y1="55" x2="50" y2="61" class="pf-axis"/><text x="50" y="70" class="pf-tick">0</text></svg>
<div class="pf-sub"><span class="pf-hi">mean 0</span>, variance 0</div>
</div>
<div class="pf-arrow"><span>signal seeking</span><svg viewBox="0 0 72 14" aria-hidden="true"><line x1="2" y1="7" x2="66" y2="7"/><path d="M60 2 L68 7 L60 12" fill="none"/></svg></div>
<div class="pf-card">
<div class="pf-title">1 of 8 succeeds</div>
<div class="pf-dots"><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot s"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span></div>
<svg viewBox="0 0 200 72" aria-hidden="true"><line x1="76" y1="58" x2="76" y2="8" class="pf-mean"/><path d="M8.0 58.0 L8.0 54.5 L11.1 54.0 L14.1 53.5 L17.2 52.9 L20.3 52.3 L23.3 51.7 L26.4 51.0 L29.5 50.4 L32.5 49.7 L35.6 49.0 L38.7 48.3 L41.7 47.6 L44.8 46.9 L47.9 46.3 L50.9 45.7 L54.0 45.1 L57.1 44.6 L60.1 44.1 L63.2 43.7 L66.3 43.4 L69.3 43.2 L72.4 43.1 L75.5 43.0 L78.5 43.0 L81.6 43.1 L84.7 43.3 L87.7 43.6 L90.8 44.0 L93.9 44.4 L96.9 44.9 L100.0 45.5 L103.1 46.1 L106.1 46.7 L109.2 47.4 L112.3 48.1 L115.3 48.8 L118.4 49.4 L121.5 50.1 L124.5 50.8 L127.6 51.5 L130.7 52.1 L133.7 52.7 L136.8 53.3 L139.9 53.8 L142.9 54.3 L146.0 54.8 L149.1 55.2 L152.1 55.5 L155.2 55.9 L158.3 56.2 L161.3 56.5 L164.4 56.7 L167.5 56.9 L170.5 57.1 L173.6 57.2 L176.7 57.4 L179.7 57.5 L182.8 57.6 L185.9 57.7 L188.9 57.7 L192.0 57.8 L192.0 58.0 Z" class="pf-fill"/><path d="M8.0 54.5 L11.1 54.0 L14.1 53.5 L17.2 52.9 L20.3 52.3 L23.3 51.7 L26.4 51.0 L29.5 50.4 L32.5 49.7 L35.6 49.0 L38.7 48.3 L41.7 47.6 L44.8 46.9 L47.9 46.3 L50.9 45.7 L54.0 45.1 L57.1 44.6 L60.1 44.1 L63.2 43.7 L66.3 43.4 L69.3 43.2 L72.4 43.1 L75.5 43.0 L78.5 43.0 L81.6 43.1 L84.7 43.3 L87.7 43.6 L90.8 44.0 L93.9 44.4 L96.9 44.9 L100.0 45.5 L103.1 46.1 L106.1 46.7 L109.2 47.4 L112.3 48.1 L115.3 48.8 L118.4 49.4 L121.5 50.1 L124.5 50.8 L127.6 51.5 L130.7 52.1 L133.7 52.7 L136.8 53.3 L139.9 53.8 L142.9 54.3 L146.0 54.8 L149.1 55.2 L152.1 55.5 L155.2 55.9 L158.3 56.2 L161.3 56.5 L164.4 56.7 L167.5 56.9 L170.5 57.1 L173.6 57.2 L176.7 57.4 L179.7 57.5 L182.8 57.6 L185.9 57.7 L188.9 57.7 L192.0 57.8" class="pf-curve" fill="none"/><line x1="8" y1="58" x2="192" y2="58" class="pf-axis"/><line x1="50" y1="55" x2="50" y2="61" class="pf-axis"/><text x="50" y="70" class="pf-tick">0</text></svg>
<div class="pf-sub">tiny mean, huge variance</div>
</div>
<div class="pf-arrow"><span>variance reduction</span><svg viewBox="0 0 72 14" aria-hidden="true"><line x1="2" y1="7" x2="66" y2="7"/><path d="M60 2 L68 7 L60 12" fill="none"/></svg></div>
<div class="pf-card">
<div class="pf-title">3 of 24 succeed</div>
<div class="pf-dots"><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot s"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot s"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot s"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span><span class="pf-dot"></span></div>
<svg viewBox="0 0 200 72" aria-hidden="true"><line x1="76" y1="58" x2="76" y2="8" class="pf-mean"/><path d="M8.0 58.0 L8.0 58.0 L11.1 58.0 L14.1 58.0 L17.2 58.0 L20.3 58.0 L23.3 58.0 L26.4 58.0 L29.5 58.0 L32.5 58.0 L35.6 57.9 L38.7 57.8 L41.7 57.6 L44.8 57.1 L47.9 56.2 L50.9 54.4 L54.0 51.5 L57.1 47.1 L60.1 41.0 L63.2 33.6 L66.3 25.5 L69.3 18.1 L72.4 12.5 L75.5 10.1 L78.5 11.3 L81.6 15.8 L84.7 22.8 L87.7 30.8 L90.8 38.6 L93.9 45.2 L96.9 50.2 L100.0 53.6 L103.1 55.7 L106.1 56.9 L109.2 57.5 L112.3 57.8 L115.3 57.9 L118.4 58.0 L121.5 58.0 L124.5 58.0 L127.6 58.0 L130.7 58.0 L133.7 58.0 L136.8 58.0 L139.9 58.0 L142.9 58.0 L146.0 58.0 L149.1 58.0 L152.1 58.0 L155.2 58.0 L158.3 58.0 L161.3 58.0 L164.4 58.0 L167.5 58.0 L170.5 58.0 L173.6 58.0 L176.7 58.0 L179.7 58.0 L182.8 58.0 L185.9 58.0 L188.9 58.0 L192.0 58.0 L192.0 58.0 Z" class="pf-fill"/><path d="M8.0 58.0 L11.1 58.0 L14.1 58.0 L17.2 58.0 L20.3 58.0 L23.3 58.0 L26.4 58.0 L29.5 58.0 L32.5 58.0 L35.6 57.9 L38.7 57.8 L41.7 57.6 L44.8 57.1 L47.9 56.2 L50.9 54.4 L54.0 51.5 L57.1 47.1 L60.1 41.0 L63.2 33.6 L66.3 25.5 L69.3 18.1 L72.4 12.5 L75.5 10.1 L78.5 11.3 L81.6 15.8 L84.7 22.8 L87.7 30.8 L90.8 38.6 L93.9 45.2 L96.9 50.2 L100.0 53.6 L103.1 55.7 L106.1 56.9 L109.2 57.5 L112.3 57.8 L115.3 57.9 L118.4 58.0 L121.5 58.0 L124.5 58.0 L127.6 58.0 L130.7 58.0 L133.7 58.0 L136.8 58.0 L139.9 58.0 L142.9 58.0 L146.0 58.0 L149.1 58.0 L152.1 58.0 L155.2 58.0 L158.3 58.0 L161.3 58.0 L164.4 58.0 L167.5 58.0 L170.5 58.0 L173.6 58.0 L176.7 58.0 L179.7 58.0 L182.8 58.0 L185.9 58.0 L188.9 58.0 L192.0 58.0" class="pf-curve" fill="none"/><line x1="8" y1="58" x2="192" y2="58" class="pf-axis"/><line x1="50" y1="55" x2="50" y2="61" class="pf-axis"/><text x="50" y="70" class="pf-tick">0</text></svg>
<div class="pf-sub">same mean, <span class="pf-hi">less variance</span></div>
</div>
</figure>

Notice that a group with all successes or all failures contributes **zero** variance, not infinite. A group with one success has a real signal, but everything it says about what a right answer looks like comes from that single answer, so the variance is huge. This is another way to see why signal seeking and variance reduction are two _separate_ algorithmic considerations. The first moves a prompt from the left box to the middle one; the second moves it from the middle box to the right one.

We find that this perspective provides a nice dichotomy of the past literature.

**From 0 to 1: signal seeking.** 

This direction seems popular already and has been adopted at frontier scale. 

For example, DAPO oversamples prompts, drops all-agree groups, and keeps sampling until the batch is full of prompts with signal ([Yu et al., 2025](#ref-yu2025)). ScaleRL keeps this zero-variance filter in a recipe it scales to a single 100,000-GPU-hour run, and also stops revisiting prompts the model already passes at least 90% of the time ([Khatri et al., 2025](#ref-khatri2025)); Olmo 3 refills its batches the same way, which it calls active sampling ([Team Olmo, 2025](#ref-olmo2025)). Never Give Up goes further and keeps sampling a prompt until one answer is correct, so easy prompts are cleared in a few attempts and the rest of the budget goes to hard ones ([Noukhovitch et al., 2026](#ref-noukhovitch2026)).

We personally think this issue is better tackled one level up, such as better midtraining data and curricula that meet the edge of the model's capabilities. How to find that edge automatically is a very interesting but separate subject in our opinion.

**From 1 to more: variance reduction.** 

This direction is obviously appealing to more theory-minded researchers (including us), but to the best of our knowledge, no open-source frontier RL pipeline has adopted such algorithms yet.

GVM-RAFT is probably the first work that has rigorous analysis on sample allocation and variance minimization in the context of LLM post-training ([Yao et al., 2025](#ref-yao2025)). Reinforce-Ada, a follow-up work, simplified and improved the algorithms with strong empirical results ([Xiong et al., 2025](#ref-xiong2025)). PODS generates many rollouts but trains only on a high-variance subset, because the update phase has costs of its own ([Xu et al., 2025](#ref-xu2025)). VIP predicts each prompt's success rate with a Gaussian process and solves for the allocation that minimizes total gradient variance under a rollout budget ([Nguyen et al., 2026](#ref-nguyen2026)), and DUET does the same under a budget of generated tokens ([Hu et al., 2026](#ref-hu2026)).

Importantly, these works budget **inference compute**. PODS already noticed that training costs compute too, but it fixes how many rollouts to train on. To the best of our knowledge, this project is the first to put **inference and training compute in one budget** and choose both together.


**The problem we solved.** 

We solve **variance minimization under a joint inference + training compute constraint**. But, wait, how could we get away with not worrying about the signal-seeking issue that we just explained? 

To avoid signal seeking, we adopt a **probe period** in our algorithm. At the start of each batch, every prompt gets $$8$$ samples. If a prompt scores either $$0/8$$ or $$8/8$$, we **give it up** for this batch and spend the rest of the compute on the other prompts.

The intuition is that these prompts simply contribute nothing to the gradient if we don't sample more. On the other hand, if we try to sample more and luckily get a success out of all failures, its gradient would be non-zero 🥳 but very noisy 😱, so we might be better off not training on it. But if we likely won't train on it anyways, why bother sampling more in the first place? 

After the probe period, every prompt is guaranteed to have non-unanimous responses, aka the regime of variance reduction! 

Now, we can explain the output of our algorithm. For each prompt, our algorithm outputs how many more rollouts $$n_i$$ to **sample**, and how many successes $$S_i$$ and failures $$F_i$$ to **train** on. This way, adaptive sampling becomes exactly a stochastic programming problem.

Schematically,

<div class="problem schematic" markdown="1">
**Problem (optimal adaptive sampling at fixed total compute).**

$$
\begin{aligned}
\{n_i,S_i,F_i\}={}&\arg\min\ \operatorname{Var}(\text{batch gradient})\\
&\text{s.t.}\ \mathbb E\big[\text{inference tokens} + 2 \times \text{training tokens}\big]\le C
\end{aligned}
$$

</div>

You might be wondering, why does the algorithm need to output $$S_i$$ and $$F_i$$? Doesn't adaptive sampling only need the allocation $$n_i$$? The answer is because our constraint is the **total compute**! The more rollouts we train on, the fewer rollouts we can sample. Given each $$\{n_i\}$$, the algorithm needs to consider all possible outcomes (e.g., all-failure, 1-success-$$(n_i-1)$$-failure, etc.). For each outcome, it then finds which rollouts to train on to minimize variance. It then takes the average over all outcomes to get the expected variance for $$\{n_i\}$$. Notice that in doing so, the algorithm has already prepared itself for _any_ outcome that could come out of the $$\{n_i\}$$ extra samples. 


We know this might sound complicated, so we will work through a simple, explicit example in Section 2 to give you some more intuition. But you should also feel free to treat the integer program as a black box: it reads a few numbers per prompt and returns $$n_i$$, $$S_i$$ and $$F_i$$, so it can sit inside an existing GRPO pipeline.


## 2. Compute-optimal adaptive sampling {#method}

Let's get into the details of our algorithm! 

Each math question has a success rate $$p$$. Each response from the model is either right ($$R=1$$) or wrong ($$R=0$$). We fix the reward baseline to the expected reward, $$b=p$$, for simplicity, so the advantage for the right response is $$(1-p)$$ and the wrong one is $$-p$$. You might wonder why we don't optimize the baseline for variance minimization together. This is a bit technical and distracting anyways; if you are interested, click this:

<details markdown="1">
<summary>Why not let variance choose the baseline too?</summary>

**That is the classic route, and it works well when the training samples are fixed.** Every $$b$$ gives the same expected gradient, so classic RL picks the $$b$$ that makes the update least noisy. When we train on every rollout, the answer is roughly $$b=p$$ ([Williams, 1992](#ref-williams1992)).

**With adaptive sampling, we also choose which responses to train on, and then the same route ends somewhere strange.** At a fixed number of training rollouts, minimizing variance over $$b$$ and over what to keep picks $$b=0$$ when $$p<1/2$$ and $$b=1$$ when $$p>1/2$$: train on successes only, or on failures only. The rarer outcome has the larger mean score ($$\mu_+=\nabla p/p$$ for successes), so each rare response carries more signal for the same noise.

**So we start from somewhere else: what $$b$$ does to the policy.** Changing $$b$$ adds a noise term to the update. Writing $$R-b=(R-p)+(p-b)$$ gives $$g_b=g_p+(p-b)\,h$$, where $$h$$ is a reweighted sum of the kept responses' scores: a maximum-likelihood step on whatever was just sampled. It is zero on average, but it still moves the policy at second order. For a step $$\theta\to\theta+\eta g$$ and any observable $$O(\theta)$$ with Hessian $$H_O$$,

$$
\begin{aligned}
\mathbb E[\Delta O]={}&\eta\,\nabla O\cdot\nabla p+\tfrac{\eta^2}{2}\,\nabla p^{\top}H_O\,\nabla p\\
&+\tfrac{\eta^2}{2}\operatorname{tr}\big(H_O\operatorname{Cov}_b(g)\big)+O(\eta^3).
\end{aligned}
$$

The first line is the same for every $$b$$. The second is not, unless $$O$$ is linear. With $$b<p$$, $$h$$ makes what was sampled more likely to be sampled again, which sharpens the policy; with $$b>p$$, it pushes probability toward responses not yet tried. Training on successes only lowers pass@$$k$$ at large $$k$$, while training on failures only keeps the policy diverse ([Zhu et al., 2025](#ref-zhu2025)), and in bandits, baselines with equal variance make the policy commit early or keep exploring ([Chung et al., 2021](#ref-chung2021)).

**For adaptive sampling, we believe the natural choice is the neutral one, $$b=p$$.** There $$h$$ drops out, and responses outside the batch get no net push. We fix it first, and then minimize variance over everything else: how many rollouts to sample and which ones to train on.

</details>

We can write the policy gradient in terms of gradients on the right and wrong responses. For a prompt with success rate $$p$$, let $$\mu_+$$ and $$\mu_-$$ be the mean gradient of log prob (aka score) $$\nabla\log\pi_\theta(y)$$ conditioned on successful and failed answer respectively. Factorizing the policy gradient into two cases, we get $$\mu_+=\nabla p/p$$ and $$\mu_-=-\nabla p/(1-p)$$. So, the policy gradient is

$$
\nabla p=p(1-p)\,\big(\mu_+-\mu_-\big).
$$

From this, it is natural to guess the score difference $$(\mu_+-\mu_-)$$ might be independent of the success rate $$p$$, and therefore the policy gradient scales like

$$
\nabla p\propto p(1-p);
$$

empirically, we do observe this trend that will be shown later.

It is useful to define

$$
w_i=u_i(1-u_i),
$$

where $$u_i$$ is the prompt's success rate estimated from its past visits. This is the empirical estimate of $$p(1-p)$$, and the empirical policy gradient is

$$
g=\sum_i w_i\big(\bar S_{i,+}-\bar S_{i,-}\big),
$$

where $$\bar S_{i,+}$$ and $$\bar S_{i,-}$$ are the empirical mean scores conditioned on success and failure respectively. Training on more of each just averages away more noise, which is exactly what we pay compute for.

### 2.1 The precise problem {#problem}

Given all the preparation, here is the precise, technical version:

<div class="problem" markdown="1">
**Problem 1 (optimal adaptive sampling at fixed total compute).** 

Suppose a batch of $$B$$ prompts, from $$i=1$$ to $$B$$. Given $$\{(\hat p_i, P_i, T_{i,+},T_{i,-}, s_i)\}$$, where $$\hat p_i$$ is the forecast success rate, $$P_i$$ is the prompt length, $$T_{i,+}$$ and $$T_{i,-}$$ are the expected rollout lengths for successes and failures, and $$s_i$$, with $$1\le s_i\le7$$, is the number of successes during the probe period, 

return $$\{(n_i, S_i(x), F_i(x))\}$$, where

- $$n_i\in\{0,4,8,12,16,24\}$$ is the number of extra rollouts allocated to prompt $$i$$
- $$S_i(x)$$ and $$F_i(x)$$ are the optimal numbers of successes and failures to train on if another $$x$$ successes are observed after extra sampling. $$x$$ ranges in $$[0, n_i]$$, and $$S_i(x)$$ in $$[1, s_i+x]$$ and $$F_i(x)$$ in $$[1,(8+n_i)-(s_i+x)]$$,

that solves the following,

$$
\begin{aligned}
\min\ & \sum_i \mathbb E\left[w_i^2\left(\frac{T_{i,+}}{S_i(X_i)}+\frac{T_{i,-}}{F_i(X_i)}\right)\right]\\
\text{s.t.}\ & \sum_i \mathbb E\big[n_iG_i+2(P_i+T_{i,+})\,S_i(X_i) +2(P_i+T_{i,-})\,F_i(X_i)\big]\le C,
\end{aligned}
$$

where $$X_i\sim\operatorname{Binomial}(n_i,\hat p_i)$$ and $$G_i=P_i+\hat p_iT_{i,+}+(1-\hat p_i)T_{i,-}$$.

</div>

To write down this objective, we need **3** assumptions.

- **A1:** the squared score norm (SSN) of a response grows linearly with its length $$T$$:

  $$
  \mathbb E\lVert\nabla\log\pi_\theta(y)\rVert^2\approx vT.
  $$

- **A2:** successes and failures have the same SSN per token: A1 holds with the same $$v$$ for $$R=1$$ and $$R=0$$.
- **A3:** rollouts are independent, and noise dominates signal.

Under A1–A3, the variance of batch gradient is $$w_i^2(T_{i,+}/S_i+T_{i,-}/F_i)$$. The constraint just counts tokens: generating a rollout costs its tokens, $$P_i$$ plus its length, training on one costs twice that, and the probes are already paid for (but training on them is not). Section 3 checks all three assumptions.

### 2.2 The solution {#solution}

The most puzzling point of this optimization problem might be:

$$ \text{How do we plan for training compute if the outcomes of extra rollouts are uncertain?} $$

Say a prompt has very high success rate $$p=0.9$$, then it is natural to expect most rollouts will be successful and plan training compute accordingly. But what if we are very unlucky and get all failures? How does the algorithm adapt to the actual outcomes?

The key is that all the outcomes of sampling $$n_i$$ rollouts are _already_ considered when the algorithm compute its expected variance. More specifically, $$(S_i(x), F_i(x))$$ is just a table for each $$n_i$$, where the rows are different $$x$$, and the columns are $$S_i$$ and $$F_i$$. The optimization problem boils down to finding all the tables and compare/aggregate them. 

We simply store these tables from the intermediate steps before extra sampling even happens, and then read off the row that actually happened afterward.

To find the tables, we simply use a **mixed-integer linear program**. Every expectation in Problem 1 is a finite sum over the outcomes $$x$$, and both the objective and the compute are linear in which $$n$$ and which counts we pick. So Problem 1 is a mixed-integer linear program, with one yes-or-no variable for each candidate $$n$$ and for each candidate count in each row. An off-the-shelf solver, HiGHS, solves it to proven optimality once per step, in a few seconds for 30 prompts (Section 2.4).

Here is Problem 1 rephrased as a mixed-integer linear program:
<div class="theorem" markdown="1">
**Theorem (Adaptive sampling is a integer linear program)**

Problem is equivalent to the following.

For every prompt $$i$$, candidate $$n$$, outcome $$x\le n$$ and count $$k$$, let $$z_{i,n}\in\{0,1\}$$ mean "buy $$n$$ follow-ups", and let $$y^{+}_{i,n,x,k},\,y^{-}_{i,n,x,k}\in\{0,1\}$$ mean "keep $$k$$ successes (failures) in row $$x$$". 

Write $$\pi_{n}(x)$$ for the Binomial$$(n,\hat p_i)$$ probability of $$x$$, $$a^\pm_{i,k}=w_i^2T_{i,\pm}/k$$ for the variance of keeping $$k$$ answers of an outcome, and $$c^\pm_i=2(P_i+T_{i,\pm})$$ for the cost of training on one. 

Sums over $$k$$ run over the counts available in row $$x$$, and sums over $$\pm$$ over the two outcomes. Then

$$
\begin{aligned}
\min\ &\sum_{i,n,x}\pi_n(x)\sum_{k,\pm}a^\pm_{i,k}\,y^\pm_{i,n,x,k}\\
\text{s.t.}\ &\sum_n z_{i,n}=1,\quad \sum_k y^\pm_{i,n,x,k}=z_{i,n},\\
&\sum_{i,n}nG_i\,z_{i,n}\\
&\ +\sum_{i,n,x}\pi_n(x)\sum_{k,\pm}k\,c^\pm_i\,y^\pm_{i,n,x,k}\le C .
\end{aligned}
$$

</div>

<div class="proof" markdown="1">
*Proof.* A plan sets $$z_{i,n_i}=1$$, $$y^+_{i,n_i,x,S_i(x)}=y^-_{i,n_i,x,F_i(x)}=1$$ for every $$x$$, and every other variable to 0; this satisfies the constraints. Conversely, the constraints force a feasible point to pick exactly one $$n$$ per prompt and exactly one count of each outcome in every row of that $$n$$, which is a plan. Under this correspondence the program's objective and compute are the expectations in Problem 1 written as sums over $$x$$, so the two problems have the same feasible plans with the same values. HiGHS returns a solution with a certified zero optimality gap. <span class="qed">∎</span>

</div>

### 2.3 A simple example {#example}

Let's make all of this concrete. Take two prompts of 200 tokens each, and say we have 140,000 tokens of budget left after the probes.

| | Hard prompt | Even prompt |
|---|---|---|
| History before this visit | 2 successes, 10 failures | 6 successes, 6 failures |
| Mean answer length | successes 3,000, failures 6,000 | 3,000 for both |

The hard prompt has been solved twice in twelve tries so far, and its failures are twice as long as its successes. The even prompt is a coin flip.

**1. Estimate $$p$$ and $$w$$.** From the history, $$u=3/14\approx0.21$$ and $$u=0.5$$, so $$w=0.168$$ and $$w=0.25$$. These stay fixed no matter what happens in this step.

**2. Probe.** Sample eight rollouts of each prompt and grade them. The hard prompt gets 1 success, and the even prompt gets 4. Both probes are mixed, so both prompts stay in the batch.

**3. Plan** Say, the solver finds the best $$\{n_1,n_2\} = \{4,0\}$$ with the following table for the first prompt

| Successful follow-ups $$x$$ | Chance | Available | Keep |
|---|---|---|---|
| 0 | 44% | 1 + 11 | 1 + 5 |
| 1 | 40% | 2 + 10 | 2 + 4 |
| 2 | 14% | 3 + 9 | 3 + 4 |
| 3 | 2% | 4 + 8 | 4 + 4 |
| 4 | 0.1% | 5 + 7 | 5 + 4 |


**4. Sample.** Draw the hard prompt's 4 follow-ups and grade them. Say one succeeds, so $$x=1$$.

**5. Keep.** Row 1 says 2 + 4, so the first prompt keep both successes and 4 of the 10 failures, chosen at random. The second prompt keeps its 4 + 4.

**7. Update.** Multiply the gradient of log prob by $$R-p$$ for each group. 


## 3. Checking the assumptions {#assumptions}

Problem 1 relies on three assumptions, so let's check them. 

Each one does one job. **A1** says the variance scale linearly in length, which is why the objective counts tokens, $$T_{i,+}/S_i$$ and $$T_{i,-}/F_i$$. **A2** puts successes and failures on one scale, so the SSN per token $$v$$ factors out. **A3** allows us to add up the variance easily, without estimating the signals for each prompt.

We test A1 and A2 on a frozen Qwen3-4B in thinking mode, with 40 hard competition-math prompts and 48 rollouts each, and A3 on the Qwen3-8B checkpoint of Section 4. The short answer: all three hold well. Intervals are 95%.

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th>Assumption</th><th>What we measured</th><th>Result</th></tr>
</thead>
<tbody>
<tr><td>A1. SSN ∝ length</td><td>fitted exponent of SSN vs. length, layers 18–35</td><td class="hl">0.97 ± 0.06</td></tr>
<tr><td>A2. Same SSN per token for both outcomes</td><td>SSN of incorrect / correct responses at matched length</td><td class="hl">1.005 ± 0.013</td></tr>
<tr><td>A3. Noise dominates</td><td>‖E g‖² / tr Cov(g) per batch, Qwen3-8B replay (§4)</td><td class="hl">at most 0.5%</td></tr>
</tbody>
</table>
</div>

Here is the plot to support A1 and A2. 
<figure class="narrow">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/score-energy-vs-length.png' | relative_url }}"><img src="{{ '/assets/images/score-energy-vs-length.png' | relative_url }}" alt="Squared score norm against response length on log-log axes, for layers 18 to 35 of Qwen3-4B. Correct and incorrect answers both lie on the diagonal from about 5,000 to 16,000 tokens." loading="lazy"></a>
</figure>

For each response, we measure its SSN $$\lVert\nabla\log\pi_\theta(y)\rVert^2$$ in layers 18–35 and divide it by its prompt's average SSN per token. If A1 holds, that number should equal the response's length, so responses fall on the dashed diagonal. 

If A2 also holds, correct and incorrect responses fall on the same diagonal. Each marker averages the responses in one length octile, with a 95% interval.



Both outcomes sit right on the diagonal across a threefold range of lengths. Responses cut off at the length cap, all of them incorrect, carry about 4% more SSN than the line predicts ($$1.036\pm0.013$$ times). And A3 holds by a wide margin: at the Qwen3-8B checkpoint, a batch's mean gradient is at most 0.5% of its noise.

As a fun aside (we don't need this), A1 and A2 actually predict the norm of policy gradient too:

$$
\operatorname{tr}\operatorname{Cov}\big((R-p)\nabla\log\pi_\theta(y)/\sqrt{T}\big)\approx v\,p(1-p).
$$

The following plot tests this on the same frozen Qwen3-4B, using LoRA gradients for 10 prompts with success rates from 0.08 to 0.94 and 1,000 to 3,000 rollouts each. Each dot is one prompt's per-token gradient variance, with a 95% interval, and the dashed line is the best fit proportional to $$p(1-p)$$.

<figure class="narrow">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/variance-vs-pq.png' | relative_url }}"><img src="{{ '/assets/images/variance-vs-pq.png' | relative_url }}" alt="Per-token gradient variance of plain policy gradient against p(1-p) for ten prompts. The points rise roughly in proportion to p(1-p); one prompt near p(1-p)=0.22 lies well above the line." loading="lazy"></a>
</figure>

The prediction holds roughly. Most prompts lie within about 15% of the line, and fitting the exponent freely gives $$0.95\pm0.17$$. The spread is wider than the error bars, though, and one prompt at $$p\approx0.68$$ is about 50% noisier than the line: $$v$$ is not quite the same for every prompt, while Problem 1 treats it as one constant.

## 4. Results {#results}

### 4.1 At equal compute, our algorithm cuts gradient variance by 17% {#replay}

The cleanest test of a variance-reduction method is to measure the variance directly, so that's what we did first.

**Setup.** We take a Qwen3-8B checkpoint after 100 steps of RL training. We sampled 64 independent snapshots, each with 32 rollouts per prompt, and replayed two rules on every snapshot: **GRPO-8**, which trains on eight rollouts of every prompt, and **our algorithm** (Section 2), with the per-prompt compute budget it was trained with. Both use StableReinforce with KL coefficient 0.01, the loss they were trained with.

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th>Per batch of eight prompts, relative to GRPO-8</th><th>GRPO-8</th><th>Our algorithm</th></tr>
</thead>
<tbody>
<tr><td>Gradient variance</td><td>1</td><td>0.85 [0.78, 0.93]</td></tr>
<tr><td>Compute</td><td>1</td><td>0.98</td></tr>
<tr class="rule-above"><td>Gradient variance at equal compute</td><td>1</td><td class="hl">0.83 [0.77, 0.91]</td></tr>
</tbody>
</table>
</div>

Our batches have 15% less gradient variance and cost 2% less compute. So **at equal compute, our gradient has 17% less variance**. Brackets are 95% bootstrap intervals over eight independent blocks of eight snapshots.

<details markdown="1">
<summary>How we measured this</summary>

**Tinker returns gradient norms, not gradients, and that is enough.** Gradients accumulate across calls until the optimizer steps, so we can read the norm of a single batch or of a sum of batches, with the optimizer at zero learning rate. We read the norm of every snapshot, of disjoint pairs and blocks of snapshots, and of all snapshots together, and checked that the policy did not change. For $$B$$ independent batch gradients $$g_b$$ with mean $$G$$,

$$
\widehat V=\frac{\sum_b\lVert g_b\rVert^2-\lVert\sum_b g_b\rVert^2/B}{B-1},
$$

$$
\widehat{\lVert G\rVert^2}=\frac{\lVert\sum_b g_b\rVert^2-\sum_b\lVert g_b\rVert^2}{B(B-1)}
$$

are unbiased for $$\operatorname{tr}\operatorname{Cov}(g)$$ and $$\lVert G\rVert^2$$. For a pair they reduce to $$\lVert g_1-g_2\rVert^2/2$$ and $$g_1\cdot g_2$$. The signal estimate is signed, so it can come out negative when the true signal is small.

**The raw numbers.**

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th>Per batch of eight prompts</th><th>GRPO-8</th><th>Our algorithm</th></tr>
</thead>
<tbody>
<tr><td>Compute (weighted tokens)</td><td>528k</td><td>516k</td></tr>
<tr><td>Variance, \(\operatorname{tr}\operatorname{Cov}(g)\), ×10⁵</td><td>11.1 [10.4, 11.9]</td><td>3.1 [2.8, 3.3]</td></tr>
<tr><td>Squared mean, \(\lVert\mathbb E g\rVert^2\), ×10³</td><td>2.6 [−0.1, 5.3]</td><td>0.3 [−0.4, 1.0]</td></tr>
</tbody>
</table>
</div>

**Putting the two on the same scale.** The two rules estimate the gradient at different scales, so the raw variances cannot be compared directly. The measured means cannot settle the scale either: both intervals include zero, because a batch's mean gradient is a fraction of a percent of its noise (A3). So we work out the scale from the rules themselves. GRPO-8 subtracts the group's mean reward, and that mean includes the response's own reward: $$r_j-\bar r=\tfrac78\,(r_j-\bar r_{-j})$$, where $$\bar r_{-j}$$ is the mean of the other seven. That leave-one-out mean is independent of response $$j$$, so each response contributes $$\tfrac78\nabla p$$ on average, and the eight sum to $$7\,\nabla p$$ per prompt. Our algorithm gives the kept successes and failures totals of $$\pm4w$$, so it contributes $$4\,m(p)\nabla p$$ per prompt when the history is right (Section 2.4), which is $$4\,\nabla p$$ for a prompt whose probe is almost always mixed. On such prompts our gradient is $$4/7$$ of GRPO-8's. Multiplying it by $$7/4$$ takes our variance from 3.1 to 9.4 [8.7, 10.2], which is 0.85 of GRPO-8's 11.1.

Two notes on the fine print. The factor $$4/7$$ matches the scales only where $$m(p)\approx1$$: these eight prompts have $$m$$ between 0.54 and 0.99, and on the more one-sided ones our update counts for less by design. And the KL term, which is charged on every kept token, is not on this scale.

</details>

### 4.2 Training {#training}

**Setup.** We train Qwen3-8B on 300 prompts from the [Reinforce-Ada hard-prompt dataset](#ref-dataset), one seed each, at learning rate $$2\times10^{-5}$$ with the [StableReinforce](#ref-tinker) loss in Tinker and KL coefficient 0.01. 

Every method draws 30 prompts per batch. GRPO-8 samples eight answers to each prompt. Our algorithm has an expected budget of 1.96 million weighted tokens per update, probes included, and uses the weight $$u(1-u)$$, as in Section 4.1. 

We have 60 held-out prompts for evaluation. Every 25 updates, we draw 16 responses to each prompt and report averaged accuracy.

Here is held-out accuracy against total compute (left) and training compute (right), measured after 25, 50, 75 and 100 updates (and 110 for GRPO-8):

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/heldout-accuracy-vs-compute.png' | relative_url }}"><img src="{{ '/assets/images/heldout-accuracy-vs-compute.png' | relative_url }}" alt="Held-out accuracy against total compute and against training compute for Seq-Balance, GRPO-8 and our algorithm. Our algorithm is above GRPO-8 at every checkpoint on both axes; at the end of training the two are level in total compute." loading="lazy"></a>
</figure>

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th>At the end of training</th><th>GRPO-8</th><th>Seq-Balance</th><th>Our algorithm</th></tr>
</thead>
<tbody>
<tr><td>Updates</td><td>110</td><td>100</td><td>100</td></tr>
<tr><td>Total compute (EFLOPs)</td><td>3.13</td><td>2.66</td><td>3.16</td></tr>
<tr><td>Training compute (EFLOPs)</td><td>2.08</td><td>0.89</td><td>1.67</td></tr>
<tr><td>Train accuracy</td><td>56.2%</td><td>52.9%</td><td>58.8%</td></tr>
<tr class="rule-above"><td>Held-out accuracy</td><td>53.2%</td><td>47.4%</td><td class="hl">53.7%</td></tr>
</tbody>
</table>
</div>

At equal total compute, our algorithm and GRPO-8 end closely on the held-out prompts, while on the training prompts our algorithm is 2.6 points ahead. At equal training compute, our algorithm is ahead on the held-out prompts too: after 75 updates, it has used 1.28 EFLOPs of training compute and reaches 53.8%, while GRPO-8 after 75 updates has used 1.36 EFLOPs and reaches 47.4%. Seq-Balance ends at 47.4% on the held-out prompts, with 16% less total compute.

On train accuracy, the win of our algorithm is more obvious:

<figure class="three-views">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}"><picture>
    <source media="(max-width: 600px)" srcset="{{ '/assets/images/train-accuracy-three-views-stacked.png' | relative_url }}">
    <img src="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}" alt="Accuracy on the training prompts for Seq-Balance, GRPO-8 and our algorithm, against optimizer updates, total compute and training compute. Our algorithm ends highest in all three panels." loading="lazy">
  </picture></a>
</figure>


## 5. A mystery {#mystery}

All our training runs use Tinker's [StableReinforce](#ref-tinker) loss, which comes with a KL term between the sampler and the learner, with coefficient 0.01. We did not expect this term to favor our algorithm over GRPO, or even to have any noticeable effect at our horizon length. However, it matters _a lot_! Here are our algorithm and GRPO-8, trained with the KL term (left) and without it (right):

<figure class="three-views">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/train-accuracy-kl.png' | relative_url }}"><picture>
    <source media="(max-width: 600px)" srcset="{{ '/assets/images/train-accuracy-kl-stacked.png' | relative_url }}">
    <img src="{{ '/assets/images/train-accuracy-kl.png' | relative_url }}" alt="Train accuracy against optimizer updates for our algorithm and GRPO-8, in two panels. With the KL term, our algorithm ends near 59 percent and GRPO-8 near 55 percent. Without it, both end near 54 to 55 percent." loading="lazy">
  </picture></a>
</figure>

With the KL term, our algorithm pulls ahead. Without it, the lead is nearly gone, and GRPO-8 barely notices the difference. All four runs share the same seed, prompts and learning rate, and each curve is the accuracy of the first four responses per prompt over the last 10 updates.


We don't understand this yet, but one thing we noticed is that 

$$ \text{adaptive sampling distorts sampling distribution.}$$

This is simply because we biased the distribution toward having more rare outcomes. 

We hope to study this scientifically in details.

## Citation {#citation}

{% raw %}
```bibtex
@misc{chen2026adaptivesampling,
  author = {Chen, Chang-Han},
  title  = {Variance reduction, adaptive sampling, and a mystery},
  year   = {2026},
  url    = {https://chang-han-chen.github.io/science-adaptive-sampling/},
  note   = {Blog post}
}
```
{% endraw %}

## References {#refs}

<div class="references" markdown="1">

1. {: #ref-chung2021} W. Chung, V. Thomas, M. C. Machado, and N. Le Roux. [Beyond variance reduction: Understanding the true impact of baselines on policy optimization](https://arxiv.org/abs/2008.13773). *ICML*, 2021.
1. {: #ref-deepseek2025} DeepSeek-AI. [DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948, 2025.
1. {: #ref-hu2026} H. Hu, X. Zhao, X. Xu, and N. Jacoby. [DUET: Optimize token-budget allocation for reinforcement learning with verifiable rewards](https://arxiv.org/abs/2605.08441). arXiv:2605.08441, 2026.
1. {: #ref-khatri2025} D. Khatri et al. [The art of scaling reinforcement learning compute for LLMs](https://arxiv.org/abs/2510.13786). arXiv:2510.13786, 2025.
1. {: #ref-lambert2024} N. Lambert et al. [Tülu 3: Pushing frontiers in open language model post-training](https://arxiv.org/abs/2411.15124). arXiv:2411.15124, 2024.
1. {: #ref-nguyen2026} H. T. Nguyen et al. [Adaptive rollout allocation for online reinforcement learning with verifiable rewards](https://arxiv.org/abs/2602.01601). *ICLR*, 2026.
1. {: #ref-noukhovitch2026} M. Noukhovitch, H. Ivison, N. Lambert, and A. Courville. [Learning to solve hard problems in RL for LLMs by never giving up](https://arxiv.org/abs/2609.13443). arXiv:2609.13443, 2026.
1. {: #ref-olmo2025} Team Olmo. [Olmo 3](https://arxiv.org/abs/2512.13961). arXiv:2512.13961, 2025.
1. {: #ref-openai2024} OpenAI. [OpenAI o1 system card](https://arxiv.org/abs/2412.16720). arXiv:2412.16720, 2024.
1. {: #ref-dataset} RLHFlow. [reinforce_ada_hard_prompt](https://huggingface.co/datasets/RLHFlow/reinforce_ada_hard_prompt) (dataset). Hugging Face.
1. {: #ref-tml2026} Thinking Machines Lab. [Introducing Inkling 2 Small](https://thinkingmachines.ai/news/inkling-2-small/). Blog post, November 2026.
1. {: #ref-tinker} Thinking Machines. [StableReinforce](https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/), Tinker documentation.
1. {: #ref-williams1992} R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning* 8, 229–256, 1992.
1. {: #ref-xiong2025} W. Xiong et al. [Reinforce-Ada: An adaptive sampling framework under non-linear RL objectives](https://arxiv.org/abs/2510.04996). arXiv:2510.04996, 2025. (Original Seq-Balance estimator: [v1, §2.3](https://arxiv.org/html/2510.04996v1#S2.SS3); importance weight: [v3, §3.2](https://arxiv.org/html/2510.04996v3#S3.SS2).)
1. {: #ref-xu2025} Y. E. Xu, Y. Savani, F. Fang, and J. Z. Kolter. [Not all rollouts are useful: Down-sampling rollouts in LLM reinforcement learning](https://arxiv.org/abs/2504.13818). *TMLR*, 2026.
1. {: #ref-yao2025} J. Yao et al. [Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and RL](https://arxiv.org/abs/2505.02391). arXiv:2505.02391, 2025.
1. {: #ref-yu2025} Q. Yu et al. [DAPO: An open-source LLM reinforcement learning system at scale](https://arxiv.org/abs/2503.14476). *NeurIPS*, 2025.
1. {: #ref-zhu2025} X. Zhu, M. Xia, Z. Wei, W.-L. Chen, D. Chen, and Y. Meng. [The surprising effectiveness of negative reinforcement in LLM reasoning](https://arxiv.org/abs/2506.01347). *NeurIPS*, 2025.

</div>
