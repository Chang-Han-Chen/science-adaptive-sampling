---
title: "Variance reduction and adaptive sampling at fixed inference and training compute"
# date: 2026-11-30         # set on the day we publish
draft: true                # shows the draft banner; remove before publishing
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

   Of course, the numbers of successes and failures at our disposal after sampling vary from run to run, so our algorithm also adapts to all possible scenarios.
3. We solve the optimization problem exactly with an integer program solver. Intuitively, the update compares the average success with the average failure, so its noise is set mostly by whichever outcome we have fewer of. With 1 success and 7 failures, one more success cuts the noise by almost half, while one more failure cuts it by under 2%. So the solver spends compute on the rarer outcome.
4. We empirically check the noise model behind this on competition-math rollouts from Qwen3-4B.
5. On a frozen Qwen3-8B checkpoint, our algorithm's gradient has **17% less variance than GRPO-8's at equal compute**.
6. On competition math, with the small KL regularizer that Tinker's loss uses by default, our algorithm reaches **higher training accuracy than both GRPO-8 and Seq-Balance** (matched by optimizer step, total compute or training compute). Held-out accuracy is similar.
7. **A mystery.** When we turn that KL term off, the gain goes away, while GRPO-8 barely changes. We show that the KL term stops being neutral once you choose what to train on by outcome, and we derive and measure the size of this distortion. It turns out to be too small, and to point the wrong way, to explain what we see. Adaptive sampling looks promising but more sensitive to the rest of the training recipe, and we want to understand why ([Section 5](#mystery)).
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
**Problem (optimal adaptive sampling at fixed compute).**

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



**Because, left free, variance would train on one outcome only.** Every $$b$$ gives the same expected gradient, so once we also choose which answers to train on, variance can pick $$b$$ as well. At a fixed number of training rollouts, it picks $$b=0$$ when $$p<1/2$$ and $$b=1$$ when $$p>1/2$$: train on successes only, or on failures only. The rarer outcome has the larger mean score ($$\mu_+=\nabla p/p$$ for successes), so each rare answer carries more signal for the same noise. This is unlike classic RL, where we train on every rollout and the variance-optimal baseline is roughly $$p$$ ([Williams, 1992](#ref-williams1992); [Appendix B](#app-baseline)).

**What $$b$$ does change is how non-linear observables drift.** For a step $$\theta\to\theta+\eta g$$ and any observable $$O(\theta)$$ with Hessian $$H_O$$,

$$
\begin{aligned}
\mathbb E[\Delta O]={}&\eta\,\nabla O\cdot\nabla p+\tfrac{\eta^2}{2}\,\nabla p^{\top}H_O\,\nabla p\\
&+\tfrac{\eta^2}{2}\operatorname{tr}\big(H_O\operatorname{Cov}_b(g)\big)+O(\eta^3).
\end{aligned}
$$

The first line is the same for every $$b$$. The second is not, unless $$O$$ is linear. Writing $$R-b=(R-p)+(p-b)$$ splits the update into $$g_b=g_p+(p-b)\,h$$, where $$h$$ is a reweighted sum of the kept answers' scores: a maximum-likelihood step on whatever was just sampled, zero on average. With $$b<p$$, $$h$$ makes what was sampled more likely to be sampled again, which sharpens the policy; with $$b>p$$, it pushes probability toward answers not yet tried. Training on successes only lowers pass@$$k$$ at large $$k$$, while training on failures only keeps the policy diverse ([Zhu et al., 2025](#ref-zhu2025)), and in bandits, baselines with equal variance make the policy commit early or keep exploring ([Chung et al., 2021](#ref-chung2021)).

**So the baseline decides which observables stay stationary, not how noisy the update is.** With $$b=p$$, $$h$$ drops out: answers outside the batch get no net push, and the only second-order drift left is the one plain REINFORCE already has.

</details>

We can write the policy gradient in terms of gradients on the right and wrong responses. For a prompt with success rate $$p$$, let $$\mu_+$$ and $$\mu_-$$ be the mean score $$\nabla\log\pi_\theta(y)$$ of a successful and of a failed answer. The score identity gives $$\mu_+=\nabla p/p$$, and since the score has mean zero, $$\mu_-=-\nabla p/(1-p)$$. So, the policy gradient is

$$
\nabla p=p(1-p)\,\big(\mu_+-\mu_-\big).
$$

From this, it is natural to guess the score difference $$(\mu_+-\mu_-)$$ might be independent of the success rate $$p$$, and therefore the policy gradient scales like

$$
\nabla p\propto p(1-p)
$$

It will turn out useful to define the weight

$$
w_i=u_i(1-u_i),
$$

where $$u_i$$ is the prompt's success rate estimated from its past visits. This is the empirical estimate of $$p(1-p)$$.

So what does our update actually look like? In plain REINFORCE averaged over $$N$$ rollouts, about $$pN$$ right responses each carry $$(1-p)/N$$, so together they carry $$p(1-p)$$, and the wrong ones together carry $$-p(1-p)$$. We keep exactly these two totals, with $$w_i$$ in place of $$p(1-p)$$, but split each one evenly over the responses we actually train on: each kept success gets advantage $$+w_i/S_i$$, and each kept failure $$-w_i/F_i$$. The update is then

$$
g=\sum_i w_i\big(\bar S_{i,+}-\bar S_{i,-}\big),
$$

where $$\bar S_{i,+}$$ and $$\bar S_{i,-}$$ are the average scores of the kept successes and failures: the average right response minus the average wrong one, times $$w_i$$. Training on more of each just averages away more noise, which is exactly what we pay compute for.

### 2.1 The precise problem {#problem}

Given all the preparation, here is the precise, technical version:

<div class="problem" markdown="1">
**Problem 1 (adaptive sampling at fixed total compute).** 

Suppose a batch of $$B$$ prompts, from $$i=1$$ to $$B$$. Given $$\{w_i,\hat p_i, P_i, T_{i,+},T_{i,-}, s_i\}$$, where $$w_i$$ is the weight above, $$\hat p_i$$ is the forecast success rate, $$P_i$$ is the prompt length, $$T_{i,+}$$ and $$T_{i,-}$$ are the expected rollout lengths for successes and failures, and $$s_i$$, with $$1\le s_i\le7$$, is the number of successes during the probe period, choose 

- a number of follow-up rollouts $$n_i\in\{0,4,8,12,16,24\}$$, and
- for every possible number $$x$$ of successful follow-ups, how many successes $$S_i(x)\in[1,\,s_i+x]$$ and failures $$F_i(x)\in[1,\,8-s_i+n_i-x]$$ to keep,

to solve

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

Under A1–A3, each kept response adds noise proportional to its length, and averaging $$S$$ of them divides it by $$S$$. That's where $$w_i^2(T_{i,+}/S_i+T_{i,-}/F_i)$$ comes from. The constraint just counts tokens: generating a rollout costs its tokens, $$P_i$$ plus its length, training on one costs twice that, and the probes are already paid for (but training on them is not). Section 3 checks all three assumptions.

### 2.2 The solution {#solution}

Problem 1 looks scary, but it turns out to be friendly. Two observations do most of the work.

**First, a plan is just a table.** The follow-ups come out at random, so the plan has to say in advance what to keep for every possible outcome. For prompt $$i$$, a plan is a follow-up count $$n_i$$ plus a table with one row for each possible number $$x$$ of successful follow-ups, saying how many successes $$S_i(x)$$ and failures $$F_i(x)$$ to keep. We fix the table before sampling anything. Afterwards, we just read off the row that actually happened.

**Second, an integer program finds the best table exactly.** Every expectation in Problem 1 is a finite sum over the outcomes $$x$$, and both the objective and the compute are linear in which $$n$$ and which counts we pick. So Problem 1 is a mixed-integer linear program, with one yes-or-no variable for each candidate $$n$$ and for each candidate count in each row. An off-the-shelf solver, HiGHS, solves it to proven optimality once per step, in a few seconds for 30 prompts (Section 2.4).

**What does the optimal table look like?** You don't need this to use the algorithm, but it is a nice sanity check. Put a price $$\lambda$$ on compute. In one row, the success part of the cost, $$w_i^2T_{i,+}/S+2\lambda(P_i+T_{i,+})S$$, is smallest at

$$
S_i^\circ=w_i\sqrt{\frac{T_{i,+}}{2\lambda\,(P_i+T_{i,+})}}\;\approx\;\frac{w_i}{\sqrt{2\lambda}},
$$

and failures work the same way. Two rules of thumb fall out:

- **Length cancels.** A longer response is noisier, but it also costs more to train on, in the same proportion. So a prompt aims for about the same number of successes and failures whatever their lengths, and prompts with larger weights aim for more of both.
- **Follow-ups buy the outcome that runs short.** A new rollout is a lottery ticket: it lands in an outcome at random. When an outcome is below its target, the plan buys tickets while the expected gain pays for them, then keeps each outcome up to its target.

If you know survey sampling, this is Neyman allocation ([Neyman, 1934](#ref-neyman1934)) with (prompt, outcome) pairs as the strata, except that you can only reach a stratum by lottery. GVM-RAFT, VIP and DUET derive square-root rules of the same kind across prompts; here the strata also split each prompt by outcome, and the price includes training. The exact solution follows these targets up to rounding, nudging a few counts so that it uses the whole budget.

### 2.3 Two prompts, step by step {#example}

Let's make all of this concrete. Take two prompts of 200 tokens each, and say we have 140,000 tokens of budget left after the probes.

| | Hard prompt | Even prompt |
|---|---|---|
| History before this visit | 2 successes, 10 failures | 6 successes, 6 failures |
| Mean answer length | successes 3,000, failures 6,000 | 3,000 for both |

The hard prompt has been solved twice in twelve tries so far, and its failures are twice as long as its successes. The even prompt is a coin flip.

**1. Freeze the weights.** From the history, $$u=3/14\approx0.21$$ and $$u=0.5$$, so $$w=0.168$$ and $$w=0.25$$. These stay fixed no matter what happens in this step.

**2. Probe.** Sample eight rollouts of each prompt and grade them. The hard prompt gets 1 success, and the even prompt gets 4. Both probes are mixed, so both prompts stay in the batch.

**3. Forecast.** Folding the probe into the decayed history gives $$\hat p=0.18$$ and $$\hat p=0.50$$. The forecast is only used to predict how the follow-ups will come out; it never enters the update.

**4. Plan, once.** The solver returns the whole plan before anything else is sampled. The even prompt buys no follow-ups and will train on its 4 + 4 probes. The hard prompt buys 4 follow-ups, with this table:

| Successful follow-ups $$x$$ | Chance | Available | Keep |
|---|---|---|---|
| 0 | 44% | 1 + 11 | 1 + 5 |
| 1 | 40% | 2 + 10 | 2 + 4 |
| 2 | 14% | 3 + 9 | 3 + 4 |
| 3 | 2% | 4 + 8 | 4 + 4 |
| 4 | 0.1% | 5 + 7 | 5 + 4 |

Each cell is successes + failures. Why this plan? At this budget, the price sets the hard prompt's targets at 4.1 successes and 4.2 failures: nearly equal, even though its failures are twice as long (length cancels!). It has only 1 success, far below its target, so it buys tickets, keeps every success it gets, and keeps about four failures. The even prompt's targets are 6.1, but it already has 4 + 4, and four more rollouts wouldn't pay for themselves. The table follows the targets except for one extra failure in row 0 and one extra success in row 4, which spend budget left over from rounding.

**5. Sample.** Draw the hard prompt's 4 follow-ups and grade them. Say one succeeds, so $$x=1$$.

**6. Keep.** Row 1 says 2 + 4: keep both successes and 4 of the 10 failures, chosen at random. The even prompt keeps its 4 + 4.

**7. Update.** The hard prompt's two successes each get advantage $$+0.168/2=+0.084$$, and its four failures each get $$-0.168/4=-0.042$$. The even prompt's responses get $$\pm0.25/4=\pm0.063$$. As promised, each prompt's advantages add up to $$+w$$ over its successes and $$-w$$ over its failures.

**8. Record.** Add all twelve rollouts of the hard prompt and all eight of the even prompt to their histories, for the next visit.

### 2.4 What is exact, and what is not {#correctness}

Before the experiments, let's be clear about what this algorithm guarantees, and what it doesn't.

**What is exact: given the grades, each prompt's update points exactly along its own accuracy gradient, whatever the plan.** Why? Once we know a response's grade, it is just a random draw from the model's responses with that grade, and keeping a random subset within each grade doesn't change that. So the kept averages have means $$\mu_+$$ and $$\mu_-$$, and by the identity above, prompt $$i$$'s part of the update, $$g_i=w_i(\bar S_{i,+}-\bar S_{i,-})$$, has

$$
\mathbb E[g_i\mid\text{grades}]=w_i\,(\mu_+-\mu_-)=\frac{w_i\,\nabla p_i}{p_i(1-p_i)},
$$

where $$p_i$$ is the prompt's true success rate. This mean depends only on the probes, not on the plan. So the plan can only change the variance, which is exactly what Problem 1 minimizes.

There is one catch: $$n_i$$, the table, and which responses we keep must depend only on grades and on things fixed before the step. So no keeping responses by their content or length (for example, the first ones to finish), no planning on the probes' actual lengths, and no weights that peek at this visit's follow-ups. The forecast $$\hat p_i$$ only shapes the plan, so a bad forecast makes the plan less efficient but never moves the mean of the update.

**What is not exact: averaged over the probe, our simple weight is slightly biased for $$\nabla p$$, and we accept that.** A prompt only reaches the update when its probe is mixed, which happens with probability $$m(p)=1-p^8-(1-p)^8$$. So, with an accurate history, a prompt contributes $$m(p_i)\nabla p_i$$ instead of $$\nabla p_i$$. Prompts in the middle count fully, and those near 0% or 100% fade out: $$m=0.99$$ at $$p=0.5$$, $$0.83$$ at $$p=0.2$$, and $$0.34$$ at $$p=0.05$$.

Could we fix this? Sure: divide the weight by $$m(u_i)$$. But that would also multiply the noisiest terms in the batch, the ones resting on one or two rare responses, by 3× at $$u=0.05$$ and 7× at $$u=0.02$$. We keep the simple weight because it lowers the variance, and in practice it trains well. Bringing the faded prompts back into play is signal seeking, which we leave to the curriculum (Section 1). We are in good company here: MaxRL targets $$\log p$$, but its estimator is unbiased only for a truncated objective, with the truncation set by the number of rollouts per prompt ([Tajwar et al., 2026](#ref-tajwar2026)).

**Finally, the solver really is exact.** For the mathematically inclined, here is the precise statement.

<div class="theorem" markdown="1">
**Theorem (the solver is exact).** Problem 1 is equivalent to the following mixed-integer linear program, and an optimal solution of either is an optimal solution of the other. For every prompt $$i$$, candidate $$n$$, outcome $$x\le n$$ and count $$k$$, let $$z_{i,n}\in\{0,1\}$$ mean "buy $$n$$ follow-ups", and let $$y^{+}_{i,n,x,k},\,y^{-}_{i,n,x,k}\in\{0,1\}$$ mean "keep $$k$$ successes (failures) in row $$x$$". Write $$\pi_{n}(x)$$ for the Binomial$$(n,\hat p_i)$$ probability of $$x$$, $$a^\pm_{i,k}=w_i^2T_{i,\pm}/k$$ for the variance of keeping $$k$$ answers of an outcome, and $$c^\pm_i=2(P_i+T_{i,\pm})$$ for the cost of training on one. Sums over $$k$$ run over the counts available in row $$x$$, and sums over $$\pm$$ over the two outcomes. Then

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

## 3. Checking the assumptions {#assumptions}

Problem 1 leans on three assumptions, so let's check them. Each one does one job. **A1** makes a kept response add variance in proportion to its length, which is why the objective counts tokens, $$T_{i,+}/S_i$$ and $$T_{i,-}/F_i$$. **A2** puts successes and failures on one scale, so the SSN per token $$v$$ factors out. **A3** lets the variances of the two kept averages add, so the plan can minimize variance without tracking the mean gradient.

We test A1 and A2 on a frozen **Qwen3-4B** in thinking mode, with 40 hard competition-math prompts and 48 rollouts each, and A3 on the Qwen3-8B checkpoint of Section 4. The short answer: all three hold well. Intervals are 95%.

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

How do we read the first plot? For each response, we take its SSN, $$\lVert\nabla\log\pi_\theta(y)\rVert^2$$ in layers 18–35, and divide it by its prompt's average SSN per token. This removes the differences in $$v$$ between prompts and leaves a number measured in tokens. If A1 holds, that number equals the response's length, so responses fall on the dashed diagonal. If A2 also holds, correct and incorrect responses fall on the same diagonal. Each marker averages the responses in one length octile, with a 95% interval, on log–log axes; responses that hit the length cap are left out.

<figure class="narrow">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/score-energy-vs-length.png' | relative_url }}"><img src="{{ '/assets/images/score-energy-vs-length.png' | relative_url }}" alt="Normalized score energy against answer length on log-log axes, for layers 18 to 35 of Qwen3-4B. Correct and incorrect answers both lie on the diagonal from about 5,000 to 16,000 tokens." loading="lazy"></a>
</figure>

Both outcomes sit right on the diagonal across a threefold range of lengths. Responses cut off at the length cap, all of them incorrect, carry about 4% more SSN than the line predicts ($$1.036\pm0.013$$ times). Earlier layers have heavier tails ([Appendix A](#app-energy)). And A3 holds by a wide margin: at the Qwen3-8B checkpoint, a batch's mean gradient is at most 0.5% of its noise.

**Bonus: a prediction we don't need, tested anyway.** Nothing in Section 2 assumes how a prompt's noise depends on its success rate. But A1 and A2 still predict it: for plain policy gradient on one rollout,

$$
\operatorname{tr}\operatorname{Cov}\big((R-p)\nabla\log\pi_\theta(y)/\sqrt{T}\big)\approx v\,p(1-p),
$$

the SSN per token times the variance of the binary reward. The second plot tests this on the same frozen Qwen3-4B, using LoRA gradients for 10 prompts with success rates from 0.08 to 0.94 and 1,000 to 3,000 rollouts each. Each dot is one prompt's per-token gradient variance, with a 95% interval, and the dashed line is the best fit proportional to $$p(1-p)$$.

<figure class="narrow">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/variance-vs-pq.png' | relative_url }}"><img src="{{ '/assets/images/variance-vs-pq.png' | relative_url }}" alt="Per-token gradient variance of plain policy gradient against p(1-p) for ten prompts. The points rise roughly in proportion to p(1-p); one prompt near p(1-p)=0.22 lies well above the line." loading="lazy"></a>
</figure>

The prediction holds roughly. Most prompts lie within about 15% of the line, and fitting the exponent freely gives $$0.95\pm0.17$$. The spread is wider than the error bars, though, and one prompt at $$p\approx0.68$$ is about 50% noisier than the line: $$v$$ is not quite the same for every prompt, while Problem 1 treats it as one constant.

## 4. Results {#results}

### 4.1 At equal compute, our algorithm cuts gradient variance by 17% {#replay}

The cleanest test of a variance-reduction method is to measure the variance directly, so that's what we did first.

**Setup.** We froze a Qwen3-8B checkpoint after 100 steps of RL training, together with eight of its training prompts and their history. We sampled 64 independent rollout tapes, each with 32 rollouts per prompt, and replayed two rules on every tape: **GRPO-8**, which trains on eight rollouts of every prompt, and **our algorithm** (Section 2), with the per-prompt compute budget it was trained with. Both use the loss they were trained with, StableReinforce with KL coefficient 0.01. Tinker returned unclipped LoRA gradient norms at zero learning rate, from which we estimate the variance of a batch's gradient and the squared norm of its mean (details below). Brackets are 95% bootstrap intervals over eight independent blocks of eight tapes.

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

Our raw variance is 72% lower! But wait: the two rules estimate the gradient at different scales, so this number alone means little. Can the mean gradient tell us the scale? Not really: both intervals include zero, because a batch's mean gradient is a fraction of a percent of its noise (A3).

Instead, we can work out the scale from the two rules themselves. GRPO-8 subtracts the group's mean reward, and that mean includes the answer's own reward: $$r_j-\bar r=\tfrac78\,(r_j-\bar r_{-j})$$, where $$\bar r_{-j}$$ is the mean of the other seven. That leave-one-out mean is independent of answer $$j$$, so each answer contributes $$\tfrac78\nabla p$$ on average, and the eight answers sum to $$7\,\nabla p$$ per prompt. Our algorithm gives the kept successes and failures totals of $$\pm4w$$, so it contributes $$4\,m(p)\nabla p$$ per prompt when the history is right (Section 2.4), which is $$4\,\nabla p$$ for a prompt whose probe is almost always mixed. On such prompts our gradient is $$4/7$$ of GRPO-8's, and multiplying it by $$7/4$$ puts the two on the same scale:

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th>Per batch, our algorithm rescaled by 7/4</th><th>GRPO-8</th><th>Our algorithm</th></tr>
</thead>
<tbody>
<tr><td>Variance, ×10⁵</td><td>11.1 [10.4, 11.9]</td><td>9.4 [8.7, 10.2]</td></tr>
<tr><td>Squared mean, ×10³</td><td>2.6 [−0.1, 5.3]</td><td>1.0 [−1.1, 3.1]</td></tr>
<tr><td>(Simple) critical batch size</td><td>431 [200, ∞)</td><td>951 [314, ∞)</td></tr>
</tbody>
</table>
</div>

Rescaled, our variance per batch is 0.85 [0.78, 0.93] of GRPO-8's, and our batches cost 2% less compute. So **at equal compute, our gradient has 17% less variance**: 0.83 [0.77, 0.91]. The last row is the simple critical batch size of [McCandlish et al. (2018)](#ref-mccandlish2018), $$\operatorname{tr}\operatorname{Cov}(g)/\lVert\mathbb E g\rVert^2$$, in batches of eight prompts; it does not depend on the scale. We report it for completeness only. Because neither mean is resolved, both intervals are unbounded above, and the point estimates are unstable: pooling all 64 tapes instead of averaging over blocks gives 309 for GRPO-8 and 260 for our algorithm.

Two notes on the fine print. The factor $$4/7$$ matches the scales only where $$m(p)\approx1$$: these eight prompts have $$m$$ between 0.54 and 0.99, and on the more one-sided ones our update counts for less by design. And the KL term, which is charged on every kept token, is not on this scale.

<details markdown="1">
<summary>Measuring variance and signal from gradient norms</summary>

For each rule, Tinker returned the unclipped LoRA gradient norm of every tape, of disjoint pairs and blocks of tapes, and of all tapes together, with the optimizer at zero learning rate; we checked that the policy did not change. For $$B$$ independent batch gradients $$g_b$$ with mean $$G$$,

$$
\widehat V=\frac{\sum_b\lVert g_b\rVert^2-\lVert\sum_b g_b\rVert^2/B}{B-1},
$$

$$
\widehat{\lVert G\rVert^2}=\frac{\lVert\sum_b g_b\rVert^2-\sum_b\lVert g_b\rVert^2}{B(B-1)}
$$

are unbiased for $$\operatorname{tr}\operatorname{Cov}(g)$$ and $$\lVert G\rVert^2$$. For a pair they reduce to $$\lVert g_1-g_2\rVert^2/2$$ and $$g_1\cdot g_2$$. The signal estimate is signed, so it can come out negative when the true signal is small.

</details>

### 4.2 Training {#training}

**Setup.** We train Qwen3-8B with each method on 300 prompts from the [Reinforce-Ada hard-prompt dataset](#ref-dataset), one seed each, at learning rate $$2\times10^{-5}$$ with the [StableReinforce](#ref-tinker) loss in Tinker and KL coefficient 0.01. Every method draws 30 prompts per update. GRPO-8 samples eight answers to each and centers rewards by the group mean. Our algorithm has an expected budget of 1.96 million weighted tokens per update, probes included, and uses the weight $$u(1-u)$$, as in Section 4.1. Seq-Balance was trained with an earlier version of the grader, which differs on two training prompts. Every 25 updates, we draw 16 responses to each of the same 60 held-out prompts for every method, graded the same way, and report accuracy among graded responses. Total compute counts prefill and sampled tokens plus twice the training tokens, and training compute counts only the latter; we convert both to FLOPs at $$2N$$ per token, with $$N=8\times10^9$$.

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

What do these numbers say? At equal total compute, our algorithm and GRPO-8 end level on the held-out prompts, 53.7% against 53.2%, a difference of 0.5 points [−3.8, +4.9], while on the training prompts our algorithm is 2.6 points ahead. At equal training compute, our algorithm is ahead on the held-out prompts too: after 75 updates, it has used 1.28 EFLOPs of training compute and reaches 53.8%, while GRPO-8 after 75 updates has used 1.36 EFLOPs and reaches 47.4%, a gap of 6.4 points [2.0, 11.0]. (Brackets are 95% paired bootstrap intervals over the 60 held-out prompts.) Seq-Balance ends at 47.4% on the held-out prompts, with 16% less total compute.

On the training prompts, the picture is clearer. Our algorithm is ahead of both baselines from about the 20th update on, pulls ahead in total compute after about 1.5 EFLOPs, and leads GRPO-8 for most of the run in training compute:

<figure class="three-views">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}"><picture>
    <source media="(max-width: 600px)" srcset="{{ '/assets/images/train-accuracy-three-views-stacked.png' | relative_url }}">
    <img src="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}" alt="Accuracy on the training prompts for Seq-Balance, GRPO-8 and our algorithm, against optimizer updates, total compute and training compute. Our algorithm ends highest in all three panels." loading="lazy">
  </picture></a>
</figure>

Each curve is the accuracy of answers sampled before each update, pooled over a trailing window: the last 10 updates on the left, the last 20M weighted tokens of total compute in the middle, and the last 8M of training compute on the right. The left panel counts the first four responses per prompt for every method; the other two count the first four for our algorithm and Seq-Balance and all eight for GRPO-8. The train accuracy in the table above is the right end of the middle panel.

These are single training runs, and the intervals cover only the choice of held-out prompts, not the training seed.

## 5. Some mysteries {#mystery}

All our training runs use Tinker's [StableReinforce](#ref-tinker) loss, which comes with a KL term between the sampler and the learner, with coefficient 0.01. We did not expect this term to favor our algorithm over GRPO, or even to have any noticeable effect at our horizon length. However, it matters _a lot_! Here are our algorithm and GRPO-8, trained with the KL term (left) and without it (right):

<figure class="three-views">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/train-accuracy-kl.png' | relative_url }}"><picture>
    <source media="(max-width: 600px)" srcset="{{ '/assets/images/train-accuracy-kl-stacked.png' | relative_url }}">
    <img src="{{ '/assets/images/train-accuracy-kl.png' | relative_url }}" alt="Train accuracy against optimizer updates for our algorithm and GRPO-8, in two panels. With the KL term, our algorithm ends near 59 percent and GRPO-8 near 55 percent. Without it, both end near 54 to 55 percent." loading="lazy">
  </picture></a>
</figure>

With the KL term, our algorithm pulls ahead. Without it, the lead is gone, and GRPO-8 barely notices the difference. All four runs share the same seed, prompts and learning rate, and each curve is the accuracy of the first four responses per prompt over the last 10 updates.

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th>After 100 updates</th><th>Ours, KL</th><th>Ours, no KL</th><th>GRPO-8, KL</th><th>GRPO-8, no KL</th></tr>
</thead>
<tbody>
<tr><td>Train accuracy, updates 61–100</td><td class="hl">58.2%</td><td>54.4%</td><td>54.3%</td><td>54.0%</td></tr>
<tr><td>Held-out accuracy</td><td class="hl">53.7%</td><td>50.6%</td><td>50.3%</td><td>51.4%</td></tr>
<tr class="rule-above"><td>Held-out responses cut off at 4,096 tokens</td><td>14.4%</td><td>23.4%</td><td>28.4%</td><td>15.2%</td></tr>
</tbody>
</table>
</div>

Turning the KL term off costs our algorithm about 3 points: 3.8 on the training prompts, and 3.1 on the held-out prompts [−0.3, +6.5]. It costs GRPO-8 nothing we can see.

We don't understand this yet. But it leaves us with three concrete scientific questions, and we find them the most exciting part of this project.

### Question 1: is it because we train on a skewed sample? {#q-skew}

**The KL term is not neutral for us, and we can compute by how much.** StableReinforce adds $$\beta\sum_t\big(\log q(x_t)-\log\pi_\theta(x_t)\big)$$ to the loss, summed over the tokens we train on, where $$q$$ is the policy that sampled them. Take its gradient, and you find something very simple: it adds $$+\beta$$ to the advantage of every response we train on, right or wrong.

For GRPO, this is harmless. GRPO trains on every response it samples, and over a fair sample of the policy the scores average to zero, $$p\,\mu_++(1-p)\,\mu_-=0$$. The pushes cancel.

But our training set is not a fair sample. That is the whole point of our algorithm! We keep about as many successes as failures, whatever the success rate. Then the pushes no longer cancel. With $$S$$ kept successes and $$F$$ kept failures, the KL term contributes, on average,

$$
\beta\,\big(S\mu_++F\mu_-\big)=\beta\,\nabla\log\big[p^{S}(1-p)^{F}\big].
$$

In words, **the KL term pulls each prompt's success rate toward the fraction of successes we kept.** We keep about half, so hard prompts get an extra push up, and easy prompts get pushed down. This happens even when the sampler and the learner are exactly the same model. It also means that with the KL term on, our runs were not exactly the $$b=p$$ update of Section 2: the kept successes carry a little more weight than the kept failures. The same thing happens, in some form, to any method that chooses what to train on by outcome.

How large is it? Take a prompt with 4 + 4 kept responses and an accurate history. The reward part of its update is about $$4\,\nabla p$$ on average (Section 4.1), and the KL part changes that by a fraction $$\beta\,(1-2p)/(p(1-p))$$:

<div class="table-wrap" markdown="1">

| Success rate $$p$$ | 0.05 | 0.1 | 0.2 | 0.5 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| Change in the expected gradient | +19% | +8.9% | +3.8% | 0 | −3.8% | −8.9% | −19% |

</div>

So it is a few percent for most prompts, and more at the extremes. We also measured it directly. On a frozen checkpoint from a second training run of our algorithm, we computed the gradient of the same kept responses with and without the KL term, for eight batches of eight prompts. The KL term adds a gradient that is 6.9% as large as the reward gradient (3.2% to 14.1% across batches) and almost perpendicular to it: it rotates the update by about 4°.

So here is a real, computable effect that GRPO does not have. It makes a clear prediction: the KL term should help on hard prompts and hurt on easy ones.

**How to find out.** Use a KL term that is neutral by construction. Tinker's other estimator, k3, has zero gradient wherever the sampler and the learner agree, so our skewed sample cannot bias it. If the gain survives with k3, it is about real off-policy drift. If it disappears, it is about the skew, and then we should make the tilt toward one outcome a deliberate choice, as one more knob in Problem 1.

### Question 2: why is the gain on the easy prompts, and tied to response length? {#q-length}

**The prediction fails.** Here are the 60 held-out prompts, split by how the untrained model did on four tries:

<div class="table-wrap" markdown="1">

| Before training | Prompts | Ours, KL 0.01 | Ours, no KL | GRPO-8, KL 0.01 | GRPO-8, no KL |
|---|---|---|---|---|---|
| 0 of 4 correct | 19 | 27.9% | 29.7% | 26.7% | 26.4% |
| 1–3 of 4 correct | 29 | 60.9% | 58.0% | 60.9% | 59.7% |
| 4 of 4 correct | 8 | 86.4% | 73.0% | 70.3% | 85.8% |
| Not all graded | 4 | 56.5% | 50.8% | 42.6% | 41.4% |

</div>

Nothing much happens on the hard prompts. The action is on the eight easiest ones, which is the opposite of what the skewed sample predicts. And it comes with response length: on those eight prompts, our run without the KL term hits the 4,096-token limit on 15.6% of its responses, against 3.9% with it. A response that runs out of tokens is graded as wrong.

GRPO-8 makes it stranger still. On the same eight prompts it flips: it does much better _without_ the KL term, and that is also the GRPO-8 run that gets cut off less often (15% of responses against 28%).

So what actually separates the four runs is whether the model learned to finish its responses within the token limit. Why would a sampler–learner KL term change that? And why in opposite directions for the two methods? We don't know yet.

**How to find out.** Train and evaluate with a longer token limit. At 8,192 tokens, "solves more problems" and "finishes in time" stop being mixed together.

### Question 3: is adaptive sampling more sensitive to the rest of the recipe? {#q-recipe}

**Our working hypothesis is yes.** Adaptive sampling trains on a deliberately skewed sample of the policy's responses. That is where its lower variance comes from. But it also means that every other part of the recipe that quietly assumes a fair sample, like the KL term here, can interact with it in ways it does not with GRPO. If that is right, adaptive sampling is a promising algorithm with a sharper edge: its off-policy corrections have to be designed with it, not borrowed.

The alternative is less exciting but just as important to rule out: run-to-run luck. Everything in this post is small scale: one 8B model, one replay checkpoint, and one seed per run. A second seed of our algorithm with the KL term on finished level with GRPO-8, so even the size of the effect is not pinned down yet.

**How to find out.** More seeds of these four runs. This is the only way to separate a real interaction from luck.

So we have positive scientific results and three sharp questions, each with a well-defined experiment. What they need is mostly compute: more seeds, longer runs and larger models.

## Citation {#citation}

{% raw %}
```bibtex
@misc{chen2026adaptivesampling,
  author = {Chen, Chang-Han},
  title  = {Variance reduction and adaptive sampling at fixed inference and training compute},
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
1. {: #ref-li2025} Z. Li et al. [Knapsack RL: Unlocking exploration of LLMs via optimizing budget allocation](https://arxiv.org/abs/2509.25849). arXiv:2509.25849, 2025.
1. {: #ref-mccandlish2018} S. McCandlish, J. Kaplan, D. Amodei, and the OpenAI Dota Team. [An empirical model of large-batch training](https://arxiv.org/abs/1812.06162). arXiv:1812.06162, 2018.
1. {: #ref-neyman1934} J. Neyman. On the two different aspects of the representative method. *Journal of the Royal Statistical Society* 97(4), 1934.
1. {: #ref-nguyen2026} H. T. Nguyen et al. [Adaptive rollout allocation for online reinforcement learning with verifiable rewards](https://arxiv.org/abs/2602.01601). *ICLR*, 2026.
1. {: #ref-noukhovitch2026} M. Noukhovitch, H. Ivison, N. Lambert, and A. Courville. [Learning to solve hard problems in RL for LLMs by never giving up](https://arxiv.org/abs/2609.13443). arXiv:2609.13443, 2026.
1. {: #ref-olmo2025} Team Olmo. [Olmo 3](https://arxiv.org/abs/2512.13961). arXiv:2512.13961, 2025.
1. {: #ref-openai2024} OpenAI. [OpenAI o1 system card](https://arxiv.org/abs/2412.16720). arXiv:2412.16720, 2024.
1. {: #ref-qu2026} Y. Qu, A. Setlur, V. Smith, R. Salakhutdinov, and A. Kumar. [POPE: Learning to reason on hard problems via privileged on-policy exploration](https://arxiv.org/abs/2601.18779). arXiv:2601.18779, 2026.
1. {: #ref-dataset} RLHFlow. [reinforce_ada_hard_prompt](https://huggingface.co/datasets/RLHFlow/reinforce_ada_hard_prompt) (dataset). Hugging Face.
1. {: #ref-tajwar2026} F. Tajwar, G. Zeng, Y. Zhou, Y. Song, D. Arora, Y. Jiang, J. Schneider, R. Salakhutdinov, H. Feng, and A. Zanette. [Maximum likelihood reinforcement learning](https://arxiv.org/abs/2602.02710). *ICML*, 2026.
1. {: #ref-tml2026} Thinking Machines Lab. [Introducing Inkling 2 Small](https://thinkingmachines.ai/news/inkling-2-small/). Blog post, November 2026.
1. {: #ref-tinker} Thinking Machines. [StableReinforce](https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/), Tinker documentation.
1. {: #ref-williams1992} R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning* 8, 229–256, 1992.
1. {: #ref-xiong2025} W. Xiong et al. [Reinforce-Ada: An adaptive sampling framework under non-linear RL objectives](https://arxiv.org/abs/2510.04996). arXiv:2510.04996, 2025. (Original Seq-Balance estimator: [v1, §2.3](https://arxiv.org/html/2510.04996v1#S2.SS3); importance weight: [v3, §3.2](https://arxiv.org/html/2510.04996v3#S3.SS2).)
1. {: #ref-xu2025} Y. E. Xu, Y. Savani, F. Fang, and J. Z. Kolter. [Not all rollouts are useful: Down-sampling rollouts in LLM reinforcement learning](https://arxiv.org/abs/2504.13818). *TMLR*, 2026.
1. {: #ref-yao2025} J. Yao et al. [Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and RL](https://arxiv.org/abs/2505.02391). arXiv:2505.02391, 2025.
1. {: #ref-yu2025} Q. Yu et al. [DAPO: An open-source LLM reinforcement learning system at scale](https://arxiv.org/abs/2503.14476). *NeurIPS*, 2025.
1. {: #ref-zheng2025} H. Zheng et al. [Act only when it pays: Efficient reinforcement learning for LLM reasoning via selective rollouts](https://arxiv.org/abs/2506.02177). *NeurIPS*, 2025.
1. {: #ref-zhu2025} X. Zhu, M. Xia, Z. Wei, W.-L. Chen, D. Chen, and Y. Meng. [The surprising effectiveness of negative reinforcement in LLM reasoning](https://arxiv.org/abs/2506.01347). *NeurIPS*, 2025.

</div>

## Appendix {#appendix}

### A. SSN across layers {#app-energy}

This is the A1 check in three parameter groups, on the same 40 Qwen3-4B prompts and 48 rollouts each as Section 3, with the prompt effect divided out. Each bin is a length octile, with ±2σ error bars. Layers 18–35 and the final norm both scale close to linearly in length. In layers 0–8, heavy tails dominate a fit on the mean (exponent 0.78 ± 1.73), while a fit on log SSN, which is robust to them, gives 0.93 ± 0.06.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/score-energy-by-length.png' | relative_url }}"><img src="{{ '/assets/images/score-energy-by-length.png' | relative_url }}" alt="Score energy rises with response length in three parameter groups. Correct and incorrect answers nearly overlap in layers 18 to 35 and in the final norm, with much noisier estimates in layers 0 to 8." loading="lazy"></a>
</figure>

### B. Why the mean baseline {#app-baseline}

Our update uses the group's success rate as its baseline: $$g_i$$ is REINFORCE with advantage $$R-p$$, reweighted to the kept answers. Section 2 explains why variance should not choose the baseline once we choose which answers to keep. For plain REINFORCE on all rollouts, a slightly different baseline would lower the variance.

**A length-weighted baseline.** For the estimator $$(R-b)S$$ on all rollouts, the variance-optimal constant baseline is

$$
b^*=\frac{\mathbb E[R\|S\|^2]}{\mathbb E[\|S\|^2]}
\;\approx\;\frac{\mathbb E[RT]}{\mathbb E[T]},
$$

which counts long answers more. In the 40-prompt Qwen3-4B study, cross-fitting it saved **0.8% ± 1.8%** of the gradient noise on average (two-standard-error band), and about **14%** on one heavily truncated prompt. The left panel shows how far the fitted optimum sits from the mean reward, and the right panel what it saves on separate rollouts.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/baseline-moments.png' | relative_url }}"><img src="{{ '/assets/images/baseline-moments.png' | relative_url }}" alt="The fitted baseline tracks the length-weighted success rate. Across prompts the out-of-sample noise savings over the mean-reward baseline are small." loading="lazy"></a>
</figure>

