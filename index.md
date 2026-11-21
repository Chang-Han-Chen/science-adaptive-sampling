---
title: "How to think about compute‑optimal adaptive sampling?"
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

We investigate adaptive sampling in RL with verifiable rewards.

1. We survey the past literature and identify two distinct focuses: **signal seeking**, which aims to get the first success on a hard prompt, and **variance reduction**, which aims to reduce the noise once the mean gradient is non-zero.
2. We pose variance reduction as a stochastic optimization problem: how many more rollouts $$n_i$$ to sample for each prompt $$i$$, and how many successes $$S_i$$ and failures $$F_i$$ to train on. Schematically,

   $$
   \begin{aligned}
   \{n_i,S_i,F_i\}={}&\arg\min\ \operatorname{Var}(\text{batch gradient})\\
   &\text{s.t.}\ \mathbb E\big[\text{inference tokens} + 2 \times \text{training tokens}\big]\le C.
   \end{aligned}
   $$

   Of course, the numbers of successes and failures at our disposal after sampling vary from run to run, so our algorithm also adapts to all possible scenarios.
3. We solve the optimization problem exactly with an integer program solver. Intuitively, the update compares the average success with the average failure, so its noise is set mostly by whichever outcome we have fewer of. With 1 success and 7 failures, one more success cuts the noise by almost half, while one more failure cuts it by under 2%. So the solver spends compute on the rarer outcome. We also prove that the resulting update always points along the accuracy gradient.
4. We empirically check the noise model behind this on competition-math rollouts from Qwen3-4B.
5. On a frozen Qwen3-8B checkpoint, our algorithm's gradient has **17% less variance than GRPO-8's at equal compute**.
6. In training on competition math, our algorithm is **6.4 points ahead of GRPO-8 in held-out accuracy at equal training compute**, and matches it at equal total compute (53.7% vs 53.2%).
</div>

Here is what this looks like in practice. We train Qwen3-8B on competition math and track accuracy on the training prompts, counting progress three ways: optimizer updates, total compute and training compute. Our algorithm ends highest in all three ([Section 4.2](#training)).

<figure class="three-views">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}"><picture>
    <source media="(max-width: 600px)" srcset="{{ '/assets/images/train-accuracy-three-views-stacked.png' | relative_url }}">
    <img src="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}" alt="Accuracy on the training prompts for Seq-Balance, GRPO-8 and our algorithm, against optimizer updates, total compute and training compute. Our algorithm ends highest in all three panels.">
  </picture></a>
</figure>

## 1. Background: adaptive sampling for LLM RLVR {#background}

Reinforcement learning with verifiable rewards (RLVR) ([Lambert et al., 2024](#ref-lambert2024)) unlocks the reasoning capabilities of large language models (LLMs) ([OpenAI, 2024](#ref-openai2024); [DeepSeek-AI, 2025](#ref-deepseek2025)). The algorithm is remarkably simple and intuitive: sample a group of answers to each prompt, let a verifier mark each one right or wrong, and make the right answers more likely and the wrong ones less likely. This is essentially the simplest RL algorithm, REINFORCE ([Williams, 1992](#ref-williams1992)),

$$
\nabla J(\theta)=\mathbb E_{y\sim\pi_\theta}\big[R(y)\,\nabla\log\pi_\theta(y)\big],
$$

with a binary reward $$R\in\{0,1\}$$. For a single prompt, $$J$$ is just its success rate $$p$$.

It is kind of mind-blowing that, if you implement this with _extreme_ attention to detail, it just works! Looking back at the major advances in RLVR over the last two years, the common theme seems to be removing as much **bias** as possible (from, e.g., numerical rounding or length normalization) that stands in the way of the learning signal. If you can do that, REINFORCE just wants to learn.

To train the best model at the frontier scale, you need faith in the training algorithm and a willingness to spend compute ($$\approx$$ money) on variance reduction. As in classic statistics, the variance of the training signal goes down like the inverse of the number of samples $$B$$,

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

Notice that a group with all successes or all failures contributes **zero** variance, not infinite variance: it is silent rather than noisy. A group with one success has a real signal, but everything it says about what a right answer looks like comes from that single answer, so the variance is huge. This is another way to see why signal seeking and variance reduction are two _separate_ algorithmic considerations. The first moves a prompt from the left box to the middle one; the second moves it from the middle box to the right one.

We find that this perspective provides a nice dichotomy of the past literature.

**From 0 to 1: signal seeking.** When every rollout of a prompt fails, every advantage is zero and the prompt contributes nothing to the batch gradient. Most work on adaptive sampling so far is about prompts like this, and its simplest form is now standard at scale. DAPO oversamples prompts, drops every group whose answers all agree, and keeps sampling until the batch is full of prompts with signal ([Yu et al., 2025](#ref-yu2025)). ScaleRL keeps this zero-variance filter in a recipe it scales to a single 100,000-GPU-hour run, and also stops revisiting prompts the model already passes at least 90% of the time ([Khatri et al., 2025](#ref-khatri2025)); Olmo 3 refills its batches the same way, which it calls active sampling ([Team Olmo, 2025](#ref-olmo2025)). Filtering pays for rollouts and then throws them away, so later work tries to spend them better: skip the prompts that are likely to come back all right or all wrong ([Zheng et al., 2025](#ref-zheng2025)), or move rollouts toward hard prompts, either under an explicit budget ([Li et al., 2025](#ref-li2025)) or by sampling a hard prompt until it is solved ([Noukhovitch et al., 2026](#ref-noukhovitch2026)). And when the model essentially never succeeds, more samples won't help; the remedy is to make the problem easier, for example by revealing part of a reference solution ([Qu et al., 2026](#ref-qu2026)).

We think much of signal seeking is better handled one level up, by designing environments and curricula that keep the model near the edge of what it can solve. How to find that edge automatically is a topic for another post.

**From 1 to more: variance reduction.** Once a prompt has at least one success and one failure, it has a real learning signal: the update pushes the correct answers up and the wrong ones down. It is also very noisy. Each answer is thousands of tokens long, and every token adds its own random contribution to the gradient. Most of those tokens have nothing to do with why a correct answer is right, and with a single success the update cannot tell the decisive step from the incidental ones. More successes and failures average these accidents away. But every extra rollout costs inference compute, and every one we train on costs training compute. Variance reduction asks how many more to buy, for which prompts, and which ones to train on. That is the question we answer.

Prior work has asked this question too. GVM-RAFT made the case for allocating samples unevenly across prompts to minimize gradient variance under a sampling budget ([Yao et al., 2025](#ref-yao2025)). Reinforce-Ada trains on balanced groups of successes and failures under a log-shaped objective ([Xiong et al., 2025](#ref-xiong2025)). PODS generates many rollouts but trains only on a high-variance subset, because the update phase has costs of its own ([Xu et al., 2025](#ref-xu2025)). VIP predicts each prompt's success rate with a Gaussian process and solves for the allocation that minimizes total gradient variance under a rollout budget ([Nguyen et al., 2026](#ref-nguyen2026)), and DUET does the same under a budget of generated tokens ([Hu et al., 2026](#ref-hu2026)).

These allocation rules budget rollouts or generated tokens. We also charge for training on the answers we keep, and we choose which ones to keep.

**The problem we solved.** For this project, we focused on the second question, namely **variance reduction** when a non-zero mean gradient is guaranteed. As we will explain in detail later, at the start of each batch, our algorithm samples $$8$$ rollouts for every prompt (the "probe"). If a prompt scores either $$0/8$$ or $$8/8$$, we give it up for this batch and spend the rest of the compute on the other prompts. For each prompt that remains, we decide how many more rollouts $$n_i$$ to sample, and how many successes $$S_i$$ and failures $$F_i$$ to train on. This way, adaptive sampling becomes exactly a stochastic programming problem.

Schematically,

<div class="problem schematic" markdown="1">
**Problem (adaptive sampling under an expected compute budget).**

$$
\begin{aligned}
\{n_i,S_i,F_i\}={}&\arg\min\ \operatorname{Var}(\text{batch gradient})\\
&\text{s.t.}\ \mathbb E\big[\text{inference tokens} + 2 \times \text{training tokens}\big]\le C
\end{aligned}
$$

</div>

Inference tokens depend on $$n_i$$, and training tokens on $$S_i$$ and $$F_i$$. The extra rollouts come out at random, so the plan has to say in advance how many of each outcome to keep for every way they can come out, and we count tokens in expectation. Section 2 makes this precise and solves it exactly.

## 2. Compute-optimal adaptive sampling {#method}

Everything rests on one identity. For a prompt with success rate $$p$$, let $$\mu_+$$ and $$\mu_-$$ be the mean score $$\nabla\log\pi_\theta(y)$$ of a successful and of a failed answer. The score identity gives $$\mu_+=\nabla p/p$$, and since the score has mean zero, $$\mu_-=-\nabla p/(1-p)$$. So

$$
\nabla p=p(1-p)\,\big(\mu_+-\mu_-\big).
$$

The accuracy gradient is a weight times the difference between two class means. We estimate the two means from the successes and failures we train on. The weight we fix before the step, from the prompt's history:

$$
w_i=\frac{u_i(1-u_i)}{1-u_i^8-(1-u_i)^8},
$$

where $$u_i$$ is the prompt's success rate estimated from its past visits. The numerator is the $$p(1-p)$$ of the identity, with the history estimate in place of the unknown $$p$$. The denominator is the chance that an eight-rollout probe is mixed. A probe that comes back all right or all wrong ends the visit, and dividing by this chance lets the visits that get through stand in for those that don't. The weight is 0.252 at $$u=0.5$$, 0.19 at $$u=0.2$$ and 0.14 at $$u=0.05$$, and never leaves $$[1/8,\,0.252]$$: once a prompt gets past its probe, it counts about the same whatever its difficulty.

Sampling more and choosing what to keep then only decide how precisely we estimate the two means.

### 2.1 The problem {#problem}

<div class="problem" markdown="1">
**Problem 1 (compute-optimal adaptive sampling).** Each prompt $$i$$ in the batch has, fixed before the step, a weight $$w_i$$ (above), a forecast success rate $$\hat p_i$$, a prompt length $$P_i$$, and mean answer lengths $$T_{i,+}$$ and $$T_{i,-}$$ for successes and failures. Its eight-rollout probe had $$s_i$$ successes, with $$1\le s_i\le7$$. Choose

- a number of follow-up rollouts $$n_i\in\{0,4,8,12,16,24\}$$, and
- for every possible number $$x$$ of successful follow-ups, how many successes $$S_i(x)\in[1,\,s_i+x]$$ and failures $$F_i(x)\in[1,\,8-s_i+n_i-x]$$ to keep,

to

$$
\begin{aligned}
\min\ & \sum_i \mathbb E\left[w_i^2\left(\frac{T_{i,+}}{S_i(X_i)}+\frac{T_{i,-}}{F_i(X_i)}\right)\right]\\
\text{s.t.}\ & \sum_i \mathbb E\big[n_iG_i+2(P_i+T_{i,+})\,S_i(X_i)\\
&\qquad\ +2(P_i+T_{i,-})\,F_i(X_i)\big]\le C,
\end{aligned}
$$

where $$X_i\sim\operatorname{Binomial}(n_i,\hat p_i)$$ and $$G_i=P_i+\hat p_iT_{i,+}+(1-\hat p_i)T_{i,-}$$.

</div>

Each piece has a plain meaning.

- **The update.** The kept answers form $$g=\sum_iw_i(\bar S_{i,+}-\bar S_{i,-})$$, where $$\bar S_{i,\pm}$$ are the average scores of the kept successes and failures. Per answer, a kept success gets advantage $$+w_i/S_i$$ and a kept failure $$-w_i/F_i$$, on every token.
- **The objective** is the expected variance of $$g$$, up to a constant, under three assumptions that Section 3 checks. **A1:** an answer's score energy grows linearly with its length, $$\mathbb E\lVert\nabla\log\pi_\theta(y)\rVert^2\approx vT$$ for $$T$$ tokens. **A2:** successes and failures have the same energy per token $$v$$. **A3:** rollouts are independent, and noise dominates signal.
- **The constraint** is the expected compute left after the probes. Generating a rollout costs its tokens, $$P_i$$ plus its length, and training on one costs twice that. Probes are already paid for, but training on them is not.

### 2.2 The solution {#solution}

**A plan is a table.** Follow-ups come out at random, so the plan has to say in advance what to keep for every possible outcome. For prompt $$i$$ it is a follow-up count $$n_i$$ and a table with one row per possible $$x$$, giving $$S_i(x)$$ and $$F_i(x)$$. The table is fixed before any follow-up is sampled; afterwards we only read the row of the outcome that happened.

**One integer program finds it exactly.** Each expectation in Problem 1 is a finite sum over the outcomes $$x$$, and both the objective and the compute are linear in which $$n$$ and which counts are chosen. So Problem 1 is a mixed-integer linear program, with one binary variable for each candidate $$n$$ and for each candidate count in each row. We solve it to proven optimality (Theorem 2) with HiGHS once per step, which takes a few seconds for 30 prompts.

**What the optimal table looks like.** Put a price $$\lambda$$ on compute. The success part of a row, $$w_i^2T_{i,+}/S+2\lambda(P_i+T_{i,+})S$$, is smallest at

$$
S_i^\circ=w_i\sqrt{\frac{T_{i,+}}{2\lambda\,(P_i+T_{i,+})}}\;\approx\;\frac{w_i}{\sqrt{2\lambda}},
$$

and failures work the same way. Two rules follow.

- **Length cancels.** A longer answer is noisier, but it also costs more to train on, in the same proportion. So a prompt aims for the same number of successes and failures whatever their lengths, and prompts with larger weights aim for more of both.
- **Follow-ups buy the outcome that runs short.** A rollout is a lottery ticket: it lands in an outcome at random. When an outcome is below its target, the plan buys tickets while the expected gain pays for them, then keeps each outcome up to its target.

This is Neyman allocation from stratified sampling ([Neyman, 1934](#ref-neyman1934)), with (prompt, outcome) pairs as the strata, except that a stratum can only be reached by lottery. GVM-RAFT, VIP and DUET derive square-root rules of the same kind across prompts; here the strata also split each prompt by outcome, and the price includes training. The exact solution follows these targets up to rounding, nudging a few counts so that it uses the whole budget.

### 2.3 Two prompts, step by step {#example}

Take two prompts of 200 tokens each and a budget of 140,000 tokens after the probes.

| | Hard prompt | Even prompt |
|---|---|---|
| History before this visit | 2 successes, 10 failures | 6 successes, 6 failures |
| Mean answer length | successes 3,000, failures 6,000 | 3,000 for both |

**1. Freeze the weights.** The history gives $$u=3/14\approx0.21$$ and $$u=0.5$$, so $$w=0.197$$ and $$w=0.252$$. They stay fixed whatever happens in this step.

**2. Probe.** Sample eight rollouts of each prompt and grade them. The hard prompt gets 1 success and the even prompt 4. Both probes are mixed, so both prompts continue.

**3. Forecast.** Folding the probe into the decayed history gives $$\hat p=0.18$$ and $$\hat p=0.50$$. The forecast only predicts the follow-ups; it never enters the update.

**4. Plan, once.** The solver returns the whole plan before anything else is sampled. The even prompt buys no follow-ups and will keep its 4 + 4 probes. The hard prompt buys 4 follow-ups, with this table:

| Successful follow-ups $$x$$ | Chance | Available | Keep |
|---|---|---|---|
| 0 | 44% | 1 + 11 | 1 + 5 |
| 1 | 40% | 2 + 10 | 2 + 4 |
| 2 | 14% | 3 + 9 | 3 + 4 |
| 3 | 2% | 4 + 8 | 4 + 4 |
| 4 | 0.1% | 5 + 7 | 5 + 4 |

Counts are successes + failures. At this budget the price sets the hard prompt's targets at 4.4 successes and 4.5 failures, nearly equal although its failures are twice as long. With 1 success against a target of 4.4, the hard prompt buys tickets, keeps every success it gets, and keeps about four failures. The even prompt's targets are 5.6, but its probes already give 4 + 4, and four more rollouts would not pay for themselves. The table matches the targets except for one extra failure in row 0 and one extra success in row 4, which spend budget the rounding left over.

**5. Sample.** Draw the hard prompt's 4 follow-ups and grade them. Say one succeeds, so $$x=1$$.

**6. Keep.** Row 1 says 2 + 4: keep both successes and 4 of the 10 failures, chosen at random. The even prompt keeps its 4 + 4.

**7. Update.** The hard prompt's two successes get advantage $$+0.197/2=+0.098$$ and its four failures $$-0.197/4=-0.049$$. The even prompt's answers get $$\pm0.252/4=\pm0.063$$. Each prompt's advantages sum to $$+w$$ over its successes and $$-w$$ over its failures.

**8. Record.** Add all twelve rollouts of the hard prompt and all eight of the even prompt to their histories, for the next visit.

### 2.4 Why it is correct {#correctness}

Write $$w(u)=u(1-u)/(1-u^8-(1-u)^8)$$, so that $$w_i=w(u_i)$$, and $$p_i$$ for prompt $$i$$'s true success rate under the current model, and $$g_i=w_i(\bar S_{i,+}-\bar S_{i,-})$$ for its part of the update, with $$g_i=0$$ when its probe is not mixed.

<div class="theorem" markdown="1">
**Theorem 1 (the update follows the accuracy gradient).** Suppose $$n_i$$, the table, and the choice of which answers to keep depend only on grades and on quantities fixed before the step. Then

1. given the grades, a prompt with a mixed probe has

   $$
   \mathbb E[g_i\mid\text{grades}]=\frac{w_i\,\nabla p_i}{p_i(1-p_i)},
   $$

   and under A1–A3 the trace of its covariance is $$v\,w_i^2\,(T_{i,+}/S_i+T_{i,-}/F_i)$$;
2. averaged over the probe as well,

   $$
   \mathbb E[g_i]=\frac{w(u_i)}{w(p_i)}\,\nabla p_i .
   $$

</div>

<div class="proof" markdown="1">
*Proof.* Fix a prompt and condition on all the grades. Rollouts are independent, so given its grade each answer is a draw from the model's answers with that grade, independent of the others, and keeping a random subset within each grade preserves this. So the kept averages have means $$\mu_+$$ and $$\mu_-$$, and

$$
\mathbb E[g_i\mid\text{grades}]=w_i(\mu_+-\mu_-)=\frac{w_i\,\nabla p_i}{p_i(1-p_i)}.
$$

By A1–A3 the two averages are independent, with covariance traces $$vT_{i,+}/S_i$$ and $$vT_{i,-}/F_i$$, which gives the variance. For the second claim, the probe is mixed with probability $$m(p_i)=1-p_i^8-(1-p_i)^8$$, and $$w_i=w(u_i)$$ is fixed before the step, so

$$
\mathbb E[g_i]=m(p_i)\,\frac{w(u_i)\,\nabla p_i}{p_i(1-p_i)}=\frac{w(u_i)}{w(p_i)}\,\nabla p_i ,
$$

because $$w(p)=p(1-p)/m(p)$$. <span class="qed">∎</span>

</div>

Three consequences. Each prompt's update points exactly along its accuracy gradient. Its size is exact when the history estimate is right, $$u_i=p_i$$, and since $$w$$ only ranges between $$1/8$$ and $$0.252$$, it is never off by more than a factor of about 2. And because $$\mathbb E[g\mid\text{grades}]$$ depends only on the probes, it is the same for every plan, so minimizing the expected variance given the grades, the objective of Problem 1, minimizes the variance of $$g$$.

The condition of Theorem 1 rules out keeping answers by their content or length (for example, the first to finish), planning on the probes' realized lengths, and weights that look at this visit's follow-ups. The forecast $$\hat p_i$$ only shapes the plan, so a wrong forecast makes the plan less efficient but never moves the mean of the update.

<div class="theorem" markdown="1">
**Theorem 2 (the solver is exact).** Problem 1 is equivalent to the following mixed-integer linear program, and an optimal solution of either is an optimal solution of the other. For every prompt $$i$$, candidate $$n$$, outcome $$x\le n$$ and count $$k$$, let $$z_{i,n}\in\{0,1\}$$ mean "buy $$n$$ follow-ups", and let $$y^{+}_{i,n,x,k},\,y^{-}_{i,n,x,k}\in\{0,1\}$$ mean "keep $$k$$ successes (failures) in row $$x$$". Write $$\pi_{n}(x)$$ for the Binomial$$(n,\hat p_i)$$ probability of $$x$$, $$a^\pm_{i,k}=w_i^2T_{i,\pm}/k$$ for the variance of keeping $$k$$ answers of an outcome, and $$c^\pm_i=2(P_i+T_{i,\pm})$$ for the cost of training on one. Sums over $$k$$ run over the counts available in row $$x$$, and sums over $$\pm$$ over the two outcomes. Then

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

Each assumption does one job in Problem 1. **A1** makes a kept answer add variance in proportion to its length, which is why the objective counts tokens, $$T_{i,+}/S_i$$ and $$T_{i,-}/F_i$$. **A2** makes the energy per token $$v$$ the same for both outcomes, so it factors out and successes and failures are priced on one scale. **A3** lets the variances of the two kept averages add, and lets the plan minimize variance without tracking the mean gradient.

We test A1 and A2 on a frozen **Qwen3-4B** in thinking mode, with 40 hard competition-math prompts and 48 rollouts each, and A3 on the Qwen3-8B checkpoint of Section 4. Intervals are 95%.

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th>Assumption</th><th>What we measured</th><th>Result</th></tr>
</thead>
<tbody>
<tr><td>A1. Energy ∝ length</td><td>fitted exponent of score energy vs. length, layers 18–35</td><td class="hl">0.97 ± 0.06</td></tr>
<tr><td>A2. Same energy per token for both outcomes</td><td>energy of incorrect / correct answers at matched length</td><td class="hl">1.005 ± 0.013</td></tr>
<tr><td>A3. Noise dominates</td><td>‖E g‖² / tr Cov(g) per batch, Qwen3-8B replay (§4)</td><td class="hl">at most 0.5%</td></tr>
</tbody>
</table>
</div>

**Reading the first plot.** For each answer we take its score energy $$\lVert\nabla\log\pi_\theta(y)\rVert^2$$ in layers 18–35 and divide it by its prompt's average energy per token. This removes the differences in $$v$$ between prompts and leaves a number measured in tokens. Under A1 that number equals the answer's length, so answers fall on the dashed diagonal; under A2, correct and incorrect answers fall on the same diagonal. Each marker averages the answers in one length bin, with a 95% interval, on log–log axes.

<figure class="narrow">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/score-energy-vs-length.png' | relative_url }}"><img src="{{ '/assets/images/score-energy-vs-length.png' | relative_url }}" alt="Normalized score energy against answer length on log-log axes, for layers 18 to 35 of Qwen3-4B. Correct and incorrect answers both lie on the diagonal from about 5,000 to 16,000 tokens." loading="lazy"></a>
  <figcaption>Score energy against length, layers 18–35. Bins are length octiles; answers that hit the length cap are left out.</figcaption>
</figure>

Both outcomes sit on the diagonal across a threefold range of lengths. Answers cut off at the length cap, all of them incorrect, carry about 4% more energy than the line predicts ($$1.036\pm0.013$$ times). Earlier layers have heavier tails ([Appendix A](#app-energy)). A3 holds by a wide margin: at the Qwen3-8B checkpoint, a batch's mean gradient is at most 0.5% of its noise.

**A prediction we don't need, tested anyway.** Nothing in Section 2 assumes how a prompt's noise depends on its success rate. A1 and A2 still predict it: for plain policy gradient on one rollout,

$$
\operatorname{tr}\operatorname{Cov}\big((R-p)\nabla\log\pi_\theta(y)/\sqrt{T}\big)\approx v\,p(1-p),
$$

the energy per token times the variance of the binary reward. The second plot tests this on the same frozen Qwen3-4B, using LoRA gradients for 10 prompts with success rates from 0.08 to 0.94 and 1,000 to 3,000 rollouts each. Each dot is one prompt, with a 95% interval, and the dashed line is the best fit proportional to $$p(1-p)$$.

<figure class="narrow">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/variance-vs-pq.png' | relative_url }}"><img src="{{ '/assets/images/variance-vs-pq.png' | relative_url }}" alt="Per-token gradient variance of plain policy gradient against p(1-p) for ten prompts. The points rise roughly in proportion to p(1-p); one prompt near p(1-p)=0.22 lies well above the line." loading="lazy"></a>
  <figcaption>Per-token gradient variance against <i>p</i>(1&#8202;−&#8202;<i>p</i>), one dot per prompt.</figcaption>
</figure>

The prediction holds roughly. Most prompts lie within about 15% of the line, and fitting the exponent freely gives $$0.95\pm0.17$$. The spread is wider than the error bars, though, and one prompt at $$p\approx0.68$$ is about 50% noisier than the line: $$v$$ is not quite the same for every prompt, while Problem 1 treats it as one constant.

## 4. Results {#results}

### 4.1 At equal compute, our algorithm cuts gradient variance by 17% {#replay}

**Setup.** We froze a Qwen3-8B checkpoint after 100 steps of RL training, together with eight of its training prompts and their history. We sampled 64 independent rollout tapes, each with 32 rollouts per prompt, and replayed two rules on every tape: **GRPO-8**, which trains on eight rollouts of every prompt, and **our algorithm** (Section 2), with the per-prompt compute budget it was trained with. Both use the loss they were trained with, StableReinforce with KL coefficient 0.01, and our algorithm uses the weight it was trained with, $$u(1-u)$$. Tinker returned unclipped LoRA gradient norms at zero learning rate, from which we estimate the variance of a batch's gradient and the squared norm of its mean (details below). Brackets are 95% bootstrap intervals over eight independent blocks of eight tapes.

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

The raw variance is 72% lower, but the two rules estimate the gradient at different scales, so this number alone means little. The mean cannot settle the scale either: both intervals include zero, because a batch's mean gradient is a fraction of a percent of its noise (A3).

The scale follows from the two rules instead. GRPO-8 subtracts the group's mean reward, and that mean includes the answer's own reward: $$r_j-\bar r=\tfrac78\,(r_j-\bar r_{-j})$$, where $$\bar r_{-j}$$ is the mean of the other seven. That leave-one-out mean is independent of answer $$j$$, so each answer contributes $$\tfrac78\nabla p$$ on average, and the eight answers sum to $$7\,\nabla p$$ per prompt. Our algorithm gives the kept successes and failures totals of $$\pm4w$$, so by Theorem 1 it contributes $$4\,\nabla p$$ per prompt when the history is right. Our gradient is therefore $$4/7$$ of GRPO-8's, and multiplying it by $$7/4$$ puts the two on the same scale:

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

Rescaled, our variance per batch is 0.85 [0.78, 0.93] of GRPO-8's, and our batches cost 2% less compute, so at equal compute our gradient has 17% less variance: 0.83 [0.77, 0.91]. The last row is the simple critical batch size of [McCandlish et al. (2018)](#ref-mccandlish2018), $$\operatorname{tr}\operatorname{Cov}(g)/\lVert\mathbb E g\rVert^2$$, in batches of eight prompts; it does not depend on the scale. We report it for completeness only. Because neither mean is resolved, both intervals are unbounded above, and the point estimates are unstable: pooling all 64 tapes instead of averaging over blocks gives 309 for GRPO-8 and 260 for our algorithm.

Two caveats. The factor $$4/7$$ is exact for the gate-corrected weight of Section 2; this run used $$u(1-u)$$, under which each of these eight prompts gets between 0.54 and 0.99 of its share, less when its probe is often one-sided, so the match is approximate. And the KL term, which is charged on every kept token, is not on this scale.

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

**Setup.** We train Qwen3-8B with each method on 300 prompts from the [Reinforce-Ada hard-prompt dataset](#ref-dataset), one seed each, at learning rate $$2\times10^{-5}$$ with the [StableReinforce](#ref-tinker) loss in Tinker and KL coefficient 0.01. Every method draws 30 prompts per update. GRPO-8 samples eight answers to each and centers rewards by the group mean. Our algorithm has an expected budget of 1.96 million weighted tokens per update, probes included, and uses the weight $$u(1-u)$$, as in Section 4.1. Seq-Balance was trained with an earlier version of the grader, which differs on two training prompts. Every 25 updates we draw 16 answers to each of the same 60 held-out prompts for every method, graded the same way, and report accuracy among graded answers. Total compute counts prefill and sampled tokens plus twice the training tokens, and training compute counts only the latter; we convert both to FLOPs at $$2N$$ per token, with $$N=8\times10^9$$.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/heldout-accuracy-vs-compute.png' | relative_url }}"><img src="{{ '/assets/images/heldout-accuracy-vs-compute.png' | relative_url }}" alt="Held-out accuracy against total compute and against training compute for Seq-Balance, GRPO-8 and our algorithm. Our algorithm is above GRPO-8 at every checkpoint on both axes; at the end of training the two are level in total compute." loading="lazy"></a>
  <figcaption>Held-out accuracy against total compute (left) and training compute (right), after 25, 50, 75 and 100 updates (110 for GRPO-8).</figcaption>
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

Train accuracy is the right end of the middle panel in the figure below. At equal total compute, our algorithm and GRPO-8 end level on the held-out prompts, 53.7% against 53.2%, a difference of 0.5 points [−3.8, +4.9], while on the training prompts our algorithm is 2.6 points ahead. At equal training compute, our algorithm is ahead on the held-out prompts too. After 75 updates it has used 1.28 EFLOPs of training compute and reaches 53.8%, while GRPO-8 after 75 updates has used 1.36 EFLOPs and reaches 47.4%, a gap of 6.4 points [2.0, 11.0]. Brackets are 95% paired bootstrap intervals over the 60 held-out prompts. Seq-Balance ends at 47.4% on the held-out prompts, with 16% less total compute. On the training prompts, our algorithm is ahead of both baselines from about the 20th update on, pulls ahead in total compute after about 1.5 EFLOPs, and leads GRPO-8 for most of the run in training compute:

<figure class="three-views">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}"><picture>
    <source media="(max-width: 600px)" srcset="{{ '/assets/images/train-accuracy-three-views-stacked.png' | relative_url }}">
    <img src="{{ '/assets/images/train-accuracy-three-views.png' | relative_url }}" alt="Accuracy on the training prompts for Seq-Balance, GRPO-8 and our algorithm, against optimizer updates, total compute and training compute. Our algorithm ends highest in all three panels." loading="lazy">
  </picture></a>
</figure>

Each curve is the accuracy of answers sampled before each update, pooled over a trailing window: the last 10 updates on the left, the last 20M weighted tokens of total compute in the middle, and the last 8M of training compute on the right. The left panel counts the first four answers per prompt for every method; the other two count the first four for our algorithm and Seq-Balance and all eight for GRPO-8.

These are single training runs, and the intervals cover only the choice of held-out prompts, not the training seed.

## 5. Next steps {#open}

Everything here is small scale: one 8B model, one replay checkpoint with eight prompts, and one training seed per method. Next:

- **Scale.** More seeds, larger models and longer runs, to see how much of the lower variance turns into faster learning.
- **Planning in rounds.** Re-plan after each block of follow-ups instead of once after the probe. Theorem 1 still applies, because the plan still depends only on grades.
- **More settings.** Replays across prompt batches and stages of training, and how well the history-based weights and forecasts track the truth.

## Citation {#citation}

{% raw %}
```bibtex
@misc{chen2026adaptivesampling,
  author = {Chen, Chang-Han},
  title  = {How to think about compute-optimal adaptive sampling?},
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

### A. Score energy across layers {#app-energy}

This is the A1 check in three parameter groups, on the same 40 Qwen3-4B prompts and 48 rollouts each as Section 3, with the prompt effect divided out. Each bin is a length octile, with ±2σ error bars. Layers 18–35 and the final norm both scale close to linearly in length. In layers 0–8, heavy tails dominate a fit on the mean (exponent 0.78 ± 1.73), while a fit on log energy, which is robust to them, gives 0.93 ± 0.06.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/score-energy-by-length.png' | relative_url }}"><img src="{{ '/assets/images/score-energy-by-length.png' | relative_url }}" alt="Score energy rises with response length in three parameter groups. Correct and incorrect answers nearly overlap in layers 18 to 35 and in the final norm, with much noisier estimates in layers 0 to 8." loading="lazy"></a>
</figure>

### B. Why the mean baseline {#app-baseline}

Our update uses the group's success rate as its baseline: $$g_i$$ is REINFORCE with advantage $$R-p$$, reweighted to the kept answers. Two other baselines would lower the variance.

**A length-weighted baseline.** For the estimator $$(R-b)S$$ on all rollouts, the variance-optimal constant baseline is

$$
b^*=\frac{\mathbb E[R\|S\|^2]}{\mathbb E[\|S\|^2]}
\;\approx\;\frac{\mathbb E[RT]}{\mathbb E[T]},
$$

which counts long answers more. In the 40-prompt Qwen3-4B study, cross-fitting it saved **0.8% ± 1.8%** of the gradient noise on average (two-standard-error band), and about **14%** on one heavily truncated prompt. The left panel shows how far the fitted optimum sits from the mean reward, and the right panel what it saves on separate rollouts.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/baseline-moments.png' | relative_url }}"><img src="{{ '/assets/images/baseline-moments.png' | relative_url }}" alt="The fitted baseline tracks the length-weighted success rate. Across prompts the out-of-sample noise savings over the mean-reward baseline are small." loading="lazy"></a>
</figure>

**A baseline chosen with what we keep.** If the weights on $$\bar S_{i,+}$$ and $$\bar S_{i,-}$$ were free, variance would choose the baseline too. Under a training budget it picks $$b=0$$ on hard prompts and $$b=1$$ on easy ones: train on successes only, or on failures only. Both are unbiased, but they push the policy differently. Training on successes alone sharpens it and lowers pass@$$k$$ at large $$k$$, while training on failures alone keeps its diversity ([Zhu et al., 2025](#ref-zhu2025)). In bandits, baselines with equal variance make the policy commit early or keep exploring ([Chung et al., 2021](#ref-chung2021)). We fix the neutral baseline, $$b=p$$, before optimizing anything else.
