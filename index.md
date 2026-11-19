---
title: "The Science of Adaptive Sampling"
subtitle: "What should we spend the next rollout on, once a prompt already has a learning signal?"
# date: 2026-11-30         # set on the day we publish
draft: true                # shows the draft banner; remove before publishing
affiliation: UC Berkeley
authors:
  - name: Chang-Han Chen
    url: https://chang-han-chen.github.io/
excerpt: "We sort adaptive sampling for RLVR into signal seeking and variance reduction, pose variance reduction as an optimization problem at fixed inference and training compute, and solve it with Neyman allocation for pairs."
---


<div class="notice--info" markdown="1">
**TL;DR**

We investigate the science of adaptive sampling in RL with verifiable rewards.

1. We survey the past literature and identify two distinct focuses: **signal seeking**, which aims to get the first success on a hard task, and **variance reduction**, which aims to reduce the noise when mean gradient is non-zero.
2. We focus on variance reduction, and pose a precise constrained optimization problem: minimize the variance of the policy-gradient update at a fixed budget of inference *and* training compute.
3. We present an adaptive sampling algorithm that provably solves this problem under four stated assumptions. The solution is to **cost-weighted Neyman allocation for pairs**. When the number of training traces is fixed, a balanced group of half success and half failure minimizes the variance. So, for each prompt, we compute the expected number of success–failure pairs per inference compute, as well as the resulting variance reduction per training compute. We then minimize variance at fixed total FLOPs (estimated as inference tokens $$+2\times$$ training tokens). This is inspired by and improved upon the Seq-Balance algorithm from Reinforce-Ada.
4. We check these four assumptions empirically for Qwen3-4B on competition math.
5. On a frozen Qwen3-8B checkpoint, our algorithm has **39% lower gradient variance at equal compute** than Seq-Balance. Most of the reduction comes from early terminating extremem cases though. Uniformly allocating compute for prompts other than the all-ailure and all-success ones results in 30% lower variance already. 
6. In a single-seed training run of Qwen3-8B, our algorithm beats Seq-Balance on held-out competition math, reaching 52.7% pass@1 against 47.4% after 100 updates each.
</div>

## 1. Background: adaptive sampling for LLM RLVR {#background}

Reinforcement learning with verifiable rewards (RLVR) ([Lambert et al., 2024](#ref-lambert2024)) unlocks the reasoning capabilities of large language models (LLM) ([OpenAI, 2024](#ref-openai2024); [DeepSeek-AI, 2025](#ref-deepseek2025)). The algorithm is remarkably simple and intuitive: sample a group of answers to each prompt, a verifier marks each one right or wrong, and the algorithm pushes the correct responses to be more likely and penalizes the wrong one. This is essentially the simples RL algorithm, called REINFORCE ([Williams, 1992](#ref-williams1992)),

$$ \nabla J(\theta) = \mathbb{E}_{x\sim \pi_\theta}[ R(x) \nabla \log \pi_\theta(x)]$$

with reward being binary $$R\in\{0,1\}$$. 

It is kind of mindblowing that, if you implement this with _extreme_ attention to details, it just works! Looking back to the major advances in RLVR in the last two years, the common theme seems to be removing as much **bias** as possible (from e.g. numerical rounding, length normalization) that stands in the way of learning signals. If you can do that for it, REINFORCE just wants to learn. 

To train the best model at the frontier scale, you need faith in the training algorithms and are willing to spend compute ($$\approx$$ money) for variance reduction. As in classic statistics, the variance of training signals goes down like the inverse of sample number $$B$$, 

$$\operatorname{Var} \sim \frac{1}{B}$$

Frontier-scale RL runs use very large batches: the main RL run of Inkling 2 Small took 1,075 steps of 16K rollouts each, over 17 million rollouts in total on 2,048 GPUs ([Thinking Machines Lab, 2026](#ref-tml2026)).

In addition, a common technique is _reward baseline_ $$b$$. The reward in REINFORCE is then replaced by the _advantage_, $$A(x) = R(x)-b$$. A simple and effective baseline is the group's mean 

$$b= \mathbb{E}[R]$$

and is widely used in practice. When all the sample in a group gets the same reward, however, every advantage is zero and the prompt teaches the model nothing. There is no learning signal in this case.

From this perspective, any RLVR sampling algorithm has two phases:
1. **Signal seeking**: sample enough to get at least one success on as many questions of interest as possible
2. **Variance reduction**: among the questions with at least one success, how should we allocate compute to reduce the batch gradient?

As we will explain later, we chose an objective that has higher variance on questions that are very hard (but not impossible) or very easy (but not aced everytime). Our algorithm then decides how much more compute to spend on them. Notice that if a group has all success or failure, then it contributes **zero** to the variance instead of infinite. This is another way to see why signal seeking and variance reduction are two _separate_ algorithm considerations. 


We find that this perspective provides a nice dicatomy on the past literature
### 1.1 Signal Seeking

This is what a prompt _without_ signal looks like. Here are four answers from a Qwen3-8B checkpoint 75 steps into RL training, each cut to its last few lines:

<figure class="trace-fig">
<div class="trace-card">
  <div class="trace-prompt"><span class="lbl"><span>Prompt</span><span class="key">Correct answer: 55</span></span>A cube with a side of 5 is made up of 125 smaller cubes with a side of 1. How many small cubes does a plane perpendicular to one of the cube's diagonals and passing through its midpoint intersect?</div>
  <div class="trace bad"><span class="n">1</span><span class="excerpt">… after researching similar problems, the correct number of cubes intersected by a plane perpendicular to a space diagonal through the center of a cube of side length n is 3n² − 3n + 1. For n = 5: 75 − 15 + 1 = 61</span><span class="ans">61 <span class="mark">✗</span><span class="tok">2,439 tokens</span></span></div>
  <div class="trace bad"><span class="n">2</span><span class="excerpt">… Total number of cubes intersected = 6 + 10 + 15 = 31</span><span class="ans">31 <span class="mark">✗</span><span class="tok">1,649 tokens</span></span></div>
  <div class="trace bad"><span class="n">3</span><span class="excerpt">… After verifying with known cases and research, the correct number of small cubes intersected … is 3n² − 3n + 1. For n = 5: 75 − 15 + 1 = 61</span><span class="ans">61 <span class="mark">✗</span><span class="tok">1,373 tokens</span></span></div>
  <div class="trace bad"><span class="n">4</span><span class="excerpt">… which is known to be: for a cube of side length n (odd), the number is 3n² − 3n + 1. Thus, for n = 5 … 61</span><span class="ans">61 <span class="mark">✗</span><span class="tok">2,446 tokens</span></span></div>
</div>
</figure>

Apparently the problem is too difficult for the model. Three answers recall the same formula, which is right for a cube of side 3 but not 5, and the fourth miscounts. Since all attempts fail, the prompt contributes nothing to the batch gradient.

Most work on adaptive sampling so far is about prompts like this one, and its simplest form is now standard at scale. DAPO oversamples prompts, drops every group whose answers all agree, and keeps sampling until the batch is full of prompts with signal ([Yu et al., 2025](#ref-yu2025)). ScaleRL keeps this zero-variance filter in a recipe it scales to a single 100,000-GPU-hour run, and also stops revisiting prompts the model already passes at least 90% of the time ([Khatri et al., 2025](#ref-khatri2025)); Olmo 3 refills its batches the same way, which it calls active sampling ([Team Olmo, 2025](#ref-olmo2025)). Filtering pays for rollouts and then throws them away, so later work tries to spend them better: skip the prompts that are likely to come back all right or all wrong ([Zheng et al., 2025](#ref-zheng2025)), or move rollouts toward hard prompts, either under an explicit budget ([Li et al., 2025](#ref-li2025)) or by sampling a hard prompt until it is solved ([Noukhovitch et al., 2026](#ref-noukhovitch2026)). And when the model essentially never succeeds, more samples won't help; the remedy is to make the problem easier, for example by revealing part of a reference solution ([Qu et al., 2026](#ref-qu2026)).

We think much of signal seeking is better handled one level up, by designing environments and curricula that keep the model near the edge of what it can solve. How to find that edge automatically is a topic for another post.

### 1.2 Variance Reduction {#sharpen}

Now take a prompt the model sometimes gets right. The same checkpoint answers another competition problem four times, and one answer is correct:

<figure class="trace-fig">
<div class="trace-card">
  <div class="trace-prompt"><span class="lbl"><span>Prompt</span><span class="key">Correct answer: 9</span></span>How many ways are there to arrange the letters A, A, A, H, H in a row so that the sequence HA appears at least once?</div>
  <div class="trace bad"><span class="n">1</span><span class="excerpt">… 10. HHAAA → no HA … Arrangements with “HA”: 2, 3, 4, 5, 6, 8, 9 → that's 7 arrangements.</span><span class="ans">7 <span class="mark">✗</span><span class="tok">620 tokens</span></span></div>
  <div class="trace bad"><span class="n">2</span><span class="excerpt">… So the arrangements without “HA” are: 1. AAAHH 10. HHAAA … 10 − 2 = 8</span><span class="ans">8 <span class="mark">✗</span><span class="tok">824 tokens</span></span></div>
  <div class="trace ok"><span class="n">3</span><span class="excerpt">… 10. HHAAA → contains HA (positions 2-3) … Only arrangement 1 (AAAHH) does not contain “HA” … 10 − 1 = 9</span><span class="ans">9 <span class="mark">✓</span><span class="tok">772 tokens</span></span></div>
  <div class="trace bad"><span class="n">4</span><span class="excerpt">… the number of ways to place 2 H's into these 3 gaps is (3 choose 2) = 3 … 10 − 3 = 7</span><span class="ans">7 <span class="mark">✗</span><span class="tok">701 tokens</span></span></div>
</div>
</figure>

All four answers count the same 10 arrangements and part ways only when they look for the ones without HA: the third notices that HHAAA contains HA, and the others miss it. So this prompt has a real learning signal. The group mean is 1/4, so the update pushes the correct answer up and the three wrong ones down, and on average that moves the model toward answers like the third.

It is also very noisy. Each answer is hundreds of tokens long, and every token adds its own random contribution to the gradient. Most of those tokens, restating the problem or counting the 10 arrangements, have nothing to do with why the third answer is right, and with a single success the update cannot tell the decisive step from the incidental ones. More successes and failures average these accidents away. But every extra rollout costs inference compute, and every one we train on costs training compute. Variance reduction asks how many more to buy, and for which prompts. That is the question our algorithm answers.

Prior work has asked this question too. GVM-RAFT made the case for allocating samples unevenly across prompts to minimize gradient variance under a sampling budget ([Yao et al., 2025](#ref-yao2025)). Reinforce-Ada trains on balanced groups of successes and failures under a log-shaped objective ([Xiong et al., 2025](#ref-xiong2025)). PODS generates many rollouts but trains only on a high-variance subset, because the update phase has costs of its own ([Xu et al., 2025](#ref-xu2025)). VIP predicts each prompt's success rate with a Gaussian process and solves for the allocation that minimizes total gradient variance under a rollout budget ([Nguyen et al., 2026](#ref-nguyen2026)), and DUET does the same under a budget of generated tokens ([Hu et al., 2026](#ref-hu2026)).

These allocation rules budget rollouts or generated tokens. We also charge for training on the responses we keep, and we buy *pairs*, the unit a balanced update actually uses.

### 1.3 The problem {#problem}

Take a batch of prompts that have passed the probe with a mixed result. Prompt $$i$$ has $$s_i$$ correct answers out of 8 (so $$1\le s_i\le 7$$), success rate $$p_i$$, and mean rollout length $$T_i$$, counting the prompt.

- **What we train on.** Like Seq-Balance, we train on balanced success–failure pairs. If prompt $$i$$ ends up with $$K_i$$ pairs, its update $$\hat g_i$$ combines those $$K_i$$ successes and $$K_i$$ failures into an unbiased estimate of $$\nabla F(p_i)$$, the gradient of a fixed objective of the success rate. We use the log odds of success (§2.1).
- **What it costs.** We count compute in units of one forward token's FLOPs. Generating a rollout costs $$T_i$$. We price a forward–backward pass at twice the sampling cost per token, so training on a pair, two sequences, costs $$4T_i$$.

<div class="problem" markdown="1">
**Problem (variance-optimal sampling at fixed compute).** Choose a number of extra rollouts $$n_i\ge 0$$ for each prompt, and an unbiased update $$\hat g_i$$ from the $$K_i$$ pairs they produce, to

$$
\min_{n,\ \hat g}\ \operatorname{tr}\operatorname{Cov}\Big(\sum_i \hat g_i\Big)
\qquad\text{subject to}\qquad
\sum_i T_i\,\big(n_i + 4K_i\big)\;\le\; C .
$$

</div>

The first term of the budget is generation and the second is training. The pair count $$K_i$$ depends on how the rollouts come out; assumption A4 below replaces it by its expected value. Because the target $$\sum_i\nabla F(p_i)$$ does not depend on how we sample, minimizing the variance is the same as minimizing the simple critical batch size $$B_{\text{simple}}=\operatorname{tr}\operatorname{Cov}(\hat g)/\lVert\mathbb E\,\hat g\rVert^2$$ of [McCandlish et al. (2018)](#ref-mccandlish2018), the batch size beyond which larger batches stop paying for themselves.

## 2. The solution: Neyman allocation for pairs {#algorithm}

Under four assumptions the problem has a closed-form answer. We state them first; §3 checks A1–A3 against data. Let $$S=\nabla_\theta\log\pi_\theta(y\mid x)$$ be the **score** of a sampled response, a sum of one gradient per token.

- **A1. Score energy grows linearly with length:** $$\mathbb E\lVert S\rVert^2\approx vT$$ for a rollout of length $$T$$. Strictly, only the response carries score, but prompts are short next to responses (about 5% of the tokens in our replay), so we let $$T$$ include the prompt.
- **A2. Successes and failures have the same size:** at matched length, correct and incorrect responses have the same per-token score energy $$v$$. Their *mean* scores differ, and that difference is the signal.
- **A3. Noise dominates, and rollouts are independent:** rollouts are i.i.d. given the prompt, prompts are independent, and a response's mean score is small next to its spread, so second moments stand in for covariances.
- **A4. Known rates, expected pairs:** $$p_i$$ and $$T_i$$ are known (in practice, estimated from the prompt's history), and the pair count $$K_i$$ is replaced by its expected value.

### 2.1 The objective: log odds, to run the last mile {#objective}

Plain policy gradient climbs the average success rate. More generally, we can pick an increasing objective $$F(p)$$ and weight the policy gradient by $$F'(p)$$. An appealing choice is $$\log p$$, as in the large-compute limit of MaxRL ([Tajwar et al., 2026](#ref-tajwar2026)): sample until an answer succeeds, and the accepted answer has expected score $$\nabla\log p$$. This puts extra weight on hard prompts.

Once both outcomes are available, though, it's hard to argue that moving from **1% to 2%** should matter more than moving from **98% to 99%**. The first makes success twice as common; the second makes failure half as common. Log odds values the two changes equally:

$$
F(p)=\log\frac{p}{1-p}.
$$

In practice we smooth it as $$F_\lambda(p)=\log[(p+\lambda)/(1-p+\lambda)]$$ with $$\lambda=0.02$$, which keeps the derivative finite at the endpoints, and write $$w_\lambda(p)=p(1-p)\,F_\lambda'(p)$$ for the resulting weight. It is close to 1 except near 0% or 100% success.

### 2.2 The best update for one prompt {#estimator}

Write $$D(p)=p^2+(1-p)^2$$, which lies between $$\tfrac12$$ and $$1$$, and let $$\bar S_+$$ and $$\bar S_-$$ be the average scores of the $$K$$ successes and the $$K$$ failures.

<div class="theorem" markdown="1">
**Proposition 1 (best pair weights).** Under A1–A3, among updates $$a_+\bar S_+-a_-\bar S_-$$ that are unbiased for $$\nabla F_\lambda(p)$$, the variance is smallest for

$$
a_+=\frac{w_\lambda(p)\,(1-p)}{D(p)},\qquad a_-=\frac{w_\lambda(p)\,p}{D(p)},
$$

where it equals

$$
\frac{a}{K},\qquad a:=\frac{vT\,w_\lambda(p)^2}{D(p)} .
$$

</div>

<p class="proof" markdown="1">*Proof.* The score identity gives $$\mathbb E[S_+]=\nabla p/p$$, and since the unconditional score has mean zero, $$\mathbb E[S_-]=-\nabla p/(1-p)$$. The target is $$\nabla F_\lambda=F'_\lambda\nabla p=\tfrac{w_\lambda}{p(1-p)}\nabla p$$, so unbiasedness means $$(1-p)a_++p\,a_-=w_\lambda$$. By A1–A3 each class average has covariance trace $$vT/K$$ and the two are independent, so the variance is $$vT(a_+^2+a_-^2)/K$$. Minimizing $$a_+^2+a_-^2$$ on the line $$(1-p)a_++p\,a_-=w_\lambda$$ picks the point closest to the origin, $$(a_+,a_-)\propto(1-p,\,p)$$, which gives the weights above and $$a_+^2+a_-^2=w_\lambda^2/D$$. <span class="qed">∎</span></p>

Notice what dropped out. **Measured per pair, the success rate barely matters**: a pair from a 10% prompt is worth about as much as a pair from a 50% prompt (up to $$D\in[\tfrac12,1]$$). The success rate still matters, but through *how fast new pairs arrive*: each new pair needs one more of the rarer outcome, which shows up about once every ten rollouts at 10% success and once every two at 50%.

Seq-Balance is already close to this update. For one success and one failure, its mean-centered sum has expectation $$\mathbb E[(1-p)S_+-pS_-]=D(p)\,\nabla\log\tfrac{p}{1-p}$$, so it estimates the log-odds gradient up to a factor $$1/D(p)$$ between 1 and 2. (A [later revision of Reinforce-Ada](#ref-xiong2025) adds an importance weight; we compare against the original estimator.)

### 2.3 Splitting the budget {#allocation}

Now the allocation. Prompt $$i$$ starts with $$K_i^0=\min(s_i,\,8-s_i)$$ pairs from its probe. Each extra pair needs about $$1/q_i$$ more rollouts, with $$q_i=\min(p_i,1-p_i)$$, plus training on two more responses, so it costs

$$
c_i=T_i\Big(\frac{1}{q_i}+4\Big).
$$

<div class="theorem" markdown="1">
**Theorem 1 (Neyman allocation for pairs).** Under A1–A4, with a constant price $$c_i$$ per extra pair, the solution of the problem in §1.3 gives prompt $$i$$

$$
K_i^\star=\max\!\Big(K_i^0,\ \theta\sqrt{a_i/c_i}\Big)
$$

pairs, where $$\theta$$ is the unique level at which the budget is spent: $$\sum_i c_i\,(K_i^\star-K_i^0)=C$$. Equivalently, every prompt that gets extra rollouts ends with the same noise removed per FLOP, $$a_i/(c_iK_i^2)=1/\theta^2$$, and every prompt that gets none is already past that point.

</div>

<p class="proof" markdown="1">*Proof.* By Proposition 1 and independence across prompts, the variance of $$\sum_i\hat g_i$$ is $$\sum_i a_i/K_i$$. So we minimize $$\sum_i a_i/K_i$$ over $$K_i\ge K_i^0$$ subject to $$\sum_i c_i(K_i-K_i^0)\le C$$. The objective is strictly convex and decreasing in each $$K_i$$, and the constraints are linear, so the budget binds and the KKT conditions are necessary and sufficient: $$a_i/K_i^2=\lambda c_i-\mu_i$$ with $$\mu_i\ge 0$$ and $$\mu_i(K_i-K_i^0)=0$$. If $$K_i>K_i^0$$, then $$\mu_i=0$$ and $$K_i=\sqrt{a_i/(\lambda c_i)}$$. If $$K_i=K_i^0$$, then $$a_i/(K_i^0)^2\le\lambda c_i$$, i.e. $$K_i^0\ge\sqrt{a_i/(\lambda c_i)}$$. Both cases read $$K_i=\max(K_i^0,\theta\sqrt{a_i/c_i})$$ with $$\theta=\lambda^{-1/2}$$. The spend $$\sum_i c_i(K_i^\star-K_i^0)$$ is continuous and strictly increasing in $$\theta$$ once it is positive, so $$\theta$$ is unique. <span class="qed">∎</span></p>

In words: **every prompt has a target number of pairs, and compute goes first to the prompts furthest below their targets.** Raising the budget raises every target in proportion. Three consequences:

- **Length cancels.** The noise per pair $$a_i$$ and its price $$c_i$$ are both proportional to $$T_i$$, so the target $$\theta\sqrt{v\,w_\lambda^2/(D\,(1/q+4))}$$ depends only on the success rate. A prompt with long answers gets as many pairs as one with short answers; they just cost more.
- **Lopsided prompts get lower targets**, because their pairs are expensive, but they also start with fewer pairs. A 1/8 probe has one pair; a 4/8 probe has four.
- **Pairs in hand matter.** A prompt whose probe already exceeds its target gets nothing more, however cheap its rollouts are.

Theorem 1 is Neyman allocation from stratified sampling ([Neyman, 1934](#ref-neyman1934)): each stratum's sample size grows with its standard deviation and shrinks with the square root of its unit cost. GVM-RAFT, VIP and DUET derive rules of the same square-root form for rollouts. Two things differ here. The unit we buy is a pair, so the probe sets a floor $$K_i^0$$: a prompt is only ever topped up to its target, so the rule is a square root with a floor rather than a pure square root. And the price of a pair includes training on it.

### 2.4 The algorithm in practice {#practice}

1. **Probe.** Every prompt visit starts with eight rollouts. We estimate $$p_i$$ from the probe and the prompt's decayed past reward counts, with Beta(1,1) smoothing, and estimate $$T_i$$ from the same history.
2. **Gate.** A prompt whose probe is all right or all wrong gets no extra rollouts on this visit; it is a signal-seeking problem. The gate gives up some learning: a prompt with a true 10% success rate still produces eight failures about 43% of the time.
3. **Allocate.** For each mixed prompt we consider $$n\in\{0,8,16,24\}$$ extra rollouts and compute the expected pair count $$\mathbb E[K]$$ exactly from the binomial distribution. A small dynamic program then minimizes $$\sum_i T_i\,w_\lambda(p_i)^2/(D(p_i)\,\mathbb E[K_i])$$ subject to $$\sum_i T_i(n_i+4\,\mathbb E[K_i])\le C$$. This is the problem of §1.3 under A1–A4 in blocks of eight, which the dynamic program solves exactly; Theorem 1 is its continuous limit. (Our code counts only response tokens in the noise term, which slightly lowers the target of a prompt with a long prompt text.)
4. **Update.** We keep every balanced pair the pooled answers provide and apply the weights of Proposition 1. For a response with reward $$r$$ in a group with $$k$$ actual pairs, the per-token advantage is
   $$
   A(r)=\frac{4}{2k}\,\frac{w_\lambda(p)}{D(p)}\,(r-p),
   $$
   where $$4/(2k)$$ gives each prompt the scale of a four-response Seq-Balance group, so extra pairs add samples rather than weight. We center with the ordinary success rate; a length-weighted baseline is optimal in theory but saved less than 1% of the noise in practice ([Appendix C](#app-baseline)).

<details markdown="1">
<summary>What the planner leaves out</summary>

Plugging the expected pair count into the $$1/K$$ law (A4) is optimistic: the exact calculation would use $$\mathbb E[1/K]$$, which is larger when pair counts vary. The planner also ignores how the gate changes the mean and variance of a prompt's contribution when it sometimes drops the prompt. And realized lengths and pair counts can push a batch above or below the planned budget, so we compare methods by cumulative compute, including discarded rollouts.

</details>

<details markdown="1">
<summary>A two-prompt example with the discrete planner</summary>

Take two prompts. A got 1/8, and its rollouts run about 500 tokens, prompt included; B got 4/8, and its rollouts run about 200 tokens. Their histories put their success rates at 25% and 50%. With 24,000 units left after the probes, the planner gives **16 more rollouts to A and none to B**, while still training on B's four probe pairs. The plan uses about 21,200 units and expects about five pairs from A and four from B; another block of eight for B would go over budget. Spending about the same on 8 rollouts for A and 16 for B leaves 16% more modeled noise. In the language of Theorem 1, B's target is about a third higher than A's ($$\sqrt{a/c}\approx0.55$$ vs $$0.42$$, in units of $$\theta$$), but B starts with four pairs and A with one, so A is much further below its target.

</details>

## 3. Checking the assumptions {#assumptions}

We test A1 and A2 with frozen-policy diagnostics on **Qwen3-4B** in thinking mode: 40 hard competition-math prompts, 48 rollouts each. For A3 we use the Qwen3-8B checkpoint of §4 and report the upper end of a 95% interval.

<div class="table-wrap">
<table class="paper-table wrap">
<thead>
<tr><th>Assumption</th><th>What we measured</th><th>Result</th></tr>
</thead>
<tbody>
<tr><td>A1. Energy ∝ length</td><td>fitted exponent of score energy vs. length, layers 18–35</td><td class="hl">0.97 ± 0.03</td></tr>
<tr><td>A2. Same size for both outcomes</td><td>energy of incorrect / correct answers at matched length</td><td class="hl">1.005 ± 0.007</td></tr>
<tr><td>A3. Noise dominates</td><td>‖E g‖² / tr Cov(g) per batch, Qwen3-8B replay (§4)</td><td>at most about 0.4%</td></tr>
</tbody>
</table>
</div>

The plot below shows A1 and A2 for layers 18–35 on log–log axes: score energy against rollout length, with each prompt's energy per token divided out and answers averaged in length bins. Correct answers, incorrect answers, and incorrect answers cut off at the length cap all fall on the same slope-one line. Earlier layers have much heavier tails ([Appendix A](#app-energy)).

<figure class="narrow">
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/score-energy-main.png' | relative_url }}"><img src="{{ '/assets/images/score-energy-main.png' | relative_url }}" alt="Score energy against rollout length on log-log axes, for layers 18 to 35. Correct answers, incorrect answers, and incorrect answers that hit the length cap all lie on the same slope-one line." loading="lazy"></a>
</figure>

A1 and A2 hold well in the later layers. Together they also predict that plain policy gradient has per-rollout noise $$\operatorname{tr}\operatorname{Cov}((R-p)S)\approx v\,p(1-p)\,T$$: the $$p(1-p)$$ is the variance of the binary reward, and the $$T$$ comes from summing over the response. [Appendix B](#app-variance) checks the $$p(1-p)$$ dependence across prompts. A3 holds by a wide margin: at the Qwen3-8B checkpoint, a batch's mean gradient is a fraction of a percent of its noise, so this checkpoint sits deep in the noise-dominated regime.

## 4. Results {#results}

### 4.1 At equal compute, our algorithm cuts gradient variance by 39% {#replay}

**Setup.** We froze the step-100 Qwen3-8B checkpoint from our algorithm's training run, eight training prompts, and their success-rate and length estimates. We then sampled 128 independent rollout tapes, in two rounds of 64, plus 8 more for calibrating budgets, and replayed three collection rules on every tape:

- **Seq-Balance**, as in Reinforce-Ada: sample in rounds of eight until there are two successes and two failures, up to 32 rollouts.
- **Uniform follow-ups**: our probe, gate and pair update, with the extra blocks of rollouts spread evenly at random over the mixed prompts.
- **Our algorithm**: the same, with the extra blocks chosen by the planner of §2.4.

All three use the same grades, the same smoothed log-odds weights from the frozen success rates, the same realized-pair normalization, and zero KL. Budgets were calibrated on the separate tapes so that mean compute roughly matches. Tinker returned LoRA gradient norms at zero learning rate, from which we estimate the gradient variance. The two rounds cost about $120.

The table gives each rule's compute and gradient variance per batch of eight prompts. The last row multiplies the two and divides by Seq-Balance's value, which compares the rules at equal compute; brackets are 95% intervals from a bootstrap over blocks of tapes.

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th>Per batch</th><th>Seq-Balance</th><th>Uniform follow-ups</th><th>Our algorithm</th></tr>
</thead>
<tbody>
<tr><td>Compute (forward-token units)</td><td>455k</td><td>449k</td><td>444k</td></tr>
<tr><td>Gradient variance, tr Cov</td><td>2.73M</td><td>1.94M</td><td>1.71M</td></tr>
<tr class="rule-above"><td>Variance × compute, vs. Seq-Balance</td><td>1</td><td>0.70 <span class="muted">[0.65, 0.75]</span></td><td class="hl">0.61 <span class="muted">[0.58, 0.65]</span></td></tr>
</tbody>
</table>
</div>

The plot below shows the same comparison: the left panel gives each rule's gradient variance per batch, and the right panel puts the rules on equal compute. Lines are 95% intervals.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/replay-128.png' | relative_url }}"><img src="{{ '/assets/images/replay-128.png' | relative_url }}" alt="Left: gradient variance per batch is 2.73 million for Seq-Balance, 1.94 million for uniform follow-ups and 1.71 million for our algorithm. Right: variance times compute relative to Seq-Balance is 0.70 for uniform follow-ups and 0.61 for our algorithm, both with intervals well below 1." loading="lazy"></a>
</figure>

Two things stand out:

1. **Our algorithm beats Seq-Balance by a wide margin**: 39% lower variance at equal compute, with an interval of 35–42%.
2. **Both parts of the design pay off.** Probing, gating and keeping every pair already cut variance by 30%, with follow-up rollouts spread uniformly. Neyman allocation of those follow-ups then removes another 13% of the variance that remains (interval 7–18%).

<details markdown="1">
<summary>Measuring variance and signal from gradient norms</summary>

For each rule and each round of 64 tapes, Tinker returned the unclipped LoRA gradient norm of every tape, of 32 disjoint pairs of tapes, of disjoint blocks of tapes (8 in the first round, 16 in the second), and of all 64 together, with the optimizer at zero learning rate; we checked that the policy stayed frozen throughout. For $$B$$ independent batch gradients $$g_1,\dots,g_B$$ with mean $$G$$,

$$
\widehat V=\frac{\sum_b\lVert g_b\rVert^2-\lVert\sum_b g_b\rVert^2/B}{B-1},
\qquad
\widehat{\lVert G\rVert^2}=\frac{\lVert\sum_b g_b\rVert^2-\sum_b\lVert g_b\rVert^2}{B(B-1)}
$$

are unbiased for $$\operatorname{tr}\operatorname{Cov}(g)$$ and $$\lVert G\rVert^2$$. For a pair they reduce to $$\lVert g_1-g_2\rVert^2/2$$ and $$g_1\cdot g_2$$. The signal estimate is signed, so it can come out negative when the true signal is small.

</details>

### 4.2 A single-seed training run {#training}

**Setup.** We train Qwen3-8B with each of three methods on 300 prompts from the [Reinforce-Ada hard-prompt dataset](#ref-dataset), one seed each, at learning rate $$2\times10^{-5}$$ with the [StableReinforce](#ref-tinker) loss in Tinker. Seq-Balance and our algorithm ran 100 updates. GRPO samples a fixed group of eight answers for each of 30 prompts per batch and centers rewards by the group mean; it ran until it had used as much compute as our algorithm, at 110 updates. Every 25 updates we draw 16 answers to each of 60 held-out prompts and report pass@1 among graded answers (1–3% of answers per checkpoint could not be graded). Compute counts prefill and sampled tokens plus twice the learner tokens.

The plot below follows held-out accuracy against cumulative training compute, with a point every 25 updates.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/heldout-three-arms.png' | relative_url }}"><img src="{{ '/assets/images/heldout-three-arms.png' | relative_url }}" alt="Held-out pass@1 against cumulative training compute for one run per method. GRPO ends at 53.2 percent at 196 million units and our algorithm at 52.7 percent at 198 million; Seq-Balance peaks at 49.2 percent at 118 million and ends at 47.4 percent at 166 million." loading="lazy"></a>
</figure>

The table compares the final checkpoints, with compute in millions of weighted tokens; brackets are 95% paired bootstrap intervals over the 60 held-out prompts.

<div class="table-wrap">
<table class="paper-table">
<thead>
<tr><th></th><th>Seq-Balance</th><th>GRPO</th><th>Our algorithm</th></tr>
</thead>
<tbody>
<tr><td>Held-out pass@1, final checkpoint</td><td>47.4% <span class="muted">at 166M</span></td><td>53.2% <span class="muted">at 196M</span></td><td>52.7% <span class="muted">at 198M</span></td></tr>
<tr><td>Best checkpoint</td><td>49.2% <span class="muted">at 118M</span></td><td>53.2% <span class="muted">at 196M</span></td><td>52.7% <span class="muted">at 198M</span></td></tr>
<tr><td>Our algorithm minus this method, final checkpoints</td><td>+5.3 pts <span class="muted">[+1.9, +8.8]</span></td><td>−0.5 pts <span class="muted">[−3.9, +3.0]</span></td><td>—</td></tr>
<tr><td>Pairs kept per million weighted tokens</td><td>29.6</td><td>—</td><td>46.3</td></tr>
</tbody>
</table>
</div>

Our algorithm beats Seq-Balance. After the same 100 updates it reaches 52.7% held-out pass@1 against Seq-Balance's 47.4%, a gap of 5.3 points with an interval of +1.9 to +8.8. It also gets more of what its update needs out of each unit of compute, keeping 46.3 success–failure pairs per million weighted tokens against Seq-Balance's 29.6. At equal compute it is on par with GRPO, the standard baseline.

## 5. Next steps {#open}

Everything in this post is small scale: one 8B model, a replay on a single checkpoint with eight prompts, and one training seed per method. Even at this scale, the scientific picture is clear and positive. The assumptions behind Neyman allocation for pairs hold, our algorithm cuts gradient variance by 39% at equal compute, and it beats Seq-Balance in training. Translating lower variance into a reliable performance gain will take more scale and more iteration:

- **Training at scale.** With one seed, our algorithm is on par with GRPO at equal compute rather than ahead of it, and Seq-Balance's 100 updates used about 18% less compute than ours (its run also used slightly different grader references). Several seeds, larger models and longer runs, with a shared reward objective across methods, would show how much of the lower variance turns into faster learning.
- **More baselines and settings.** A GRPO arm in the gradient replay, and the replay repeated across prompt batches and stages of training.
- **Assumption A4.** How well the history-based estimates of $$p_i$$ and $$T_i$$ track the truth, and how much the expected-pair approximation costs.

## Citation {#citation}

{% raw %}
```bibtex
@misc{chen2026adaptivesampling,
  author = {Chen, Chang-Han},
  title  = {The Science of Adaptive Sampling},
  year   = {2026},
  url    = {https://chang-han-chen.github.io/science-adaptive-sampling/},
  note   = {Blog post}
}
```
{% endraw %}

## References {#refs}

<div class="references" markdown="1">

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
1. {: #ref-tajwar2026} F. Tajwar et al. [Maximum likelihood reinforcement learning](https://arxiv.org/abs/2602.02710). arXiv:2602.02710, 2026. (Large-compute limit: [§4](https://arxiv.org/html/2602.02710#S4).)
1. {: #ref-tml2026} Thinking Machines Lab. [Introducing Inkling 2 Small](https://thinkingmachines.ai/news/inkling-2-small/). Blog post, November 2026.
1. {: #ref-tinker} Thinking Machines. [StableReinforce](https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/), Tinker documentation.
1. {: #ref-williams1992} R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning* 8, 229–256, 1992.
1. {: #ref-xiong2025} W. Xiong et al. [Reinforce-Ada: An adaptive sampling framework under non-linear RL objectives](https://arxiv.org/abs/2510.04996). arXiv:2510.04996, 2025. (Original Seq-Balance estimator: [v1, §2.3](https://arxiv.org/html/2510.04996v1#S2.SS3); importance weight: [v3, §3.2](https://arxiv.org/html/2510.04996v3#S3.SS2).)
1. {: #ref-xu2025} Y. E. Xu, Y. Savani, F. Fang, and J. Z. Kolter. [Not all rollouts are useful: Down-sampling rollouts in LLM reinforcement learning](https://arxiv.org/abs/2504.13818). *TMLR*, 2026.
1. {: #ref-yao2025} J. Yao et al. [Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and RL](https://arxiv.org/abs/2505.02391). arXiv:2505.02391, 2025.
1. {: #ref-yu2025} Q. Yu et al. [DAPO: An open-source LLM reinforcement learning system at scale](https://arxiv.org/abs/2503.14476). *NeurIPS*, 2025.
1. {: #ref-zheng2025} H. Zheng et al. [Act only when it pays: Efficient reinforcement learning for LLM reasoning via selective rollouts](https://arxiv.org/abs/2506.02177). *NeurIPS*, 2025.

</div>

## Appendix {#appendix}

### A. Score energy across layers {#app-energy}

Here is the A1 check in three parameter groups, on the same 40 Qwen3-4B prompts and 48 rollouts each as §3, with the prompt effect divided out. Each bin is a length octile, with ±2σ error bars. Layers 18–35 and the final norm both scale close to linearly in length. In layers 0–8, heavy tails dominate a fit on the mean (exponent 0.78 ± 1.73), while a fit on log energy, which is robust to them, gives 0.93 ± 0.06.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/score-energy-by-length.png' | relative_url }}"><img src="{{ '/assets/images/score-energy-by-length.png' | relative_url }}" alt="Score energy rises with response length in three parameter groups. Correct and incorrect answers nearly overlap in layers 18 to 35 and in the final norm, with much noisier estimates in layers 0 to 8." loading="lazy"></a>
</figure>

### B. Variance and success rate across prompts {#app-variance}

A1 and A2 predict that plain policy gradient has per-rollout noise $$\operatorname{tr}\operatorname{Cov}((R-p)S)\approx v\,p(1-p)\,T$$ (§3). To check the $$p(1-p)$$ part, we froze Qwen3-4B in thinking mode and, for selected prompts, measured the length-normalized, approximately centered LoRA gradient $$h=(R-p)\widetilde S/\sqrt{T}$$. The left panel below shows its covariance trace, which broadly follows the reference line proportional to $$p(1-p)$$, with visible exceptions. The right panel shows the squared norm of its mean, which varies substantially between prompts.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/gradient-variance-by-reward.png' | relative_url }}"><img src="{{ '/assets/images/gradient-variance-by-reward.png' | relative_url }}" alt="Across selected prompts, length-normalized gradient variance broadly follows reward variance. The squared mean gradient varies substantially between prompts." loading="lazy"></a>
</figure>

### C. Why we keep the ordinary mean baseline {#app-baseline}

For the IID score estimator $$(R-b)S$$ with a constant baseline $$b$$, the mean gradient does not depend on $$b$$, because $$\mathbb E[S]=0$$. Minimizing its variance gives

$$
b^*=\frac{\mathbb E[R\|S\|^2]}{\mathbb E[\|S\|^2]}
\;\approx\;\frac{\mathbb E[RT]}{\mathbb E[T]},
$$

so a long answer counts for more when estimating the baseline. In practice this made little difference on most prompts in the 40-prompt Qwen3-4B study: cross-fitting the length-weighted baseline saved **0.8% ± 1.8%** of the gradient noise on average (two-standard-error band). On one heavily truncated prompt in a deeper follow-up, it saved about **14%**. So our algorithm uses the ordinary, unweighted success rate, with shrinkage for small sample counts, and tracks length only to estimate noise and cost.

The left panel below shows how far the fitted optimum sits from the ordinary mean reward, and the right panel shows how much it saves over that mean when evaluated on separate rollouts. Both describe the IID estimator; they motivate, rather than test, our choice for the balanced update.

<figure>
  <a class="figure-link" target="_blank" rel="noopener" href="{{ '/assets/images/baseline-moments.png' | relative_url }}"><img src="{{ '/assets/images/baseline-moments.png' | relative_url }}" alt="The fitted baseline tracks the length-weighted success rate. Across prompts the out-of-sample noise savings over the mean-reward baseline are small." loading="lazy"></a>
</figure>
