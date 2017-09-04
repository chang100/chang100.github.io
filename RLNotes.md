# Reinforcement Learning

## RL Basics

### Value Iteration

1. Initialize $V_0(s_i) =0$ for all states $s_i$,

2. Set $k=1$

3. Loop until convergence:
   $$
   V_{k+1}(s) = \max_a \left[r(s,a) + \gamma\sum_{s'\in S}p(s'|a,s)V_k(s')\right]
   $$

4. Extract Policy

### Policy Iteration

1. $i=0$; Initialize $\pi_0(s)$ randomly for all states $s$
2. Converged = 0
3. While $i == 0$ or $|pi_i - \pi_{i-1}| > 0$:
   1. i++
   2. Policy Evaluation
   3. Policy improvement

#### Policy Evaluation

- $V_{k+1}(s) = \max_a \left[r(s,a) + \gamma \sum_{s' \in S} p(s'|s,a)) V_k(s')\right]$
- $V_{k+1}^\pi (s) = r(s,\pi(s)) = r(s,\pi(s)) + \gamma \sum_{s' \in S} p(s'|s,\pi(s)) V_k^\pi(s')$

#### Policy Improvement

- Find Q-value for each state-action pair. 
- Take argmax of Qs

### Policy Iteration vs Value Iteration

- Policy iteration has **fewer** iterations but is more expensive per iteration

- Value iteration has **more** iterations but is cheaper per iteration

  ​