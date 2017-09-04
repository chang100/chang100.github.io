# Reinforcement Learning

## Lec 2: RL Basics

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

### TD Learning

- Maintain an estimate of $V^\pi(s)$ for all states

  - Update $V^\pi(s)$ each time after each transition $(s,a,s',r)$

  - Likely outcomes $s'$ will contribute updates more often

  - Approximating expectation over next state with samples

  - Running average:

    ​
    $$
    V_\text{samp}(s) = r + \gamma V^\pi(s') \\
    V^\pi(s) = (1-\alpha)V^\pi(s) + \alpha V_\text{samp}(s)
    $$




## Lec 3: Monte Carlo and Generalization

### Monte Carlo Methods

- **Monte Carlo** methods are learning methods

  - Experience $\rightarrow$ values, policy

  - Monte Carlo uses the simples possible idea: value = mean return

  - Monte carlo can be used in two ways:

    - **Model-free:** No model necessary and still attains optimality
    - **Simulation:** Needs only a simulation, not a full model

- **First-Visit MC Policy Evaluation:**
  - The **first** time-step $t$ that state $s$ is visited in an episode,
    - **Increment counter:** $N(s) \leftarrow N(s) + 1$
    - **Increment total return:** $S(s) \leftarrow S(s) + G_t$ 
  - Value is estimated by mean return $V(s) = \frac{S(s)}{N(s)}$
- **Every-Visit MC Policy Evaluation**
  - **Every time step** $t$ that state $s$ is visited in an episode,
    - **Increment counter:** $N(s) \leftarrow N(s) + 1$
    - **Increment total return:** $S(s) \leftarrow S(s) + G_t$
  - Value is estimated by mean return $V(s) = \frac{S(s)}{N(s)}$
- Monte Carlo is most useful when a **model is not available**
- Benefits over DP:
  - Can learn directly from interacting with environment
  - No need for full models
  - No need to learn about all states
  - Less harmed by violating MArkov property
- One issue is maintaining sufficient exploration

### Function Approximation

- We want to be able to generalize without having to explicitly store information for every single state
- Generalization should work because of **smoothness assumption**
- **Value Function Approximation**:
  - Represent each state using a feature vector $\mathbf{x}$
  - Have loss function $J$ 
  - Perform gradient descent on $J$



## Deep Reinforcement Learning

- Approximate function using a neural net
- Build a dataset from agent's own experience, then train neural net from this dataset



### Double Q-Learning

- Train 2 action-value functions $Q_1$ and $Q_2$

- Do $Q$-learning on both, but

  - pick **one of the two functions** to be updated at each time step

    ​

















