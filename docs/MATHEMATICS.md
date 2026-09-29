# From matrices to learning

Every equation below maps to an explicit NumPy operation in this repository. NumPy supplies arrays, random samples, and matrix multiplication; it does not supply differentiation, layers, or a training algorithm.

## Shapes and forward propagation

One row represents one example. For a batch of size $B$, input width $D$, and output width $H$:

| Symbol | Shape | Meaning |
|---|---|---|
| $X$ | $B \times D$ | Inputs or previous-layer activations |
| $W$ | $D \times H$ | Trainable weights |
| $b$ | $1 \times H$ | Biases, broadcast across rows |
| $Z=XW+b$ | $B \times H$ | Affine pre-activations |
| $A=f(Z)$ | $B \times H$ | Nonlinear activations |

ReLU is $f(z)=\max(0,z)$. Sigmoid is $s(z)=1/(1+e^{-z})$. The sigmoid implementation branches by sign so it never evaluates an overflowing exponential for large negative inputs.

For the last layer, softmax converts logits into class probabilities:

$$p_{ic}=\frac{\exp(z_{ic}-m_i)}{\sum_k\exp(z_{ik}-m_i)},\qquad m_i=\max_k z_{ik}.$$

Subtracting a row constant does not change softmax. It prevents exponentiating large positive values.

## Objective and stable cross-entropy

For target distributions $Y$ (including one-hot labels), the objective is:

$$L=-\frac1B\sum_i\sum_c Y_{ic}\log p_{ic}+\sum_l\frac{\lambda_l}{2}\lVert W_l\rVert_F^2.$$

Training computes log probabilities directly from shifted logits:

$$\log p_{ic}=(z_{ic}-m_i)-\log\sum_k\exp(z_{ik}-m_i).$$

This log-sum-exp form preserves the true loss even if a very unlikely class has a probability that underflows to zero. For example, logits `[1000, -1000]` with target class 1 have loss 2000 and gradient `[1, -1]`; an epsilon-clipped probability loss would hide that scale.

The separate probability API implements $-\mathrm{mean}(\sum Y\log(p+\epsilon))$ and its exact derivative $-Y/[B(p+\epsilon)]$. It is useful for inspecting the separate loss and softmax stages. Its epsilon smoothing differs slightly from the unsmoothed logit objective; the training loop always uses the logit form.

## Backpropagation: the chain rule in reverse

Combining the softmax Jacobian with cross-entropy gives:

$$\frac{\partial L}{\partial Z_{\mathrm{last}}}=\frac{P-Y}{B}.$$

The factor $1/B$ appears here **once**. Averaging again in a Dense layer would make gradients too small by another factor of the batch size.

Let $G=\partial L/\partial Z$ be a Dense layer's incoming gradient. Then:

$$dW=X^T G+\lambda W,\qquad db=\sum_{i=1}^{B}G_i,\qquad dX=GW^T.$$

`Dense.backward()` explicitly stores `dW`, `db`, and `dX`. The backward traversal finishes before any optimizer changes the weights, so `dX` uses the same weights as the forward pass.

For ReLU, multiply by $\mathbb{1}[Z>0]$. The chosen subgradient at zero is zero. For sigmoid, multiply by $s(Z)(1-s(Z))$.

The standalone softmax backward method computes a vector-Jacobian product for each row, without building a quadratic-size Jacobian:

$$J=\operatorname{diag}(p)-pp^T,\qquad J^Tg=p\odot\left(g-\sum_c g_cp_c\right).$$

`NeuralNetwork.backward()` starts with the fused loss gradient and traverses all layers except the final Softmax in reverse. Applying `Softmax.backward()` again here would double-count that derivative.

## Gradient descent, momentum, and Adam

SGD updates each parameter $\theta$ by $\theta\leftarrow\theta-\eta g$. Each shuffled mini-batch estimates the data gradient; the last, possibly smaller batch is normalized by its actual size. The L2 term is the gradient of the complete penalty on every step.

Classical momentum accumulates a velocity:

$$v_t=\mu v_{t-1}+g_t,\qquad\theta_t=\theta_{t-1}-\eta v_t.$$

This implementation does not multiply the new gradient by $1-\mu$ and does not implement Nesterov momentum.

Adam keeps exponential moving averages of the gradient and its elementwise square:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

Since both start at zero, divide by their accumulated mass:

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

The time step increments once per mini-batch. Each weight and bias array has its own moment state. The L2 derivative enters the gradient before Adam; this is ordinary L2 regularization, **not decoupled AdamW weight decay**. See the [original Adam paper](https://arxiv.org/abs/1412.6980).

## Initialization and regularization

All three initialization choices are zero-centered normal distributions. Biases are zero:

| Strategy | Weight standard deviation |
|---|---:|
| Random | $0.01$ |
| Xavier / Glorot normal | $\sqrt{2/(D+H)}$ |
| He normal | $\sqrt{2/D}$ |

The fan-based scales address changes in signal magnitude across layers. The He scale accounts for rectifier behavior. References: [Glorot and Bengio](https://proceedings.mlr.press/v9/glorot10a.html) and [He et al.](https://arxiv.org/abs/1502.01852).

Inverted dropout with drop rate $r$ samples $M\sim\mathrm{Bernoulli}(1-r)$ during training:

$$A'=A\odot\frac{M}{1-r},\qquad dA=dA'\odot\frac{M}{1-r}.$$

The same mask is reused in the backward pass. At inference the layer is the identity, with no additional scaling. `fit()` enables dropout for updates; `evaluate()`, `predict()`, and `predict_proba()` disable it automatically. Epoch-end training metrics also disable it, making them comparable to validation metrics.

## Numerical gradient checking

For a scalar parameter $\theta_j$, central finite differences approximate its derivative:

$$g_j^{\mathrm{numeric}}=\frac{L(\theta_j+h)-L(\theta_j-h)}{2h}.$$

The checker compares this with the manually backpropagated value, reports maximum absolute error and

$$\mathrm{relative\ error}=\frac{\lVert g^{\mathrm{analytic}}-g^{\mathrm{numeric}}\rVert_2}{\max(10^{-12},\lVert g^{\mathrm{analytic}}\rVert_2+\lVert g^{\mathrm{numeric}}\rVert_2)},$$

and applies an elementwise absolute/relative tolerance. This catches transposed matrices, missing bias sums, incorrect regularization, and double batch averaging. Checks include input derivatives and restore every perturbed value.

Use small float64 networks: cancellation makes finite differences unreliable in float32, and checking $P$ parameters needs approximately $2P$ objective evaluations. Avoid ReLU's nondifferentiable origin. Dropout must be disabled or its mask held fixed; the network checker rejects nonzero dropout, while a separate unit test checks dropout with an identical mask on both perturbations. These checks establish local derivative correctness, not convergence on every task.
