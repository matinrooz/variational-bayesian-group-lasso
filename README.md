# variational-bayesian-group-lasso

Linear regression with group sparsity, where the posterior over the weights is learned with
variational inference. Instead of a plain Gaussian posterior, the samples go through 16
planar normalizing flows, so the posterior can have a more flexible shape.

## Model

- y = Xw + ε, with the noise level learned from the data.
- q(w): a diagonal Gaussian, then 16 planar flows (Rezende & Mohamed, 2015). The flow
  parameter u is projected so that every flow stays invertible.
- Loss = Gaussian negative log-likelihood + KL(q ‖ N(0, I)) / n + λ · Σ_g ‖w_g‖.
  The KL and the group penalty are both computed on the weights after the flows,
  estimated with 8 samples per step.
- Prediction: mean and standard deviation over 100 posterior samples, plus the noise.

## Example

`example.py` makes synthetic data: 1000 samples, 20 features in 4 groups of 5, and only
groups 0 and 2 have non-zero weights (noise std 0.1). 800 samples for training, 200 for
testing, 1000 epochs with Adam (lr 0.01, λ = 0.1).

Results with seed 42:

| | |
|---|---|
| test MSE | 0.0104 |
| coverage of the 95% prediction interval | 95.0% |
| learned noise std | 0.098 (true 0.1) |

Norm of each weight group, true vs. posterior mean:

| group | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| true | 2.94 | 0 | 2.52 | 0 |
| learned | 2.93 | 0.01 | 2.52 | 0.01 |

So the two inactive groups are pushed almost to zero and the intervals are well calibrated
on this data.

## Limitations

- The prior is a standard normal. The group sparsity comes from the extra penalty term, not
  from a group-sparse prior, so it is not a fully Bayesian group lasso.
- The example is easy: few features, little noise, groups of equal size. I have not tested
  it on real data yet.
- The groups must have the same size (`input_dim` divisible by `num_groups`).

## Run

```bash
pip install -r requirements.txt
python example.py
```

Files: `bayesian_group_lasso.py` (flows, model, training), `example.py` (synthetic data and
evaluation), `visualization.py` (weight heatmaps, loss curve, group norms, predictions).

## References

- D. Rezende and S. Mohamed. Variational Inference with Normalizing Flows. ICML 2015.
- M. Yuan and Y. Lin. Model selection and estimation in regression with grouped variables.
  Journal of the Royal Statistical Society B, 2006.
