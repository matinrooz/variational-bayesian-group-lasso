import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PlanarFlow(nn.Module):
    # f(z) = z + u_hat * tanh(w^T z + b)
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim) * 0.1)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def u_hat(self):
        # the planar flow is only invertible if w^T u >= -1, so we
        # project u (Rezende & Mohamed, 2015, appendix A)
        wu = (self.w @ self.u.t()).squeeze()
        m = -1 + F.softplus(wu)
        return self.u + (m - wu) * self.w / (self.w.pow(2).sum() + 1e-8)

    def forward(self, z):
        u = self.u_hat()
        h = torch.tanh(z @ self.w.t() + self.b)          # [S, 1]
        psi = (1 - h ** 2) * self.w                       # [S, D]
        log_det = torch.log(torch.abs(1 + psi @ u.t()) + 1e-8).squeeze(-1)
        return z + u * h, log_det


class NormalizingFlow(nn.Module):
    def __init__(self, dim, flow_length):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(flow_length)])

    def forward(self, z):
        sum_log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.flows:
            z, ld = flow(z)
            sum_log_det = sum_log_det + ld
        return z, sum_log_det


class BayesianGroupLasso(nn.Module):
    def __init__(self, input_dim, num_groups, flow_length=16, group_weights=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_size = input_dim // num_groups

        # base distribution q0(w) = N(q_mu, softplus(q_rho)^2)
        self.q_mu = nn.Parameter(torch.randn(input_dim) * 0.1)
        self.q_rho = nn.Parameter(torch.full((input_dim,), -3.0))

        if group_weights is None:
            group_weights = torch.ones(num_groups)
        self.register_buffer("group_weights", group_weights)

        self.flow = NormalizingFlow(input_dim, flow_length)

        # observation noise y = Xw + e, e ~ N(0, noise_std^2), learned from the data
        self.log_noise_std = nn.Parameter(torch.tensor(0.0))

    def noise_std(self):
        return self.log_noise_std.exp()

    def _sample(self, num_samples):
        # returns flowed weights w_K and log q_K(w_K) for each sample
        sigma = F.softplus(self.q_rho)
        eps = torch.randn(num_samples, self.input_dim, device=self.q_mu.device)
        w0 = self.q_mu + sigma * eps   # reparameterization trick

        log_q0 = (-0.5 * eps ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(-1)
        wK, sum_log_det = self.flow(w0)
        log_qK = log_q0 - sum_log_det   # change of variables
        return wK, log_qK

    def sample_weights(self, num_samples=1):
        w, _ = self._sample(num_samples)
        return w if num_samples > 1 else w.squeeze(0)

    def forward(self, X, num_samples=1):
        w, log_q = self._sample(num_samples)   # [S, D], [S]
        y_pred = X @ w.t()                      # [N, S]
        return y_pred.t(), w, log_q             # [S, N]

    def kl_divergence(self, w, log_q):
        # Monte Carlo estimate of KL(q_K || p) with a standard normal prior.
        # Before, this was computed on the base distribution only, so the
        # flows never showed up in the KL.
        log_p = (-0.5 * w ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)
        return (log_q - log_p).mean()

    def group_lasso_penalty(self, w):
        # penalty on the weights the model actually uses (after the flow),
        # averaged over the posterior samples
        w = w.view(w.shape[0], self.num_groups, self.group_size)
        group_norms = w.norm(dim=-1)                   # [S, G]
        return (group_norms * self.group_weights).sum(-1).mean()

    def predict(self, X, num_samples=100):
        # predictive std = uncertainty about w (spread of the samples)
        # plus the observation noise
        with torch.no_grad():
            preds, _, _ = self.forward(X, num_samples)
            var = preds.var(0) + self.noise_std() ** 2
        return preds.mean(0), var.sqrt()


def train_model(model, X, y, num_epochs=1000, lr=0.01, group_penalty=0.1, num_samples=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'reconstruction_loss': [], 'kl_loss': [], 'group_penalty': []}
    n = X.shape[0]

    pbar = tqdm(range(num_epochs), desc='Training')
    for epoch in pbar:
        optimizer.zero_grad()

        y_pred, w, log_q = model(X, num_samples)
        # gaussian negative log-likelihood instead of plain MSE, so the
        # noise level is learned too (otherwise the intervals are too narrow)
        s = model.noise_std()
        recon = (0.5 * ((y - y_pred) / s) ** 2 + torch.log(s)
                 + 0.5 * math.log(2 * math.pi)).mean()
        # recon is a mean over the data, so the KL is scaled by 1/n
        # to keep both terms on the same scale
        kl = model.kl_divergence(w, log_q) / n
        penalty = group_penalty * model.group_lasso_penalty(w)

        loss = recon + kl + penalty
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['reconstruction_loss'].append(recon.item())
        history['kl_loss'].append(kl.item())
        history['group_penalty'].append(penalty.item())

        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return history
