import torch
import torch.nn as nn


def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == 'warmup10':
        betas = _warmup_beta(start, end, n_timesteps, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(start, end, n_timesteps, 0.5)
    elif schedule == "exp":
        e_range = torch.linspace(torch.log(torch.tensor(start)), torch.log(torch.tensor(end)), n_timesteps)
        betas = torch.exp(e_range)
    else:
        raise NotImplementedError(schedule)
    return betas


def _warmup_beta(start, end, n_timestep, warmup_frac):

    betas               = end * torch.ones(n_timestep)
    warmup_time         = int(n_timestep * warmup_frac)
    betas[:warmup_time] = torch.linspace(start, end, warmup_time)

    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class GaussianDiffusion(nn.Module):
    def __init__(self, betas):
        super().__init__()

        betas = betas.type(torch.float64)
        timesteps = betas.shape[0]
        self.n_steps = int(timesteps)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        alphas = 1 - betas
        alphas = alphas
        alphas_prod = torch.cumprod(alphas, 0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        self.register("betas", betas.to(self.device))
        self.register("alphas", alphas.to(self.device))
        self.register("alphas_rod", alphas_prod.to(self.device))
        self.register("alphas_bar_sqrt", alphas_bar_sqrt.to(self.device))
        self.register("one_minus_alphas_bar_log", one_minus_alphas_bar_log.to(self.device))
        self.register("one_minus_alphas_bar_sqrt", one_minus_alphas_bar_sqrt.to(self.device))


    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))


    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0).to(self.device)
        alphas_t = extract(self.alphas_bar_sqrt, t, x_0)
        alphas_1_m_t = extract(self.one_minus_alphas_bar_sqrt, t, x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)
    
    
    def p_sample(self, model, x, t):
        t = torch.tensor([t]).to(self.device)
        # Factor to the model output
        eps_factor = ((1 - extract(self.alphas, t, x)) / extract(self.one_minus_alphas_bar_sqrt, t, x)).to(self.device)
        eps_theta = model(x, t).to(self.device)
        mean = (1 / extract(self.alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta)).to(self.device)
        z = torch.randn_like(x).to(self.device)
        # Fixed sigma
        sigma_t = extract(self.betas, t, x).sqrt().to(self.device)
        sample = mean + sigma_t * z
        return (sample)
    

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        # noise from Gauss distribution
        cur_x = torch.randn(shape).to(self.device)
        x_seq = [cur_x.detach().cpu()]
        # gradually denoise distribution
        for i in reversed(range(self.n_steps)):
            cur_x = self.p_sample(model, cur_x, i)
            x_seq.append(cur_x.detach().cpu())
        return x_seq
    

    def noise_estimation_loss(self, model, x_0):
        batch_size = x_0.shape[0]
        # Select a random step for each example
        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,))
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size].long().to(self.device)
        # x0 multiplier
        a = extract(self.alphas_bar_sqrt, t, x_0).to(self.device)
        # eps multiplier
        am1 = extract(self.one_minus_alphas_bar_sqrt, t, x_0).to(self.device)
        e = torch.randn_like(x_0).to(self.device)
        # model input
        x = x_0 * a + e * am1
        x = x.to(self.device)
        output = model(x, t)
        return (e - output).square().mean()