import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Sequence
import numpy as np
import torch.distributions as dist
from torch.distributions import Independent, Normal, MultivariateNormal

def variance_scaling_init_(
    tensor: torch.Tensor,
    scale: float = 1.0,
    mode: str = "fan_in",
    distribution: str = "truncated_normal"
) -> torch.Tensor:
    distribution = distribution.lower()
    if distribution not in {"truncated_normal","untruncated_normal","uniform"}:
        raise ValueError("distribution must be 'truncated_normal','untruncated_normal', or 'uniform'")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
        raise ValueError("mode must be 'fan_in', 'fan_out', or 'fan_avg'")
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        n = fan_in
    elif mode == "fan_out":
        n = fan_out
    else:
        n = (fan_in + fan_out) / 2.0
    variance = scale / max(1.0, n)
    with torch.no_grad():
        if distribution == "truncated_normal":
            correction = 0.87962566103423978
            std = math.sqrt(variance) / correction
            return nn.init.trunc_normal_(tensor,mean=0.0,std=std,a=-2.0 * std,b=2.0 * std)
        if distribution == "untruncated_normal":
            std = math.sqrt(variance)
            return tensor.normal_(mean=0.0, std=std)
        if distribution == "uniform":
            limit = math.sqrt(3.0 * variance)
            return tensor.uniform_(-limit, limit)

class NearZeroInitializedLinear(nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        scale: float = 1e-4,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features=input_size,
            out_features=output_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        # Overrides nn.Linear's default initialization.
        variance_scaling_init_(
            self.weight,
            scale=scale,
            mode="fan_in",
            distribution="truncated_normal",
        )

        if self.bias is not None:
            nn.init.zeros_(self.bias)


    

class AcmeLayerNormMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        w_init: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        activation: type[nn.Module] = nn.ELU,
        activate_final: bool = False,
    ) -> None:
        super().__init__()

        if len(layer_sizes) == 0:
            raise ValueError("layer_sizes must contain at least one size")

        self._w_init = w_init
        # self._activate_final = activate_final KEEP OR DELETE THSI LINE?

        layers: list[nn.Module] = []
        first_linear = nn.Linear(in_features=input_size,out_features=layer_sizes[0])
        self._initialize_linear(first_linear)

        layers.extend([
            first_linear,
            nn.LayerNorm(
                # shape of final dimension being normalized is 512 - output dim of the 1st linear layer
                normalized_shape=layer_sizes[0],
                elementwise_affine=True,
            ),nn.Tanh()])
        previous_size = layer_sizes[0]

        for index, output_size in enumerate(layer_sizes[1:]):
            linear = nn.Linear(in_features=previous_size,out_features=output_size)
            self._initialize_linear(linear)
            layers.append(linear)
            is_final_layer = (index == len(layer_sizes[1:]) - 1)
            
            if not is_final_layer or activate_final:
                layers.append(activation())
            previous_size = output_size
        self._network = nn.Sequential(*layers)

    def _initialize_linear(self, linear: nn.Linear) -> None:
        if self._w_init is None:
            variance_scaling_init_(linear.weight,scale=0.333,mode="fan_out",distribution="uniform")
        else:
            self._w_init(linear.weight)
        nn.init.zeros_(linear.bias)
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self._network(observations)

class MultivariateNormalDiagHead(nn.Module):
    """Module that produces a multivariate normal distribution."""

    def __init__(
        self,
        input_size: int,
        num_dimensions: int,
        init_scale: float = 0.3,
        min_scale: float = 1e-6,
        tanh_mean: bool = False,
        fixed_scale: bool = False,
        use_independent: bool = True,
    ) -> None:
        super().__init__()
        self._init_scale = float(init_scale)
        self._min_scale = float(min_scale)
        self._fixed_scale = fixed_scale
        self._use_independent = use_independent

        self._mean_layer = nn.Linear(in_features=input_size, out_features=num_dimensions)
        self._initialize_linear(self._mean_layer)
        if not fixed_scale:
            self._scale_layer = nn.Linear(in_features=input_size, out_features=num_dimensions)
            self._initialize_linear(self._scale_layer)
    
    @staticmethod
    def _initialize_linear(linear: nn.Linear) -> None:
        variance_scaling_init_(linear.weight,scale=1e-4,mode="fan_in",distribution="truncated_normal")
        nn.init.zeros_(linear.bias)
    def forward(self, inputs: torch.Tensor) -> dist.Distribution:
        mean = self._mean_layer(inputs)
        if self._fixed_scale:
            scale = torch.full_like(mean, self._init_scale)
        else:
            raw_scale = self._scale_layer(inputs)
            softplus_zero = F.softplus(torch.zeros((), dtype=raw_scale.dtype, device=raw_scale.device))
            scale = (
                F.softplus(raw_scale)
                * self._init_scale
                / softplus_zero
                + self._min_scale
            )
        if self._use_independent:
            return dist.Independent(dist.Normal(loc=mean, scale=scale),reinterpreted_batch_ndims=1)

        return dist.MultivariateNormal(loc=mean,scale_tril=torch.diag_embed(scale))

class DiscreteValuedDistribution:
    # categorical distribution. values = fixed return values / atoms ranging from vmin to vmax
    # logits = raw neural net output for each atom
    def __init__(self, values: torch.Tensor, logits: torch.Tensor):
        self.values = values
        self.logits = logits

# softmax convert logits into probabilities
    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    # mean of the distrib = the expected Q value. Q(s, a)= E[Z(s,a)]
    def mean(self) -> torch.Tensor:
        return (self.probs * self.values).sum(dim=-1, keepdim=True)

# critic is only allowed to output probabilities on fixed grid of return vals
# Z = [logit(return =vmin), logit(return = vmax)], bins = num atoms 
def l2_project(Zp: torch.Tensor, P: torch.Tensor, Zq: torch.Tensor) -> torch.Tensor:
    """Project distribution (Zp, P) onto support Zq under the L2 metric over CDFs.

    Zp: (B, Kp) source support
    P:  (B, Kp) source probabilities
    Zq: (Kq,) target support
    returns: (B, Kq)
    """
    #Extracts vmin and vmax and construct helper tensors from Zq
    vmin, vmax = Zq[0], Zq[-1]
    d_pos = torch.cat([Zq, vmin[None]], dim=0)[1:]
    d_neg = torch.cat([vmax[None], Zq], dim=0)[:-1]

    # Clips Zp to be in new support range (vmin, vmax).
    clipped_zp = torch.clamp(Zp, vmin, vmax)[:, None, :]
    clipped_zq = Zq[None, :, None]

    # Gets the distance between atom values in support
    d_pos = (d_pos - Zq)[None, :, None]
    d_neg = (Zq - d_neg)[None, :, None]

    delta_qp = clipped_zp - clipped_zq
    d_sign = (delta_qp >= 0.0).to(P.dtype)

    delta_hat = (
        d_sign * delta_qp / d_pos
        - (1.0 - d_sign) * delta_qp / d_neg
    )
    P = P[:, None, :]
    # returns the L2 projection of (Zp, P) onto Zq. --> note it's Zq!! not Zp!!

    return torch.sum(torch.clamp(1.0 - delta_hat, 0.0, 1.0) * P, dim=2)


# computes categorical distributional loss
# q_t.values = the target atoms (the return bins, i.e. -150, -145, .. +150)
# target for current = Z_target_t = r_t + gamma * Z_target(S_{t+1}, a'), where a' is from target policy
# p_t is the probabilities for the logit values, which here, each logit val correspond to a target atom from q_t
def categorical(
    q_tm1: DiscreteValuedDistribution,
    r_t: torch.Tensor,
    d_t: torch.Tensor,
    q_t: DiscreteValuedDistribution,
) -> torch.Tensor:
    """Categorical distributional loss. Returns per-sample loss (B,)."""
    values = q_t.values

    # r_t.reshape change reward shape from (B,) to (B,1)
    # since values have shape (k,), where k = num atoms
    # this makes every atom get the reward and d_t bellman shift.
    z_t = r_t.reshape(-1, 1) + d_t.reshape(-1, 1) * values  # shape (B,K)
    # in order to do l2_project, the atoms need probability mass p_t attached
    # to each atom, not logits. 
    p_t = torch.softmax(q_t.logits, dim=-1)

    # need to project shifted distrib back onto atom values
    target = l2_project(z_t, p_t, values).detach()

    # pytorch equivalent of tf.nn.softmax_cross_entropy_with_logits
    # loss = (-labels * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
    log_p_tm1 = torch.log_softmax(q_tm1.logits, dim=-1)
    
    # target = the projected (Zp, P) onto Zq - Z_w'(St, At)
    # q_tm1 logits = online critic's prediction.
    # cross entropy formula
    return -(target * log_p_tm1).sum(dim=-1)

def td_learning(v_tm1, r_t, pcont_t, v_t):
    target = (r_t + pcont_t * v_t).detach()
    td_error = target - v_tm1
    loss = 0.5 * td_error.square()

    return loss, target, td_error


class DiscreteValuedHead(nn.Module):
    ''' maps hidden features to logits over a fixed support of return atoms,
    then returns a discretevalueddistribution '''
    def __init__(
        self,
        input_size: int,
        vmin: float,
        vmax: float,
        num_atoms: int,
        w_init: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        b_init: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> None:
        super().__init__()

        if vmax <= vmin:
            raise ValueError(f"vmax must be greater than vmin, got vmin={vmin}, vmax={vmax}")

        self.num_atoms = num_atoms
        self.register_buffer("atom_values",torch.linspace(vmin,vmax,num_atoms,dtype=torch.float32))
        
        self._distributional_layer = nn.Linear(in_features=input_size,out_features=num_atoms)
        # edit the weights and biases to match snt.Linear's default: bias = 0. 
        # weight = truncated random normal values with stddev = 1 / sqrt(input-feature-size)
        # truncated at: no more than 2 stddevs from the mean.
        with torch.no_grad():
            if w_init is None:
                std = 1.0 / math.sqrt(input_size)
                nn.init.trunc_normal_(
                    self._distributional_layer.weight,
                    mean=0.0,
                    std=std,
                    a=-2.0 * std,
                    b=2.0 * std,
                )
            else:
                w_init(self._distributional_layer.weight)

            if self._distributional_layer.bias is not None:
                if b_init is None:
                    nn.init.zeros_(self._distributional_layer.bias)
                else:
                    b_init(self._distributional_layer.bias)

    def forward(self,inputs: torch.Tensor) -> DiscreteValuedDistribution:
        # Input:(B, input_size)
        # Output logits:(B, num_atoms)
        logits = self._distributional_layer(inputs)

        # Cast support to match logits, equivalent to:
        # tf.cast(self._values, logits.dtype)
        atom_values = self.atom_values.to(dtype=logits.dtype,device=logits.device)
        return DiscreteValuedDistribution(values=atom_values,logits=logits)
  
class AcmeActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        layer_sizes: Sequence[int] = (256, 256, 256),
        init_scale: float = 0.3,   
        min_scale: float = 1e-6,    # originally 1e-6
    ) -> None:
        super().__init__()
        if len(layer_sizes) == 0:
            raise ValueError("layer_sizes must contain at least one size")
        self.register_buffer("action_low",torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("action_high",torch.as_tensor(act_high, dtype=torch.float32))
        
        self.torso = AcmeLayerNormMLP(
            input_size=obs_dim,
            layer_sizes=layer_sizes,
            activate_final=True,
        )
        self.policy_head = MultivariateNormalDiagHead(
            input_size=layer_sizes[-1],
            num_dimensions=act_dim,
            init_scale=init_scale,
            min_scale=min_scale,
            fixed_scale=False,
            use_independent=True,
        )
    def forward(self, observations: torch.Tensor) -> dist.Distribution:
        observations = observations.reshape(observations.shape[0],-1)
        features = self.torso(observations)
        return self.policy_head(features)

class ScalarAcmeCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        layer_sizes: Sequence[int] = (512, 512, 256),
    ) -> None:
        super().__init__()

        self.register_buffer("action_low",torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high",torch.as_tensor(action_high, dtype=torch.float32))
        
        self.torso = AcmeLayerNormMLP(
            input_size=obs_dim + act_dim,
            layer_sizes=layer_sizes,
            activate_final=True)

        self.value_head = NearZeroInitializedLinear(
            input_size=layer_sizes[-1],
            output_size=1,
            scale=1e-4)

    def forward(self,observation: torch.Tensor,action: torch.Tensor) -> torch.Tensor:
        # Clip action
        observation = observation.reshape(observation.shape[0],-1)
        action = action.reshape(action.shape[0],-1)

        action = torch.clamp(action,self.action_low,self.action_high)
        action = action.to(dtype=observation.dtype,device=observation.device)
        inputs = torch.cat([observation, action],dim=-1)

        # LayerNormMLP and value head
        torso_output = self.torso(inputs)  # (B, 256)
        value = self.value_head(torso_output)
        return value

class AcmeCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        layer_sizes: Sequence[int] = (512, 512, 256),
        vmin: float = -500.0,
        vmax: float = 20.0,
        num_atoms: int = 101,
    ) -> None:
        super().__init__()

        self.register_buffer("action_low",torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high",torch.as_tensor(action_high, dtype=torch.float32))
        
        self.torso = AcmeLayerNormMLP(
            input_size=obs_dim + act_dim,
            layer_sizes=layer_sizes,
            activate_final=True)

        self.distributional_head = DiscreteValuedHead(
            input_size=layer_sizes[-1],
            vmin=vmin,
            vmax=vmax,
            num_atoms=num_atoms)
    
    @property
    def atom_values(self) -> torch.Tensor:
        return self.distributional_head.atom_values
    def forward(self,observation: torch.Tensor,action: torch.Tensor) -> DiscreteValuedDistribution:
        # Clip action
        observation = observation.reshape(observation.shape[0],-1)
        action = action.reshape(action.shape[0],-1)

        action = torch.clamp(action,self.action_low,self.action_high)
        action = action.to(dtype=observation.dtype,device=observation.device)
        inputs = torch.cat([observation, action],dim=-1)

        # LayerNormMLP and distribution head
        torso_output = self.torso(inputs)  # (B, 256)
        distribution = self.distributional_head(torso_output)
        return distribution
