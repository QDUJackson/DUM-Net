import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SoftplusSoftErf(nn.Module):
    """
    SoftplusTanhApprox activation function defined as:

        f(x) = softplus(x) * tanh( sqrt(2/π) * ( (x/eps) + 0.044517*(x/eps)^3 ) )

    where softplus(x) = ln(1+exp(x)) and eps is a learnable parameter.
    To guarantee eps remains positive, we learn log_eps.
    """

    def __init__(self, init_eps=1.0):
        super(SoftplusSoftErf, self).__init__()
        # Learn the logarithm of eps to ensure positivity
        self.log_eps = nn.Parameter(torch.log(torch.tensor(init_eps, dtype=torch.float32)))

    def forward(self, x):
        # Compute eps as exp(log_eps) to ensure eps > 0
        eps = torch.exp(self.log_eps)
        # Compute x1 = x/eps
        x1 = x / eps
        # Use F.softplus(x) for numerical stability: ln(1+exp(x))
        sp = F.softplus(x)
        # Compute the constant factor sqrt(2/π)
        factor = torch.sqrt(torch.tensor(2.0 / 3.141592653589793, dtype=torch.float32, device=x.device))
        # Compute the tanh term: tanh( sqrt(2/π) * (x1 + 0.044517 * x1^3) )
        tanh_term = torch.tanh(factor * (x1 + 0.044517 * x1 ** 3))
        return sp * tanh_term


# ======= Testing and visualization =======
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate test data over a range
    x = torch.linspace(-5, 5, 400)
    activation = SoftplusSoftErf(init_eps=1.0)
    y = activation(x)

    # Plot the activation function curve
    plt.figure(figsize=(8, 5))
    plt.plot(x.detach().numpy(), y.detach().numpy(),
             label=f"SoftplusTanhApprox (eps={torch.exp(activation.log_eps).item():.3f})",
             color='purple')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Learnable SoftplusTanhApprox Activation Function")
    plt.legend()
    plt.grid(True)
    plt.show()
