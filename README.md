# GAI_loss

Generative models have various loss functions depending on their architecture and objectives. Here are some common generative models and their associated loss functions:

### 1. **Variational Autoencoders (VAEs)**

**Loss Function**: Evidence Lower Bound (ELBO)

The VAE loss consists of two main components: the reconstruction loss and the KL divergence.

- **Reconstruction Loss**: Measures how well the decoder reconstructs the input data from the latent representation.
- **KL Divergence**: Measures how much the learned latent distribution diverges from the prior distribution (usually a standard normal distribution).

<img width="503" alt="Screenshot 2024-07-26 at 9 58 03 AM" src="https://github.com/user-attachments/assets/cd0265ca-9dc4-4b09-a7eb-59de45e2a05c">


### 2. **Generative Adversarial Networks (GANs)**

**Loss Function**: Adversarial Loss

GANs consist of two networks: a generator \( G \) and a discriminator \( D \). The objective is for \( G \) to generate realistic data that \( D \) cannot distinguish from real data.

<img width="495" alt="Screenshot 2024-07-26 at 9 58 18 AM" src="https://github.com/user-attachments/assets/9d692ef8-cd6e-41d2-a4b1-2e67c4fde72b">

### 3. **Autoregressive Models (e.g., RNNs, Transformers)**

**Loss Function**: Cross-Entropy Loss

These models predict the next token in a sequence given the previous tokens. The loss function is the cross-entropy between the predicted and actual next tokens.

<img width="328" alt="Screenshot 2024-07-26 at 9 58 30 AM" src="https://github.com/user-attachments/assets/1db3fce4-03d9-40bd-9024-061086033226">


### 4. **Flow-based Models**

**Loss Function**: Negative Log-Likelihood (NLL)

Flow-based models explicitly model the data distribution and can compute exact log-likelihoods.

<img width="261" alt="Screenshot 2024-07-26 at 9 58 38 AM" src="https://github.com/user-attachments/assets/53f73194-fab5-4c46-9cd6-f07d61ec7d11">


### Example Implementation: VAE Loss in Python

Here’s an example implementation of the VAE loss function in Python:

```python
import torch
import torch.nn.functional as F

def vae_loss_function(reconstructed_x, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')

    # KL Divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Example usage
# Assume we have a batch of data x, reconstructed_x from the decoder,
# and mu and logvar from the encoder
x = torch.randn(64, 784)  # Example input batch (batch_size=64, input_dim=784)
reconstructed_x = torch.randn(64, 784)  # Example reconstructed batch
mu = torch.randn(64, 20)  # Example mu from encoder (latent_dim=20)
logvar = torch.randn(64, 20)  # Example logvar from encoder

loss = vae_loss_function(reconstructed_x, x, mu, logvar)
print("VAE Loss:", loss.item())
```

### Summary

Different generative models use different loss functions tailored to their specific architectures and objectives:

1. **VAEs**: Use the ELBO, combining reconstruction loss and KL divergence.
2. **GANs**: Use adversarial loss for both the generator and discriminator.
3. **Autoregressive Models**: Use cross-entropy loss for sequence prediction.
4. **Flow-based Models**: Use negative log-likelihood.

Each loss function plays a crucial role in guiding the training process to achieve the desired generative behavior.
