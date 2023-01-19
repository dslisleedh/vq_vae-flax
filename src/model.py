import jax
from jax import lax
import jax.numpy as jnp

import flax
from flax import linen as nn


class ResidualBlock(nn.Module):

    @nn.compact
    def __call__(self, x):
        _, _, _, c = x.shape
        y = nn.relu(x)
        y = nn.Conv(c, (3, 3), padding='SAME')(y)
        y = nn.relu(y)
        y = nn.Conv(c, (1, 1), padding='VALID')(y)
        return x + y


class Encoder(nn.Module):
    n_filters: int
    n_latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.n_filters // 2, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.Conv(self.n_filters, (4, 4), strides=(2, 2), padding='SAME')(x)

        x = ResidualBlock()(x)
        x = ResidualBlock()(x)
        x = nn.Dense(self.n_latents)(x)
        return x


class Decoder(nn.Module):
    n_filters: int
    n_output: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_filters)(x)
        x = ResidualBlock()(x)
        x = ResidualBlock()(x)

        x = nn.ConvTranspose(self.n_filters // 2, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.ConvTranspose(self.n_filters // 4, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.Conv(self.n_output, (3, 3), padding='SAME')(x)
        return x


class VectorQuantizer(nn.Module):
    n_embeddings: int
    n_latent: int
    beta: float = 0.25

    @nn.compact
    def __call__(self, x, training:bool = False):
        embedding = self.param('embedding', nn.initializers.uniform(), (self.n_embeddings, self.n_latent))

        b, h, w, c = x.shape
        x_flatten = jnp.reshape(x, (b * h * w, c))
        dist = jnp.sum(x_flatten ** 2, axis=1, keepdims=True) + jnp.sum(embedding ** 2, axis=1)\
               - 2 * jnp.dot(x_flatten, embedding.T)
        encoding_indices = jnp.argmin(dist, axis=1)
        encodings = jax.nn.one_hot(encoding_indices, self.n_embeddings)
        encodings = jnp.reshape(encodings, (b, h, w, self.n_embeddings))
        quantized = jnp.reshape(jnp.dot(encodings, embedding), (b, h, w, self.n_latent))
        if training:
            e_latent_loss = jnp.mean((lax.stop_gradient(quantized) - x) ** 2)
            q_latent_loss = jnp.mean((quantized - lax.stop_gradient(x)) ** 2)
            loss = q_latent_loss + self.beta * e_latent_loss
            return quantized, loss
        else:
            return quantized


class VQVAE(nn.Module):
    n_filters: int
    n_latents: int
    n_embeddings: int
    beta: float = 0.25

    @nn.compact
    def __call__(self, x, training: bool = False):
        _, _, _, c = x.shape
        x = Encoder(self.n_filters, self.n_latents)(x)
        _, h, w, _ = x.shape
        x = VectorQuantizer(self.n_embeddings, self.n_latents, self.beta)(x, training)
        if training:
            x, loss = x
        x = Decoder(self.n_filters, c)(x)
        if training:
            return x, loss
        else:
            return x


if __name__ == '__main__':
    vqvae = VQVAE(128, 32, 512)
    print(vqvae.tabulate(jax.random.PRNGKey(0), jnp.ones((16, 28, 28, 1))))
