# from https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/

import torch
import torch.nn as nn
from Utils.Mnist_Utils import patchify, get_positional_embeddings


class VIT_Block(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(VIT_Block, self).__init__()

        self.hidden_dim = hidden_d
        self.n_heads = n_heads

        """
        Layer normalization is applied to the last dimension only. We can thus make each of our 50x8 matrices 
        (representing a single sequence) have mean 0 and std 1. After we run our (N, 50, 8) tensor through LN, 
        we still get the same dimensionality.
        """
        self.norm1 = nn.LayerNorm(hidden_d)  # tokens size


class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7, hidden_dim=8):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches  # 28x28 / 7x7 means 49 patches if size 4x4 (16 floats)

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1]/n_patches, chw[2]/n_patches)  # 4x4

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])    # 16
        self.hidden_dim = hidden_dim

        """
        Notice that we run an (N, 49, 16) tensor through a (16, 8) linear mapper (or matrix).
         The linear operation only happens on the last dimension.
        """
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_dim)

        # 2) Learnable classifiation token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        """
        In particular, positional encoding adds high-frequency values to the first dimensions and low-frequency values 
        to the latter dimensions. In each sequence, for token i we add to its j-th coordinate the following value:
        """
        # 3) Positional embedding
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_dim)))  # 50X8
        self.pos_embed.requires_grad = False

    def forward(self, images):
        n = images.shape[0]
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        # add the classification token to the tokens (shares data about all the other tokens...)
        # he comes in first! (keep in mined)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        pos_embedding_per_sample = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embedding_per_sample


        return out


if __name__ == '__main__':
    # Current model
    model = MyViT(
        chw=(1, 28, 28),
        n_patches=7
    )

    x = torch.randn(7, 1, 28, 28)  # Dummy images
    print(model(x).shape)  # torch.Size([7, 49, 16])
