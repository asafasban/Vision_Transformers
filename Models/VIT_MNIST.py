# from https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/
# https://medium.com/@anushka.sonawane/query-key-value-and-multi-head-attention-transformers-part-2-ba8d3db0db75 #BEST SO FAT
import torch
import torch.nn as nn
from Utils.Mnist_Utils import patchify, get_positional_embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        self.d = d # tokens dimension
        self.n_heads = n_heads # number of attention heads (all in parallel does not affect each other)
        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)   # each head is working on another part of the token (embedded)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1) # all dims...

    def forward(self, x):
        # vectorized
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)  # [batch, seq_len, 3 * d]
        qkv = qkv.reshape(batch_size, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.permute(2, 0, 1, 3)  # [n_heads, batch, seq_len, 3 * d_head]

        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each: [n_heads, batch, seq_len, d_head]

        # Transpose k for matmul
        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [n_heads, batch, seq, seq]
        attn_weights = self.softmax(attn_scores)
        attn_output = attn_weights @ v  # [n_heads, batch, seq_len, d_head]

        # Concatenate heads
        attn_output = attn_output.permute(1, 2, 0, 3).reshape(batch_size, seq_len, self.d)

        return self.out(attn_output)

    # def forward(self, sequences):
    #     """
    #     :param sequences:   Sequences has shape (N, seq_length, token_dim) ->
    #                         then we split to (N, seq_length, n_heads, token_dim / n_heads)
    #     :return: # And come back to    (N, seq_length, item_dim)  (through concatenation)
    #     """
    #     result = []
    #     for seq in sequences:
    #         seq_result = []
    #         for head in range(self.n_heads):
    #             q_mapping = self.q_mappings[head]
    #             k_mapping = self.k_mappings[head]
    #             v_mapping = self.v_mappings[head]
    #
    #             seq_part = seq[:, head*self.d_head : (head + 1)*self.d_head]
    #             q, k, v = q_mapping(seq_part), k_mapping(seq_part), v_mapping(seq_part)
    #
    #             attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
    #             seq_result.append(attention @ v)
    #
    #         result.append(torch.hstack(seq_result))
    #     return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class VIT_Block(nn.Module):

    """
    Multi-head Self Attention
    We now need to implement sub-figure c of the architecture picture. What’s happening there?
    Simply put: we want, for a single image, each patch to get updated based on some similarity measure with the other
    patches. We do so by linearly mapping each patch (that is now an 8-dimensional vector in our example)
    to 3 distinct vectors: q, k, and v (query, key, value).
    """
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
        self.mhsa = MultiHeadSelfAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28),  n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches  # 28x28 / 7x7 means 49 patches if size 4x4 (16 floats)

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1]/n_patches, chw[2]/n_patches)  # 4x4

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])    # 16
        self.hidden_dim = hidden_d

        """
        Notice that we run an (N, 49, 16) tensor through a (16, 8) linear mapper (or matrix).
         The linear operation only happens on the last dimension.
        """
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_dim)

        # 2) Learnable classifiation token
        """
        this is a special token that we add to our model that has the role of capturing information about the other
        tokens. This will happen with the MSA block (later on).
        """
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        """
        In particular, positional encoding adds high-frequency values to the first dimensions and low-frequency values 
        to the latter dimensions. In each sequence, for token i we add to its j-th coordinate the following value:
        """
        # 3) Positional embedding
        self.pos_embed = nn.Parameter(
            get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_dim),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([VIT_Block(self.hidden_dim, n_heads) for _ in range(n_blocks)])

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        n = images.shape[0]
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        # add the classification token to the tokens (shares data about all the other tokens...)
        # he comes in first! (keep in mined)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        pos_embedding_per_sample = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embedding_per_sample

        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)


if __name__ == '__main__':
    model = VIT_Block(hidden_d=8, n_heads=2)

    x = torch.randn(7, 50, 8)  # Dummy sequences
    print(model(x).shape)  # torch.Size([7, 50, 8])

    #
    # # Current model
    # model = MyViT(
    #     chw=(1, 28, 28),
    #     n_patches=7
    # )
    #
    # x = torch.randn(7, 1, 28, 28)  # Dummy images
    # print(model(x).shape)  # torch.Size([7, 49, 16])
