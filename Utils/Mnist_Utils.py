import os
import struct
import numpy as np
import torch


def save_model_ckpt(dir, step, model, accuracy):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    ckpt_name = 'iter_{:d}.pth'.format(step)
    ckpt_path = os.path.join(dir, ckpt_name)
    states = {
        'global_step': step,
        'model_state': model.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(states, ckpt_path)


def read_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)


def read_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def convert_to_pt(raw_dir, processed_dir):
    train_x = read_images(os.path.join(raw_dir, 'train-images.idx3-ubyte'))
    train_y = read_labels(os.path.join(raw_dir, 'train-labels.idx1-ubyte'))
    test_x = read_images(os.path.join(raw_dir, 't10k-images.idx3-ubyte'))
    test_y = read_labels(os.path.join(raw_dir, 't10k-labels.idx1-ubyte'))

    os.makedirs(processed_dir, exist_ok=True)
    torch.save((torch.tensor(train_x), torch.tensor(train_y)), os.path.join(processed_dir, 'training.pt'))
    torch.save((torch.tensor(test_x), torch.tensor(test_y)), os.path.join(processed_dir, 'test.pt'))

def patchify(imgs, n_patches):
    # imgs: [B, C, H, W]  (H=W=28 for MNIST)
    B, C, H, W = imgs.shape
    ph, pw = H // n_patches, W // n_patches          # 4×4 when n_patches=7
    patches = imgs.unfold(2, ph, ph).unfold(3, pw, pw)  # [B,C,7,7,ph,pw]
    patches = patches.flatten(2, 3)                     # [B,C,49,ph,pw]
    patches = patches.permute(0, 2, 1, 3, 4)            # [B,49,C,ph,pw]
    return patches.reshape(B, n_patches**2, -1)                   # [B,49,16]


# def patchify(images, n_patches):
#     n, c, h, w = images.shape
#     assert h == w, "Patchify method is implemented for square images only"
#     patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
#     patch_size = h // n_patches
#
#     for idx, image in enumerate(images):
#         for i in range(n_patches):
#             for j in range(n_patches):
#                 patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
#                 patches[idx, i * n_patches + j] = patch.flatten()
#     return patches.to('cuda')


# def get_positional_embeddings(sequence_length, d):
#     """
#     From the heatmap we have plotted, we see that all ‘horizontal lines’ are all different from each other,
#      and thus samples can be distinguished.
#     :param sequence_length: number of tokens
#     :param d: dimension of token
#     :return: tensor to be added to tokens tensor
#     """
#     result = torch.ones(sequence_length, d)
#     for i in range(sequence_length):
#         for j in range(d):
#             result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
#     return result
#
import math


def get_positional_embeddings(sequence_length, d_model):
    position = torch.arange(sequence_length).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model/2]

    pe = torch.zeros(sequence_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

    return pe  # [seq_len, d_model]

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  plt.imshow(get_positional_embeddings(49, 16), cmap="hot", interpolation="nearest")
  plt.show()