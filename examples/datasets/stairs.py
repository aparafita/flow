"""
Create stairs dataset.

Creates a matrix of n x d^2 dimensions which, reshaped to (n, d, d),
contains n grayscale images of stairs. 
Stairs and background also contain some noise in each pixel,
and stair pixels are darker than background pixels. 

Along with the dataset, a sample image 
with m^2 samples arranged in a square grid is generated.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


default_folder = 'stairs'
labels = True # comes along with labels


# Defaults:
_n = 1000 # number of samples
_d = 5 # images are d x d pixels
_stairs_noise = .05 # noise applied to stairs pixels
_background_noise = .01 # noise applied to background pixels

    
def _generate_masks(d):
    """Generate a different mask for every type of stair."""

    # right-down to left-up stairs: type 1
    masks = []
    
    a = np.arange(d)
    unsq = lambda x, dim: np.expand_dims(x, dim)

    for k in range(d - 1):
        mask = unsq(a, 1) - unsq(a, 0) >= k
        masks.append(mask)

    # left-down to right-up stairs: type 0
    for m in list(masks): # we're adding to the mask, so we need a new iterator
        masks.append(m[:, ::-1])

    masks = np.stack(masks, 0).reshape(len(masks), d ** 2)

    # labels: type (0 or 1) + one-hot k
    labels = np.array([
        [t] + [ int(i == k) for i in range(d - 1) ]
        for t in range(2)
        for k in range(d - 1)
    ])

    return masks, labels


def generate_samples(
    n=_n, d=_d, 
    stairs_noise=_stairs_noise, background_noise=_background_noise,
    return_labels=False
):
    """Generate n samples of dxd stair images."""
    masks, labels = _generate_masks(d)

    d2 = d ** 2

    stair = np.random.uniform(size=(n, 1)) * .5 + .5
    stair = stair + np.random.normal(scale=stairs_noise, size=(n, d2))
    stair = stair.clip(0.5, 1.)

    bg = np.random.uniform(size=(n, 1)) * .1
    bg = bg + np.random.normal(scale=background_noise, size=(n, d2))
    bg = bg.clip(0., .1)

    idx = np.random.choice(range(len(masks)), size=(n,))
    
    mask = masks.reshape(len(masks), d2)[idx]
    samples = mask * stair + (1 - mask) * bg

    labels = labels[idx]

    if return_labels:
        return samples, labels
    else:
        return samples


def plot_samples(samples, nr=None, nc=None, figure_width=10):
    """Plot samples in a squared grid and return the resulting figure."""
    n, d2 = samples.shape

    if nr is None and nc is None:
        nc = int(np.sqrt(n))
        nr = (n - 1) // nc + 1
    elif nr is None:
        nr = (n - 1) // nc + 1
    elif nc is None:
        nc = (n - 1) // nr + 1
    else:
        assert nr * nc >= n

    d = int(np.sqrt(d2))
    assert d ** 2 == d2, 'Needs to be a square image'

    fig, axes_mat = plt.subplots(nr, nc, figsize=(figure_width, figure_width))
    axes = axes_mat.flatten()

    for ax, x in zip(axes, samples):
        ax.imshow(x.reshape(d, d), cmap='gray_r', vmin=0, vmax=1)

    for ax in axes:
        ax.axis('off') # axis off for ALL cells, even the ones without an image

    return axes_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n', type=int, default=_n, help='#samples')
    parser.add_argument('--d', type=int, default=_d, help='#width pixels')
    parser.add_argument(
        '--m', type=int, default=10, 
        help='m^2 is the number of images in the sample image'
    )

    parser.add_argument(
        '--output_folder', type=str, default=default_folder,
        help='Output folder where to store the result'
    )

    parser.add_argument('--seed', type=int, default=123, help='Random seed')

    parser.add_argument(
        '--stairs-noise', dest='stairs_noise', type=float, 
        default=_stairs_noise,
        help='Scale for the normal noise added to stair pixels'
    )

    parser.add_argument(
        '--background-noise', dest='background_noise', type=float, 
        default=_background_noise,
        help='Scale for the normal noise added to background pixels'
    )

    args = parser.parse_args()

    # Create folder if it doesn't exist
    try:
        os.mkdir(args.output_folder)
    except:
        pass # already exists

    # Generate dataset
    np.random.seed(args.seed)

    dataset, labels = generate_samples(
        args.n, args.d, 
        args.stairs_noise, args.background_noise,
        return_labels=True
    )

    np.save(os.path.join(args.output_folder, 'data.npy'), dataset)
    np.save(os.path.join(args.output_folder, 'labels.npy'), labels)

    # Generate samples image
    samples = generate_samples(
        args.m ** 2, args.d, 
        args.stairs_noise, args.background_noise,
    )

    axes = plot_samples(samples)
    plt.savefig(os.path.join(args.output_folder, 'sample.png'))