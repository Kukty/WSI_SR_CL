import numpy as np


def low_entropy_filter(patch_size: int):
    def low_entropy(item) -> bool:
        patch_src, patch_dst = item
        p = np.mean(patch_src, axis=2).flatten()
        entropy = -np.sum(p * np.log2(p + 0.001)) / (patch_size * patch_size)
        if entropy > 0.2:
            return True
        return False

    return low_entropy
