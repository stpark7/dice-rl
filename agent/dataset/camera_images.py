"""Read only selected cameras and frames from a converted image archive."""

import numpy as np
import torch
import zarr


class CameraImages:
    """Tensor-like frame slicing without loading the complete archive onto GPU."""

    def __init__(self, path, image_keys, length, device):
        self.store = zarr.open_group(str(path), mode="r")
        if not self.store.attrs.get("complete", False):
            raise ValueError(f"Incomplete camera archive: {path}")
        available = self.store.attrs["camera_names"]
        self.image_keys = list(available if image_keys is None else image_keys)
        if (not self.image_keys or len(set(self.image_keys)) != len(self.image_keys)
                or any(name not in available for name in self.image_keys)):
            raise ValueError(f"Invalid camera selection {self.image_keys}; available cameras: {available}")
        self.arrays = [self.store[key] for key in self.image_keys]
        first_shape = self.arrays[0].shape
        for array in self.arrays:
            if (array.shape != first_shape or len(array.shape) != 4 or array.shape[1] != 3
                    or array.shape[0] < length or array.dtype != np.uint8):
                raise ValueError(f"Inconsistent camera array {array.name}: {array.shape}, {array.dtype}")
        self.shape = (int(length), 3 * len(self.arrays), *first_shape[2:])
        self.dtype = torch.uint8
        self.device = device

    def __getitem__(self, index):
        frames = [array[index] for array in self.arrays]
        channel_axis = 0 if isinstance(index, (int, np.integer)) else 1
        return torch.from_numpy(np.concatenate(frames, axis=channel_axis)).to(self.device)
