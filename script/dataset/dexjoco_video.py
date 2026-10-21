"""Decode every recorded camera into one RGB array per camera."""

from pathlib import Path

import av
import numpy as np
from tqdm import tqdm


def load_episode_videos(raw_path, report, image_size=96):
    """Use the state converter's exact episode order and prefix cuts.

    Camera names come from videos/*.mp4, including arbitrary additional wrist
    cameras. All retained episodes must expose the same camera set. Returns
    ``{camera: uint8 (T, 3, image_size, image_size)}`` over all retained frames,
    decoding at most one video frame at a time.
    """
    if image_size < 1:
        raise ValueError("image_size must be positive")
    directories = [Path(raw_path) / name for name in report["episodes"]]
    videos = [dict((p.stem, p) for p in sorted((ep.parent / "videos").glob("*.mp4")))
              for ep in directories]
    cameras = sorted(videos[0])
    if not cameras:
        raise ValueError(f"No camera MP4 files under {directories[0].parent / 'videos'}")
    for episode, paths in zip(directories, videos):
        if sorted(paths) != cameras:
            raise ValueError(f"Inconsistent camera set for {episode}: {sorted(paths)} != {cameras}")
    total = sum(report["traj_lengths"])
    frames = {name: np.empty((total, 3, image_size, image_size), dtype=np.uint8)
              for name in cameras}
    offset = 0
    for paths, details in tqdm(zip(videos, report["episode_details"]),
                               total=len(videos), desc="Convert camera videos"):
        length = details["original_length"]
        start = details["retained_start_index"]
        retained = details["retained_length"]
        for camera, path in paths.items():
            count = 0
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    count += 1
                    if count > length or index - start >= retained:
                        raise ValueError(f"{path}: more video frames than {length} state samples")
                    if index < start:
                        continue
                    rgb = frame.reformat(width=image_size, height=image_size,
                                         format="rgb24", interpolation="AREA").to_ndarray()
                    frames[camera][offset + index - start] = rgb.transpose(2, 0, 1)
            if count != length:
                raise ValueError(f"{path}: {count} video frames != {length} state samples")
        offset += retained
    return frames
