"""Stream all recorded cameras into a named, frame-addressable RGB archive."""

from pathlib import Path

import av
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm


def save_episode_videos(raw_path, report, output_path, image_size=96):
    """Use the state converter's exact episode order and prefix cuts.

    Camera names come from videos/*.mp4, including arbitrary additional wrist
    cameras. All retained episodes must expose the same camera set. Decode at
    most one video frame at a time; write small batches to bounded Zarr chunks.
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
    store = zarr.open_group(str(output_path), mode="w")
    total = sum(report["traj_lengths"])
    store.attrs.update(camera_names=cameras, image_size=[image_size, image_size],
                       layout="TCHW", color_space="RGB", complete=False)
    arrays = {name: store.create_dataset(
        name, shape=(total, 3, image_size, image_size),
        chunks=(16, 3, image_size, image_size), dtype="u1",
        compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
    ) for name in cameras}
    offset = 0
    for paths, details in tqdm(zip(videos, report["episode_details"]),
                               total=len(videos), desc="Convert camera videos"):
        length = details["original_length"]
        start = details["retained_start_index"]
        for camera, path in paths.items():
            count, written, batch = 0, 0, []
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    count += 1
                    if count > length:
                        raise ValueError(f"{path}: more video frames than {length} state samples")
                    if index < start:
                        continue
                    rgb = frame.reformat(width=image_size, height=image_size,
                                         format="rgb24", interpolation="AREA").to_ndarray()
                    batch.append(rgb.transpose(2, 0, 1))
                    if len(batch) == 16:
                        arrays[camera][offset + written:offset + written + len(batch)] = np.stack(batch)
                        written += len(batch)
                        batch.clear()
            if count != length:
                raise ValueError(f"{path}: {count} video frames != {length} state samples")
            if batch:
                arrays[camera][offset + written:offset + written + len(batch)] = np.stack(batch)
        offset += details["retained_length"]
    store.attrs["complete"] = True
    return cameras
