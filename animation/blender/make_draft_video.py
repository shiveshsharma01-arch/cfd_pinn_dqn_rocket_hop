"""Assemble Blender's sampled preview frames into a review MP4."""

from pathlib import Path

import cv2


folder = Path(__file__).resolve().parent
frames = sorted((folder / "draft_frames").glob("frame_*.png"))
if len(frames) < 250:
    raise RuntimeError(f"Expected the full flight; found only {len(frames)} frames")
first = cv2.imread(str(frames[0]))
if first is None:
    raise RuntimeError("Could not read first preview frame")
height, width = first.shape[:2]
output = folder / "rocket_hop_v04_draft.mp4"
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                         10.0, (width, height))
if not writer.isOpened():
    raise RuntimeError("Could not open MP4 video writer")
for path in frames:
    image = cv2.imread(str(path))
    if image is None or image.shape[:2] != (height, width):
        raise RuntimeError(f"Invalid preview frame: {path}")
    writer.write(image)
writer.release()
reader = cv2.VideoCapture(str(output))
count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
fps = reader.get(cv2.CAP_PROP_FPS)
reader.release()
if abs(count - len(frames)) > 1 or abs(fps - 10.0) > 0.1:
    raise RuntimeError("MP4 frame count or frame rate did not survive encoding")
print(f"VIDEO={output}")
print(f"FRAMES={count}, FPS={fps:.1f}, DURATION_S={count / fps:.2f}")
