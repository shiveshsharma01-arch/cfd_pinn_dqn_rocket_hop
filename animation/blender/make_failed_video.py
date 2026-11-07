"""Encode the sampled early-DQN failure frames as a review MP4."""
from pathlib import Path
import cv2

folder = Path(__file__).resolve().parent
frames = sorted((folder / "failed_frames").glob("frame_*.png"))
if len(frames) < 275:
    raise RuntimeError(f"Expected the full failed flight; found {len(frames)} frames")
first = cv2.imread(str(frames[0]))
if first is None:
    raise RuntimeError("Could not read the first frame")
height, width = first.shape[:2]
output = folder / "rocket_hop_failed_step000000_draft.mp4"
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                         10.0, (width, height))
if not writer.isOpened():
    raise RuntimeError("Could not open MP4 writer")
for path in frames:
    frame = cv2.imread(str(path))
    if frame is None or frame.shape[:2] != (height, width):
        raise RuntimeError(f"Invalid frame: {path}")
    writer.write(frame)
writer.release()
reader = cv2.VideoCapture(str(output))
count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
fps = reader.get(cv2.CAP_PROP_FPS)
reader.release()
if abs(count - len(frames)) > 1 or abs(fps - 10) > 0.1:
    raise RuntimeError("Encoded video frame count or rate mismatch")
print(f"VIDEO={output}; FRAMES={count}; FPS={fps}; DURATION_S={count/fps:.2f}")
