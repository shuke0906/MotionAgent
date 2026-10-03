"""Deterministic, scenario-aware frame evidence without action annotations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_agent.verification.artifacts import file_digest
from motion_agent.verification.backends import BackendUnavailable


class StoryboardBuilder:
    def __init__(self, root: str, max_frames: int = 240, contact_sheet: bool = False):
        self.root, self.max_frames = Path(root), max_frames
        self.contact_sheet = contact_sheet

    def _sheets(self, records, folder):
        import cv2
        folder = folder / "sheets_v2_512"
        folder.mkdir(parents=True, exist_ok=True)
        images = []
        for start in range(0, len(records), 4):
            tiles = records[start:start + 4]
            decoded = [cv2.imread(tile["path"]) for tile in tiles]
            if any(frame is None for frame in decoded):
                raise ValueError("contact sheet source frame cannot be decoded")
            height, width = decoded[0].shape[:2]
            sheet = np.full((2 * (height + 24), 2 * width, 3), 255, dtype=np.uint8)
            for index, (frame, tile) in enumerate(zip(decoded, tiles)):
                row, col = divmod(index, 2)
                top, left = row * (height + 24), col * width
                sheet[top + 24:top + 24 + height, left:left + width] = frame
                cv2.putText(sheet, f"{tile['frame_index']} | {tile['timestamp_s']:.3f}s", (left + 4, top + 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            scale = min(1, 512 / max(sheet.shape[:2]))
            if scale < 1:
                sheet = cv2.resize(sheet, (round(sheet.shape[1] * scale), round(sheet.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
            path = folder / f"sheet_{start // 4:04d}.png"
            if not cv2.imwrite(str(path), sheet):
                raise OSError("contact sheet write failed")
            images.append({"path": str(path.resolve()), "sha256": file_digest(path),
                           "tiles": [{"frame_index": t["frame_index"], "timestamp_s": t["timestamp_s"]} for t in tiles],
                           "layout": "2x2_row_major_512", "blank_tiles": 4 - len(tiles),
                           "width": sheet.shape[1], "height": sheet.shape[0]})
        return images

    def build(self, request, check, *, dense=False):
        import cv2
        if not request.visual_evidence_uri or request.visual_evidence_candidate_id != request.candidate_id:
            raise BackendUnavailable("candidate-bound rendered video evidence is missing")
        source = Path(request.visual_evidence_uri)
        if not source.is_file():
            raise BackendUnavailable("candidate render is not available locally")
        digest = file_digest(source)
        if request.visual_evidence_fingerprint and request.visual_evidence_fingerprint != digest:
            raise ValueError("render fingerprint mismatch")
        video = cv2.VideoCapture(str(source))
        try:
            fps = video.get(cv2.CAP_PROP_FPS)
            frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if not video.isOpened() or fps <= 0 or frames < 2:
                raise ValueError("invalid candidate video")
            if abs(frames / fps - request.motion_spec.duration_s) > max(0.1, 1 / fps):
                raise ValueError("candidate render duration does not match specification")
            event_check = check.direction.startswith("event_")
            sampling_hz = 8.0 if event_check else 2.0
            count = check.metadata.get("expected_count", 0)
            if check.direction == "event_frequency":
                # At least eight samples per expected cycle, over the full clip.
                sampling_hz = max(sampling_hz, count * 8 / request.motion_spec.duration_s)
            sampling_hz *= 2 if dense else 1
            sampling_hz = min(fps, sampling_hz)
            indices = np.unique(np.append(np.rint(np.arange(0, (frames - 1) / fps, 1 / sampling_hz) * fps), frames - 1)).astype(int)
            if len(indices) > self.max_frames:
                raise BackendUnavailable(f"storyboard needs {len(indices)} frames, exceeding configured evidence budget {self.max_frames}")
            folder = self.root / request.candidate_id / digest[:16] / f"{check.direction}_{'dense' if dense else 'base'}"
            folder.mkdir(parents=True, exist_ok=True)
            records = []
            for index in indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, frame = video.read()
                if not success:
                    raise ValueError(f"cannot decode video frame {index}")
                height, width = frame.shape[:2]
                scale = min(1, 512 / max(height, width))
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                path = folder / f"frame_{index:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    raise OSError("storyboard frame write failed")
                records.append({"frame_index": int(index), "timestamp_s": float(index / fps),
                                "relative_time": float(index / max(1, frames - 1)), "path": str(path.resolve()),
                                "sha256": file_digest(path)})
            result = {"candidate_id": request.candidate_id, "source_sha256": digest, "fps": fps,
                    "source_frames": frames, "sampling_hz": sampling_hz, "dense": dense,
                    "profile": request.storyboard_profile, "frames": records}
            if self.contact_sheet:
                result["images"] = self._sheets(records, folder)
                result["image_layout"] = "2x2_timestamped_original_frames"
                result["limitations"] = "four frames per image reduce spatial detail; return uncertain for unresolved anatomy/cycles"
            return result
        finally:
            video.release()
