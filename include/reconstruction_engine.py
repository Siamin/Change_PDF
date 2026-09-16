# include/reconstruction_engine.py
import cv2
import numpy as np

from .constants import (
    INPAINT_RADIUS, INPAINT_MIN_EDGE_SIMILARITY,
    SEAMLESS_CLONE_ENABLED,
    REMOVE_PADDING_PX,
    MODE_INPAINT, MODE_REDACT, MODE_AUTO,
)
from .bbox_utils import BBoxUtils


class ReconstructionEngine:

    def __init__(self, mode: str = MODE_AUTO):
        self.mode = mode

    @staticmethod
    def sample_border_color(crop_bgr, mask):
        try:
            border = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            )
            border = cv2.subtract(border, mask)
            pixels = crop_bgr[border > 0]
            if pixels.size == 0:
                mean = crop_bgr.reshape(-1, 3).mean(axis=0)
            else:
                mean = pixels.reshape(-1, 3).mean(axis=0)
            return tuple(int(np.clip(v, 0, 255)) for v in mean)
        except Exception:
            return (255, 255, 255)

    @staticmethod
    def edge_similarity(original, reconstructed, mask):
        try:
            if original.size == 0 or reconstructed.size == 0:
                return 0.0
            g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
            border = cv2.dilate(
                mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )
            border = cv2.subtract(border, mask)
            if cv2.countNonZero(border) == 0:
                return 1.0
            e1 = cv2.Canny(g1, 50, 150)
            e2 = cv2.Canny(g2, 50, 150)
            a = cv2.bitwise_and(e1, border)
            b = cv2.bitwise_and(e2, border)
            denom = max(1, cv2.countNonZero(a) + cv2.countNonZero(b))
            return float(
                2.0 * cv2.countNonZero(cv2.bitwise_and(a, b)) / denom
            )
        except Exception:
            return 0.0

    def _inpaint(self, crop_bgr, mask):
        telea = cv2.inpaint(crop_bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        telea_q = self.edge_similarity(crop_bgr, telea, mask)

        chosen = telea
        method = "TELEA"

        if telea_q < INPAINT_MIN_EDGE_SIMILARITY:
            ns = cv2.inpaint(crop_bgr, mask, INPAINT_RADIUS, cv2.INPAINT_NS)
            ns_q = self.edge_similarity(crop_bgr, ns, mask)
            if ns_q > telea_q:
                chosen = ns
                method = "NS"

        border_color = self.sample_border_color(crop_bgr, mask)
        target = mask > 0

        if np.count_nonzero(target) > 0:
            chosen_mean = chosen[target].mean(axis=0)
            border_arr = np.array(border_color, dtype=np.float32)
            if np.linalg.norm(chosen_mean.astype(np.float32) - border_arr) > 45.0:
                soft = cv2.GaussianBlur(mask, (0, 0), 2.0).astype(
                    np.float32
                ) / 255.0
                alpha = soft[..., None] * 0.20
                color_img = np.empty_like(chosen)
                color_img[:] = np.array(border_color, dtype=np.uint8)
                chosen = (
                    chosen.astype(np.float32) * (1.0 - alpha)
                    + color_img.astype(np.float32) * alpha
                ).clip(0, 255).astype(np.uint8)

        if SEAMLESS_CLONE_ENABLED:
            try:
                ys, xs = np.where(mask > 0)
                if len(xs) >= 10:
                    center = (
                        int(round(float(xs.mean()))),
                        int(round(float(ys.mean()))),
                    )
                    center = (
                        max(0, min(crop_bgr.shape[1] - 1, center[0])),
                        max(0, min(crop_bgr.shape[0] - 1, center[1])),
                    )
                    cloned = cv2.seamlessClone(
                        chosen, crop_bgr, mask.copy(),
                        center, cv2.NORMAL_CLONE,
                    )
                    if cloned is not None:
                        if (self.edge_similarity(crop_bgr, cloned, mask)
                                > self.edge_similarity(crop_bgr, chosen, mask)):
                            chosen = cloned
                            method += "+CLONE"
            except Exception:
                pass

        return chosen, method

    def _redact(self, crop_bgr, mask):
        border_color = self.sample_border_color(crop_bgr, mask)
        filled = crop_bgr.copy()
        filled[mask > 0] = border_color
        return filled, "REDACT"

    def build_patch(self, page_rgb, bbox):
        try:
            h, w = page_rgb.shape[:2]
            patch_bbox = BBoxUtils.expand(bbox, REMOVE_PADDING_PX, w, h)
            mx1, my1, mx2, my2 = patch_bbox

            crop_rgb = page_rgb[my1:my2, mx1:mx2].copy()
            if crop_rgb.size == 0:
                return None

            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            ch, cw = crop_bgr.shape[:2]

            mask = np.zeros((ch, cw), dtype=np.uint8)
            x1, y1, x2, y2 = bbox

            lx1 = max(0, x1 - mx1)
            ly1 = max(0, y1 - my1)
            lx2 = min(cw, x2 - mx1)
            ly2 = min(ch, y2 - my1)
            if lx2 <= lx1 or ly2 <= ly1:
                return None

            cv2.rectangle(mask, (lx1, ly1), (lx2 - 1, ly2 - 1), 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=1)

            if self.mode == MODE_REDACT:
                chosen, method = self._redact(crop_bgr, mask)
            elif self.mode == MODE_INPAINT:
                chosen, method = self._inpaint(crop_bgr, mask)
            else:  # MODE_AUTO
                inpainted, method = self._inpaint(crop_bgr, mask)
                quality = self.edge_similarity(crop_bgr, inpainted, mask)
                if quality < INPAINT_MIN_EDGE_SIMILARITY:
                    redacted, rmethod = self._redact(crop_bgr, mask)
                    rquality = self.edge_similarity(crop_bgr, redacted, mask)
                    if rquality >= quality:
                        chosen = redacted
                        method = rmethod
                    else:
                        chosen = inpainted
                else:
                    chosen = inpainted

            result_rgb = cv2.cvtColor(chosen, cv2.COLOR_BGR2RGB)
            return result_rgb, patch_bbox, method

        except Exception as exc:
            print(f"    Reconstruction error: {exc}")
            return None