# include/detection_engine.py
import math
import cv2
import numpy as np

from .constants import (
    SIFT_RATIO, SIFT_MIN_MATCHES, SIFT_MIN_INLIERS,
    ORB_RATIO, ORB_MIN_MATCHES, ORB_MIN_INLIERS,
    TEMPLATE_MIN_SCORE, TEMPLATE_STRONG_SCORE,
    EDGE_MIN_SCORE, EDGE_STRONG_SCORE,
    MAX_DETECTIONS_PER_METHOD, MAX_FINAL_DETECTIONS,
    IOU_MERGE_THRESHOLD, OVERLAP_SMALLER_THRESHOLD,
    FINAL_DUPLICATE_IOU, MIN_DETECTION_SIZE,
    TEMPLATE_SCALES,
    MIN_QUAD_AREA_RATIO, MAX_QUAD_AREA_RATIO,
    MIN_QUAD_EDGE, MAX_ASPECT_RATIO,
)
from .bbox_utils import BBoxUtils


class DetectionEngine:

    # ---------- SIFT / ORB ----------
    def _feature_loop(self, page_gray, ref_gray, method):
        detections = []
        try:
            page_h, page_w = page_gray.shape[:2]
            ref_h, ref_w = ref_gray.shape[:2]
            if ref_h < 10 or ref_w < 10:
                return detections

            if method == "SIFT":
                detector = cv2.SIFT_create(nfeatures=5000)
                ratio, min_matches, min_inliers = (
                    SIFT_RATIO, SIFT_MIN_MATCHES, SIFT_MIN_INLIERS
                )
                norm, ransac_t = cv2.NORM_L2, 5.0
            else:
                detector = cv2.ORB_create(
                    nfeatures=8000, scaleFactor=1.2, nlevels=8
                )
                ratio, min_matches, min_inliers = (
                    ORB_RATIO, ORB_MIN_MATCHES, ORB_MIN_INLIERS
                )
                norm, ransac_t = cv2.NORM_HAMMING, 6.0

            kp_ref, des_ref = detector.detectAndCompute(ref_gray, None)
            kp_page, des_page = detector.detectAndCompute(page_gray, None)
            if des_ref is None or des_page is None:
                return detections
            if len(kp_ref) < 4 or len(kp_page) < 4:
                return detections

            matcher = cv2.BFMatcher(norm)
            raw = matcher.knnMatch(des_ref, des_page, k=2)

            good = []
            for pair in raw:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)

            if len(good) < min_matches:
                return detections

            active = good[:]
            max_iter = min(
                MAX_DETECTIONS_PER_METHOD,
                max(1, len(active) // max(min_inliers, 1)),
            )

            for _ in range(max_iter):
                if len(active) < min_matches:
                    break

                src = np.float32([
                    kp_ref[m.queryIdx].pt for m in active
                ]).reshape(-1, 1, 2)
                dst = np.float32([
                    kp_page[m.trainIdx].pt for m in active
                ]).reshape(-1, 1, 2)

                try:
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_t)
                except Exception:
                    break
                if H is None or mask is None:
                    break

                inlier_mask = mask.ravel().astype(bool)
                inliers = int(inlier_mask.sum())
                if inliers < min_inliers:
                    break

                bbox = self._bbox_from_homography(
                    H, ref_gray.shape, page_w, page_h
                )
                if bbox is None:
                    active = [
                        m for i, m in enumerate(active)
                        if not inlier_mask[i]
                    ]
                    continue

                score = inliers / float(max(1, len(active)))
                detections.append({
                    "bbox": bbox,
                    "method": method,
                    "score": float(score),
                    "matches": len(active),
                    "inliers": inliers,
                    "homography": H,
                })

                active = [
                    m for i, m in enumerate(active)
                    if not inlier_mask[i]
                ]
                if len(detections) >= MAX_DETECTIONS_PER_METHOD:
                    break

        except Exception as exc:
            print(f"  [{method}] error: {exc}")
        return detections[:MAX_DETECTIONS_PER_METHOD]

    def _validate_quad(self, transformed, page_w, page_h):
        try:
            pts = transformed.reshape(-1, 2).astype(np.float32)
            if len(pts) != 4 or not np.isfinite(pts).all():
                return False

            x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
            y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

            if x_max - x_min < MIN_QUAD_EDGE:
                return False
            if y_max - y_min < MIN_QUAD_EDGE:
                return False

            area = abs(float(cv2.contourArea(pts.reshape(-1, 1, 2))))
            page_area = float(page_w * page_h)
            if page_area <= 0:
                return False
            ratio = area / page_area
            if ratio < MIN_QUAD_AREA_RATIO or ratio > MAX_QUAD_AREA_RATIO:
                return False

            rw = max(x_max - x_min, 1.0)
            rh = max(y_max - y_min, 1.0)
            return max(rw / rh, rh / rw) <= MAX_ASPECT_RATIO
        except Exception:
            return False

    def _bbox_from_homography(self, H, ref_shape, page_w, page_h):
        try:
            rh, rw = ref_shape[:2]
            corners = np.float32([
                [0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1],
            ]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, H)
            if not self._validate_quad(transformed, page_w, page_h):
                return None
            return BBoxUtils.from_points(
                transformed.reshape(-1, 2), page_w, page_h
            )
        except Exception:
            return None

    # ---------- Template / Edge ----------
    def _resize(self, image, scale):
        try:
            h, w = image.shape[:2]
            tw, th = int(round(w * scale)), int(round(h * scale))
            if tw < 8 or th < 8:
                return None
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            return cv2.resize(image, (tw, th), interpolation=interp)
        except Exception:
            return None

    def _local_maxima(self, result, threshold, sw, sh):
        locs = []
        try:
            work = result.copy()
            if work.size == 0:
                return locs
            sw = max(8, sw)
            sh = max(8, sh)
            for _ in range(MAX_DETECTIONS_PER_METHOD):
                _, mv, _, ml = cv2.minMaxLoc(work)
                if mv < threshold:
                    break
                x, y = ml
                locs.append((x, y, float(mv)))
                x1, y1 = max(0, x - sw), max(0, y - sh)
                x2 = min(work.shape[1], x + sw + 1)
                y2 = min(work.shape[0], y + sh + 1)
                work[y1:y2, x1:x2] = -1.0
        except Exception:
            pass
        return locs

    def _template_match(self, page_gray, ref_gray, is_edge=False):
        detections = []
        page_h, page_w = page_gray.shape[:2]
        threshold = EDGE_MIN_SCORE if is_edge else TEMPLATE_MIN_SCORE
        method_name = "EDGE" if is_edge else "TEMPLATE"

        for scale in TEMPLATE_SCALES:
            template = self._resize(ref_gray, float(scale))
            if template is None:
                continue
            th, tw = template.shape[:2]
            if tw >= page_w or th >= page_h:
                continue

            try:
                result = cv2.matchTemplate(
                    page_gray, template, cv2.TM_CCOEFF_NORMED
                )
                locs = self._local_maxima(
                    result, threshold, tw // 2, th // 2
                )
                for x, y, score in locs:
                    bbox = BBoxUtils.clip(
                        (x, y, x + tw, y + th), page_w, page_h
                    )
                    if bbox is None:
                        continue
                    detections.append({
                        "bbox": bbox,
                        "method": method_name,
                        "score": float(score),
                        "scale": float(scale),
                        "matches": 0,
                        "inliers": 0,
                    })
            except Exception:
                continue

        detections.sort(key=lambda d: d["score"], reverse=True)
        final = []
        for d in detections:
            if any(
                BBoxUtils.iou(d["bbox"], e["bbox"]) >= FINAL_DUPLICATE_IOU
                for e in final
            ):
                continue
            final.append(d)
            if len(final) >= MAX_DETECTIONS_PER_METHOD:
                break
        return final

    # ---------- Verification ----------
    def _verify(self, page_gray, ref_gray, bbox):
        try:
            x1, y1, x2, y2 = bbox
            crop = page_gray[y1:y2, x1:x2]
            if crop.size == 0:
                return 0.0
            target = cv2.resize(
                ref_gray, (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            result = cv2.matchTemplate(crop, target, cv2.TM_CCOEFF_NORMED)
            if result.size == 0:
                return 0.0
            score = float(result[0, 0])
            return score if math.isfinite(score) else 0.0
        except Exception:
            return 0.0

    def _accept(self, group):
        methods = {d.get("method") for d in group}

        for d in group:
            if (d.get("method") == "SIFT"
                    and d.get("inliers", 0) >= SIFT_MIN_INLIERS
                    and d.get("score", 0.0) >= 0.30):
                return True, "SIFT"
        for d in group:
            if (d.get("method") == "ORB"
                    and d.get("inliers", 0) >= ORB_MIN_INLIERS
                    and d.get("score", 0.0) >= 0.30):
                return True, "ORB"
        if len(methods) >= 2:
            return True, "MULTI"
        if any(d.get("method") == "TEMPLATE"
               and d.get("score", 0.0) >= TEMPLATE_STRONG_SCORE
               for d in group):
            return True, "TEMPLATE"
        if any(d.get("method") == "EDGE"
               and d.get("score", 0.0) >= EDGE_STRONG_SCORE
               for d in group):
            return True, "EDGE"
        return False, ""

    def _merge(self, detections, page_gray, ref_gray):
        if not detections:
            return []

        priority = {"SIFT": 3, "ORB": 2, "TEMPLATE": 1, "EDGE": 1}
        detections = sorted(
            detections,
            key=lambda d: (
                d.get("score", 0.0),
                priority.get(d.get("method"), 0),
                d.get("inliers", 0),
            ),
            reverse=True,
        )

        groups = []
        for det in detections:
            bbox = det["bbox"]
            found = None
            for g in groups:
                if any(
                    BBoxUtils.iou(bbox, e["bbox"]) >= IOU_MERGE_THRESHOLD
                    or BBoxUtils.overlap_smaller(bbox, e["bbox"])
                    >= OVERLAP_SMALLER_THRESHOLD
                    for e in g
                ):
                    found = g
                    break
            if found is None:
                groups.append([det])
            else:
                found.append(det)

        final = []
        for g in groups:
            ok, reason = self._accept(g)
            if not ok:
                continue

            x1 = min(d["bbox"][0] for d in g)
            y1 = min(d["bbox"][1] for d in g)
            x2 = max(d["bbox"][2] for d in g)
            y2 = max(d["bbox"][3] for d in g)
            bbox = BBoxUtils.clip(
                (x1, y1, x2, y2),
                page_gray.shape[1], page_gray.shape[0],
            )
            if bbox is None:
                continue
            if (bbox[2] - bbox[0] < MIN_DETECTION_SIZE
                    or bbox[3] - bbox[1] < MIN_DETECTION_SIZE):
                continue

            verification = self._verify(page_gray, ref_gray, bbox)
            has_feature = any(
                (d.get("method") == "SIFT"
                 and d.get("inliers", 0) >= SIFT_MIN_INLIERS
                 and d.get("score", 0.0) >= 0.30)
                or (d.get("method") == "ORB"
                    and d.get("inliers", 0) >= ORB_MIN_INLIERS
                    and d.get("score", 0.0) >= 0.30)
                for d in g
            )
            has_direct = any(
                (d.get("method") == "TEMPLATE"
                 and d.get("score", 0.0) >= TEMPLATE_STRONG_SCORE)
                or (d.get("method") == "EDGE"
                    and d.get("score", 0.0) >= EDGE_STRONG_SCORE)
                for d in g
            )
            if (verification < 0.20 and not has_feature and not has_direct):
                continue

            best = max(g, key=lambda d: (d.get("inliers", 0), d.get("score", 0.0)))
            methods = sorted({d.get("method", "?") for d in g})

            final.append({
                "bbox": bbox,
                "methods": methods,
                "score": float(best.get("score", 0.0)),
                "verification": float(verification),
                "inliers": int(best.get("inliers", 0)),
                "acceptance": reason,
                "group": g,
            })
            if len(final) >= MAX_FINAL_DETECTIONS:
                break

        final.sort(
            key=lambda d: (
                d.get("inliers", 0),
                d.get("score", 0.0),
                d.get("verification", 0.0),
            ),
            reverse=True,
        )

        dedup = []
        for d in final:
            if any(
                BBoxUtils.iou(d["bbox"], e["bbox"]) >= FINAL_DUPLICATE_IOU
                for e in dedup
            ):
                continue
            dedup.append(d)
            if len(dedup) >= MAX_FINAL_DETECTIONS:
                break
        return dedup

    # ---------- Public ----------
    def detect_on_page(self, page_rgb, ref_gray):
        page_gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
        page_blur = cv2.GaussianBlur(page_gray, (3, 3), 0)

        print("    SIFT...")
        sift = self._feature_loop(page_gray, ref_gray, "SIFT")
        print(f"      {len(sift)} candidate(s)")

        print("    ORB...")
        orb = self._feature_loop(page_gray, ref_gray, "ORB")
        print(f"      {len(orb)} candidate(s)")

        print("    Template...")
        tpl = self._template_match(page_blur, ref_gray, is_edge=False)
        print(f"      {len(tpl)} candidate(s)")

        print("    Edge...")
        ref_edges = cv2.Canny(ref_gray, 50, 150)
        page_edges = cv2.Canny(page_gray, 50, 150)
        edge = self._template_match(page_edges, ref_edges, is_edge=True)
        print(f"      {len(edge)} candidate(s)")

        all_det = sift + orb + tpl + edge
        return self._merge(all_det, page_gray, ref_gray)