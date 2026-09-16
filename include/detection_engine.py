# include/detection_engine.py
import math
import cv2
import numpy as np

from .bbox_utils import BBoxUtils


class DetectionEngine:
    """
    Conservative SIFT-only detector with strong validation.

    Why SIFT-only?
        SIFT is robust to rotation, scale, brightness, and JPEG noise.
        Template/Edge matching produce too many false positives on
        pages with repeating patterns (numbers, table cells, etc.).

    Strong validation:
        1. Minimum inliers (matches verified by homography).
        2. Aspect ratio must be close to reference aspect ratio.
        3. Detected area must be in a reasonable range.
        4. Max 5 detections per page.
        5. Minimum SIFT keypoints on reference (avoid degenerate refs).
    """

    # ----- Tunable thresholds -----
    SIFT_RATIO = 0.70
    MIN_INLIERS = 12
    MIN_MATCHES = 15
    RANSAC_THRESHOLD = 3.0

    # Aspect-ratio tolerance: detected/reference must be within ±35%
    ASPECT_RATIO_TOLERANCE = 0.35

    # Size limits relative to page area
    MIN_AREA_RATIO = 0.0005
    MAX_AREA_RATIO = 0.50

    # Per-page limits
    MAX_DETECTIONS_PER_PAGE = 5

    # Minimum keypoints required on reference image
    MIN_REF_KEYPOINTS = 20

    def __init__(self):
        self.report = {}

    # --------------------------------------------------------
    # Public: main entry
    # --------------------------------------------------------
    def detect_on_page(self, page_rgb, ref_gray):
        """
        Returns a list of detections for one page.
        Each detection:
            {
                "bbox": (x1, y1, x2, y2),
                "methods": ["SIFT"],
                "score": inliers / matches,
                "verification": aspect_ratio_match (0..1),
                "inliers": int,
                "matches": int,
            }
        """
        page_gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
        page_h, page_w = page_gray.shape[:2]

        detections = self._sift_detect(page_gray, ref_gray, page_w, page_h)

        # Apply hard limits
        detections = detections[: self.MAX_DETECTIONS_PER_PAGE]

        return detections

    # --------------------------------------------------------
    # SIFT detection
    # --------------------------------------------------------
    def _sift_detect(self, page_gray, ref_gray, page_w, page_h):
        results = []

        try:
            # --- Reference keypoints ---
            sift = cv2.SIFT_create(nfeatures=4000)
            kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

            if des_ref is None or len(kp_ref) < self.MIN_REF_KEYPOINTS:
                print(
                    f"      Reference too small: "
                    f"{0 if kp_ref is None else len(kp_ref)} keypoints "
                    f"(need >= {self.MIN_REF_KEYPOINTS})"
                )
                return results

            print(f"      Reference keypoints: {len(kp_ref)}")

            # --- Page keypoints ---
            kp_page, des_page = sift.detectAndCompute(page_gray, None)

            if des_page is None or len(kp_page) < 4:
                print("      Page has no keypoints.")
                return results

            print(f"      Page keypoints: {len(kp_page)}")

            # --- Match ---
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            raw = matcher.knnMatch(des_ref, des_page, k=2)

            good = []
            for pair in raw:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.SIFT_RATIO * n.distance:
                    good.append(m)

            if len(good) < self.MIN_MATCHES:
                print(
                    f"      Not enough good matches: "
                    f"{len(good)} < {self.MIN_MATCHES}"
                )
                return results

            print(f"      Good matches: {len(good)}")

            # --- Aspect ratio of reference ---
            ref_h, ref_w = ref_gray.shape[:2]
            ref_aspect = ref_w / float(ref_h) if ref_h > 0 else 1.0

            # --- Iterative homography to find multiple instances ---
            active = good[:]
            iteration = 0

            while iteration < self.MAX_DETECTIONS_PER_PAGE:
                iteration += 1

                if len(active) < self.MIN_MATCHES:
                    break

                src = np.float32([
                    kp_ref[m.queryIdx].pt for m in active
                ]).reshape(-1, 1, 2)

                dst = np.float32([
                    kp_page[m.trainIdx].pt for m in active
                ]).reshape(-1, 1, 2)

                try:
                    H, mask = cv2.findHomography(
                        src, dst,
                        cv2.RANSAC,
                        self.RANSAC_THRESHOLD,
                    )
                except Exception:
                    break

                if H is None or mask is None:
                    break

                inlier_mask = mask.ravel().astype(bool)
                inliers = int(inlier_mask.sum())

                if inliers < self.MIN_INLIERS:
                    print(
                        f"      Iteration {iteration}: "
                        f"inliers {inliers} < {self.MIN_INLIERS}, stop"
                    )
                    break

                # --- Compute quad corners in page coords ---
                corners = np.float32([
                    [0, 0],
                    [ref_w - 1, 0],
                    [ref_w - 1, ref_h - 1],
                    [0, ref_h - 1],
                ]).reshape(-1, 1, 2)

                transformed = cv2.perspectiveTransform(corners, H)
                pts = transformed.reshape(-1, 2)

                if not np.isfinite(pts).all():
                    active = [
                        m for i, m in enumerate(active)
                        if not inlier_mask[i]
                    ]
                    continue

                bbox = BBoxUtils.from_points(pts, page_w, page_h)
                if bbox is None:
                    active = [
                        m for i, m in enumerate(active)
                        if not inlier_mask[i]
                    ]
                    continue

                x1, y1, x2, y2 = bbox
                bw = x2 - x1
                bh = y2 - y1

                if bw < 15 or bh < 15:
                    active = [
                        m for i, m in enumerate(active)
                        if not inlier_mask[i]
                    ]
                    continue

                # --- Area check ---
                area = bw * bh
                page_area = page_w * page_h
                area_ratio = area / page_area

                if area_ratio < self.MIN_AREA_RATIO:
                    print(
                        f"      Reject: area too small "
                        f"({area_ratio:.5f})"
                    )
                    active = [
                        m for i, m in enumerate(active)
                        if not inlier_mask[i]
                    ]
                    continue

                if area_ratio > self.MAX_AREA_RATIO:
                    print(
                        f"      Reject: area too big "
                        f"({area_ratio:.3f})"
                    )
                    active = [
                        m for i, m in enumerate(active)
                        if not inlier_mask[i]
                    ]
                    continue

                # --- Aspect ratio check ---
                det_aspect = bw / float(bh)
                aspect_diff = abs(det_aspect - ref_aspect) / ref_aspect

                if aspect_diff > self.ASPECT_RATIO_TOLERANCE:
                    print(
                        f"      Reject: aspect mismatch "
                        f"(det={det_aspect:.2f} ref={ref_aspect:.2f} "
                        f"diff={aspect_diff:.0%})"
                    )
                    active = [
                        m for i, m in enumerate(active)
                        if not inlier_mask[i]
                    ]
                    continue

                # --- Verify with template matching ---
                verify_score = self._verify(page_gray, ref_gray, bbox)

                # Accept if verification >= 0.25 OR inliers very high
                if verify_score < 0.25 and inliers < 25:
                    print(
                        f"      Reject: weak verification "
                        f"({verify_score:.2f}), inliers={inliers}"
                    )
                    active = [
                        m for i, m in enumerate(active)
                        if not inlier_mask[i]
                    ]
                    continue

                score = inliers / float(max(1, len(active)))

                print(
                    f"      ACCEPT: inliers={inliers} "
                    f"matches={len(active)} "
                    f"bbox={bbox} "
                    f"verify={verify_score:.2f}"
                )

                results.append({
                    "bbox": bbox,
                    "methods": ["SIFT"],
                    "score": float(score),
                    "verification": float(verify_score),
                    "inliers": inliers,
                    "matches": len(active),
                })

                # Remove inliers of this instance
                active = [
                    m for i, m in enumerate(active)
                    if not inlier_mask[i]
                ]

        except Exception as exc:
            print(f"  [SIFT] error: {exc}")

        return results

    # --------------------------------------------------------
    # Template verification (only used to CONFIRM, never to find)
    # --------------------------------------------------------
    @staticmethod
    def _verify(page_gray, ref_gray, bbox):
        try:
            x1, y1, x2, y2 = bbox
            crop = page_gray[y1:y2, x1:x2]
            if crop.size == 0:
                return 0.0

            target = cv2.resize(
                ref_gray,
                (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

            # Normalize both to reduce lighting differences
            crop_n = cv2.equalizeHist(crop)
            target_n = cv2.equalizeHist(target)

            result = cv2.matchTemplate(
                crop_n, target_n, cv2.TM_CCOEFF_NORMED
            )
            if result.size == 0:
                return 0.0
            score = float(result[0, 0])
            return score if math.isfinite(score) else 0.0
        except Exception:
            return 0.0