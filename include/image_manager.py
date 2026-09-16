# include/image_manager.py
import os
import cv2
import numpy as np
import fitz

from .constants import RENDER_DPI, CREATE_DEBUG_IMAGES
from .input_utils import InputUtils
from .detection_engine import DetectionEngine


class PDFImageManager:
    """
    Bulletproof logo removal via FULL PAGE RASTERIZATION.

    Why this works:
        We render the entire page as an image (which we know the exact
        pixel coordinates of), paint over the logo with background color,
        and replace the whole page with the modified image.

        No PDF coordinate system. No rotation issues. No overlays.
        No PDF object deletion. Just pixels.

    Tradeoff:
        Text becomes an image. But the logo is guaranteed to disappear.
        At 300 DPI, printed quality is still excellent.
    """

    # High quality raster replacement
    REPLACE_DPI = 300

    # Safety: bbox must not exceed this fraction of the page
    MAX_BBOX_AREA_RATIO = 0.60

    def __init__(self, doc, removal_mode="auto"):
        self.doc = doc
        self.detector = DetectionEngine()
        self.report = {
            "pages_scanned": 0,
            "pages_with_matches": 0,
            "matches": 0,
            "removed": 0,
            "replaced": 0,
            "rejected_oversized": 0,
            "pages_rasterized": 0,
            "debug_images": 0,
        }

    # ============================================================
    # RENDER
    # ============================================================
    @staticmethod
    def _render_page(page, dpi):
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ============================================================
    # BACKGROUND COLOR
    # ============================================================
    @staticmethod
    def _sample_bg(page_rgb, bbox, pad=40):
        x1, y1, x2, y2 = bbox
        h, w = page_rgb.shape[:2]
        samples = []

        for rx1, ry1, rx2, ry2 in [
            (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), max(1, y1 - 5)),
            (max(0, x1 - pad), min(h - 1, y2 + 5), min(w, x2 + pad), min(h, y2 + pad)),
            (max(0, x1 - pad), max(0, y1 - pad), max(1, x1 - 5), min(h, y2 + pad)),
            (min(w - 1, x2 + 5), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)),
        ]:
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            region = page_rgb[ry1:ry2, rx1:rx2]
            if region.size == 0:
                continue
            samples.append(region.reshape(-1, 3))

        if not samples:
            return (255, 255, 255)

        all_px = np.concatenate(samples, axis=0)
        if all_px.size == 0:
            return (255, 255, 255)

        brightness = all_px.sum(axis=1)
        thr = np.percentile(brightness, 40)
        light = all_px[brightness >= thr]
        if light.size == 0:
            light = all_px

        median = np.median(light, axis=0)
        return tuple(int(np.clip(v, 0, 255)) for v in median)

    # ============================================================
    # DEBUG
    # ============================================================
    @staticmethod
    def _save_debug(page_rgb, detections, path):
        try:
            dbg = page_rgb.copy()
            for i, d in enumerate(detections, 1):
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(
                    dbg, f"{i}", (x1 + 10, max(40, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4,
                )
            cv2.imwrite(path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
            return True
        except Exception as e:
            print(f"  debug error: {e}")
            return False

    # ============================================================
    # DETECT (strict)
    # ============================================================
    def _detect(self, ref_gray, debug_dir):
        results = []

        for page_index in range(len(self.doc)):
            page = self.doc[page_index]
            self.report["pages_scanned"] += 1

            print()
            print("-" * 70)
            print(f"IMAGE SEARCH - PAGE {page_index + 1}")

            try:
                page_rgb = self._render_page(page, RENDER_DPI)
            except Exception as e:
                print(f"  render error: {e}")
                continue

            try:
                dets = self.detector.detect_on_page(page_rgb, ref_gray)
            except Exception as e:
                print(f"  detect error: {e}")
                continue

            if not dets:
                print("  no match")
                continue

            page_h, page_w = page_rgb.shape[:2]
            page_area = page_h * page_w

            accepted = []
            for d in dets:
                bbox = d["bbox"]
                x1, y1, x2, y2 = bbox
                bw = x2 - x1
                bh = y2 - y1
                area = bw * bh
                ratio = area / float(page_area)

                if ratio > self.MAX_BBOX_AREA_RATIO:
                    self.report["rejected_oversized"] += 1
                    print(
                        f"  REJECTED oversized bbox {bbox} "
                        f"({ratio:.0%} of page, max {self.MAX_BBOX_AREA_RATIO:.0%})"
                    )
                    continue

                if bw < 30 or bh < 30:
                    print(f"  REJECTED too small {bbox}")
                    continue

                accepted.append(d)
                print(
                    f"  ACCEPTED bbox={bbox} "
                    f"({ratio:.1%} of page, inliers={d.get('inliers', 0)})"
                )

            if not accepted:
                print("  all detections rejected")
                continue

            self.report["pages_with_matches"] += 1
            self.report["matches"] += len(accepted)

            if CREATE_DEBUG_IMAGES and debug_dir:
                path = os.path.join(
                    debug_dir, f"page_{page_index + 1:04d}_matches.png"
                )
                if self._save_debug(page_rgb, accepted, path):
                    self.report["debug_images"] += 1

            for d in accepted:
                results.append({
                    "page_index": page_index,
                    "bbox": d["bbox"],
                })

        return results

    # ============================================================
    # RASTER REPLACEMENT (guaranteed to work)
    # ============================================================
    def _rasterize_page_with_removal(self, page_index, detections_for_page):
        """
        Render page at high DPI, paint over logos, replace page with
        the modified image. 100% guaranteed to remove the logo.
        """
        page = self.doc[page_index]

        # 1) Render at high DPI
        try:
            page_rgb = self._render_page(page, self.REPLACE_DPI)
        except Exception as e:
            print(f"    render error: {e}")
            return False

        # 2) Compute scale factor from RENDER_DPI to REPLACE_DPI
        scale = self.REPLACE_DPI / float(RENDER_DPI)

        # 3) Paint over each detection
        modified = page_rgb.copy()

        for det in detections_for_page:
            x1, y1, x2, y2 = det["bbox"]

            # Scale bbox to new DPI
            sx1 = int(x1 * scale)
            sy1 = int(y1 * scale)
            sx2 = int(x2 * scale)
            sy2 = int(y2 * scale)

            # Expand slightly for safety
            sx1 = max(0, sx1 - 4)
            sy1 = max(0, sy1 - 4)
            sx2 = min(modified.shape[1], sx2 + 4)
            sy2 = min(modified.shape[0], sy2 + 4)

            # Sample background
            bg = self._sample_bg(
                page_rgb,
                (x1, y1, x2, y2),
            )

            # Paint
            modified[sy1:sy2, sx1:sx2] = bg
            print(
                f"    painted bbox=({sx1},{sy1},{sx2},{sy2}) "
                f"with bg={bg}"
            )

        # 4) Encode as JPEG
        bgr = cv2.cvtColor(modified, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok:
            print("    encode failed")
            return False

        # 5) Replace page content
        try:
            # Get the page rect in the current orientation
            page_rect = page.rect

            # Remove ALL existing content on the page
            page.add_redact_annot(page_rect, fill=(1, 1, 1))
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_REMOVE,
                graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )

            # Insert our modified image as the ONLY content
            page.insert_image(
                page_rect,
                stream=encoded.tobytes(),
                keep_proportion=False,
                overlay=True,
            )

            print(f"    page {page_index + 1} replaced with raster ✓")
            return True

        except Exception as e:
            print(f"    page replacement failed: {e}")
            return False

    # ============================================================
    # PROCESS
    # ============================================================
    def process(self, reference_path, replacement_path, debug_dir):
        # Load reference
        try:
            path = os.path.expanduser(
                reference_path.strip().strip('"').strip("'")
            )
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("cannot decode reference")
            ref_gray = cv2.GaussianBlur(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0
            )
            print(
                f"Reference loaded: {img.shape[1]}x{img.shape[0]}"
            )
        except Exception as e:
            print(f"Reference error: {e}")
            return self.report

        if CREATE_DEBUG_IMAGES and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        # ---------------- DETECTION ----------------
        print()
        print("=" * 70)
        print("DETECTION PASS")
        print("=" * 70)

        detections = self._detect(ref_gray, debug_dir)

        if not detections:
            print("\nNo valid detections. Nothing to remove.")
            return self.report

        # ---------------- SUMMARY ----------------
        print()
        print("=" * 70)
        print(f"DETECTED {len(detections)} REGION(S)")
        print("=" * 70)
        by_page = {}
        for d in detections:
            by_page.setdefault(d["page_index"], []).append(d)
        for pi, dets in sorted(by_page.items()):
            print(f"  Page {pi + 1}: {len(dets)} detection(s)")

        print()
        print("!! NOTE: Affected pages will be converted to a raster image.")
        print("!! Text will no longer be selectable on those pages.")
        print("!! Other pages remain untouched.")

        if not InputUtils.ask_yes_no(
            f"Proceed with removal on {len(by_page)} page(s)?"
        ):
            print("Aborted.")
            return self.report

        # ---------------- REMOVAL ----------------
        print()
        print("=" * 70)
        print("REMOVAL PASS (raster replacement)")
        print("=" * 70)

        for page_index, dets in sorted(by_page.items()):
            print()
            print(f"  Processing page {page_index + 1}...")

            ok = self._rasterize_page_with_removal(page_index, dets)

            if ok:
                self.report["removed"] += len(dets)
                self.report["pages_rasterized"] += 1

        print()
        print(
            f"Removed: {self.report['removed']}  "
            f"Pages rasterized: {self.report['pages_rasterized']}"
        )

        return self.report

    # ============================================================
    # INTERACTIVE
    # ============================================================
    def interactively_process(self, debug_dir):
        if not InputUtils.ask_yes_no("Delete/replace any image or logo?"):
            return self.report

        reference_path = InputUtils.ask_existing_file("Reference image path: ")

        return self.process(reference_path, None, debug_dir)