# include/image_manager.py
import os
import cv2
import numpy as np
import fitz

from .constants import (
    RENDER_DPI, CREATE_DEBUG_IMAGES,
    FINAL_DUPLICATE_IOU,
    MODE_INPAINT, MODE_REDACT, MODE_AUTO,
)
from .bbox_utils import BBoxUtils
from .input_utils import InputUtils
from .detection_engine import DetectionEngine
from .reconstruction_engine import ReconstructionEngine


class PDFImageManager:
    """
    Reliable image/logo removal for PDFs using PDF Redaction.

    Key insight: PDF redaction removes ALL content (text, image
    pixels, vector graphics) inside the target region. This works
    regardless of whether the logo is an image object, a vector
    drawing, or part of a larger image.
    """

    def __init__(self, doc, removal_mode=MODE_AUTO):
        self.doc = doc
        self.removal_mode = removal_mode
        self.detector = DetectionEngine()
        self.reconstructor = ReconstructionEngine(mode=removal_mode)
        self.report = self._new_report()

    def _new_report(self):
        return {
            "pages_scanned": 0,
            "pages_with_matches": 0,
            "matches": 0,
            "removed": 0,
            "replaced": 0,
            "underlying_images_deleted": 0,
            "background_reconstructed": 0,
            "vector_regions_reconstructed": 0,
            "debug_images": 0,
            "redactions_applied": 0,
        }

    # ============================================================
    # REFERENCE
    # ============================================================
    @staticmethod
    def load_reference(path):
        try:
            path = os.path.expanduser(path.strip().strip('"').strip("'"))
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                raise RuntimeError("Reference image is empty.")
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Cannot decode reference image.")
            return img
        except Exception as exc:
            raise RuntimeError(f"Cannot load reference: {exc}") from exc

    @staticmethod
    def prepare_reference(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    # ============================================================
    # RENDER
    # ============================================================
    @staticmethod
    def _render_page(page, dpi=RENDER_DPI):
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ============================================================
    # DEBUG
    # ============================================================
    @staticmethod
    def save_debug(image_rgb, detections, output_path):
        try:
            debug = image_rgb.copy()
            for i, d in enumerate(detections, 1):
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = (
                    f"{i}: {','.join(d.get('methods', []))} "
                    f"s={d.get('score', 0.0):.2f} "
                    f"v={d.get('verification', 0.0):.2f}"
                )
                cv2.putText(
                    debug, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2, cv2.LINE_AA,
                )
            bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
            return bool(cv2.imwrite(output_path, bgr))
        except Exception as exc:
            print(f"  Debug save error: {exc}")
            return False

    # ============================================================
    # BACKGROUND COLOR SAMPLING
    # ============================================================
    @staticmethod
    def _sample_bg(page_rgb, bbox, pad=25):
        """
        Sample the dominant background color around the logo.
        Uses median to be robust against text near the logo.
        """
        x1, y1, x2, y2 = bbox
        h, w = page_rgb.shape[:2]

        samples = []

        regions = [
            # top strip
            (max(0, x1 - pad), max(0, y1 - pad),
             min(w, x2 + pad), max(1, y1 - 2)),
            # bottom strip
            (max(0, x1 - pad), min(h - 1, y2 + 2),
             min(w, x2 + pad), min(h, y2 + pad)),
            # left strip
            (max(0, x1 - pad), max(0, y1 - pad),
             max(1, x1 - 2), min(h, y2 + pad)),
            # right strip
            (min(w - 1, x2 + 2), max(0, y1 - pad),
             min(w, x2 + pad), min(h, y2 + pad)),
        ]

        for rx1, ry1, rx2, ry2 in regions:
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            region = page_rgb[ry1:ry2, rx1:rx2]
            if region.size == 0:
                continue
            flat = region.reshape(-1, 3)

            # Use the brightest ~40% of pixels (background is usually light)
            brightness = flat.sum(axis=1)
            threshold = np.percentile(brightness, 60)
            light_pixels = flat[brightness >= threshold]

            if light_pixels.size > 0:
                samples.append(np.median(light_pixels, axis=0))
            else:
                samples.append(np.median(flat, axis=0))

        if not samples:
            return (255, 255, 255)

        combined = np.median(np.array(samples), axis=0)
        return tuple(int(np.clip(v, 0, 255)) for v in combined)

    # ============================================================
    # PIXEL BBOX -> PDF RECT
    # ============================================================
    @staticmethod
    def _pixel_bbox_to_pdf(bbox, page_rgb, page):
        return BBoxUtils.pdf_rect_from_pixels(
            bbox,
            page_rgb.shape[1],
            page_rgb.shape[0],
            page.rect,
        )

    # ============================================================
    # DETECTION PASS
    # ============================================================
    def _detect_all(self, ref_gray, debug_dir):
        results = []

        for page_index in range(len(self.doc)):
            page = self.doc[page_index]
            self.report["pages_scanned"] += 1

            print()
            print("-" * 70)
            print(f"IMAGE SEARCH - PAGE {page_index + 1}")

            try:
                page_rgb = self._render_page(page)
            except Exception as exc:
                print(f"  Render error: {exc}")
                continue

            try:
                detections = self.detector.detect_on_page(page_rgb, ref_gray)
            except Exception as exc:
                print(f"  Detection error: {exc}")
                continue

            if not detections:
                print("  No confirmed match.")
                continue

            self.report["pages_with_matches"] += 1
            self.report["matches"] += len(detections)
            print(f"  CONFIRMED: {len(detections)} match(es)")

            if CREATE_DEBUG_IMAGES and debug_dir:
                path = os.path.join(
                    debug_dir,
                    f"page_{page_index + 1:04d}_matches.png",
                )
                if self.save_debug(page_rgb, detections, path):
                    self.report["debug_images"] += 1

            for det in detections:
                results.append({
                    "page_index": page_index,
                    "bbox": det["bbox"],
                    "methods": det.get("methods", []),
                    "score": det.get("score", 0.0),
                    "verification": det.get("verification", 0.0),
                    "page_rgb": page_rgb,
                })

        return results

    # ============================================================
    # REMOVAL USING PDF REDACTION
    # ============================================================
    def _remove_with_redaction(self, detections, replacement_bytes):
        """
        Strategy:
            1. Sample background color around each detection.
            2. Add a redaction annotation covering the bbox.
            3. Apply redactions per page with aggressive removal:
                 - images = PDF_REDACT_IMAGE_PIXELS (safe pixel removal)
                 - graphics = PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED
                 - text = PDF_REDACT_TEXT_REMOVE
            4. If PIXELS mode fails, fall back to IMAGE_REMOVE.
        """
        # Group by page
        page_redactions = {}

        for det in detections:
            page_index = det["page_index"]
            page = self.doc[page_index]
            page_rgb = det["page_rgb"]
            bbox = det["bbox"]

            page_rect = self._pixel_bbox_to_pdf(bbox, page_rgb, page)
            bg_rgb = self._sample_bg(page_rgb, bbox)
            fill = tuple(c / 255.0 for c in bg_rgb)

            page_redactions.setdefault(page_index, []).append({
                "rect": page_rect,
                "fill": fill,
                "bbox": bbox,
                "bg_rgb": bg_rgb,
            })

        # Apply redactions page by page
        for page_index, items in page_redactions.items():
            page = self.doc[page_index]

            print()
            print(
                f"  PAGE {page_index + 1}: "
                f"adding {len(items)} redaction(s)"
            )

            for item in items:
                page.add_redact_annot(item["rect"], fill=item["fill"])
                print(
                    f"    Redact bbox={item['bbox']} "
                    f"fill RGB={item['bg_rgb']}"
                )

            # Try PIXELS mode first (safer, keeps image objects intact
            # but removes overlapping pixels)
            applied = False
            try:
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_PIXELS,
                    graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                    text=fitz.PDF_REDACT_TEXT_REMOVE,
                )
                applied = True
                print("    -> applied (pixels mode)")
            except Exception as exc:
                print(f"    pixels mode failed: {exc}")

            # Fallback: remove entire image objects
            if not applied:
                try:
                    page.apply_redactions(
                        images=fitz.PDF_REDACT_IMAGE_REMOVE,
                        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                        text=fitz.PDF_REDACT_TEXT_REMOVE,
                    )
                    applied = True
                    print("    -> applied (remove mode)")
                except Exception as exc:
                    print(f"    remove mode failed: {exc}")

            if applied:
                self.report["redactions_applied"] += len(items)
                self.report["removed"] += len(items)
                self.report["underlying_images_deleted"] += len(items)
            else:
                print("    !! Redaction failed entirely for this page.")

        # Optional replacement
        if replacement_bytes:
            for det in detections:
                try:
                    page = self.doc[det["page_index"]]
                    page_rect = self._pixel_bbox_to_pdf(
                        det["bbox"], det["page_rgb"], page
                    )
                    page.insert_image(
                        page_rect,
                        stream=replacement_bytes,
                        keep_proportion=False,
                        overlay=True,
                    )
                    self.report["replaced"] += 1
                except Exception as exc:
                    print(f"    Replacement failed: {exc}")

    # ============================================================
    # PROCESS
    # ============================================================
    def process(self, reference_path, replacement_path, debug_dir):
        # Load reference
        try:
            ref_img = self.load_reference(reference_path)
            ref_gray = self.prepare_reference(ref_img)
        except Exception as exc:
            print(f"Reference error: {exc}")
            return self.report

        # Load replacement
        replacement_bytes = None
        if replacement_path:
            try:
                with open(replacement_path, "rb") as f:
                    replacement_bytes = f.read()
                if not replacement_bytes:
                    replacement_bytes = None
            except Exception as exc:
                print(f"Replacement load error: {exc}")
                replacement_bytes = None

        if CREATE_DEBUG_IMAGES and debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except Exception as exc:
                print(f"Debug dir error: {exc}")

        # ---------- DETECTION PASS ----------
        print()
        print("=" * 70)
        print("DETECTION PASS")
        print("=" * 70)

        detections = self._detect_all(ref_gray, debug_dir)

        if not detections:
            print()
            print("No detections found. Nothing to remove.")
            return self.report

        # ---------- SUMMARY ----------
        print()
        print("=" * 70)
        print(f"DETECTION SUMMARY: {len(detections)} region(s)")
        print("=" * 70)

        for i, det in enumerate(detections, 1):
            print(
                f"  #{i}: Page {det['page_index'] + 1}  "
                f"bbox={det['bbox']}  "
                f"methods={','.join(det['methods'])}  "
                f"score={det['score']:.3f}"
            )

        if not InputUtils.ask_yes_no(
            f"Proceed with removal of {len(detections)} region(s)?"
        ):
            print("Aborted by user.")
            return self.report

        # ---------- REMOVAL PASS ----------
        print()
        print("=" * 70)
        print("REMOVAL PASS (PDF REDACTION)")
        print("=" * 70)

        self._remove_with_redaction(detections, replacement_bytes)

        print()
        print(f"Removed: {self.report['removed']} region(s)")

        return self.report

    # ============================================================
    # INTERACTIVE
    # ============================================================
    def interactively_process(self, debug_dir):
        if not InputUtils.ask_yes_no("Delete/replace any image or logo?"):
            return self.report

        reference_path = InputUtils.ask_existing_file("Reference image path: ")

        replacement_path = None
        if InputUtils.ask_yes_no("Insert a replacement image?"):
            replacement_path = InputUtils.ask_existing_file(
                "Replacement image path: "
            )

        return self.process(reference_path, replacement_path, debug_dir)