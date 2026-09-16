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

    def __init__(
        self,
        doc: fitz.Document,
        removal_mode: str = MODE_AUTO,
        force_delete_image_objects: bool = False,
    ):
        self.doc = doc
        self.removal_mode = removal_mode
        self.force_delete_image_objects = force_delete_image_objects
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
        }

    @staticmethod
    def load_reference(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                raise RuntimeError("Reference image is empty.")
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Cannot decode: {path}")
            return img
        except Exception as exc:
            raise RuntimeError(f"Cannot load reference: {exc}") from exc

    @staticmethod
    def prepare_reference(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    @staticmethod
    def get_image_objects(page):
        try:
            return page.get_image_info(xrefs=True)
        except Exception:
            return []

    @staticmethod
    def image_object_overlap(image_rect, detection_rect):
        try:
            inter = image_rect & detection_rect
            if inter.is_empty:
                return 0.0
            area = image_rect.get_area()
            if area <= 0:
                return 0.0
            return inter.get_area() / float(area)
        except Exception:
            return 0.0

    def delete_image_objects(self, page, detection_rect):
        deleted = 0
        for info in self.get_image_objects(page):
            bbox = info.get("bbox")
            xref = info.get("xref")
            if not bbox or not xref:
                continue
            try:
                image_rect = fitz.Rect(bbox)
            except Exception:
                continue

            overlap = self.image_object_overlap(image_rect, detection_rect)
            if self.force_delete_image_objects:
                if overlap <= 0.0:
                    continue
            else:
                if overlap < 0.60:
                    continue

            try:
                page.delete_image(int(xref))
                deleted += 1
            except Exception as exc:
                print(f"    delete_image failed: {exc}")
        return deleted

    @staticmethod
    def has_vector_drawing(page, rect):
        try:
            for d in page.get_drawings():
                dr = d.get("rect")
                if dr is None:
                    continue
                try:
                    drect = fitz.Rect(dr)
                except Exception:
                    continue
                if not (drect & rect).is_empty:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def image_to_png(image_rgb):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("PNG encode failed.")
        return enc.tobytes()

    def insert_patch(self, page, patch_rgb, patch_bbox, full_w, full_h):
        try:
            rect = BBoxUtils.pdf_rect_from_pixels(
                patch_bbox, full_w, full_h, page.rect
            )
            png = self.image_to_png(patch_rgb)
            page.insert_image(
                rect, stream=png,
                keep_proportion=False, overlay=True,
            )
            return True
        except Exception as exc:
            print(f"    Patch insert failed: {exc}")
            return False

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

    def process(self, reference_path, replacement_path, debug_dir):
        try:
            ref_img = self.load_reference(reference_path)
            ref_gray = self.prepare_reference(ref_img)
        except Exception as exc:
            print(f"Reference error: {exc}")
            return self.report

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

            detections = self.detector.detect_on_page(page_rgb, ref_gray)

            if not detections:
                print("  No confirmed match.")
                continue

            self.report["pages_with_matches"] += 1
            self.report["matches"] += len(detections)
            print(f"  CONFIRMED: {len(detections)}")

            if CREATE_DEBUG_IMAGES and debug_dir:
                path = os.path.join(
                    debug_dir,
                    f"page_{page_index + 1:04d}_matches.png",
                )
                if self.save_debug(page_rgb, detections, path):
                    self.report["debug_images"] += 1

            detections = sorted(
                detections,
                key=lambda d: BBoxUtils.area(d["bbox"]),
                reverse=True,
            )

            processed = []

            for mi, det in enumerate(detections, 1):
                try:
                    bbox = det["bbox"]

                    if any(
                        BBoxUtils.iou(bbox, p) >= FINAL_DUPLICATE_IOU
                        or BBoxUtils.overlap_smaller(bbox, p) >= 0.80
                        for p in processed
                    ):
                        continue

                    print()
                    print(f"  Match #{mi}")
                    print(f"    bbox: {bbox}")
                    print(f"    methods: {','.join(det.get('methods', []))}")
                    print(f"    score: {det.get('score', 0.0):.3f}")
                    print(f"    verification: {det.get('verification', 0.0):.3f}")

                    page_rect = BBoxUtils.pdf_rect_from_pixels(
                        bbox,
                        page_rgb.shape[1],
                        page_rgb.shape[0],
                        page.rect,
                    )

                    is_vector = self.has_vector_drawing(page, page_rect)
                    if is_vector:
                        print("    Vector drawing detected.")

                    deleted = self.delete_image_objects(page, page_rect)
                    self.report["underlying_images_deleted"] += deleted
                    if deleted:
                        print(f"    Image objects deleted: {deleted}")

                    result = self.reconstructor.build_patch(page_rgb, bbox)
                    if result is None:
                        print("    Reconstruction failed.")
                        continue

                    patch_rgb, patch_bbox, method = result

                    if not self.insert_patch(
                        page, patch_rgb, patch_bbox,
                        page_rgb.shape[1], page_rgb.shape[0],
                    ):
                        continue

                    self.report["background_reconstructed"] += 1
                    self.report["removed"] += 1

                    if is_vector:
                        self.report["vector_regions_reconstructed"] += 1

                    print(f"    Region reconstructed via {method}.")

                    if replacement_bytes:
                        try:
                            page.insert_image(
                                page_rect, stream=replacement_bytes,
                                keep_proportion=False, overlay=True,
                            )
                            self.report["replaced"] += 1
                            print("    Replacement inserted.")
                        except Exception as exc:
                            print(f"    Replacement failed: {exc}")

                    processed.append(bbox)

                except Exception as exc:
                    print(f"    Match error: {exc}")
                    continue

        return self.report

    def interactively_process(self, debug_dir):
        if not InputUtils.ask_yes_no("Delete/replace any image or logo?"):
            return self.report

        print()
        print("Removal mode:")
        mode = InputUtils.ask_choice(
            "How should the region be filled?",
            [
                "Auto (inpaint, fall back to redact)",
                "Inpaint (reconstruct background)",
                "Redact (fill with border color) - recommended for large logos",
            ],
        )
        if "Inpaint" in mode:
            self.removal_mode = MODE_INPAINT
            self.reconstructor.mode = MODE_INPAINT
        elif "Redact" in mode:
            self.removal_mode = MODE_REDACT
            self.reconstructor.mode = MODE_REDACT
        else:
            self.removal_mode = MODE_AUTO
            self.reconstructor.mode = MODE_AUTO

        self.force_delete_image_objects = InputUtils.ask_yes_no(
            "Force-delete any Image Object inside detected region? "
            "(Recommended for logos that are PDF image objects)"
        )

        reference_path = InputUtils.ask_existing_file("Reference image path: ")

        replacement_path = None
        if InputUtils.ask_yes_no("Insert a replacement image?"):
            replacement_path = InputUtils.ask_existing_file(
                "Replacement image path: "
            )

        return self.process(reference_path, replacement_path, debug_dir)