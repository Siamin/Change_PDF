# python3
# -*- coding: utf-8 -*-

"""
============================================================
PDF IMAGE / TEXT EDITOR
============================================================

Features:

1) Delete pages by number or range
   Example:
       2 5 8-12

2) Delete text
   - Multiple different text strings
   - Find all occurrences
   - Permanent deletion using PDF Redaction

3) Replace text
   - The original text is deleted first
   - The new text is inserted at the same location

4) Delete image / part of a page
   - The reference image can be only a crop / screenshot
   - It does not need to be a standalone PDF Image Object
   - SIFT
   - ORB
   - Template Matching
   - Edge Matching
   - Multiple occurrences
   - Multiple pages

5) Delete image without replacement
   - Detect the target region
   - Create a mask
   - Reconstruct the background using OpenCV Inpainting
   - Place the reconstructed background over the same region

6) Replace image
   - First reconstruct the original region
   - Then insert the new image into the same region

7) Pages without modifications remain unchanged.

============================================================
DEPENDENCIES
============================================================

pip install pymupdf opencv-python numpy

============================================================
"""

import os
import sys
import math
import io
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import fitz


# ============================================================
# CONFIG
# ============================================================

RENDER_DPI = 220

# SIFT
SIFT_RATIO = 0.76
SIFT_MIN_MATCHES = 6
SIFT_MIN_INLIERS = 5

# ORB
ORB_RATIO = 0.80
ORB_MIN_MATCHES = 8
ORB_MIN_INLIERS = 5

# Template
TEMPLATE_MIN_SCORE = 0.68
TEMPLATE_STRONG_SCORE = 0.80

# Edge
EDGE_MIN_SCORE = 0.58
EDGE_STRONG_SCORE = 0.72

# Detection
MAX_DETECTIONS_PER_METHOD = 40
MAX_FINAL_DETECTIONS = 50

# Merge
IOU_MERGE_THRESHOLD = 0.25
OVERLAP_SMALLER_THRESHOLD = 0.55

# Minimum detected size in rendered pixels
MIN_DETECTION_SIZE = 8

# Padding around image while removing
REMOVE_PADDING_PX = 3

# Inpainting
INPAINT_RADIUS = 5

# Template scales
TEMPLATE_SCALES = np.linspace(0.30, 2.50, 23)

# Create debug images
CREATE_DEBUG_IMAGES = True

# Output suffix
OUTPUT_SUFFIX = "_edited"


# ============================================================
# BASIC UTILITIES
# ============================================================

def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()

        if answer in ("y", "yes", "1"):
            return True

        if answer in ("n", "no", "0"):
            return False

        print("Please enter y or n.")


def ask_existing_file(prompt: str) -> str:
    while True:
        path = input(prompt).strip().strip('"').strip("'")

        if not path:
            print("Path is empty")
            continue

        path = os.path.expanduser(path)

        if os.path.isfile(path):
            return path

        print(f"File not found: \n{path}")


def parse_page_ranges(text: str, total_pages: int) -> List[int]:
    """
    Input:
        2 5 8-12

    Output:
        zero-based page indexes
    """

    result = set()

    parts = text.replace(",", " ").split()

    for part in parts:

        if "-" in part:
            try:
                a, b = part.split("-", 1)

                a = int(a)
                b = int(b)

                if a > b:
                    a, b = b, a

                for page_num in range(a, b + 1):
                    if 1 <= page_num <= total_pages:
                        result.add(page_num - 1)

            except ValueError:
                print(f"Invalid range: {part}")

        else:
            try:
                page_num = int(part)

                if 1 <= page_num <= total_pages:
                    result.add(page_num - 1)

            except ValueError:
                print(f"Invalid page number: {part}")

    return sorted(result)


# ============================================================
# TEXT INPUT
# ============================================================

def collect_texts() -> List[str]:

    texts = []

    print()
    print("=" * 60)
    print("TEXTS TO DELETE")
    print("=" * 60)

    print("Enter the text strings that should be deleted, one at a time.")
    print("Press Enter on an empty line when you are finished.")
    print()

    while True:

        value = input("Text to delete: ")

        if not value:
            break

        texts.append(value)

    return texts


def collect_replacements(texts: List[str]) -> Dict[str, str]:

    replacements = {}

    print()
    print("=" * 60)
    print("TEXT REPLACEMENTS")
    print("=" * 60)

    print("Enter the replacement text for each original text.")
    print("Press Enter on an empty line if you only want to delete the text.")
    print()

    for text in texts:

        print(f'Original: "{text}"')

        replacement = input("Replacement: ")

        replacements[text] = replacement

    return replacements


# ============================================================
# PDF RENDER
# ============================================================

def render_page(page: fitz.Page, dpi: int = RENDER_DPI):

    zoom = dpi / 72.0

    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    if pix.n == 4:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2RGB
        )

    else:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

    return img


def pdf_rect_from_pixels(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    page_rect: fitz.Rect
) -> fitz.Rect:

    x1, y1, x2, y2 = bbox

    sx = page_rect.width / image_width
    sy = page_rect.height / image_height

    return fitz.Rect(
        page_rect.x0 + x1 * sx,
        page_rect.y0 + y1 * sy,
        page_rect.x0 + x2 * sx,
        page_rect.y0 + y2 * sy
    )


# ============================================================
# BBOX UTILITIES
# ============================================================

def clip_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int
):

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))

    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1:
        return None

    if y2 <= y1:
        return None

    return x1, y1, x2, y2


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox

    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    union = bbox_area(a) + bbox_area(b) - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def overlap_smaller(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    smaller = min(
        bbox_area(a),
        bbox_area(b)
    )

    if smaller <= 0:
        return 0.0

    return intersection / smaller


def bbox_center(bbox):

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


# ============================================================
# REFERENCE IMAGE PREPARATION
# ============================================================

def load_reference_image(path: str):

    data = np.fromfile(
        path,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read reference image:\n{path}"
        )

    return image


def prepare_reference(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Slight normalization improves screenshot matching
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


# ============================================================
# SIFT
# ============================================================

def detect_sift(
    page_gray,
    ref_gray
):

    detections = []

    if (
        ref_gray.shape[0] < 10
        or ref_gray.shape[1] < 10
    ):
        return detections

    try:
        sift = cv2.SIFT_create(
            nfeatures=5000
        )

        kp1, des1 = sift.detectAndCompute(
            ref_gray,
            None
        )

        kp2, des2 = sift.detectAndCompute(
            page_gray,
            None
        )

        if des1 is None or des2 is None:
            return detections

        if len(kp1) < 4 or len(kp2) < 4:
            return detections

        matcher = cv2.BFMatcher(
            cv2.NORM_L2
        )

        matches = matcher.knnMatch(
            des1,
            des2,
            k=2
        )

        good = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < SIFT_RATIO * n.distance:
                good.append(m)

        if len(good) < SIFT_MIN_MATCHES:
            return detections

        src_pts = np.float32([
            kp1[m.queryIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp2[m.trainIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if H is None or mask is None:
            return detections

        inliers = int(mask.ravel().sum())

        if inliers < SIFT_MIN_INLIERS:
            return detections

        h, w = ref_gray.shape[:2]

        corners = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(
            corners,
            H
        ).reshape(-1, 2)

        x1 = int(np.floor(transformed[:, 0].min()))
        y1 = int(np.floor(transformed[:, 1].min()))
        x2 = int(np.ceil(transformed[:, 0].max()))
        y2 = int(np.ceil(transformed[:, 1].max()))

        bbox = clip_bbox(
            (x1, y1, x2, y2),
            page_gray.shape[1],
            page_gray.shape[0]
        )

        if bbox is None:
            return detections

        if (
            bbox[2] - bbox[0] < MIN_DETECTION_SIZE
            or
            bbox[3] - bbox[1] < MIN_DETECTION_SIZE
        ):
            return detections

        score = inliers / max(1, len(good))

        detections.append({
            "bbox": bbox,
            "method": "SIFT",
            "score": float(score),
            "inliers": inliers,
            "matches": len(good)
        })

    except Exception as e:

        print(
            f"  [SIFT] error: {e}"
        )

    return detections


# ============================================================
# ORB
# ============================================================

def detect_orb(
    page_gray,
    ref_gray
):

    detections = []

    try:

        orb = cv2.ORB_create(
            nfeatures=8000,
            scaleFactor=1.2,
            nlevels=8
        )

        kp1, des1 = orb.detectAndCompute(
            ref_gray,
            None
        )

        kp2, des2 = orb.detectAndCompute(
            page_gray,
            None
        )

        if des1 is None or des2 is None:
            return detections

        if len(kp1) < 4 or len(kp2) < 4:
            return detections

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        matches = matcher.knnMatch(
            des1,
            des2,
            k=2
        )

        good = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < ORB_RATIO * n.distance:
                good.append(m)

        if len(good) < ORB_MIN_MATCHES:
            return detections

        src_pts = np.float32([
            kp1[m.queryIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp2[m.trainIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            6.0
        )

        if H is None or mask is None:
            return detections

        inliers = int(mask.ravel().sum())

        if inliers < ORB_MIN_INLIERS:
            return detections

        h, w = ref_gray.shape[:2]

        corners = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(
            corners,
            H
        ).reshape(-1, 2)

        x1 = int(np.floor(transformed[:, 0].min()))
        y1 = int(np.floor(transformed[:, 1].min()))
        x2 = int(np.ceil(transformed[:, 0].max()))
        y2 = int(np.ceil(transformed[:, 1].max()))

        bbox = clip_bbox(
            (x1, y1, x2, y2),
            page_gray.shape[1],
            page_gray.shape[0]
        )

        if bbox is None:
            return detections

        if (
            bbox[2] - bbox[0] < MIN_DETECTION_SIZE
            or
            bbox[3] - bbox[1] < MIN_DETECTION_SIZE
        ):
            return detections

        score = inliers / max(1, len(good))

        detections.append({
            "bbox": bbox,
            "method": "ORB",
            "score": float(score),
            "inliers": inliers,
            "matches": len(good)
        })

    except Exception as e:

        print(
            f"  [ORB] error: {e}"
        )

    return detections


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def local_maxima(
    result,
    threshold,
    max_results=MAX_DETECTIONS_PER_METHOD
):

    locations = []

    work = result.copy()

    for _ in range(max_results):

        _, max_val, _, max_loc = cv2.minMaxLoc(
            work
        )

        if max_val < threshold:
            break

        x, y = max_loc

        locations.append(
            (x, y, float(max_val))
        )

        # suppress neighborhood
        radius_x = max(
            8,
            result.shape[1] // 20
        )

        radius_y = max(
            8,
            result.shape[0] // 20
        )

        x1 = max(0, x - radius_x)
        y1 = max(0, y - radius_y)

        x2 = min(
            work.shape[1],
            x + radius_x
        )

        y2 = min(
            work.shape[0],
            y + radius_y
        )

        work[y1:y2, x1:x2] = 0

    return locations


def detect_template(
    page_gray,
    ref_gray
):

    detections = []

    rh, rw = ref_gray.shape[:2]

    if rh < 8 or rw < 8:
        return detections

    page_h, page_w = page_gray.shape[:2]

    for scale in TEMPLATE_SCALES:

        tw = int(rw * scale)
        th = int(rh * scale)

        if tw < 8 or th < 8:
            continue

        if tw >= page_w or th >= page_h:
            continue

        template = cv2.resize(
            ref_gray,
            (tw, th),
            interpolation=cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_CUBIC
        )

        try:

            result = cv2.matchTemplate(
                page_gray,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            locations = local_maxima(
                result,
                TEMPLATE_MIN_SCORE
            )

            for x, y, score in locations:

                bbox = (
                    x,
                    y,
                    x + tw,
                    y + th
                )

                bbox = clip_bbox(
                    bbox,
                    page_w,
                    page_h
                )

                if bbox is None:
                    continue

                detections.append({
                    "bbox": bbox,
                    "method": "TEMPLATE",
                    "score": score,
                    "scale": scale
                })

        except Exception:
            continue

    # Keep strongest candidates
    detections.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return detections[:MAX_DETECTIONS_PER_METHOD]


# ============================================================
# EDGE MATCHING
# ============================================================

def detect_edge(
    page_gray,
    ref_gray
):

    detections = []

    ref_edges = cv2.Canny(
        ref_gray,
        50,
        150
    )

    page_edges = cv2.Canny(
        page_gray,
        50,
        150
    )

    rh, rw = ref_edges.shape[:2]

    page_h, page_w = page_gray.shape[:2]

    if rh < 8 or rw < 8:
        return detections

    for scale in TEMPLATE_SCALES:

        tw = int(rw * scale)
        th = int(rh * scale)

        if tw < 8 or th < 8:
            continue

        if tw >= page_w or th >= page_h:
            continue

        template = cv2.resize(
            ref_edges,
            (tw, th),
            interpolation=cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_CUBIC
        )

        try:

            result = cv2.matchTemplate(
                page_edges,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            locations = local_maxima(
                result,
                EDGE_MIN_SCORE
            )

            for x, y, score in locations:

                bbox = (
                    x,
                    y,
                    x + tw,
                    y + th
                )

                bbox = clip_bbox(
                    bbox,
                    page_w,
                    page_h
                )

                if bbox is None:
                    continue

                detections.append({
                    "bbox": bbox,
                    "method": "EDGE",
                    "score": score,
                    "scale": scale
                })

        except Exception:
            continue

    detections.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return detections[:MAX_DETECTIONS_PER_METHOD]


# ============================================================
# CANDIDATE VERIFICATION
# ============================================================

def verify_template_candidate(
    page_gray,
    ref_gray,
    bbox
):

    x1, y1, x2, y2 = bbox

    crop = page_gray[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return 0.0

    target = cv2.resize(
        ref_gray,
        (crop.shape[1], crop.shape[0]),
        interpolation=cv2.INTER_AREA
    )

    try:

        result = cv2.matchTemplate(
            crop,
            target,
            cv2.TM_CCOEFF_NORMED
        )

        score = float(result[0, 0])

        return score

    except Exception:

        return 0.0


# ============================================================
# MERGE DETECTIONS
# ============================================================

def merge_detections(
    detections,
    page_gray,
    ref_gray
):

    if not detections:
        return []

    # Sort strongest first
    detections = sorted(
        detections,
        key=lambda d: (
            d.get("score", 0),
            1 if d["method"] in ("SIFT", "ORB") else 0
        ),
        reverse=True
    )

    groups = []

    for detection in detections:

        bbox = detection["bbox"]

        found_group = None

        for group in groups:

            for existing in group:

                iou = bbox_iou(
                    bbox,
                    existing["bbox"]
                )

                overlap = overlap_smaller(
                    bbox,
                    existing["bbox"]
                )

                if (
                    iou >= IOU_MERGE_THRESHOLD
                    or
                    overlap >= OVERLAP_SMALLER_THRESHOLD
                ):
                    found_group = group
                    break

            if found_group is not None:
                break

        if found_group is None:
            groups.append(
                [detection]
            )
        else:
            found_group.append(
                detection
            )

    final = []

    for group in groups:

        methods = {
            d["method"]
            for d in group
        }

        best = max(
            group,
            key=lambda d: d.get(
                "score",
                0
            )
        )

        # ----------------------------------------------------
        # Candidate acceptance
        # ----------------------------------------------------

        accepted = False

        # Strong feature-based match
        for d in group:

            if d["method"] == "SIFT":

                if (
                    d.get("inliers", 0)
                    >= SIFT_MIN_INLIERS
                    and
                    d.get("score", 0)
                    >= 0.30
                ):
                    accepted = True

            elif d["method"] == "ORB":

                if (
                    d.get("inliers", 0)
                    >= ORB_MIN_INLIERS
                    and
                    d.get("score", 0)
                    >= 0.30
                ):
                    accepted = True

        # Multiple independent methods
        if len(methods) >= 2:
            accepted = True

        # Strong template
        if (
            "TEMPLATE" in methods
            and
            any(
                d.get("score", 0)
                >= TEMPLATE_STRONG_SCORE
                for d in group
                if d["method"] == "TEMPLATE"
            )
        ):
            accepted = True

        # Strong edge
        if (
            "EDGE" in methods
            and
            any(
                d.get("score", 0)
                >= EDGE_STRONG_SCORE
                for d in group
                if d["method"] == "EDGE"
            )
        ):
            accepted = True

        if not accepted:
            continue

        # ----------------------------------------------------
        # Build union bbox
        # ----------------------------------------------------

        xs1 = [
            d["bbox"][0]
            for d in group
        ]

        ys1 = [
            d["bbox"][1]
            for d in group
        ]

        xs2 = [
            d["bbox"][2]
            for d in group
        ]

        ys2 = [
            d["bbox"][3]
            for d in group
        ]

        bbox = (
            min(xs1),
            min(ys1),
            max(xs2),
            max(ys2)
        )

        bbox = clip_bbox(
            bbox,
            page_gray.shape[1],
            page_gray.shape[0]
        )

        if bbox is None:
            continue

        # ----------------------------------------------------
        # Remove duplicate final candidates
        # ----------------------------------------------------

        already = False

        for existing in final:

            if (
                bbox_iou(
                    bbox,
                    existing["bbox"]
                )
                >= 0.40
            ):
                already = True
                break

        if already:
            continue

        final.append({
            "bbox": bbox,
            "methods": sorted(methods),
            "score": best.get(
                "score",
                0
            ),
            "group": group
        })

        if len(final) >= MAX_FINAL_DETECTIONS:
            break

    return final


# ============================================================
# DEBUG IMAGE
# ============================================================

def save_debug_image(
    image,
    detections,
    output_path
):

    debug = image.copy()

    for index, detection in enumerate(detections, 1):

        x1, y1, x2, y2 = detection["bbox"]

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        label = (
            f"{index}: "
            + ",".join(
                detection["methods"]
            )
        )

        cv2.putText(
            debug,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            debug,
            cv2.COLOR_RGB2BGR
        )
    )


# ============================================================
# BACKGROUND RECONSTRUCTION
# ============================================================

def make_removal_patch(
    page_rgb,
    bbox
):

    h, w = page_rgb.shape[:2]

    x1, y1, x2, y2 = bbox

    # Expand region slightly for better inpainting
    px = REMOVE_PADDING_PX

    mx1 = max(
        0,
        x1 - px
    )

    my1 = max(
        0,
        y1 - px
    )

    mx2 = min(
        w,
        x2 + px
    )

    my2 = min(
        h,
        y2 + px
    )

    # Crop
    crop_rgb = page_rgb[
        my1:my2,
        mx1:mx2
    ].copy()

    if crop_rgb.size == 0:
        return None

    crop_bgr = cv2.cvtColor(
        crop_rgb,
        cv2.COLOR_RGB2BGR
    )

    ch, cw = crop_bgr.shape[:2]

    # Mask only actual target area
    mask = np.zeros(
        (ch, cw),
        dtype=np.uint8
    )

    local_x1 = x1 - mx1
    local_y1 = y1 - my1
    local_x2 = x2 - mx1
    local_y2 = y2 - my1

    local_x1 = max(0, local_x1)
    local_y1 = max(0, local_y1)
    local_x2 = min(cw, local_x2)
    local_y2 = min(ch, local_y2)

    cv2.rectangle(
        mask,
        (
            local_x1,
            local_y1
        ),
        (
            local_x2,
            local_y2
        ),
        255,
        -1
    )

    # Slightly soften / enlarge mask edge
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # OpenCV Telea inpainting
    # --------------------------------------------------------

    result_bgr = cv2.inpaint(
        crop_bgr,
        mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA
    )

    result_rgb = cv2.cvtColor(
        result_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Return:
    # patch
    # destination bbox including padding
    return (
        result_rgb,
        (
            mx1,
            my1,
            mx2,
            my2
        )
    )


# ============================================================
# INSERT RASTER PATCH INTO PDF
# ============================================================

def image_to_png_bytes(image_rgb):

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    success, encoded = cv2.imencode(
        ".png",
        image_bgr
    )

    if not success:
        raise RuntimeError(
            "Could not encode image patch."
        )

    return encoded.tobytes()


def insert_background_patch(
    page,
    patch_rgb,
    patch_bbox,
    full_width,
    full_height
):

    rect = pdf_rect_from_pixels(
        patch_bbox,
        full_width,
        full_height,
        page.rect
    )

    png_bytes = image_to_png_bytes(
        patch_rgb
    )

    page.insert_image(
        rect,
        stream=png_bytes,
        keep_proportion=False,
        overlay=True
    )


# ============================================================
# TRY DELETE REAL PDF IMAGE OBJECT
# ============================================================

def try_delete_underlying_images(
    page,
    detection_rect
):

    deleted = 0

    try:

        infos = page.get_image_info(
            xrefs=True
        )

    except Exception:

        return deleted

    for info in infos:

        bbox = info.get("bbox")

        xref = info.get("xref")

        if not bbox or not xref:
            continue

        image_rect = fitz.Rect(
            bbox
        )

        intersection = image_rect & detection_rect

        if intersection.is_empty:
            continue

        intersection_area = intersection.get_area()

        image_area = image_rect.get_area()

        if image_area <= 0:
            continue

        overlap = (
            intersection_area
            / image_area
        )

        # If most of the PDF image object is
        # inside our detected region, delete it.
        if overlap >= 0.60:

            try:

                page.delete_image(
                    xref
                )

                deleted += 1

            except Exception:
                pass

    return deleted


# ============================================================
# IMAGE DETECTION FOR ONE PAGE
# ============================================================

def find_image_matches(
    page,
    reference_gray
):

    page_rgb = render_page(
        page,
        RENDER_DPI
    )

    page_gray = cv2.cvtColor(
        page_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Slight blur for more stable template matching
    page_gray_blur = cv2.GaussianBlur(
        page_gray,
        (3, 3),
        0
    )

    detections = []

    # --------------------------------------------------------
    # SIFT
    # --------------------------------------------------------

    print("    SIFT...")

    sift = detect_sift(
        page_gray,
        reference_gray
    )

    print(
        f"      {len(sift)} candidate(s)"
    )

    detections.extend(
        sift
    )

    # --------------------------------------------------------
    # ORB
    # --------------------------------------------------------

    print("    ORB...")

    orb = detect_orb(
        page_gray,
        reference_gray
    )

    print(
        f"      {len(orb)} candidate(s)"
    )

    detections.extend(
        orb
    )

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------

    print("    Template matching...")

    template = detect_template(
        page_gray_blur,
        reference_gray
    )

    print(
        f"      {len(template)} candidate(s)"
    )

    detections.extend(
        template
    )

    # --------------------------------------------------------
    # Edge
    # --------------------------------------------------------

    print("    Edge matching...")

    edge = detect_edge(
        page_gray,
        reference_gray
    )

    print(
        f"      {len(edge)} candidate(s)"
    )

    detections.extend(
        edge
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    final = merge_detections(
        detections,
        page_gray,
        reference_gray
    )

    return page_rgb, final


# ============================================================
# IMAGE OPERATION
# ============================================================

def process_image_operations(
    doc,
    reference_path: str,
    replacement_path: Optional[str],
    debug_dir: Optional[str]
):

    reference_image = load_reference_image(
        reference_path
    )

    reference_gray = prepare_reference(
        reference_image
    )

    replacement_bytes = None

    if replacement_path:

        with open(
            replacement_path,
            "rb"
        ) as f:
            replacement_bytes = f.read()

    report = {
        "pages_scanned": 0,
        "pages_with_matches": 0,
        "matches": 0,
        "removed": 0,
        "replaced": 0,
        "underlying_images_deleted": 0
    }

    for page_index in range(
        len(doc)
    ):

        page = doc[page_index]

        report["pages_scanned"] += 1

        print()
        print(
            "-" * 70
        )

        print(
            f"IMAGE SEARCH - PAGE {page_index + 1}"
        )

        try:

            page_rgb, detections = find_image_matches(
                page,
                reference_gray
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

            continue

        if not detections:

            print(
                "  No image match."
            )

            continue

        report["pages_with_matches"] += 1

        print(
            f"  ACCEPTED MATCHES: {len(detections)}"
        )

        if CREATE_DEBUG_IMAGES and debug_dir:

            os.makedirs(
                debug_dir,
                exist_ok=True
            )

            debug_path = os.path.join(
                debug_dir,
                f"page_{page_index + 1:04d}_matches.png"
            )

            save_debug_image(
                page_rgb,
                detections,
                debug_path
            )

        # ----------------------------------------------------
        # Process strongest / largest first
        # ----------------------------------------------------

        detections = sorted(
            detections,
            key=lambda d: bbox_area(
                d["bbox"]
            ),
            reverse=True
        )

        for match_index, detection in enumerate(
            detections,
            1
        ):

            bbox = detection["bbox"]

            print()
            print(
                f"  Match #{match_index}"
            )

            print(
                f"    bbox: {bbox}"
            )

            print(
                "    methods:",
                ", ".join(
                    detection["methods"]
                )
            )

            print(
                f"    score: {detection['score']:.3f}"
            )

            report["matches"] += 1

            # ------------------------------------------------
            # Convert bbox to PDF
            # ------------------------------------------------

            page_rect = pdf_rect_from_pixels(
                bbox,
                page_rgb.shape[1],
                page_rgb.shape[0],
                page.rect
            )

            # ------------------------------------------------
            # Try deleting actual PDF image object
            # ------------------------------------------------

            deleted_xrefs = try_delete_underlying_images(
                page,
                page_rect
            )

            report[
                "underlying_images_deleted"
            ] += deleted_xrefs

            if deleted_xrefs:
                print(
                    f"    Underlying PDF image object(s) deleted: "
                    f"{deleted_xrefs}"
                )

            # ------------------------------------------------
            # Build background
            # ------------------------------------------------

            removal_result = make_removal_patch(
                page_rgb,
                bbox
            )

            if removal_result is None:

                print(
                    "    Could not create background patch."
                )

                continue

            patch_rgb, patch_bbox = removal_result

            # ------------------------------------------------
            # Put reconstructed background over original
            # ------------------------------------------------

            insert_background_patch(
                page,
                patch_rgb,
                patch_bbox,
                page_rgb.shape[1],
                page_rgb.shape[0]
            )

            report["removed"] += 1

            print(
                "    Original image region removed/reconstructed."
            )

            # ------------------------------------------------
            # Replacement image
            # ------------------------------------------------

            if replacement_bytes:

                try:

                    page.insert_image(
                        page_rect,
                        stream=replacement_bytes,
                        keep_proportion=False,
                        overlay=True
                    )

                    report["replaced"] += 1

                    print(
                        "    Replacement image inserted."
                    )

                except Exception as e:

                    print(
                        f"    Replacement failed: {e}"
                    )

    return report


# ============================================================
# TEXT OPERATIONS
# ============================================================

def find_text_rects(
    page,
    text
):

    try:

        rects = page.search_for(
            text
        )

        return rects

    except Exception as e:

        print(
            f"Text search error: {e}"
        )

        return []


def process_text_operations(
    doc,
    texts,
    replacements
):

    report = {
        "found": 0,
        "deleted": 0,
        "replaced": 0
    }

    # Store replacements before deleting
    replacement_targets = []

    print()
    print("=" * 70)
    print("TEXT PROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # FIND ALL TEXT
    # --------------------------------------------------------

    for page_index in range(
        len(doc)
    ):

        page = doc[page_index]

        for text in texts:

            rects = find_text_rects(
                page,
                text
            )

            if not rects:
                continue

            print(
                f"Page {page_index + 1}: "
                f'"{text}" -> {len(rects)} occurrence(s)'
            )

            report["found"] += len(rects)

            replacement = replacements.get(
                text,
                ""
            )

            for rect in rects:

                replacement_targets.append({
                    "page": page_index,
                    "rect": fitz.Rect(rect),
                    "replacement": replacement
                })

    # --------------------------------------------------------
    # DELETE ALL ORIGINAL TEXT FIRST
    # --------------------------------------------------------

    if replacement_targets:

        print()
        print(
            "Applying text redactions..."
        )

        pages_with_redactions = {}

        for target in replacement_targets:

            page_index = target["page"]

            pages_with_redactions.setdefault(
                page_index,
                []
            ).append(
                target["rect"]
            )

        for page_index, rects in pages_with_redactions.items():

            page = doc[page_index]

            for rect in rects:

                page.add_redact_annot(
                    rect,
                    fill=(1, 1, 1)
                )

                report["deleted"] += 1

            page.apply_redactions()

        print(
            f"Deleted text occurrences: "
            f"{report['deleted']}"
        )

    # --------------------------------------------------------
    # INSERT REPLACEMENT TEXT
    # --------------------------------------------------------

    for target in replacement_targets:

        replacement = target["replacement"]

        if not replacement:
            continue

        page = doc[
            target["page"]
        ]

        rect = target["rect"]

        # Estimate font size from rectangle
        fontsize = max(
            6,
            min(
                72,
                rect.height * 0.85
            )
        )

        try:

            result = page.insert_textbox(
                rect,
                replacement,
                fontname="helv",
                fontsize=fontsize,
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_LEFT,
                overlay=True
            )

            if result >= 0:

                report["replaced"] += 1

        except Exception as e:

            print(
                f"Replacement text error: {e}"
            )

    return report


# ============================================================
# DELETE PAGES
# ============================================================

def delete_pages(
    doc,
    page_indexes
):

    if not page_indexes:
        return

    # Delete backwards
    for index in sorted(
        page_indexes,
        reverse=True
    ):

        print(
            f"Deleting page {index + 1}"
        )

        doc.delete_page(
            index
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AUTOMATIC PDF / IMAGE / TEXT EDITOR")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf_path = ask_existing_file(
        "Main PDF path: "
    )

    pdf_path_obj = Path(
        pdf_path
    )

    output_path = str(
        pdf_path_obj.with_name(
            pdf_path_obj.stem
            + OUTPUT_SUFFIX
            + ".pdf"
        )
    )

    debug_dir = str(
        pdf_path_obj.with_name(
            pdf_path_obj.stem
            + "_debug"
        )
    )

    print()
    print(
        "Opening PDF..."
    )

    try:

        doc = fitz.open(
            pdf_path
        )

    except Exception as e:

        print(
            f"Cannot open PDF: {e}"
        )

        sys.exit(1)

    original_page_count = len(doc)

    print(
        f"Pages: {original_page_count}"
    )

    # ========================================================
    # DELETE PAGES
    # ========================================================

    pages_to_delete = []

    print()

    if ask_yes_no(
        "Are there pages to delete?"
    ):

        value = input(
            "Page numbers/ranges "
            "(example: 2 5 8-12): "
        ).strip()

        pages_to_delete = parse_page_ranges(
            value,
            len(doc)
        )

        if pages_to_delete:

            print(
                "Pages selected:",
                [
                    x + 1
                    for x in pages_to_delete
                ]
            )

            delete_pages(
                doc,
                pages_to_delete
            )

        else:

            print(
                "No valid pages selected."
            )

    # ========================================================
    # TEXT
    # ========================================================

    text_report = {
        "found": 0,
        "deleted": 0,
        "replaced": 0
    }

    if ask_yes_no(
        "Is there text to delete?"
    ):

        texts = collect_texts()

        if texts:

            replacements = {}

            if ask_yes_no(
                "Is there text to replace?"
            ):

                replacements = collect_replacements(
                    texts
                )

            else:

                replacements = {
                    text: ""
                    for text in texts
                }

            text_report = process_text_operations(
                doc,
                texts,
                replacements
            )

    # ========================================================
    # IMAGE
    # ========================================================

    image_report = {
        "pages_scanned": 0,
        "pages_with_matches": 0,
        "matches": 0,
        "removed": 0,
        "replaced": 0,
        "underlying_images_deleted": 0
    }

    if ask_yes_no(
        "Is there an image / image area to delete?"
    ):

        print()
        print(
            "The reference image can be only a cropped part of the page."
        )

        print(
            "It does not need to be recognized as a standalone PDF Image Object."
        )

        reference_path = ask_existing_file(
            "Reference image path: "
        )

        replacement_path = None

        if ask_yes_no(
            "Is there a replacement image?"
        ):

            replacement_path = ask_existing_file(
                "Replacement image path: "
            )

        # Remove old debug directory
        if os.path.isdir(
            debug_dir
        ):

            try:
                shutil.rmtree(
                    debug_dir
                )
            except Exception:
                pass

        image_report = process_image_operations(
            doc,
            reference_path,
            replacement_path,
            debug_dir
        )

    # ========================================================
    # SAVE
    # ========================================================

    print()
    print("=" * 70)
    print("SAVING")
    print("=" * 70)

    print(
        f"Output: {output_path}"
    )

    try:

        # garbage=4:
        # remove unused objects
        #
        # clean=True:
        # clean PDF structure
        #
        # deflate=True:
        # compress streams

        doc.save(
            output_path,
            garbage=4,
            deflate=True,
            clean=True
        )

        doc.close()

    except Exception as e:

        print()
        print(
            "SAVE ERROR:"
        )

        print(
            e
        )

        try:
            doc.close()
        except Exception:
            pass

        sys.exit(1)

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print()
    print(
        f"Original pages: "
        f"{original_page_count}"
    )

    print(
        f"Final pages: "
        f"{original_page_count - len(pages_to_delete)}"
    )

    print(
        f"Deleted pages: "
        f"{len(pages_to_delete)}"
    )

    print()
    print(
        "TEXT"
    )

    print(
        f"  Found: "
        f"{text_report['found']}"
    )

    print(
        f"  Deleted: "
        f"{text_report['deleted']}"
    )

    print(
        f"  Replaced: "
        f"{text_report['replaced']}"
    )

    print()
    print(
        "IMAGE"
    )

    print(
        f"  Pages scanned: "
        f"{image_report['pages_scanned']}"
    )

    print(
        f"  Pages with matches: "
        f"{image_report['pages_with_matches']}"
    )

    print(
        f"  Matches: "
        f"{image_report['matches']}"
    )

    print(
        f"  Removed/reconstructed: "
        f"{image_report['removed']}"
    )

    print(
        f"  Replaced: "
        f"{image_report['replaced']}"
    )

    print(
        f"  PDF image objects deleted: "
        f"{image_report['underlying_images_deleted']}"
    )

    print()
    print(
        f"Output file:"
    )

    print(
        output_path
    )

    if CREATE_DEBUG_IMAGES and os.path.isdir(
        debug_dir
    ):

        print()
        print(
            "Debug images:"
        )

        print(
            debug_dir
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
