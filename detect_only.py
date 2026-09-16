#python3
# -*- coding: utf-8 -*-
"""
Detection + fill box with sampled BG color.
No backup. Debug folder is deleted at the end.
"""

import os
import sys
import shutil
import cv2
import numpy as np
import fitz


# ---------------- CONFIG ----------------
RENDER_DPI = 220
REPLACE_DPI = 300
SIFT_RATIO = 0.70
MIN_INLIERS = 10
MIN_MATCHES = 12
RANSAC_THRESHOLD = 3.0
ASPECT_TOLERANCE = 0.40
MIN_AREA_RATIO = 0.0003
MAX_AREA_RATIO = 0.60
MAX_PER_PAGE = 10


def ask_file(prompt):
    while True:
        try:
            p = input(prompt).strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            sys.exit(1)
        if not p:
            continue
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            return p
        print(f"Not found: {p}")


# ----------------------------------------------------------------
# RENDER  (no BGR->RGB conversion)
# ----------------------------------------------------------------
def render_page(page, dpi):
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    ).copy()
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


# ----------------------------------------------------------------
# BACKGROUND COLOR SAMPLING
# ----------------------------------------------------------------
def sample_background_rgb(page_rgb, bbox, pad=40):
    x1, y1, x2, y2 = bbox
    h, w = page_rgb.shape[:2]
    samples = []

    strips = [
        (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), max(1, y1 - 5)),
        (max(0, x1 - pad), min(h - 1, y2 + 5), min(w, x2 + pad), min(h, y2 + pad)),
        (max(0, x1 - pad), max(0, y1 - pad), max(1, x1 - 5), min(h, y2 + pad)),
        (min(w - 1, x2 + 5), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)),
    ]
    for rx1, ry1, rx2, ry2 in strips:
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


# ----------------------------------------------------------------
# SIFT DETECTION
# ----------------------------------------------------------------
def detect_sift_multiple(page_gray, ref_gray):
    results = []
    try:
        ph, pw = page_gray.shape[:2]
        rh, rw = ref_gray.shape[:2]

        sift = cv2.SIFT_create(nfeatures=5000)
        kp_r, des_r = sift.detectAndCompute(ref_gray, None)
        kp_p, des_p = sift.detectAndCompute(page_gray, None)

        if des_r is None or des_p is None:
            return results
        if len(kp_r) < 8 or len(kp_p) < 8:
            print(f"      ref kp: {len(kp_r)}, page kp: {len(kp_p)} (too few)")
            return results

        print(f"      ref kp: {len(kp_r)}, page kp: {len(kp_p)}")

        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw = bf.knnMatch(des_r, des_p, k=2)

        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < SIFT_RATIO * n.distance:
                good.append(m)

        print(f"      good matches: {len(good)}")
        if len(good) < MIN_MATCHES:
            return results

        ref_aspect = rw / float(rh)
        active = good[:]

        for iteration in range(MAX_PER_PAGE):
            if len(active) < MIN_MATCHES:
                break

            src = np.float32([kp_r[m.queryIdx].pt for m in active]).reshape(-1, 1, 2)
            dst = np.float32([kp_p[m.trainIdx].pt for m in active]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
            if H is None or mask is None:
                break

            inl_mask = mask.ravel().astype(bool)
            inliers = int(inl_mask.sum())
            print(f"      iter {iteration+1}: inliers={inliers}")

            if inliers < MIN_INLIERS:
                break

            corners = np.float32([
                [0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]
            ]).reshape(-1, 1, 2)
            tf = cv2.perspectiveTransform(corners, H)
            pts = tf.reshape(-1, 2)

            if not np.isfinite(pts).all():
                active = [m for i, m in enumerate(active) if not inl_mask[i]]
                continue

            x1 = int(np.floor(pts[:, 0].min()))
            y1 = int(np.floor(pts[:, 1].min()))
            x2 = int(np.ceil(pts[:, 0].max()))
            y2 = int(np.ceil(pts[:, 1].max()))

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(pw, x2); y2 = min(ph, y2)

            if x2 <= x1 or y2 <= y1:
                active = [m for i, m in enumerate(active) if not inl_mask[i]]
                continue

            bw = x2 - x1
            bh = y2 - y1
            area_ratio = (bw * bh) / float(pw * ph)

            if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                active = [m for i, m in enumerate(active) if not inl_mask[i]]
                continue

            det_aspect = bw / float(bh)
            if abs(det_aspect - ref_aspect) / ref_aspect > ASPECT_TOLERANCE:
                active = [m for i, m in enumerate(active) if not inl_mask[i]]
                continue

            print(f"        ACCEPT bbox=({x1},{y1},{x2},{y2}) inliers={inliers}")
            results.append((x1, y1, x2, y2))
            active = [m for i, m in enumerate(active) if not inl_mask[i]]

    except Exception as e:
        print(f"      SIFT error: {e}")

    return results


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
def main():
    print()
    print("=" * 60)
    print("DETECTION + FILL WITH BG COLOR + NEW PDF")
    print("=" * 60)

    pdf_path = ask_file("PDF path: ")
    ref_path = ask_file("Reference image path: ")

    data = np.fromfile(ref_path, dtype=np.uint8)
    ref_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if ref_bgr is None:
        print("Cannot decode reference.")
        sys.exit(1)
    ref_gray = cv2.GaussianBlur(
        cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0
    )
    print(f"Reference size: {ref_bgr.shape[1]}x{ref_bgr.shape[0]}")

    base, _ = os.path.splitext(pdf_path)
    out_pdf = base + "_edited.pdf"
    debug_dir = base + "_fill_debug"

    os.makedirs(debug_dir, exist_ok=True)

    src_doc = fitz.open(pdf_path)
    print(f"PDF pages: {len(src_doc)}")

    out_doc = fitz.open()

    total_found = 0
    pages_replaced = 0

    for page_index in range(len(src_doc)):
        src_page = src_doc[page_index]
        print(f"\n--- PAGE {page_index + 1} ---")

        rect = src_page.rect
        pw, ph = rect.width, rect.height

        page_rgb_detect = render_page(src_page, RENDER_DPI)
        page_gray = cv2.GaussianBlur(
            cv2.cvtColor(page_rgb_detect, cv2.COLOR_RGB2GRAY), (3, 3), 0
        )
        bboxes = detect_sift_multiple(page_gray, ref_gray)

        if not bboxes:
            print("  no detection -> copying original page")
            out_doc.insert_pdf(
                src_doc, from_page=page_index, to_page=page_index
            )
            continue

        total_found += len(bboxes)
        print(f"  detections: {len(bboxes)} -> rasterizing this page")

        page_rgb_final = render_page(src_page, REPLACE_DPI)
        scale = REPLACE_DPI / float(RENDER_DPI)

        for i, (x1, y1, x2, y2) in enumerate(bboxes, 1):
            bg = sample_background_rgb(page_rgb_detect, (x1, y1, x2, y2))
            print(f"  #{i} bbox=({x1},{y1},{x2},{y2})  BG_RGB={bg}")

            sx1 = int(round(x1 * scale))
            sy1 = int(round(y1 * scale))
            sx2 = int(round(x2 * scale))
            sy2 = int(round(y2 * scale))

            sx1 = max(0, sx1)
            sy1 = max(0, sy1)
            sx2 = min(page_rgb_final.shape[1], sx2)
            sy2 = min(page_rgb_final.shape[0], sy2)

            # page_rgb_final is RGB -> pass RGB directly
            cv2.rectangle(
                page_rgb_final,
                (sx1, sy1),
                (sx2, sy2),
                bg,
                -1,
            )
            print(f"    filled RGB{bg} at ({sx1},{sy1})-({sx2},{sy2})")

        # Save debug image (will be deleted at the end)
        dbg_path = os.path.join(debug_dir, f"page_{page_index+1:04d}.png")
        cv2.imwrite(
            dbg_path,
            cv2.cvtColor(page_rgb_final, cv2.COLOR_RGB2BGR),
        )
        print(f"  debug saved: {dbg_path}")

        bgr = cv2.cvtColor(page_rgb_final, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok:
            print("  JPEG encode failed, copying original page")
            out_doc.insert_pdf(
                src_doc, from_page=page_index, to_page=page_index
            )
            continue

        new_page = out_doc.new_page(width=pw, height=ph)
        new_page.insert_image(
            new_page.rect,
            stream=encoded.tobytes(),
            keep_proportion=False,
            overlay=True,
        )
        pages_replaced += 1
        print(f"  page {page_index + 1} rebuilt as image")

    print()
    print(f"Saving to: {out_pdf}")
    out_doc.save(out_pdf, garbage=4, deflate=True)
    out_doc.close()
    src_doc.close()

    # ---- Delete debug folder ----
    try:
        if os.path.isdir(debug_dir):
            shutil.rmtree(debug_dir)
            print(f"Debug folder deleted: {debug_dir}")
    except Exception as e:
        print(f"Could not delete debug folder: {e}")

    print()
    print("=" * 60)
    print(f"TOTAL DETECTIONS: {total_found}")
    print(f"PAGES REPLACED:   {pages_replaced}")
    print(f"Output PDF:       {out_pdf}")
    print("=" * 60)


if __name__ == "__main__":
    main()