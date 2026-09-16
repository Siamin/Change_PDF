# include/bbox_utils.py
import math
import fitz


class BBoxUtils:

    @staticmethod
    def clip(bbox, width, height):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = BBoxUtils.area(a) + BBoxUtils.area(b) - inter
        if union <= 0:
            return 0.0
        return inter / float(union)

    @staticmethod
    def overlap_smaller(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        smaller = min(BBoxUtils.area(a), BBoxUtils.area(b))
        if smaller <= 0:
            return 0.0
        return inter / float(smaller)

    @staticmethod
    def center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def expand(bbox, padding, width, height):
        x1, y1, x2, y2 = bbox
        return (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding),
        )

    @staticmethod
    def from_points(points, width, height):
        try:
            xs = points[:, 0]
            ys = points[:, 1]
            bbox = (
                int(math.floor(float(xs.min()))),
                int(math.floor(float(ys.min()))),
                int(math.ceil(float(xs.max()))),
                int(math.ceil(float(ys.max()))),
            )
            return BBoxUtils.clip(bbox, width, height)
        except Exception:
            return None

    @staticmethod
    def pdf_rect_from_pixels(bbox, image_width, image_height, page_rect):
        x1, y1, x2, y2 = bbox
        sx = page_rect.width / float(image_width)
        sy = page_rect.height / float(image_height)
        return fitz.Rect(
            page_rect.x0 + x1 * sx,
            page_rect.y0 + y1 * sy,
            page_rect.x0 + x2 * sx,
            page_rect.y0 + y2 * sy,
        )