#python3
# -*- coding: utf-8 -*-

"""
============================================================
PDF EDITOR  -  MAIN CLASS
============================================================

Install:
    pip install pymupdf opencv-python numpy

Run:
    python pdf_editor.py

Directory layout:
    pdf_editor.py
    include/
        __init__.py
        constants.py
        bbox_utils.py
        input_utils.py
        page_manager.py
        text_manager.py
        detection_engine.py
        reconstruction_engine.py
        image_manager.py
"""

import os
import sys
import shutil
from pathlib import Path

import fitz

# --- import from include package ---
from include import (
    InputUtils,
    PDFPageManager,
    PDFTextManager,
    PDFImageManager,
)
from include.constants import OUTPUT_SUFFIX, CREATE_DEBUG_IMAGES


# ============================================================
# MAIN CLASS
# ============================================================

class PDFEditor:
    """
    Main orchestrator.

    Uses:
        PDFPageManager   -> delete pages
        PDFTextManager   -> delete / replace text
        PDFImageManager  -> detect / delete / replace images and logos
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_path_obj = Path(pdf_path)

        self.output_path = str(
            self.pdf_path_obj.with_name(
                self.pdf_path_obj.stem + OUTPUT_SUFFIX + ".pdf"
            )
        )
        self.debug_dir = str(
            self.pdf_path_obj.with_name(
                self.pdf_path_obj.stem + "_debug"
            )
        )

        try:
            self.doc = fitz.open(pdf_path)
        except Exception as exc:
            print(f"Cannot open PDF: {exc}")
            sys.exit(1)

        self.original_page_count = len(self.doc)

    # ---------- debug ----------
    def _reset_debug_dir(self):
        try:
            if os.path.isdir(self.debug_dir):
                shutil.rmtree(self.debug_dir)
            os.makedirs(self.debug_dir, exist_ok=True)
            return True
        except Exception as exc:
            print(f"Debug dir error: {exc}")
            return False

    # ---------- run ----------
    def run(self):
        print()
        print("=" * 70)
        print("PDF EDITOR")
        print("=" * 70)
        print(f"Pages: {self.original_page_count}")

        # 1) Pages
        page_manager = PDFPageManager(self.doc)
        deleted_pages_count = page_manager.interactively_delete()

        # 2) Text
        text_manager = PDFTextManager(self.doc)
        text_report = text_manager.interactively_process()

        # 3) Images / logos
        image_manager = PDFImageManager(self.doc)
        if CREATE_DEBUG_IMAGES:
            self._reset_debug_dir()
        image_report = image_manager.interactively_process(self.debug_dir)

        # 4) Capture final page count BEFORE saving/closing
        final_page_count = len(self.doc)

        # 5) Save
        self._save()

        # 6) Report
        self._print_report(
            deleted_pages_count,
            final_page_count,
            text_report,
            image_report,
        )

    def _save(self):
        print()
        print("=" * 70)
        print("SAVING")
        print("=" * 70)
        print(f"Output: {self.output_path}")

        try:
            self.doc.save(
                self.output_path,
                garbage=4,
                deflate=True,
                clean=True,
            )
        except Exception as exc:
            print(f"Save error: {exc}")
            try:
                self.doc.close()
            except Exception:
                pass
            sys.exit(1)

        # Close after successful save
        try:
            self.doc.close()
        except Exception:
            pass

    def _print_report(
        self,
        deleted_pages,
        final_pages,
        text_report,
        image_report,
    ):
        print()
        print("=" * 70)
        print("FINAL REPORT")
        print("=" * 70)

        print()
        print("PAGES")
        print(f"  Original: {self.original_page_count}")
        print(f"  Final:    {final_pages}")
        print(f"  Deleted:  {deleted_pages}")

        print()
        print("TEXT")
        print(f"  Found:      {text_report.get('found', 0)}")
        print(f"  Deleted:    {text_report.get('deleted', 0)}")
        print(f"  Replaced:   {text_report.get('replaced', 0)}")

        print()
        print("IMAGE")
        print(f"  Pages scanned:     {image_report.get('pages_scanned', 0)}")
        print(f"  Pages with match:  {image_report.get('pages_with_matches', 0)}")
        print(f"  Confirmed matches: {image_report.get('matches', 0)}")
        print(f"  Removed:           {image_report.get('removed', 0)}")
        print(f"  Reconstructed:     {image_report.get('background_reconstructed', 0)}")
        print(f"  Vector regions:    {image_report.get('vector_regions_reconstructed', 0)}")
        print(f"  Replaced:          {image_report.get('replaced', 0)}")
        print(f"  Image objects deleted: {image_report.get('underlying_images_deleted', 0)}")
        print(f"  Debug images:      {image_report.get('debug_images', 0)}")

        print()
        print("OUTPUT")
        print(f"  {self.output_path}")
        if CREATE_DEBUG_IMAGES and os.path.isdir(self.debug_dir):
            print(f"  Debug: {self.debug_dir}")

        print()
        print("=" * 70)
        print("DONE")
        print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    print()
    print("=" * 70)
    print("AUTOMATIC PDF / IMAGE / TEXT EDITOR")
    print("=" * 70)
    print()

    pdf_path = InputUtils.ask_existing_file("Main PDF path: ")
    editor = PDFEditor(pdf_path)
    editor.run()


if __name__ == "__main__":
    main()