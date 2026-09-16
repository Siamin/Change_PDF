# include/text_manager.py
import fitz
from .input_utils import InputUtils


class PDFTextManager:

    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.report = {"found": 0, "deleted": 0, "replaced": 0}

    def collect_texts(self):
        print()
        print("=" * 60)
        print("TEXTS TO DELETE")
        print("=" * 60)
        print("Enter one string per line. Empty line to finish.")
        texts = []
        while True:
            try:
                val = input("Text: ")
            except (EOFError, KeyboardInterrupt):
                break
            if not val:
                break
            texts.append(val)
        return texts

    def collect_replacements(self, texts):
        print()
        print("=" * 60)
        print("TEXT REPLACEMENTS")
        print("=" * 60)
        print("Empty line = delete without replacement.")
        repl = {}
        for t in texts:
            print(f'Original: "{t}"')
            try:
                r = input("Replacement: ")
            except (EOFError, KeyboardInterrupt):
                r = ""
            repl[t] = r
        return repl

    def process(self, texts, replacements):
        targets = []

        print()
        print("=" * 60)
        print("TEXT PROCESSING")
        print("=" * 60)

        for page_index in range(len(self.doc)):
            try:
                page = self.doc[page_index]
                for text in texts:
                    try:
                        rects = page.search_for(text)
                    except Exception:
                        rects = []
                    if not rects:
                        continue
                    print(
                        f'Page {page_index + 1}: "{text}" '
                        f'-> {len(rects)} occurrence(s)'
                    )
                    self.report["found"] += len(rects)
                    replacement = replacements.get(text, "")
                    for r in rects:
                        targets.append({
                            "page": page_index,
                            "rect": fitz.Rect(r),
                            "replacement": replacement,
                        })
            except Exception as exc:
                print(f"Page {page_index + 1} error: {exc}")

        if not targets:
            return self.report

        print()
        print("Applying redactions...")
        by_page = {}
        for t in targets:
            by_page.setdefault(t["page"], []).append(t["rect"])

        for page_index, rects in by_page.items():
            try:
                page = self.doc[page_index]
                for r in rects:
                    page.add_redact_annot(r, fill=(1, 1, 1))
                    self.report["deleted"] += 1
                page.apply_redactions()
            except Exception as exc:
                print(f"Redaction error page {page_index + 1}: {exc}")

        print(f"Deleted: {self.report['deleted']}")

        for t in targets:
            if not t["replacement"]:
                continue
            try:
                page = self.doc[t["page"]]
                rect = t["rect"]
                fontsize = max(6, min(72, rect.height * 0.85))
                result = page.insert_textbox(
                    rect,
                    t["replacement"],
                    fontname="helv",
                    fontsize=fontsize,
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                    overlay=True,
                )
                if result >= 0:
                    self.report["replaced"] += 1
            except Exception as exc:
                print(f"Replacement error page {t['page'] + 1}: {exc}")

        print(f"Replaced: {self.report['replaced']}")
        return self.report

    def interactively_process(self):
        if not InputUtils.ask_yes_no("Delete any text?"):
            return self.report
        texts = self.collect_texts()
        if not texts:
            return self.report
        if InputUtils.ask_yes_no("Replace any text?"):
            replacements = self.collect_replacements(texts)
        else:
            replacements = {t: "" for t in texts}
        return self.process(texts, replacements)