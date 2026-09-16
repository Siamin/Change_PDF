# include/page_manager.py
import fitz
from .input_utils import InputUtils


class PDFPageManager:

    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.deleted_indexes = []

    def parse_ranges(self, text: str):
        total = len(self.doc)
        result = set()
        for part in text.replace(",", " ").split():
            if "-" in part:
                try:
                    a, b = part.split("-", 1)
                    a, b = int(a), int(b)
                    if a > b:
                        a, b = b, a
                    for n in range(a, b + 1):
                        if 1 <= n <= total:
                            result.add(n - 1)
                        else:
                            print(f"Page {n} outside range 1-{total}")
                except ValueError:
                    print(f"Invalid range: {part}")
            else:
                try:
                    n = int(part)
                    if 1 <= n <= total:
                        result.add(n - 1)
                    else:
                        print(f"Page {n} outside range 1-{total}")
                except ValueError:
                    print(f"Invalid page: {part}")
        return sorted(result)

    def delete(self, indexes):
        if not indexes:
            return 0
        count = 0
        for idx in sorted(indexes, reverse=True):
            try:
                print(f"Deleting page {idx + 1}")
                self.doc.delete_page(idx)
                self.deleted_indexes.append(idx)
                count += 1
            except Exception as exc:
                print(f"  Failed: {exc}")
        return count

    def interactively_delete(self):
        if not InputUtils.ask_yes_no("Delete any pages?"):
            return 0
        try:
            raw = input("Page numbers (e.g. 2 5 8-12): ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        indexes = self.parse_ranges(raw)
        if indexes:
            print("Selected:", [i + 1 for i in indexes])
            return self.delete(indexes)
        print("No valid pages selected.")
        return 0