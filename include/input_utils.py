# include/input_utils.py
import os
import sys


class InputUtils:

    @staticmethod
    def ask_yes_no(question: str) -> bool:
        while True:
            try:
                answer = input(f"{question} [y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            if answer in ("y", "yes", "1"):
                return True
            if answer in ("n", "no", "0"):
                return False
            print("Please enter y or n.")

    @staticmethod
    def ask_existing_file(prompt: str) -> str:
        while True:
            try:
                path = input(prompt).strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                print()
                sys.exit(1)
            if not path:
                print("Path is empty.")
                continue
            path = os.path.expanduser(path)
            if os.path.isfile(path):
                return path
            print(f"File not found:\n{path}")

    @staticmethod
    def ask_choice(prompt: str, choices):
        print(prompt)
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        while True:
            try:
                raw = input("Choose: ").strip()
                idx = int(raw) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except (ValueError, EOFError, KeyboardInterrupt):
                pass
            print("Invalid choice.")