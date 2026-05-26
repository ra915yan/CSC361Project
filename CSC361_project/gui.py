import os
import tkinter as tk
from tkinter import messagebox, ttk

import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_detector_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Email Detector")
        self.root.geometry("760x560")
        self.root.minsize(620, 480)
        self.root.configure(bg="#f4f7fb")

        self.model = None
        self.vectorizer = None

        self._configure_styles()
        self._build_ui()
        self._load_artifacts()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f7fb")
        style.configure("Header.TLabel", background="#f4f7fb", foreground="#1d2939", font=("Segoe UI", 22, "bold"))
        style.configure("Body.TLabel", background="#f4f7fb", foreground="#475467", font=("Segoe UI", 11))
        style.configure("Result.TLabel", background="#ffffff", foreground="#1d2939", font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#667085", font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 9))
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(12, 8))

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=24)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Spam Email Detector", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            main,
            text="Paste an email message below, then check whether the trained model classifies it as spam.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(6, 18))

        input_frame = tk.Frame(main, bg="#ffffff", bd=1, relief="solid", highlightthickness=0)
        input_frame.pack(fill="both", expand=True)

        self.email_text = tk.Text(
            input_frame,
            wrap="word",
            height=14,
            bg="#ffffff",
            fg="#101828",
            insertbackground="#101828",
            relief="flat",
            font=("Segoe UI", 11),
            padx=14,
            pady=14,
        )
        self.email_text.pack(fill="both", expand=True)

        button_row = ttk.Frame(main)
        button_row.pack(fill="x", pady=16)

        ttk.Button(button_row, text="Check Email", style="Primary.TButton", command=self.check_email).pack(side="left")
        ttk.Button(button_row, text="Clear", style="Secondary.TButton", command=self.clear_text).pack(side="left", padx=(10, 0))

        result_frame = tk.Frame(main, bg="#ffffff", bd=1, relief="solid")
        result_frame.pack(fill="x")

        self.result_label = tk.Label(
            result_frame,
            text="Result will appear here",
            bg="#ffffff",
            fg="#1d2939",
            font=("Segoe UI", 18, "bold"),
            anchor="w",
        )
        self.result_label.pack(anchor="w", padx=16, pady=(14, 4))

        self.detail_label = tk.Label(
            result_frame,
            text="Model files loaded from trained_models.",
            bg="#ffffff",
            fg="#667085",
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.detail_label.pack(anchor="w", padx=16, pady=(0, 14))

    def _load_artifacts(self):
        try:
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.model = joblib.load(MODEL_PATH)
        except FileNotFoundError as exc:
            messagebox.showerror(
                "Missing model files",
                f"Could not find required file:\n{exc.filename}\n\nRun test.py first to create the trained_models files.",
            )
            self.detail_label.config(text="Model files are missing.")
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not load the trained model:\n{exc}")
            self.detail_label.config(text="The trained model could not be loaded.")

    def check_email(self):
        email = self.email_text.get("1.0", "end").strip()

        if not email:
            messagebox.showwarning("No email text", "Please paste or type an email message first.")
            return

        if self.model is None or self.vectorizer is None:
            messagebox.showerror("Model unavailable", "The model is not loaded. Check the trained_models folder.")
            return

        try:
            email_vector = self.vectorizer.transform([email])
            if email_vector.nnz == 0:
                self.result_label.config(text="Not Spam", fg="#027a48")
                self.detail_label.config(text="This text is too short/simple for a confident model score.")
                return

            prediction = self.model.predict(email_vector)[0]
        except Exception as exc:
            messagebox.showerror("Prediction error", f"Could not check this email:\n{exc}")
            return

        probability_text = ""
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(email_vector)[0]
            classes = list(self.model.classes_)
            predicted_index = classes.index(prediction)
            confidence = probabilities[predicted_index] * 100
            probability_text = f"Confidence: {confidence:.1f}%"

        prediction_text = str(prediction).strip().lower()
        is_spam = prediction_text in {"1", "spam"}

        if is_spam:
            self.result_label.config(text="Spam", fg="#b42318")
            self.detail_label.config(text=probability_text or "The message looks like spam.")
        else:
            self.result_label.config(text="Not Spam", fg="#027a48")
            self.detail_label.config(text=probability_text or "The message looks safe.")

    def clear_text(self):
        self.email_text.delete("1.0", "end")
        self.result_label.config(text="Result will appear here", fg="#1d2939")
        self.detail_label.config(text="Paste another email to check it.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()
