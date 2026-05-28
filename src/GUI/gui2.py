import os
import random
import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd

import config


SAMPLE_EMAILS = [
    "Congratulations! You won a cash prize click here to claim.",
    "Hey, are we still meeting up for lunch today at 1 PM?",
    "URGENT: Verify your bank credentials immediately to stop account suspension.",
    "Can you please review the attached machine learning project layout?"
]

class SimpleSpamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Detector Engine")
        self.root.geometry("700x500")

        # Attempt to load raw data for the genuine random sample extraction
        self.dataset = None
        
        if os.path.exists("spam_or_not_spam.csv"):
            try:
                self.dataset = pd.read_csv("spam_or_not_spam.csv").dropna()
            except Exception:
                print("couldn't load the dataset")
                pass

        # Master Notebook setup for structural tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Initialize the Layout Tabs
        self.tab_model = ttk.Frame(self.notebook)
        self.tab_plot = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_model, text="Model Predictor")
        self.notebook.add(self.tab_plot, text="Analytics Plot")

        # Build individual layouts
        self.setup_model_tab()
        self.setup_plot_tab()

    def setup_model_tab(self):
        """Builds a layout with left side button panel and right side text field."""
        # Left Side Sub-Container for Buttons
        left_button_frame = tk.Frame(self.tab_model)
        left_button_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.btn_random = tk.Button(left_button_frame, text="Random Sample", width=15, command=self.get_random_sample)
        self.btn_random.pack(side="top", pady=5)
        

        self.btn_predict = tk.Button(left_button_frame, text="Predict / Run Model", width=15, command=self.predict_text)
        self.btn_predict.pack(side="top", pady=5)
        
        # Bottom area within the button frame to print simple output text
        self.lbl_result = tk.Label(left_button_frame, text="Prediction: None", font=("Arial", 11, "bold"))
        self.lbl_result.pack(side="top", pady=20)

        # Right Side Big Text Area
        self.text_input = tk.Text(self.tab_model, wrap="word")
        self.text_input.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def setup_plot_tab(self):
        """Loads and draws the pre-generated image artifact on tab activation."""
        # Simple refreshing descriptive label button
        btn_load_plot = tk.Button(self.tab_plot, text="Render Chart Image", command=self.render_saved_plot)
        btn_load_plot.pack(anchor="n", pady=5)

        self.image_container = tk.Label(self.tab_plot, text="[ Image Asset Not Loaded Yet ]")
        self.image_container.pack(fill="both", expand=True, pady=10)

    # ==========================================
    # LOGIC COMPONENT FUNCTIONALITIES
    # ==========================================
    def get_random_sample(self):
        """Pulls a random string record from the CSV or fallback list."""
        if self.dataset is not None and not self.dataset.empty:
            random_row = self.dataset.sample(n=1).iloc[0]
            email_text = str(random_row['email'])
        else:
            email_text = random.choice(SAMPLE_EMAILS)

        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", email_text)

    def predict_text(self):
        """Reads input text string and runs placeholder validation logic."""
        raw_string = self.text_input.get("1.0", tk.END).strip()
        if not raw_string:
            messagebox.showwarning("Empty String", "Please input text or generate a sample first.")
            return

        # Simple placeholder word triggers until model artifacts are piped into src/core/
        spam_keywords = ["win", "prize", "cash", "urgent", "click", "claim", "money", "free"]
        is_spam_match = any(word in raw_string.lower() for word in spam_keywords)
        
        if is_spam_match:
            self.lbl_result.config(text="Prediction: SPAM (1)")
        else:
            self.lbl_result.config(text="Prediction: SAFE (0)")

    def render_saved_plot(self):
        """Reads the distribution chart generated during data exploration steps."""
        target_img = "spam_distribution.png"
        
        if not os.path.exists(target_img):
            self.image_container.config(
                text=f"Error: image code couldn't locate '{target_img}' in local directory.\n"
                     f"Please run your training/explore_data.py pipeline script to save it first!"
            )
            return

        try:
            # Using PhotoImage to draw raw image data elements without external PIL library dependency
            self.plot_img = tk.PhotoImage(file=target_img)
            self.image_container.config(image=self.plot_img, text="")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to open image element:\n{e}")

if __name__ == "__main__":
    app_root = tk.Tk()
    app = SimpleSpamGUI(app_root)
    app_root.mainloop()