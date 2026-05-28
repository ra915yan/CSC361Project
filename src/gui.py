import os
import random
import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
from SpamNotSpamDataset import SpamNotSpamDataset
import config
import joblib

class SimpleSpamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Detector Engine")
        self.root.geometry("850x650")
        
        # Initialize the dataset instance safely
        try:
            self.dataset = SpamNotSpamDataset()
        except Exception as e:
            print(f"Failed to load dataset on startup: {e}")
            self.dataset = None
            
        # 2. Load the ML Model and Vectorizer safely from config paths
        try:
            self.vectorizer = joblib.load(config.VECTORIZER_PATH)
            self.model = joblib.load(config.MODEL_PATH)
            self.model_loaded = True
        except Exception as e:
            print(f"Failed to load ML artifacts: {e}")
            self.model_loaded = False
            self.vectorizer = None
            self.model = None

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
        left_button_frame = tk.Frame(self.tab_model)
        left_button_frame.pack(side="left", fill="y", padx=10, pady=10)

        random_frame = tk.Frame(left_button_frame)
        random_frame.pack(side="top", fill="x", pady=5)

        self.btn_random = tk.Button(random_frame, text="Random Sample", width=15, command=self.get_random_sample)
        self.btn_random.pack(side="left")
        
        self.lbl_isSpam = tk.Label(random_frame, text="[Ground Truth]", fg="blue", font=("Arial", 10, "bold"))
        self.lbl_isSpam.pack(side="left", padx=10)

        self.btn_predict = tk.Button(left_button_frame, text="Predict / Run Model", width=30, command=self.predict_text)
        self.btn_predict.pack(side="top", pady=5)
        
        self.lbl_result = tk.Label(left_button_frame, text="Prediction: None", font=("Arial", 11, "bold"))
        self.lbl_result.pack(side="top", pady=20)

        self.text_input = tk.Text(self.tab_model, wrap="word")
        self.text_input.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def setup_plot_tab(self):
        """Builds a pure viewing layout with a dropdown selector for existing images."""
        # Top control bar to hold the dropdown picker selection mechanics
        top_bar = tk.Frame(self.tab_plot)
        top_bar.pack(side="top", fill="x", padx=10, pady=10)

        lbl_select = tk.Label(top_bar, text="Select Dataset Plot to View:", font=("Arial", 10, "bold"))
        lbl_select.pack(side="left", padx=5)

        # Dropdown Combobox Menu mapping options directly to your data images
        self.plot_options = {
            "Spam vs Not Spam Distribution": "spam_distribution.png",
            "Top Frequent Words (Spam)": "top_words_spam.png",
            "Top Frequent Words (Ham)": "top_words_ham.png"
        }

        self.dropdown_view = ttk.Combobox(
            top_bar, 
            values=list(self.plot_options.keys()), 
            state="readonly", 
            width=30
        )
        self.dropdown_view.set("Spam vs Not Spam Distribution") # Initialize Default view
        self.dropdown_view.pack(side="left", padx=10)

        # Automatically update the image container whenever a new selection is highlighted
        self.dropdown_view.bind("<<ComboboxSelected>>", lambda event: self.render_saved_plot())

        # Main display frame area for rendering the graphic matrix lines
        self.image_container = tk.Label(self.tab_plot, text="[ Image Asset Loading... ]")
        self.image_container.pack(fill="both", expand=True, pady=10)

        # Fire once initially right after rendering finishes to show the default option image
        self.root.after(150, self.render_saved_plot)

    # ==========================================
    # LOGIC COMPONENT FUNCTIONALITIES
    # ==========================================
    def get_random_sample(self):
        """Pulls a random string record and sets both text field and truth label."""
        if self.dataset is None:
            self.text_input.delete("1.0", tk.END)
            self.text_input.insert("1.0", "ERROR: Dataset not loaded.")
            self.lbl_isSpam.config(text="Label: Error", fg="red")
            return

        try:
            sample_row = self.dataset.get_random_sample()
            email_text = sample_row['email'] 
            true_label = str(sample_row['label']).strip()
            
            if true_label == "1" or "spam" in true_label.lower():
                label_display = "(SPAM)"
                label_color = "red"
            else:
                label_display = "(Not SPAM)"
                label_color = "green"
            
        except Exception as e:
            email_text = f"ERROR: Failed to load data row: {e}"
            label_display = "(ERROR)"
            label_color = "red"

        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", email_text)
        self.lbl_isSpam.config(text=label_display, fg=label_color)

    def predict_text(self):
        """Reads input text string, vectorizes it, and runs the loaded model."""
        if not self.model_loaded or self.model is None or self.vectorizer is None:
            messagebox.showerror("Model Error", "Model or Vectorizer artifacts are not loaded.")
            return

        raw_string = self.text_input.get("1.0", tk.END).strip()
        if not raw_string:
            messagebox.showwarning("Empty String", "Please input text or generate a sample first.")
            return

        try:
            vectorized_text = self.vectorizer.transform([raw_string])
            prediction = self.model.predict(vectorized_text)[0]
            
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(vectorized_text)[0]
                confidence = probabilities[prediction] * 100
                confidence_str = f" ({confidence:.1f}% Confidence)"
            else:
                confidence_str = ""

            if str(prediction) == "1" or (isinstance(prediction, str) and "spam" in prediction.lower()):
                self.lbl_result.config(text=f"Prediction: SPAM (1){confidence_str}", fg="red")
            else:
                self.lbl_result.config(text=f"Prediction: SAFE (0){confidence_str}", fg="green")
                
        except Exception as e:
            messagebox.showerror("Inference Error", f"Failed to execute model prediction:\n{e}")

    def render_saved_plot(self):
        """Reads the dropdown selection and pulls the image out of the data directory."""
        selected_view = self.dropdown_view.get()
        target_filename = self.plot_options.get(selected_view)

        if not target_filename:
            return

        target_img_path = os.path.join(config.DATA_DIR, target_filename)
        
        if not os.path.exists(target_img_path):
            self.image_container.config(
                image="",
                text=f"Asset Missing: Could not find image target file at:\n'{target_img_path}'\n\n"
                     f"Please make sure you have generated the plots in your dataset analysis scripts first!"
            )
            return

        try:
            # 1. Load the raw image binary
            raw_img = tk.PhotoImage(file=target_img_path)
            
            # 2. Downsample the image dynamically by a factor of 2 to make it fit the screen
            # (e.g., if it's 1400x1000, it becomes a crisp 700x500 window size)
            self.plot_img = raw_img.subsample(2, 2)
            
            # 3. Bind the scaled asset to your display container label
            self.image_container.config(image=self.plot_img, text="")
            
        except Exception as e:
            self.image_container.config(image="", text=f"Rendering Failure: Failed to decode image file.\n{e}")
            
            
            
if __name__ == "__main__":
    app_root = tk.Tk()
    app = SimpleSpamGUI(app_root)
    app_root.mainloop()