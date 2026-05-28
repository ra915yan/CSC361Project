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
        self.root.geometry("700x500")
        
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
        # Left Side Sub-Container for Buttons
        left_button_frame = tk.Frame(self.tab_model)
        left_button_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Container row inside the left panel to hold the button and label side-by-side
        random_frame = tk.Frame(left_button_frame)
        random_frame.pack(side="top", fill="x", pady=5)

        self.btn_random = tk.Button(random_frame, text="Random Sample", width=15, command=self.get_random_sample)
        self.btn_random.pack(side="left")
        
        # The true indicator label placed directly beside the random button
        self.lbl_isSpam = tk.Label(random_frame, text="[Ground Truth]", fg="blue", font=("Arial", 10, "bold"))
        self.lbl_isSpam.pack(side="left", padx=10)

        # Other action buttons inside the left panel
        self.btn_predict = tk.Button(left_button_frame, text="Predict / Run Model", width=30, command=self.predict_text)
        self.btn_predict.pack(side="top", pady=5)
        
        self.lbl_result = tk.Label(left_button_frame, text="Prediction: None", font=("Arial", 11, "bold"))
        self.lbl_result.pack(side="top", pady=20)

        # Right Side Big Text Area
        self.text_input = tk.Text(self.tab_model, wrap="word")
        self.text_input.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def setup_plot_tab(self):
        """Loads and draws the pre-generated image artifact on tab activation."""
        btn_load_plot = tk.Button(self.tab_plot, text="Render Chart Image", command=self.render_saved_plot)
        btn_load_plot.pack(anchor="n", pady=5)

        self.image_container = tk.Label(self.tab_plot, text="[ Image Asset Not Loaded Yet ]")
        self.image_container.pack(fill="both", expand=True, pady=10)

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
            # 1. Fetch the raw series from the dataset class
            sample_row = self.dataset.get_random_sample()
            
            # 2. Extract values mapped directly to your CSV headers ('email', 'label')
            email_text = sample_row['email'] 
            true_label = str(sample_row['label']).strip()
            
            # 3. Format the status label display text (Handles both 1/0 and Text classifications)
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

        # Update text input UI component
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", email_text)
        
        # Update ground truth classification indicator label beside button
        self.lbl_isSpam.config(text=label_display, fg=label_color)

    def predict_text(self):
        """Reads input text string, vectorizes it, and runs the loaded model."""
        # 1. Structural Guard: Ensure model artifacts exist
        if not self.model_loaded or self.model is None or self.vectorizer is None:
            messagebox.showerror("Model Error", "Model or Vectorizer artifacts are not loaded.")
            return

        # 2. Read input from the text widget
        raw_string = self.text_input.get("1.0", tk.END).strip()
        if not raw_string:
            messagebox.showwarning("Empty String", "Please input text or generate a sample first.")
            return

        try:
            # 3. Transform the raw text into numerical features
            # Input must be inside a list/array structure: [raw_string]
            vectorized_text = self.vectorizer.transform([raw_string])
            
            # 4. Generate prediction (returns an array, e.g., [1] or [0])
            prediction = self.model.predict(vectorized_text)[0]
            
            # 5. Check if probabilities are available for extra UI confidence detail
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(vectorized_text)[0]
                confidence = probabilities[prediction] * 100
                confidence_str = f" ({confidence:.1f}% Confidence)"
            else:
                confidence_str = ""

            # 6. Update the UI Label display based on prediction output
            if str(prediction) == "1" or (isinstance(prediction, str) and "spam" in prediction.lower()):
                self.lbl_result.config(
                    text=f"Prediction: SPAM (1){confidence_str}", 
                    fg="red"
                )
            else:
                self.lbl_result.config(
                    text=f"Prediction: SAFE (0){confidence_str}", 
                    fg="green"
                )
                
        except Exception as e:
            messagebox.showerror("Inference Error", f"Failed to execute model prediction:\n{e}")

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
            self.plot_img = tk.PhotoImage(file=target_img)
            self.image_container.config(image=self.plot_img, text="")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to open image element:\n{e}")

if __name__ == "__main__":
    app_root = tk.Tk()
    app = SimpleSpamGUI(app_root)
    app_root.mainloop()