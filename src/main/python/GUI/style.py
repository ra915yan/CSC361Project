# src/ui/styles.py
from tkinter import ttk

def apply_styles():
    style = ttk.Style()
    style.theme_use("clam")
    
    # Define your "CSS Classes"
    style.configure("Primary.TButton", 
                    background="#4f46e5", 
                    foreground="white", 
                    font=("Segoe UI", 10, "bold"))
    
    style.configure("Header.TLabel", 
                    background="#f4f7fb", 
                    foreground="#1d2939", 
                    font=("Segoe UI", 16, "bold"))