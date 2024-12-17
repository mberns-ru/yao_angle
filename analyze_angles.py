import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading
import os
from src.angle_analysis import *

class AnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Run Analysis Tool")
        self.geometry("500x200")
        
        # Widgets
        self.folder_label = tk.Label(self, text="Selected Folder: None", wraplength=400)
        self.folder_label.pack(pady=10)
        
        self.select_folder_button = tk.Button(self, text="Select Data Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=5)
        
        self.run_button = tk.Button(self, text="Run Analysis", command=self.run_analysis_thread, state=tk.DISABLED)
        self.run_button.pack(pady=10)
        
        self.status_label = tk.Label(self, text="")
        self.status_label.pack(pady=10)
        
        self.selected_folder = None

    def select_folder(self):
        """Opens a folder dialog to select the data folder."""
        folder_path = filedialog.askdirectory(title="Select Data Folder")
        if folder_path:
            self.selected_folder = folder_path
            self.folder_label.config(text=f"Selected Folder: {folder_path}")
            self.run_button.config(state=tk.NORMAL)
        else:
            self.folder_label.config(text="Selected Folder: None")
            self.run_button.config(state=tk.DISABLED)

    def run_analysis_thread(self):
        """Runs the analysis in a separate thread to avoid freezing the GUI."""
        if self.selected_folder:
            self.status_label.config(text="Running analysis... Please wait.")
            self.run_button.config(state=tk.DISABLED)
            thread = threading.Thread(target=self.run_analysis_safe)
            thread.start()

    def run_analysis_safe(self):
        """Runs the analysis and handles completion or errors."""
        try:
            run_analysis(self.selected_folder)
            self.status_label.config(text="Analysis complete!")
            messagebox.showinfo("Success", "Analysis completed successfully!")
        except Exception as e:
            self.status_label.config(text="Error occurred during analysis.")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            self.run_button.config(state=tk.NORMAL)

# Main Function to Run the GUI
if __name__ == "__main__":
    app = AnalysisGUI()
    app.mainloop()