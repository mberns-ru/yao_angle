import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
from src.angle_analysis import *

class AnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Run Analysis Tool")
        self.root.geometry("500x300")
        
        # Initialize selected folder variable
        self.data_folder = tk.StringVar()
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Label
        label = tk.Label(self.root, text="Analyze Pre- and Post-HL Angle Data", font=("Helvetica", 16))
        label.pack(pady=10)
        
        # Folder selection
        folder_label = tk.Label(self.root, text="Select Data Folder:")
        folder_label.pack(pady=(10, 5))
        
        # Entry box to show selected folder
        folder_entry = tk.Entry(self.root, textvariable=self.data_folder, width=50, state="readonly")
        folder_entry.pack()
        
        # Browse button
        browse_button = tk.Button(self.root, text="Browse", command=self.browse_folder)
        browse_button.pack(pady=(5, 15))
        
        # Run button
        self.run_button = tk.Button(self.root, text="Run Analysis", command=self.run_analysis, bg="green", fg="white")
        self.run_button.pack(pady=10)
        self.run_button.config(state=tk.DISABLED)
        
        # Output text area
        output_label = tk.Label(self.root, text="Output:")
        output_label.pack(pady=(10, 5))
        
        self.output_text = tk.Text(self.root, height=8, width=60, wrap=tk.WORD, state="disabled")
        self.output_text.pack()
    
    def browse_folder(self):
        """ Open file dialog to select the data directory """
        folder_path = filedialog.askdirectory(title="Select Data Folder")
        if folder_path:
            self.data_folder.set(folder_path)
            self.run_button.config(state=tk.NORMAL)
        else:
            self.data_folder.set("")
            self.run_button.config(state=tk.DISABLED)
    
    def run_analysis(self):
        """ Run the analysis script """
        data_folder = self.data_folder.get()
        if not data_folder:
            messagebox.showwarning("Input Error", "Please select a data folder.")
            return
        
        # Disable GUI elements during execution
        self.disable_widgets()
        self.log_output("Running analysis...\n")
        
        # Run the analysis in a separate thread
        thread = threading.Thread(target=self.execute_analysis, args=(data_folder,))
        thread.start()
    
    def execute_analysis(self, folder_path):
        """ Execute the analysis and handle output or errors """
        try:
            self.log_output("Starting analysis...\n")
            run_analysis(folder_path)
            self.log_output("Analysis complete!\n")
            messagebox.showinfo("Success", "Analysis completed successfully!")
        except Exception as e:
            self.log_output(f"Error: {str(e)}\n")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.enable_widgets()
    
    def log_output(self, message):
        """ Display output in the text area """
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)
        self.output_text.config(state="disabled")
    
    def disable_widgets(self):
        """ Disable widgets during execution """
        self.root.update()
        for widget in self.root.winfo_children():
            widget.configure(state="disabled")
    
    def enable_widgets(self):
        """ Re-enable widgets after execution """
        self.root.update()
        for widget in self.root.winfo_children():
            widget.configure(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()
