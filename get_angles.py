import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import threading

class AngleDataGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Generate Angle Data")
        self.root.geometry("500x300")
        
        # Initialize selected path variable
        self.data_path = tk.StringVar()
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Label
        label = tk.Label(self.root, text="Generate Angle Data CSV", font=("Helvetica", 16))
        label.pack(pady=10)
        
        # Path selection
        path_label = tk.Label(self.root, text="Select Data Path:")
        path_label.pack(pady=(10, 5))
        
        # Entry box to show selected path
        path_entry = tk.Entry(self.root, textvariable=self.data_path, width=50, state="readonly")
        path_entry.pack()
        
        # Browse button
        browse_button = tk.Button(self.root, text="Browse", command=self.browse_data_path)
        browse_button.pack(pady=(5, 15))
        
        # Run button
        run_button = tk.Button(self.root, text="Run Script", command=self.run_script, bg="green", fg="white")
        run_button.pack(pady=10)
        
        # Output text area
        output_label = tk.Label(self.root, text="Output:")
        output_label.pack(pady=(10, 5))
        
        self.output_text = tk.Text(self.root, height=8, width=60, wrap=tk.WORD, state="disabled")
        self.output_text.pack()
    
    def browse_data_path(self):
        """ Open file dialog to select the data directory """
        path = filedialog.askdirectory(title="Select Data Directory")
        if path:
            self.data_path.set(path)
    
    def run_script(self):
        """ Run the generate_angle_speed_data.py script """
        data_path = self.data_path.get()
        if not data_path:
            messagebox.showwarning("Input Error", "Please select a data path.")
            return
        
        # Disable GUI elements during execution
        self.disable_widgets()
        self.log_output("Running script...\n")
        
        # Run the script in a separate thread
        thread = threading.Thread(target=self.execute_script, args=(data_path,))
        thread.start()
    
    def execute_script(self, data_path):
        """ Execute the script and capture the output """
        try:
            # Path to the external script in the src subfolder
            script_path = os.path.join("src", "generate_angle_speed_data.py")
            
            # Check if the script exists
            if not os.path.isfile(script_path):
                raise FileNotFoundError(f"Script not found: {script_path}")
            
            # Run the script using subprocess
            process = subprocess.Popen(
                ["python", script_path, "--data_path", data_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Log the output and errors
            for line in process.stdout:
                self.log_output(line)
            for line in process.stderr:
                self.log_output(line)
            
            # Wait for the process to complete
            process.wait()
            if process.returncode == 0:
                self.log_output("Script finished successfully.\n")
                messagebox.showinfo("Success", "Angle data generation complete!")
            else:
                self.log_output("Error: Script did not complete successfully.\n")
                messagebox.showerror("Error", "An error occurred during execution.")
        
        except Exception as e:
            self.log_output(f"Exception: {str(e)}\n")
            messagebox.showerror("Error", f"An exception occurred: {str(e)}")
        finally:
            self.enable_widgets()
    
    def log_output(self, message):
        """ Display output in the text area """
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)
        self.output_text.config(state="disabled")
    
    def disable_widgets(self):
        """ Disable widgets during script execution """
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
    app = AngleDataGUI(root)
    root.mainloop()