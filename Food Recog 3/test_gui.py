import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2 
import predict_final as pd 
import nutri_estimate as nes

selected_file_path = ""
method = 0 # 0 = none, 1 = file, 2 = camera
MAX_SIZE = (800, 800) 
tk_image_reference = None 

# HANDLER

def update_tkinter_frame(frame_overlay, max_size, image_label_widget, result_label_widget, prediction_text=""):
    global tk_image_reference
    prediction_text += "\n \n" + nes.getDataByLabel(pd.result)
    result_label_widget.config(text=prediction_text)
    rgb_image = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(rgb_image)
    img.thumbnail(max_size) 
    
    tk_image = ImageTk.PhotoImage(img)
    
    tk_image_reference = tk_image 
    image_label_widget.config(image=tk_image)
    image_label_widget.image = tk_image

def browse_file():
    global selected_file_path, method
    
    pd.stop_cam_thread()
    
    file_label.config(text="File mode selected. Browse file to proceed.")
    
    file_path = filedialog.askopenfilename(
        title="Choose image file",
        filetypes=(("Image Files", "*.jpg;*.png;*.jpeg"),) 
    )
    
    if file_path:
        selected_file_path = file_path
        method = 1 
        
        file_label.config(text=f"Chosen: {selected_file_path}")
        result_label.config(text="Result")
        
        try:
            frame = cv2.imread(file_path)
            resized_cv2 = cv2.resize(frame, (600, 400)) 

            rgb_img = cv2.cvtColor(resized_cv2, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_img)
            img.thumbnail(MAX_SIZE)
            
            tk_image = ImageTk.PhotoImage(img)
            
            image_display_label.config(image=tk_image)
            image_display_label.image = tk_image
            
        except Exception as e:
            image_display_label.config(image='')
            image_display_label.config(text="Error loading image.")
            messagebox.showerror("Image Error", f"Could not load image: {e}")
            
    else:
        if method == 1:
            method = 0
            selected_file_path = ""
            file_label.config(text="Choose input method.")
            image_display_label.config(image='')
            image_display_label.config(text="")
            result_label.config(text="Result")

def choose_cam():
    global method
    method = 2
    file_label.config(text="Camera mode selected. Processing...")
    
    pd.start_cam_thread(root, update_tkinter_frame, MAX_SIZE, image_display_label, file_label, result_label)

def predict_handler():
    global method
    
    if method == 1:
        pd.predict(selected_file_path, update_tkinter_frame, MAX_SIZE, image_display_label, result_label)
    elif method == 2:
        pd.stop_cam_thread()
    else: 
        messagebox.showwarning("Warning", "Please choose an input method (File or Camera) to proceed.")

# GUI

root = tk.Tk()
root.title("Food Recognition and Calories Estimation Program")
root.geometry("1200x600") 

main_frame = ttk.Frame(root, padding="10 10 10 10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

main_frame.columnconfigure(0, weight=3)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=0) 
main_frame.rowconfigure(1, weight=1) 
main_frame.rowconfigure(2, weight=1)
main_frame.rowconfigure(3, weight=1) 
main_frame.rowconfigure(4, weight=3) 

# Row 0
file_label = ttk.Label(main_frame, text="Choose input method.", wraplength=850) 
file_label.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)

# Col 0
image_frame = ttk.Frame(main_frame, padding="5", relief="groove")
image_frame.grid(row=1, column=0, rowspan=4, padx=5, pady=5, sticky=tk.N+tk.S+tk.W+tk.E) 

image_display_label = ttk.Label(image_frame, background="gray", anchor="center") 
image_display_label.pack(expand=True, fill='both')

# Col 1 
# Row 1 
btn_file = ttk.Button(main_frame, text="Upload Image", command=browse_file)
btn_file.grid(row=1, column=1, padx=5, pady=5, sticky=tk.N+tk.S+tk.W+tk.E)

# Row 2 
btn_camera = ttk.Button(main_frame, text="Start Camera", command=choose_cam)
btn_camera.grid(row=2, column=1, padx=5, pady=5, sticky=tk.N+tk.S+tk.W+tk.E)

# Row 3
btn_recog = ttk.Button(main_frame, text="Start / Stop Recognition", command=predict_handler)
btn_recog.grid(row=3, column=1, padx=5, pady=15, sticky=tk.N+tk.S+tk.W+tk.E)

# Row 4
result_label = ttk.Label(main_frame, text="Result", wraplength=200, justify=tk.LEFT, foreground="blue")
result_label.grid(row=4, column=1, padx=5, pady=5, sticky=tk.N+tk.W+tk.E)


root.mainloop()