import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import json
import threading
import time
import grad_cam as gc

model = load_model(r"Food Recog Real Final\Food Recog 3\model\food_model_final.keras")
labels = json.load(open(r"Food Recog Real Final\Food Recog 3\model\labels.json"))
label_map = {i: label for i, label in enumerate(labels)}
stop_thread = False 
result =""


print("Model loaded")


def predict(img_path, update_func, max_size, image_label, result_label):
    global result
    orig = cv2.imread(img_path)
    orig_h, orig_w = orig.shape[:2]
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    resized_for_model = cv2.resize(rgb, (224, 224))

    img = resized_for_model.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]
    idx = np.argmax(pred)
    label = label_map[idx]
    conf = pred[idx] * 10
    
    prediction_text = f"Result: {label}\nConfidence: {conf:.2f}%"
    result = label
    print("\n===== PREDICTION =====")
    for i in np.argsort(pred)[::-1][:3]:
        print(f"{label_map[i]:20s}: {pred[i]*100:.2f}%")
    
    # Grad-CAM và Bounding Box
    cam = gc.grad_cam(model, img, gc.gradcam_layer)
    cam = cv2.resize(cam, (orig.shape[1], orig.shape[0])) 
    heat = (cam*255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    threshold = 0.5
    mask = cam > threshold
    mask = mask.astype(np.uint8)
    overlay = orig.copy()
    overlay[mask==1] = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)[mask==1]

    thresh = (cam>0.5).astype(np.uint8)*255
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        scale_x = 1280 / orig_w
        scale_y = 720 / orig_h
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        overlay = cv2.resize(overlay, (1280, 720))
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(overlay, f"{label} ({conf:.2f}%)", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    update_func(overlay, max_size, image_label, result_label, prediction_text)


def _predict_cam_loop(root_tk, update_func, max_size, image_label, status_label, result_label):
    global stop_thread, result
    if model is None:
        root_tk.after(100, lambda: status_label.config(text="Model is not loaded. Camera stopped."))
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        root_tk.after(100, lambda: status_label.config(text="ERROR: Could not open camera."))
        return

    while not stop_thread:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_for_model = cv2.resize(rgb, (224, 224))
        img = resized_for_model.astype(np.float32)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0]
        idx = np.argmax(pred)
        label = label_map[idx]
        conf = pred[idx] * 100
        
        prediction_text = f"Result: {label}\nConfidence: {conf:.2f}%"
        result = label
        print("\n===== PREDICTION =====")
        for i in np.argsort(pred)[::-1][:3]:
            print(f"{label_map[i]:20s}: {pred[i]*100:.2f}%")

        # Grad-CAM, Bounding Box
        cam = gc.grad_cam(model, img, gc.gradcam_layer)
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        heat = (cam*255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        threshold = 0.5
        mask = cam > threshold
        mask = mask.astype(np.uint8)
        overlay = frame.copy()
        overlay[mask==1] = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)[mask==1]

        thresh = (cam>0.5).astype(np.uint8)*255
        contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(overlay, f"{label} ({conf:.2f}%)", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        # UPDATE TKINTER
        root_tk.after(0, update_func, overlay, max_size, image_label, result_label, prediction_text)
        time.sleep(0.01) 

    cap.release()
    cv2.destroyAllWindows()
    root_tk.after(100, lambda: status_label.config(text="Camera stopped. Choose input method."))
    root_tk.after(100, lambda: result_label.config(text="Result"))


def start_cam_thread(root_tk, update_func, max_size, image_label, status_label, result_label):
    global stop_thread
    stop_thread = False
    thread = threading.Thread(target=_predict_cam_loop, args=(root_tk, update_func, max_size, image_label, status_label, result_label))
    thread.daemon = True
    thread.start()

def stop_cam_thread():
    global stop_thread
    stop_thread = True