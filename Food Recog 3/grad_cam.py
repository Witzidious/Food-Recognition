import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import json

model = load_model(r"Food Recog Real Final\Food Recog 3\model\food_model_final.keras")
labels = json.load(open(r"Food Recog Real Final\Food Recog 3\model\labels.json"))
label_map = {i: label for i, label in enumerate(labels)}

print("Model loaded")

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print("Using Grad-CAM layer:", layer.name)
            return layer.name
    return None

gradcam_layer = find_last_conv_layer(model)

def grad_cam(model, img, target_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img, tf.float32)
        conv_out, preds = grad_model(img_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grad = tape.gradient(loss, conv_out)

    weights = tf.reduce_mean(grad, axis=(0, 1, 2))
    cam = tf.reduce_sum(weights * conv_out[0], axis=-1)
    cam = np.maximum(cam, 0)
    # cam = cam.numpy()

    if cam.max() > 0:
        cam /= cam.max()

    return cam

def predict_cam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224,224))

        img = resized.astype(np.float32)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0]
        idx = np.argmax(pred)
        label = label_map[idx]
        conf = pred[idx]*100

        # Grad-CAM
        cam = grad_cam(model, img, gradcam_layer)
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        heat = (cam*255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        threshold = 0.5
        mask = cam > threshold
        mask = mask.astype(np.uint8)
        overlay = frame.copy()
        overlay[mask==1] = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)[mask==1]

        # Bounding box
        thresh = (cam>0.5).astype(np.uint8)*255
        contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(overlay,f"{label} ({conf:.2f}%)", (x, y+25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.imshow("Food Recognition + Grad-CAM", overlay)

        if cv2.waitKey(1)==27:
            break

    cap.release()
    cv2.destroyAllWindows()

def predict(img_path):
    orig = cv2.imread(img_path)
    orig_h, orig_w = orig.shape[:2]
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))

    img = resized.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    label = label_map[idx]
    conf = pred[idx] * 100

    print("\n===== PREDICTION =====")
    for i in np.argsort(pred)[::-1]:
        print(f"{label_map[i]:20s}: {pred[i]*100:.2f}%")

    # Grad-CAM
    cam = grad_cam(model, img, gradcam_layer)
    cam = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
    heat = (cam*255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    threshold = 0.5
    mask = cam > threshold
    mask = mask.astype(np.uint8)
    overlay = orig.copy()
    overlay[mask==1] = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)[mask==1]

    # Bounding box
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
    else:
        overlay = cv2.resize(overlay, (1280, 720))

    cv2.imshow("Prediction + Grad-CAM", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# predict(r"dataset\test\banana.jpg")
# predict_cam()
