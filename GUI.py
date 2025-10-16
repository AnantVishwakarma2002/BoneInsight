import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0)
        cam = cam / cam.max()
        return cam

    def overlay_cam(self, image, cam, alpha=0.5):
        cam = cam.detach().cpu().numpy()
        cam = np.uint8(cam * 255)
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        image = np.uint8(image * 255)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        blended = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
        return blended

# === Load model ===
model = models.densenet121(pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load('mura_densenet121.pth', map_location=torch.device('cpu')))
model.eval()
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === GUI Setup ===
class MURAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MURA X-Ray Predictor")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Upload a medical X-ray image", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Choose Image", command=self.load_image)
        self.upload_btn.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_text = tk.StringVar()
        self.result_label = tk.Label(root, textvariable=self.result_text, font=("Helvetica", 16, "bold"))
        self.result_label.pack(pady=10)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        image = Image.open(file_path).convert('RGB')
        display_img = image.resize((300, 300))
        self.image_tk = ImageTk.PhotoImage(display_img)
        self.image_label.configure(image=self.image_tk)

        input_tensor = transform(image).unsqueeze(0)

        # === Predict Abnormality ===
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, 1).item()
            label = "Positive" if pred == 1 else "Negative"

        # === Infer Body Part from Filename/Path ===
        body_parts = ["SHOULDER", "HUMERUS", "ELBOW", "FOREARM", "WRIST", "HAND", "FINGER"]
        detected_part = "Unknown"
        for part in body_parts:
            if part.lower() in file_path.lower():
                detected_part = part
                break

        self.result_text.set(f"Prediction: {label} | Body Part: {detected_part}")

        # Clear previous canvases
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        if label == "Positive":
            cam = gradcam.generate(input_tensor)

            # Prepare unnormalized image
            img_np = input_tensor.squeeze().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            # Grad-CAM overlay
            overlay = gradcam.overlay_cam(img_np, cam)

            # Segmentation
            cam_resized = cv2.resize(cam.cpu().numpy(), (img_np.shape[1], img_np.shape[0]))
            thresh = np.percentile(cam_resized, 95)
            mask = (cam_resized > thresh).astype(np.uint8) * 255
            edges = cv2.Canny(mask, 100, 200)
            seg_ov = img_np.copy()
            seg_ov[edges > 0] = [1, 0, 0]

            # Plot
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_np); axs[0].axis('off'); axs[0].set_title('Original')
            axs[1].imshow(overlay); axs[1].axis('off'); axs[1].set_title('Grad-CAM')
            axs[2].imshow(seg_ov); axs[2].axis('off'); axs[2].set_title('Segmentation')

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

# === Launch App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MURAApp(root)
    root.mainloop()
