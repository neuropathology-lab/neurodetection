import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import cv2
import numpy as np
import os
import time

# Windows DPI-awareness fix
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
except Exception:
    pass

def resize_img(img, max_width):
    h, w = img.shape[:2]
    if w > max_width:
        new_w = max_width
        new_h = int(h * max_width / w)
        return cv2.resize(img, (new_w, new_h)), new_w, new_h
    return img, w, h

def cv2_to_ImageTk(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 1) * 255 if img_rgb.max() <= 1 else np.clip(img_rgb, 0, 255)
        img_rgb = img_rgb.astype(np.uint8)
    return ImageTk.PhotoImage(Image.fromarray(img_rgb))

def prepare_image(img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_original_based_on_mean_square(original_img, gui_display_shape, mean_rel_area):
    target_area = 100 * 100
    H_orig, W_orig = original_img.shape[:2]
    scale_factor = (target_area / (mean_rel_area * W_orig * H_orig)) ** 0.5
    new_w = int(round(W_orig * scale_factor))
    new_h = int(round(H_orig * scale_factor))
    resized = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale_factor, (new_h, new_w)

def grab_full_window_screenshot(window):
    window.lift()
    window.attributes('-topmost', True)
    window.update()
    time.sleep(0.2)  # Let the system render everything fully
    window.attributes('-topmost', False)

    # Get actual screen coordinates
    x = window.winfo_rootx()
    y = window.winfo_rooty()
    width = window.winfo_width()
    height = window.winfo_height()

    # Final grab
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    return screenshot

class ImageGUI:
    def __init__(self, root, img1, img2_gui, img2_original, on_done, out_dir, max_width=800):
        self.root = root
        self.img2_original = img2_original
        self.img2_gui_shape = img2_gui.shape[:2]
        self.on_done = on_done
        self.out_dir = out_dir
        self.root.title("Neuron Size Annotation Tool")
        self.root.state('zoomed')
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

        self.label = tk.Label(
            root,
            text="Please indicate on the right picture the size of the neuron similarly to the left picture",
            font=("Helvetica", 12)
        )
        self.label.pack()

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.img1_resized, self.img1_w, self.img1_h = resize_img(img1, max_width)
        self.img2_resized, self.img2_w, self.img2_h = resize_img(img2_gui, max_width)

        self.tk_img1 = cv2_to_ImageTk(self.img1_resized)
        self.tk_img2 = cv2_to_ImageTk(self.img2_resized)

        self.canvas1 = tk.Canvas(self.frame, width=self.img1_w, height=self.img1_h)
        self.canvas1.pack(side=tk.LEFT)
        self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.tk_img1)

        self.canvas2 = tk.Canvas(self.frame, width=self.img2_w, height=self.img2_h)
        self.canvas2.pack(side=tk.LEFT)
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.tk_img2)

        self.start_x = None
        self.start_y = None
        self.rect = None
        self.rectangles = []
        self.square_sizes = []

        self.canvas2.bind("<Button-1>", self.on_click)
        self.canvas2.bind("<B1-Motion>", self.on_drag)
        self.canvas2.bind("<ButtonRelease-1>", self.on_release)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.result_label.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)
        self.confirm_button = tk.Button(self.button_frame, text="✅ Confirm", command=self.confirm_selection, state=tk.DISABLED)
        self.redo_button = tk.Button(self.button_frame, text="🔁 Redo", command=self.reset_selection, state=tk.DISABLED)
        self.confirm_button.pack(side=tk.LEFT, padx=10)
        self.redo_button.pack(side=tk.LEFT, padx=10)

    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas2.delete(self.rect)
        self.rect = self.canvas2.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_drag(self, event):
        if self.rect:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            side = min(abs(dx), abs(dy))
            end_x = self.start_x + side if dx >= 0 else self.start_x - side
            end_y = self.start_y + side if dy >= 0 else self.start_y - side
            self.canvas2.coords(self.rect, self.start_x, self.start_y, end_x, end_y)

    def on_release(self, event):
        coords = self.canvas2.coords(self.rect)
        width = abs(coords[2] - coords[0])
        height = abs(coords[3] - coords[1])
        area = width * height / (self.img2_w * self.img2_h)
        self.square_sizes.append(area)

        # Draw and store permanent square
        final_rect = self.canvas2.create_rectangle(*coords, outline='red', width=2)
        self.rectangles.append(final_rect)
        self.canvas2.delete(self.rect)
        self.rect = None

        if len(self.square_sizes) == 3:
            self.result_label.config(text=f"Mean relative square size: {np.mean(self.square_sizes):.4f}")
            self.canvas2.unbind("<Button-1>")
            self.canvas2.unbind("<B1-Motion>")
            self.canvas2.unbind("<ButtonRelease-1>")
            self.confirm_button.config(state=tk.NORMAL)
            self.redo_button.config(state=tk.NORMAL)

    def reset_selection(self):
        # Remove rectangles and reset
        for r in self.rectangles:
            self.canvas2.delete(r)
        self.rectangles = []
        self.square_sizes = []
        self.result_label.config(text="")
        self.confirm_button.config(state=tk.DISABLED)
        self.redo_button.config(state=tk.DISABLED)
        self.canvas2.bind("<Button-1>", self.on_click)
        self.canvas2.bind("<B1-Motion>", self.on_drag)
        self.canvas2.bind("<ButtonRelease-1>", self.on_release)



    def confirm_selection(self):
        self.root.update()

        screenshot = grab_full_window_screenshot(self.root)
        screenshot.save(os.path.join(self.out_dir, "annotated_right_image.png"))

        mean_rel_area = np.mean(self.square_sizes)
        resized, scale_factor, target_shape = resize_original_based_on_mean_square(
            self.img2_original, self.img2_gui_shape, mean_rel_area
        )

        self.root.destroy()
        self.on_done(mean_rel_area, resized, scale_factor, target_shape)