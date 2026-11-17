"""
Simple click-and-drag annotation helper for one YOLO image.
Run from your backend folder:
    python pear_annotation_editor.py

It opens the original 256x256 image, lets you draw a box around the pear and
visible stem, and prints the normalized YOLO label. It does NOT edit the dataset.
Controls:
  Left mouse drag: draw box
  Enter: accept box and print label
  R: reset box
  Esc: quit without output
"""
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

IMAGE = Path(r"C:\Users\bdhul\Documents\Fruit_Grading_Project\Dataset_V3_Detection3Class_CLEAN\valid\images\r0_139_jpg.rf.4f3099d810c81ff3297c0e9ac4ee26c1.jpg")
CLASS_ID = 0
DISPLAY_SIZE = 768

if not IMAGE.is_file():
    raise SystemExit(
        f"Image not found:\n{IMAGE}\n"
        "Edit IMAGE at the top of this script if your dataset is stored elsewhere."
    )

original = Image.open(IMAGE).convert("RGB")
W, H = original.size
scale = min(DISPLAY_SIZE / W, DISPLAY_SIZE / H)
disp_w, disp_h = round(W * scale), round(H * scale)
display_image = original.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

root = tk.Tk()
root.title("Pear annotation helper - draw box, then press Enter")
root.resizable(False, False)

help_text = (
    "Drag a rectangle around the FULL visible pear and stem.\n"
    "Then press Enter to print the YOLO coordinates. R resets, Esc cancels.\n"
    "This tool does not modify your dataset label."
)
tk.Label(root, text=help_text, justify="left", padx=8, pady=8).pack()
canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="crosshair", highlightthickness=0)
canvas.pack()
photo = ImageTk.PhotoImage(display_image)
canvas.create_image(0, 0, image=photo, anchor="nw")

state = {"start": None, "rect": None, "box": None}

def mouse_down(event):
    state["start"] = (max(0, min(disp_w - 1, event.x)), max(0, min(disp_h - 1, event.y)))
    if state["rect"] is not None:
        canvas.delete(state["rect"])
    state["rect"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=3)
    state["box"] = None

def mouse_move(event):
    if state["start"] is None or state["rect"] is None:
        return
    x0, y0 = state["start"]
    x1, y1 = max(0, min(disp_w - 1, event.x)), max(0, min(disp_h - 1, event.y))
    canvas.coords(state["rect"], x0, y0, x1, y1)

def mouse_up(event):
    if state["start"] is None:
        return
    x0, y0 = state["start"]
    x1, y1 = max(0, min(disp_w - 1, event.x)), max(0, min(disp_h - 1, event.y))
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right - left < 2 or bottom - top < 2:
        state["box"] = None
        return
    # Convert displayed rectangle to original pixel coordinates.
    ox1, ox2 = left / scale, right / scale
    oy1, oy2 = top / scale, bottom / scale
    cx = ((ox1 + ox2) / 2) / W
    cy = ((oy1 + oy2) / 2) / H
    bw = (ox2 - ox1) / W
    bh = (oy2 - oy1) / H
    state["box"] = (cx, cy, bw, bh)

def accept(_event=None):
    if state["box"] is None:
        messagebox.showwarning("No box", "Draw a rectangle first.")
        return
    cx, cy, bw, bh = state["box"]
    label = f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    print("\nPROPOSED YOLO LABEL (NOT SAVED):")
    print(label)
    messagebox.showinfo(
        "Coordinates ready",
        "Proposed label (not saved):\n\n" + label +
        "\n\nCopy this line only after checking the box visually."
    )

def reset(_event=None):
    if state["rect"] is not None:
        canvas.delete(state["rect"])
    state.update({"start": None, "rect": None, "box": None})

canvas.bind("<ButtonPress-1>", mouse_down)
canvas.bind("<B1-Motion>", mouse_move)
canvas.bind("<ButtonRelease-1>", mouse_up)
root.bind("<Return>", accept)
root.bind("<Escape>", lambda _e: root.destroy())
root.bind("r", reset)
root.bind("R", reset)
root.mainloop()
