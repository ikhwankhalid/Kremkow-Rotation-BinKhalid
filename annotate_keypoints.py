# 'cam0_2023-03-22-12-49-51_R.mp4'

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class KeypointAnnotator:
    def __init__(self, master, video_path):
        self.master = master
        self.video_path = video_path
        self.current_frame = 0

        # Key points to mark
        self.keypoint_list = ['eye(back)', 'eye(bottom)', 'eye(front)', 'eye(top)', 'lowerlip', 'mouth',
                              'nose(bottom)', 'nose(r)', 'nose(tip)', 'nose(top)', 'nosebridge', 'paw',
                              'whisker(I)', 'whisker(II)', 'whisker(III)']

        self.selected_keypoint_idx = 0

        # Create canvas
        self.canvas = tk.Canvas(master, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_mouse_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_motion)

        # Create keypoint legend and color map
        self.keypoint_colors = {}
        for idx, kp in enumerate(self.keypoint_list):
            color = "#{:06x}".format(idx*1118481 % (2**24))  # Generate unique color code
            self.keypoint_colors[kp] = color
            self.canvas.create_rectangle(10 + (idx*80), 10, 25 + (idx*80), 25, fill=color)
            self.canvas.create_text(40 + (idx*80), 20, text=kp, anchor="w")

        # Create keypoints for each frame
        self.video_capture = cv2.VideoCapture(self.video_path)
        _, _ = self.video_capture.read()
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_keypoints = np.zeros((self.frame_count, len(self.keypoint_list), 2), dtype=np.float32)

        # Create buttons
        self.previous_button = tk.Button(master, text="Previous frame", command=self.previous_frame)
        self.previous_button.pack(side="left", padx=(20, 10), pady=20)

        self.next_button = tk.Button(master, text="Next frame", command=self.next_frame)
        self.next_button.pack(side="left", padx=(10, 20), pady=20)

        self.save_button = tk.Button(master, text="Save keypoints", command=self.save_keypoints_to_numpy_file)
        self.save_button.pack(side="left", padx=(20, 10), pady=20)

        self.canvas.create_text(150, 570, text="Current keypoint: ", anchor="w")
        self.canvas.create_text(250, 570, text=self.keypoint_list[self.selected_keypoint_idx], anchor="w", tag="current_keypoint")

        self.reset_video()

    def reset_video(self):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.canvas.create_image(20, 50, image=self.current_image, anchor=tk.NW)
            self.draw_keypoints()
            self.canvas.create_text(600, 570, text=f">> Frame: {self.current_frame}/{self.frame_count-1}", anchor="w", tag="frame_number")
        else:
            print("No video frame")

    def previous_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
        self.reset_video()

    def next_frame(self):
        if self.current_frame < self.frame_count - 1:
            self.current_frame += 1
        self.reset_video()

    def on_canvas_mouse_click(self, event):
        pt = np.array([event.x - 20, event.y - 50], dtype=np.float32)
        self.video_keypoints[self.current_frame][self.selected_keypoint_idx] = pt
        self.draw_keypoints()

    def on_canvas_motion(self, event):
        pt = np.array([event.x - 20, event.y - 50], dtype=np.float32)
        self.video_keypoints[self.current_frame][self.selected_keypoint_idx] = pt
        self.draw_keypoints()

    def draw_keypoints(self):
        self.canvas.delete("keypoint")
        for idx, point in enumerate(self.video_keypoints[self.current_frame]):
            color = self.keypoint_colors[self.keypoint_list[idx]]
            self.canvas.create_oval(point[0] + 15, point[1] + 45, point[0] + 25, point[1] + 55, fill=color, tags=("keypoint"))

    def save_keypoints_to_numpy_file(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".npy")
        np.save(save_path, self.video_keypoints)
        print("Keypoints saved to", save_path)

def main():
    video_file = 'cam0_2023-03-22-12-49-51_R.mp4'
    root = tk.Tk()
    root.title("Video Keypoint Annotator")
    root.geometry("800x600")

    annotator = KeypointAnnotator(root, video_file)

    root.mainloop()

if __name__ == "__main__":
    main()