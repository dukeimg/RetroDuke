import numpy as np


class ImageProcessor:
    def __init__(self, process_image_format, render_format=None, viewer=None):
        self.viewer = viewer
        self.render_format = render_format
        self.process_image_format = process_image_format

    @classmethod
    def stack_frames(cls, stacked_frames, frame):
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_frame = np.stack(stacked_frames, axis=2)

        return stacked_frame

    def render(self, img):
        if self.viewer:
            f = self.render_format
            img = self.slice_image(img, f)
            self.viewer.imshow(img)

    def process_image(self, img):
        f = self.process_image_format
        img = self.slice_image(img, f)
        return np.mean(img, -1)

    @staticmethod
    def slice_image(img, f):
        slice_1 = slice(f[0][0], f[0][1])
        slice_2 = slice(f[1][0], f[1][1])
        return img[slice_1, slice_2]
