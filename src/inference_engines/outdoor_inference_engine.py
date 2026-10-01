import src
import torch
import cv2
import numpy as np


class InferenceOutdoor():
    def __init__(self, model, heigh=192, width=640, device='cpu'):
        self.model = model
        self.width = width
        self.heigh = heigh
        self.device = device


    @torch.no_grad
    def image_inference(self, in_path):
        self.model.eval()

        in_tensor = src.dataloaders.get_inference_tensor_kitti(in_path, self.heigh, self.width, self.device)
        out_tensor = self.model(in_tensor)
        prediction = out_tensor[0]
        return in_tensor, prediction


    @torch.no_grad()
    def video_inference(self, in_path, out_path):
        self.model.eval()
        transform = src.dataloaders.built_one_img_transform(self.heigh, self.width)

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.width, 2 * self.heigh))
        # writer2 = cv2.VideoWriter(f"clear_korona.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.width, self.heigh))
        vmax = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            in_tensor = src.dataloaders.get_inference_frame_kitti(frame, transform, self.device)
            disp = self.model(in_tensor)[0][0, 0].cpu().numpy()

            if vmax is None:
                vmax = np.percentile(disp, 95)
            disp_u8 = (np.clip(disp / vmax, 0, 1) * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_MAGMA)

            frame_small = cv2.resize(frame, (self.width, self.heigh), interpolation=cv2.INTER_AREA)
            writer.write(np.vstack([frame_small, disp_color]))
            # writer2.write(disp_color)

        cap.release()
        writer.release()
        # writer2.release()


    def pipeline(self, in_path):

        type, out_path = src.utils.parse_extension(in_path)
        assert type in ["image", "video"]

        if type == "image":
            rgb, depth = self.image_inference(in_path)
            self.draw_image(rgb, depth, out_path)
        elif type == "video":
            self.video_inference(in_path, out_path)


    def draw_image(self, rgb, depth, out_path):
        src.utils.draw_prediction_outdoor(rgb, depth, out_path)
