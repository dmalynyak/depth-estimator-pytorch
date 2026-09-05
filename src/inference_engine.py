import src
import torch
import cv2


class Inference():
    def __init__(self, model, heigh=240, width=320, device='cpu'):
        self.model = model
        self.width = width
        self.heigh = heigh
        self.device = device


    @torch.no_grad
    def image_inference(self, in_path):
        self.model.eval()

        in_tensor = src.dataloaders.get_inference_tensor(in_path, self.heigh, self.width, self.device)
        out_tensor = self.model(in_tensor)
        prediction = out_tensor[0]
        return in_tensor, prediction


    def pipeline(self, in_path):

        type, out_path = src.utils.parse_extension(in_path)
        assert type in ["image", "video"]

        if type == "image":
            rgb, depth = self.image_inference(in_path)
            self.draw_image(rgb, depth, out_path)
        elif type == "video":
            self.video_inference()


    def draw_image(self, rgb, depth, out_path):
        src.utils.draw_prediction(rgb, depth, out_path)



def main(args):
    device = args.device
    model = args.model_path
    file_path = args.file_path

    type, out_path = src.utils.parse_extension(file_path)
    assert type in ["image", "video"]

    if type == "image":
        image_inference(model, file_path, out_path, device)
    elif type == "video":
        video_inference(model, file_path, out_path, device)

if __name__ == "__main__":
    args = src.utils.parse_inference_args()
    print(args)
    main(args)