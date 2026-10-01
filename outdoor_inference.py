# download kitti.pt weights:
# wget https://github.com/dmalynyak/depth-estimator-pytorch/releases/download/kitti_weights/kitti.pt -O weights/kitti.pt

# run inference: (both for photoes and videos)
# python -m outdoor_inference --device cuda --file_path 'your_file_path' --model_path weights/kitti.pt

import src
import torch


def main(args):
    device = src.utils.parse_device(args.device)
    model = src.KITTIdepthNET().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True)['depth_net'])
    file_path = args.file_path

    inference = src.InferenceOutdoor(model, heigh=192, width=640, device=device)
    print(file_path)
    inference.pipeline(file_path)

if __name__ == "__main__":
    args = src.utils.parse_inference_args()
    print(args)
    main(args)