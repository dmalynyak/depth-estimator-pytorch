import src
import torch

def main(args):
    device = src.utils.parse_device(args.device)
    model = src.NYUmodel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    file_path = args.file_path

    inference = src.Inference(model, heigh=240, width=320, device=device)
    print(file_path)
    inference.pipeline(file_path)

if __name__ == "__main__":
    args = src.utils.parse_inference_args()
    print(args)
    main(args)