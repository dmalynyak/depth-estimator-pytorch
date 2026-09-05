import torch
import argparse
import os

def parse_device(name):
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("testCUDA requested but not available on this machine")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device '{name}'")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--chkpt_path", help="path to folder where model states will be saved")
    parser.add_argument("--log_path", help="path to scv file where model validation metrics will be saved")
    
    return parser.parse_args()

def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--file_path", help="path to file")
    parser.add_argument("--model_path", help="path to checkpoint")
    
    return parser.parse_args()

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".webm"}

def parse_extension(name):
    extension = os.path.splitext(name)[1].lower()
    base = os.path.splitext(name)[0]

    if extension in IMAGE_EXTS:
        type = "image"
        out_path = f"{base}_out.png"
        return type, out_path
    elif extension in VIDEO_EXTS:
        type = "video"
        out_path = f"{base}_out.mp4"
        return type, out_path
    else:
        print(f"Unsupported file type. Supported: {IMAGE_EXTS}, {VIDEO_EXTS}")