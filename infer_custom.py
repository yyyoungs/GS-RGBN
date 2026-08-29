import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]= "1"
os.environ['CUDA_VISIBLE_DEVICES']='0'
import cv2
import time
import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import kiui
from safetensors.torch import load_file
from dataset.traindata import *
from dataset.testdata import GSODataset,CustomDataset
from core.Network import Network
from core.gs_enderer import Renderer
import argparse
import kiui
# from memory_profiler import profile
from kiui.lpips import LPIPS
from omegaconf import OmegaConf

def main(opt):  
    renderer = Renderer(sh_degree=opt.sh_degree,view_color=opt.view_color)
    model = Network(cfg=opt)
    if len(opt.resume)>0:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        print(f'[INFO] Loaded checkpoint from {opt.resume}')
    else:
        Warning("There is no resume!")
    test_dataset = CustomDataset(opt=opt)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    for i, data in enumerate(test_dataloader):
            gs_position,gs_opacity,gs_scales,gs_rots,gs_colors = model(data,True)
            name = data["name"][0]
            if opt.save_gs:
                model.save_ply(gs_position,gs_opacity,gs_scales,gs_rots,gs_colors,f'{opt.outdir}/{name}')
            out = renderer.render(data,gs_position,gs_opacity,gs_scales,gs_rots,gs_colors,is_fortest=True)
            pred_images = out["rend_image"]
            # pred_images = out["surf_normal"]
            pred_images = pred_images.squeeze(0).detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            path = opt.save_imgs + f"/{name}"
            os.makedirs(path,exist_ok=True)
            for i in range(pred_images.shape[0]):
                save_pre = pred_images[i].transpose(1,2,0)
                save_path = path+f"/{i:03d}.png"
                kiui.write_image(save_path, save_pre)
            print(f"Saving images of name: {name}. \n")
            
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image.yaml")
    parser.add_argument("--save_gs", default=False)
    parser.add_argument("--save_imgs", default="./output")
    parser.add_argument("--output_size", default=256)
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    opt.outdir = os.path.join(opt.outdir,formatted_time)
    main(opt)