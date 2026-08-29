import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]= "1"
import cv2
import time
import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kiui
from safetensors.torch import load_file
import rembg
from dataset.traindata import *
from dataset.testdata import *
from core.Network import Network
# from core.gs_enderer import Renderer
from torch.utils.tensorboard import SummaryWriter
import argparse
import math
import kiui
# from memory_profiler import profile
from kiui.lpips import LPIPS
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr_func, structural_similarity as ssim_func
from accelerate import Accelerator
from accelerate.utils import set_seed

def check_grad(model):
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                return True
    return False

def cal_psnr_and_ssim(pre_imgs,gt_imgs):
    psnr_values = 0
    ssim_values = 0
    pre_imgs = pre_imgs.reshape(-1, pre_imgs.shape[2], pre_imgs.shape[3], pre_imgs.shape[4])
    gt_imgs = gt_imgs.reshape(-1, gt_imgs.shape[2], gt_imgs.shape[3], gt_imgs.shape[4])
    for i in range(pre_imgs.shape[0]):
        psnr_value = psnr_func(pre_imgs[i], gt_imgs[i])
        ssim_value = ssim_func(pre_imgs[i], gt_imgs[i], channel_axis=0,data_range=1)
        psnr_values = psnr_values + psnr_value
        ssim_values = ssim_values + ssim_value
    return psnr_values/(pre_imgs.shape[0]), ssim_values/(pre_imgs.shape[0])
    
def validation_step(val_dataloader,model,epoch,opt,device):
    total_psnr = 0
    total_ssim = 0
    for i, data in enumerate(val_dataloader):
            gt_images = data["gt_images"].to(device)
            gt_masks = data["gt_masks"].to(device)
            out = model(data)
            name = data["name"][0]
            # rgb loss
            pred_images = out["rend_image"]
            bg_color = torch.ones(3, dtype=torch.float32).to(device)
            gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
            ###
            gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            psnr,ssim = cal_psnr_and_ssim(pred_images,gt_images)
            #
            psnr = torch.tensor(psnr).to(device)
            ssim = torch.tensor(ssim).to(device)
            #
            total_psnr = total_psnr+psnr
            total_ssim = total_ssim+ssim
            gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
            pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
            pre_gt_imgs = np.concatenate((pred_images, gt_images), axis=0)
            kiui.write_image(f'{opt.outdir}/{name}/train_pre_gt_images_{epoch}.jpg', pre_gt_imgs)
            ###
            input_imgs = data["input_images"].detach().cpu().numpy()
            input_normals = data["input_normals"].detach().cpu().numpy()
            input_imgs = input_imgs.transpose(0, 3, 1, 4, 2).reshape(-1, input_imgs.shape[1] * input_imgs.shape[3], 3) # [B*output_size, V*output_size, 3]
            input_normals = input_normals.transpose(0, 3, 1, 4, 2).reshape(-1, input_normals.shape[1] * input_normals.shape[3], 3)
            input_imgs = np.concatenate((input_imgs, input_normals), axis=0)
            kiui.write_image(f'{opt.outdir}/{name}/train_input_images.jpg', input_imgs)
            # kiui.write_image(f'{opt.outdir}/imgs/train_pred_images_{epoch}_{i}.jpg', pred_images)
            ###
    return total_psnr,total_ssim

       
def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image.yaml")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    opt.outdir = os.path.join(opt.outdir,formatted_time)
    accelerator = Accelerator(mixed_precision='bf16',gradient_accumulation_steps=1)
    model = Network(cfg=opt)
    if len(opt.resume)>0:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        print(f'[INFO] Loaded checkpoint from {opt.resume}')
    train_dataset = ObjaverseDataset(opt=opt)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = GSODataset(opt=opt)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=5e-4, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.T_max)
    if accelerator.is_main_process:
        tb_writer = SummaryWriter(log_dir=opt.outdir+"/tf_log")
    # initialize accelerator and auto move data/model to accelerator.device
    accelerator.print(f'device {str(accelerator.device)} is used!')
    model, optimizer,scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer,scheduler, train_dataloader, val_dataloader)
    if opt.lambda_lpips >= 0:
        lpips_loss = LPIPS(net='vgg').to(accelerator.device)
        lpips_loss.requires_grad_(False)
    #======================================================================
    save_psnr = 0
    save_txt_path = opt.outdir+"/logs.txt"
    for epoch in tqdm.trange(opt.total_epoch):
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_lpips_loss = 0
        total_depth_loss = 0
        total_reg_dist_loss = 0
        total_reg_normal_loss = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                gt_images = data["gt_images"].to(accelerator.device)
                gt_masks = data["gt_masks"].to(accelerator.device)
                gt_depths = data["gt_depths"].to(accelerator.device)
                optimizer.zero_grad()
                out = model(data)
                loss = 0
                # rgb loss
                pred_images = out["rend_image"]
                pred_alphas = out["rend_alpha"]
                pred_depths = out["surf_depth"]
                bg_color = torch.ones(3, dtype=torch.float32).to(accelerator.device)
                gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
                loss_alpha = F.l1_loss(pred_alphas, gt_masks)
                loss_mse = F.l1_loss(pred_images, gt_images) + loss_alpha
                loss_depth = F.l1_loss(gt_depths, pred_depths)
                loss = loss + loss_mse + loss_depth * opt.lambda_depth
                #lpips loss
                if opt.lambda_lpips >= 0:
                    loss_lpips = lpips_loss(
                        # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                        # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                        # downsampled to at most 256 to reduce memory cost
                        F.interpolate(gt_images.view(-1, 3, opt.output_size, opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                        F.interpolate(pred_images.view(-1, 3, opt.output_size, opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                    ).mean()
                loss = loss + opt.lambda_lpips * loss_lpips
                ###
                # regularization
                lambda_normal = opt.lambda_normal if epoch > 100 else 0.0
                lambda_dist = opt.lambda_dist if epoch > 100 else 0.0 
                rend_dist = out["rend_dist"]
                rend_normal  = out['rend_normal']
                surf_normal = out['surf_normal']
                ### dist loss
                dist_loss = lambda_dist * rend_dist.mean()
                ### normal loss
                # tmp = (rend_normal * surf_normal).sum(dim=2).unsqueeze(2)
                normal_error = ((1 - (rend_normal * surf_normal).sum(dim=2).unsqueeze(2))*pred_alphas).mean() 
                normal_loss = lambda_normal * normal_error
                loss = loss + opt.lambda_reg*(normal_loss + dist_loss)
                ###
                # optimize step
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gradient_clip = 1.0
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clip)
                # if i>0 and i%16==0:
                if check_grad(model):
                    print("NaN detected in gradients, skipping step")
                    continue
                optimizer.step()
                scheduler.step()
                total_loss = total_loss+loss.detach()
                total_mse_loss = total_mse_loss +loss_mse.detach()
                total_lpips_loss = total_lpips_loss + loss_lpips.detach()
                total_depth_loss = total_depth_loss + loss_depth.detach()
                total_reg_dist_loss = total_reg_dist_loss + dist_loss.detach()
                total_reg_normal_loss = total_reg_normal_loss + normal_loss.detach()
                # torch.cuda.empty_cache()
                
            if accelerator.is_main_process:
                # if epoch % 10 == 0:
                mem_free, mem_total = torch.cuda.mem_get_info()    
                print(f"[INFO] epoch: {epoch} data:{len(train_dataloader)}/{i} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} total_loss: {loss.item():.6f} mse_loss: {loss_mse.item():.6f} lpips_loss: {loss_lpips.item():.6f} depth_loss: {loss_depth.item():.6f} alpha_loss: {loss_alpha.item():.6f} dist_loss: {dist_loss.item():.6f} nor_loss: {normal_loss.item():.6f}")
                save_txt = open(save_txt_path, 'a')
                save_txt.write(f"[INFO] epoch: {epoch} data:{len(train_dataloader)}/{i} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} total_loss: {loss.item():.6f} mse_loss: {loss_mse.item():.6f} lpips_loss: {loss_lpips.item():.6f} depth_loss: {loss_depth.item():.6f} alpha_loss: {loss_alpha.item():.6f} dist_loss: {dist_loss.item():.6f} nor_loss: {normal_loss.item():.6f}\n")
                save_txt.close()
            if i % opt.imgs_iter == 0 or i == len(train_dataloader)-1:
                gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                pre_gt_imgs = np.concatenate((pred_images, gt_images), axis=0)
                kiui.write_image(f'{opt.outdir}/temp/imgs_{i}/train_pre_gt_images_{epoch}_{str(accelerator.device)}.jpg', pre_gt_imgs)

        ###
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_mse_loss = accelerator.gather_for_metrics(total_mse_loss).mean()
        total_depth_loss = accelerator.gather_for_metrics(total_depth_loss).mean()
        total_lpips_loss = accelerator.gather_for_metrics(total_lpips_loss).mean()
        total_reg_dist_loss = accelerator.gather_for_metrics(total_reg_dist_loss).mean()
        total_reg_normal_loss = accelerator.gather_for_metrics(total_reg_normal_loss).mean()
        ###
        if accelerator.is_main_process:
            tb_writer.add_scalar('Loss/Total', total_loss/len(train_dataloader), epoch)
            tb_writer.add_scalar('Loss/MSE', total_mse_loss/len(train_dataloader), epoch)
            tb_writer.add_scalar('Loss/Depth', total_depth_loss/len(train_dataloader), epoch)
            tb_writer.add_scalar('Loss/LPIPS', total_lpips_loss/len(train_dataloader), epoch)
            tb_writer.add_scalar('Loss/Dist', total_reg_dist_loss/len(train_dataloader), epoch)
            tb_writer.add_scalar('Loss/Normal', total_reg_normal_loss/len(train_dataloader), epoch)
            if epoch % opt.save_iter == 0:
                epoch_path = os.path.join(opt.outdir,f"model/{epoch}")
                if not os.path.exists(epoch_path):
                    # 如果路径不存在，则创建该路径
                    os.makedirs(epoch_path,exist_ok=True)
                accelerator.save_model(model, epoch_path)
        accelerator.wait_for_everyone()
        with torch.no_grad():
            model.eval()
            psnr,ssim = validation_step(val_dataloader=val_dataloader,model=model,epoch=epoch,opt=opt,device=accelerator.device)
            psnr = accelerator.gather_for_metrics(psnr).mean()
            ssim = accelerator.gather_for_metrics(ssim).mean()
            psnr = psnr/len(val_dataloader)
            ssim = ssim/len(val_dataloader)
            if accelerator.is_main_process:
                tb_writer.add_scalar('Metric/PSNR', psnr, epoch)
                tb_writer.add_scalar('Metric/SSIM', ssim, epoch)
            if psnr > save_psnr:
                save_psnr = psnr
                # tmp = torch.round(save_psnr * 2) / 2
                save_path = os.path.join(opt.outdir,f"model/best")
                if not os.path.exists(save_path):
                    # 如果路径不存在，则创建该路径
                    os.makedirs(save_path,exist_ok=True)
                # save_path = os.path.join(save_path)
                accelerator.save_model(model, save_path)
                print("Finish saving best model for psnr: {:.4f}!".format(psnr))
        torch.cuda.empty_cache()
        
        
            
    
if __name__ == "__main__":
    main()