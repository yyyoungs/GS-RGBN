import os
import math
import torch
import numpy as np
import torchvision.utils as vutils
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def depths_to_points(rays, depthmap):
    points = rays[...,:3].view(-1,3)  + depthmap.view(-1, 1) * rays[...,3:].view(-1,3)
    return points

def depth_to_normal(rays, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(rays, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

class Renderer:
    def __init__(self, sh_degree=1,white_background=True,view_color=True):
        
        
        self.white_background = white_background
        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        ) 
        self.sh_degree = sh_degree
        self.view_color = view_color
        
    def render(
        self,
        data,
        gs_position,gs_opacity,gs_scales,gs_rots,gs_colors,
        scaling_modifier=1.0,
        bg_color=None,
        is_fortest=False
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means


        # Set up rasterization configuration
        
        if not is_fortest:
            B,V,_,H,W = data["gt_images"].shape[:]
        else: 
            B,V,_,H,W = data["input_images"].shape[:]
        # V = 1
        rendered_images = []
        render_alphas = []
        render_normals = []
        render_dists = []
        surf_depths = []
        surf_normals = []
        visibility_filters = []
        radiis = []
        for i in range(B):
            tanfovx = math.tan(data["FovX"][i] * 0.5)
            tanfovy = math.tan(data["FovY"][i] * 0.5)
            means3D = gs_position[i]
            screenspace_points = (
                    torch.zeros_like(
                        means3D,
                        dtype=means3D.dtype,
                        requires_grad=True,
                        device="cuda",
                    )
                    + 0
                )
            try:
                screenspace_points.retain_grad()
            except:
                pass
            means2D = screenspace_points
            opacity = gs_opacity[i]
            scales = gs_scales[i]
            rotations = gs_rots[i]
            #
            if self.view_color:
                shs = gs_colors[i]
                rgbs = None
            else:
                shs = None
                rgbs = gs_colors[i]
            #
            for j in range(V):
                world_view_transform = data["world_view_transforms"][i,j].cuda()
                full_proj_transform = data["full_proj_transforms"][i,j].cuda()
                camera_center = data["camera_centers"][i,j].cuda()
                raster_settings = GaussianRasterizationSettings(
                    image_height=int(H),
                    image_width=int(W),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scaling_modifier,
                    viewmatrix=world_view_transform,
                    projmatrix=full_proj_transform,
                    sh_degree=self.sh_degree,
                    campos=camera_center,
                    prefiltered=False,
                    debug=False,
                    # pipe.debug
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                rendered_image, radii, allmap = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = rgbs,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = None
                )

                rendered_image = rendered_image.clamp(0, 1)
                
               # additional regularizations
                render_alpha = allmap[1:2]
                # get normal map
                # transform normal from view space to world space
                render_normal = allmap[2:5]
                render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)

                # get median depth map
                render_depth_median = allmap[5:6]
                render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

                # get expected depth map
                render_depth_expected = allmap[0:1]
                render_depth_expected = (render_depth_expected / render_alpha)
                render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
                
                # get depth distortion map
                render_dist = allmap[6:7]

                # psedo surface attributes
                # surf depth is either median or expected by setting depth_ratio to 1 or 0
                # for bounded scene, use median depth, i.e., depth_ratio = 1; 
                # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
                depth_ratio = 0.0
                surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
                # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
                rays_ori = data["rays_ori"][i,j].cuda()
                surf_normal,_ = depth_to_normal(rays_ori,surf_depth)
                surf_normal = surf_normal.permute(2,0,1)
                # remember to multiply with accum_alpha since render_normal is unnormalized.
                surf_normal = surf_normal * (render_alpha).detach()
                ###
                # vutils.save_image(surf_normal, 'output.png')
                ###
                rendered_images.append(rendered_image)
                render_alphas.append(render_alpha)
                render_normals.append(render_normal)
                render_dists.append(render_dist)
                surf_depths.append(surf_depth)
                surf_normals.append(surf_normal)
                visibility_filters.append(radii > 0)
                radiis.append(radii)

       
        return {
            "rend_image": torch.stack(rendered_images, dim=0).view(B, V, 3, H, W),
            'rend_alpha': torch.stack(render_alphas,dim=0).view(B, V, 1, H, W),
            'rend_normal': torch.stack(render_normals,dim=0).view(B, V, 3, H, W),
            'rend_dist': torch.stack(render_dists,dim=0).view(B, V, 1, H, W),
            'surf_depth': torch.stack(surf_depths,dim=0).view(B, V, 1, H, W),
            'surf_normal': torch.stack(surf_normals,dim=0).view(B, V, 3, H, W),
            "visibility_filter": torch.stack(visibility_filters, dim=0).view(B, V, -1),
            "radii": torch.stack(radiis, dim=0).view(B, V, -1),
        }

    
