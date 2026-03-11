import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix
from utils.sh_utils import eval_sh


def get_smallest_axis(rotation, scaling, return_idx=False):
    """
    Returns the smallest axis of the Gaussians.
    """
    rotation = F.normalize(rotation, dim=1, eps=1e-8)
    rotation_matrices = quaternion_to_matrix(rotation)
    smallest_axis_idx = scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
    smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)

    if return_idx:
        return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    
    return smallest_axis.squeeze(dim=2)

def get_normals(rotation, scaling):
    normals = get_smallest_axis(rotation, scaling)
    N = F.normalize(normals, dim=1)
    return N

def get_view_dirs(xyz, camera_center):
    dir_pp_camera = (xyz - camera_center.repeat(xyz.shape[0], 1))
    V = -F.normalize(dir_pp_camera, dim=1)
    return V

def get_light_dirs_and_dist(xyz, light_center, spatial_lr_scale):
    dir_pp_light = (xyz - light_center.repeat(xyz.shape[0], 1))
    L = -F.normalize(dir_pp_light, dim=1) 

    dir_gauss_lightcenter = (xyz - light_center.repeat(xyz.shape[0], 1))
    light_gauss_dist = dir_gauss_lightcenter.norm(dim=1, keepdim=True) / spatial_lr_scale

    return L, light_gauss_dist

def flip_normals_to_view(normals, view_dirs):
    # normals always towards cameras
    N_dot_V = torch.sum(normals * view_dirs, dim=1, keepdim=True)
    normals = torch.where(N_dot_V < 0, -normals, normals)
    return normals

def inverse_square_attenuation(dist, tau=1e-5, clamp_max=None):
    """
    Compute inverse-square light attenuation.

    Formula:
        att = 1 / (tau + dist^2)

    Args:
        dist: [N, 1], distance from point to light
        tau: small positive constant to avoid division by zero
        clamp_max: optional upper bound for attenuation

    Returns:
        attenuation: [N, 1]
    """
    attenuation = 1.0 / (tau + dist ** 2)

    if clamp_max is not None:
        attenuation = torch.clamp(attenuation, max=clamp_max)

    return attenuation

def compute_half_vectors(light_dirs, view_dirs, eps=1e-8):
    """
    Compute half-vectors H = normalize(L + V).

    Args:
        light_dirs: [N, 3]
        view_dirs: [N, 3]

    Returns:
        H: [N, 3]
    """
    H = F.normalize(light_dirs + view_dirs, dim=1, eps=eps)
    return H

def compute_fresnel_schlick(H, V, F0):
    """
    Compute Fresnel term using Schlick approximation.
    """
    H_dot_V = torch.sum(H * V, dim=1, keepdim=True)
    fresnel = F0 + (1.0 - F0) * (1.0 - H_dot_V) ** 5
    return fresnel

def compute_ggx_distribution(N, H, roughness, eps=1e-8):
    """
    Compute GGX / Trowbridge-Reitz distribution term D.
    """
    N_dot_H = torch.clamp(torch.sum(N * H, dim=1, keepdim=True), min=0.0)
    alpha = roughness ** 2
    denom_D = (torch.pi * ((N_dot_H ** 2) * (alpha ** 2 - 1) + 1) ** 2) 
    D = (alpha ** 2) / (denom_D + eps)
    return D

def compute_smith_geometry(N, V, L, roughness, eps=1e-6):
    """
    Compute Smith-Schlick geometry term G.
    """
    N_dot_V = torch.clamp(torch.sum(N * V, dim=1, keepdim=True), min=0.0)
    N_dot_L = torch.clamp(torch.sum(N * L, dim=1, keepdim=True), min=0.0)

    k = ((roughness + 1.0) ** 2) / 8.0

    denom_V = N_dot_V * (1 - k) + k
    denom_L = N_dot_L * (1 - k) + k
    G1_V = N_dot_V / (denom_V + (denom_V == 0).float() * eps)
    G1_L = N_dot_L / (denom_L + (denom_L == 0).float() * eps)
    
    G = G1_V * G1_L
    return G

def compute_specular_rgb(xyz, rotation, scaling, camera_center, light_center, roughness, F0, spatial_lr_scale, light_intensity=1.0, tau=1e-5, clamp_att_max=None, eps=1e-6):
    """
    Compute Cook-Torrance specular RGB for each Gaussian.

    Args:
        xyz: [N, 3]
        rotation: [N, 4]
        scaling: [N, 3]
        camera_center: [1, 3] or [N, 3]
        light_center: [1, 3] or [N, 3]
        roughness: [N, 1]
        F0: [N, 1]
        spatial_lr_scale: scalar
        light_intensity: scalar
        tau: inverse-square attenuation epsilon
        clamp_att_max: optional float
        eps: small constant

    Returns:
        I_specular: [N, 3]
        aux_dict: dict for debugging
    """
    V = get_view_dirs(xyz, camera_center)
    N = get_normals(rotation, scaling)
    N = flip_normals_to_view(N, V)

    L, light_gauss_dist = get_light_dirs_and_dist(xyz, light_center, spatial_lr_scale)
    attenuation = inverse_square_attenuation(light_gauss_dist, tau=tau, clamp_max=clamp_att_max)

    H = compute_half_vectors(L, V, eps=eps)

    fresnel = compute_fresnel_schlick(H, V, F0)
    D = compute_ggx_distribution(N, H, roughness, eps=eps)
    G = compute_smith_geometry(N, V, L, roughness, eps=eps)

    N_dot_V = torch.clamp(torch.sum(N * V, dim=1, keepdim=True), min=0.0)
    N_dot_L = torch.clamp(torch.sum(N * L, dim=1, keepdim=True), min=0.0)
    spec_denom = (4 * N_dot_V * N_dot_L)
    specular_component_coeffs = (fresnel * D * G) / (spec_denom + (spec_denom == 0).float() * eps)
    I_specular_coeffs = light_intensity * attenuation * specular_component_coeffs * N_dot_L

    I_specular = torch.clamp(I_specular_coeffs, min=0.0)

    aux_dict = {
        "N": N,
        "V": V,
        "L": L,
        "H": H,
        "attenuation": attenuation,
        "fresnel": fresnel,
        "D": D,
        "G": G,
        "N_dot_V": N_dot_V,
        "N_dot_L": N_dot_L,
    }

    return I_specular, aux_dict

def compute_diffuse_rgb(xyz, features, max_sh_degree, active_sh_degree, camera_center):
        # Returns base color

        shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
        dir_pp = (xyz - camera_center.repeat(features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
        return sh2rgb + 0.5 #torch.clamp_min(sh2rgb + 0.5, 0.0)

def compute_final_rgb(diffuse_rgb, specular_rgb, fresnel):
    reflected_rgb = (1 - fresnel) * diffuse_rgb + specular_rgb
    return reflected_rgb.clamp(min=1e-6)**(1/2.2)