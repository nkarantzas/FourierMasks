import numpy as np
import torch
from skimage import draw

def get_masks(paths):
    masks = torch.zeros(len(paths), 128, 128)
    for i, p in enumerate(paths):
        masks[i] = torch.fft.fftshift(torch.load(p, map_location='cpu').squeeze())
    return masks
    
def get_mask_diffs(masks, diffs):
    Dm = torch.zeros(len(diffs), 128, 128)
    for i, diff in enumerate(diffs):
        Dm[i] = masks[diff[0]] - masks[diff[1]]
    Dm = (Dm - Dm.min()) / (Dm.max() - Dm.min())
    Dm -= 0.5
    return Dm

def radial_band(
    radius_in, 
    radius_out, 
    mask_size=(128, 128), 
    mask_value=1., 
    p=2
):
    
    # get the center of the image
    c_x = mask_size[0]//2
    c_y = mask_size[1]//2
    
    # band mask init
    mask = torch.ones(mask_size, dtype=bool)
    
    x = torch.arange(0, mask_size[0])
    y = torch.arange(0, mask_size[0])
    x_dist = torch.abs(x - c_x)
    y_dist = torch.abs(y - c_y)
    xx, yy = torch.meshgrid(x_dist, y_dist)
    
    if p==2:
        # L2:
        dist = torch.sqrt(xx**2 + yy**2)
    elif p==1:
        # L1: 
        dist = (torch.abs(xx) + torch.abs(yy))
    else:
        raise ValueError("Invalid choice for p")
        
    if radius_out is not None:
        mask[dist >= radius_out] = False
        
    mask[dist < radius_in] = False
    mask = mask * mask_value
    return mask

def circular_band(
    angle1, 
    angle2, 
    mask_size=(128, 128), 
    mask_value=1.
):
    
    # get the center and max radius considered
    r0, c0 = mask_size[0]//2, mask_size[1]//2
    R = np.sqrt(mask_size[0]**2 + mask_size[1]**2)
    
    theta0 = np.deg2rad(angle1)
    theta1 = np.deg2rad(angle2)

    r1, c1 = r0 - 1.5 * R * np.sin(theta0), c0 + 1.5 * R * np.cos(theta0)
    r2, c2 = r0 - 1.5 * R * np.sin(theta1), c0 + 1.5 * R * np.cos(theta1)

    mask_circle = torch.zeros(mask_size, dtype=bool)
    mask_poly = torch.zeros(mask_size, dtype=bool)

    rr, cc = draw.disk((r0, c0), R, shape=mask_circle.shape)
    mask_circle[rr, cc] = 1

    rr, cc = draw.polygon(
        [r0, r1, r2, r0],
        [c0, c1, c2, c0], 
        shape=mask_poly.shape
    )

    mask_poly[rr, cc] = 1
    mask = mask_circle & mask_poly
    mask = mask * mask_value
    return mask

def bandmasks(num_bands, band_intensities):
    radii = torch.linspace(0, np.sqrt(2*64**2), num_bands+1)
    angles = torch.linspace(0, 360, num_bands+1)
    
    rmasks = []
    amasks = []
    
    for i in range(num_bands):
        r = radial_band(radii[i], radii[i+1], mask_value=band_intensities[i])
        a = circular_band(angles[i], angles[i+1], mask_value=band_intensities[i])
        if i==0:
            rmasks.append(r)
            amasks.append(a)
        else:
            r[r==rmasks[i-1]] = 0.
            a[a==amasks[i-1]] = 0.
            if i==num_bands-1:
                r[r==rmasks[0]] = 0.
                a[a==amasks[0]] = 0.
                
            rmasks.append(r)
            amasks.append(a)
            
    return rmasks, amasks

def energies(img, num_bands=32):
    img = img/torch.norm(img, p=2)
    rmasks, amasks = bandmasks(num_bands, torch.ones(num_bands))
    radial = torch.zeros(num_bands)
    angular = torch.zeros(num_bands)
    
    for i in range(num_bands):
        radial[i] = torch.norm(rmasks[i] * img, p=2)
        angular[i] = torch.norm(amasks[i] * img, p=2)
        
    return radial, angular

def get_sectors(num_bands):
    band_intensities = np.linspace(0, 1, num_bands)
    radial_masks, angular_masks = bandmasks(num_bands, band_intensities)
    return sum(radial_masks), sum(angular_masks)