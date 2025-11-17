import os
import random
import torch
import rasterio
from pyproj import Transformer
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split

        if split == 'train':
            with open(args.dataset_txt_paths, 'r') as f:
                self.lr_list = sorted([line.strip() for line in f.readlines()])
            with open(args.highquality_dataset_txt_paths, 'r') as f:
                self.hr_list = sorted([line.strip() for line in f.readlines()])
            with open(args.dem_dataset_txt_paths, 'r') as f:
                self.dem_list = sorted([line.strip() for line in f.readlines()])
            with open(args.landcover_dataset_txt_paths, 'r') as f:
                self.landcover_list = sorted([line.strip() for line in f.readlines()])
            with open(args.s1_dataset_txt_paths, 'r') as f:
                self.s1_list = sorted([line.strip() for line in f.readlines()])

            assert len(self.lr_list) == len(self.hr_list) == len(self.landcover_list) == len(self.dem_list) == len(self.s1_list)

            self.random_crop = transforms.RandomCrop((args.resolution_ori, args.resolution_ori))
            self.resize = transforms.Resize((args.resolution_tgt, args.resolution_tgt))
            self.flip = transforms.RandomHorizontalFlip()
        
        elif split == 'eval':
            self.input_folder = os.path.join(args.dataset_test_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(args.dataset_test_folder, "test_HR")
            self.lr_list = sorted([
                os.path.join(self.input_folder, f)
                for f in os.listdir(self.input_folder)
                if f.endswith('L8.tif')
            ])
            self.hr_list = sorted([
                os.path.join(self.output_folder, f)
                for f in os.listdir(self.output_folder)
                if f.endswith('S2.tif')
            ])
            self.dem_list = sorted([
                os.path.join(self.input_folder, f)
                for f in os.listdir(self.input_folder)
                if f.endswith('DEM.tif')
            ])
            self.landcover_list = sorted([
                os.path.join(self.input_folder, f)
                for f in os.listdir(self.input_folder)
                if f.endswith('LC.tif')
            ])
            self.s1_list = sorted([
                os.path.join(self.input_folder, f)
                for f in os.listdir(self.input_folder)
                if f.endswith('S1.tif')
            ])

            assert len(self.lr_list) == len(self.hr_list) == len(self.landcover_list) == len(self.dem_list) == len(self.s1_list)

            self.resize = transforms.Compose([
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])

        elif split == 'test':
            with open(args.dataset_txt_paths, 'r') as f:
                self.lr_list = sorted([line.strip() for line in f.readlines()])
            with open(args.highquality_dataset_txt_paths, 'r') as f:
                self.hr_list = sorted([line.strip() for line in f.readlines()])
            with open(args.dem_dataset_txt_paths, 'r') as f:
                self.dem_list = sorted([line.strip() for line in f.readlines()])
            with open(args.landcover_dataset_txt_paths, 'r') as f:
                self.landcover_list = sorted([line.strip() for line in f.readlines()])
            with open(args.s1_dataset_txt_paths, 'r') as f:
                self.s1_list = sorted([line.strip() for line in f.readlines()])

            assert len(self.lr_list) == len(self.hr_list) == len(self.landcover_list) == len(self.dem_list) == len(self.s1_list)

            self.resize = transforms.Resize((args.resolution_tgt, args.resolution_tgt))

    def __len__(self):
        return len(self.lr_list)

    def __getitem__(self, idx):
        lr_img = self.load_tiff(self.lr_list[idx])
        hr_img = self.load_tiff(self.hr_list[idx])
        dem_img = self.load_dem(self.dem_list[idx])  # (1, H, W)
        landcover_img = self.load_landcover(self.landcover_list[idx])  # (1, H, W)
        s1_img = self.load_s1(self.s1_list[idx])                  # (2,H,W), [VV, VH] → [-1,1]

        # time info
        month_index = self.parse_month_index_from_filename(self.lr_list[idx])  # int 0~5
        month_index = torch.tensor([month_index], dtype=torch.long)  # Tensor (1,)

        if self.split == 'train':
            # resize
            lr_img = self.resize(lr_img)
            hr_img = self.resize(hr_img)
            dem_img = self.resize(dem_img)
            landcover_img = self.resize(landcover_img)
            s1_img = self.resize(s1_img)

            # Horizontal flip
            if random.random() < 0.5:
                lr_img = F.hflip(lr_img)
                hr_img = F.hflip(hr_img)
                dem_img = F.hflip(dem_img)
                landcover_img = F.hflip(landcover_img)
                s1_img = F.hflip(s1_img)

            # Vertical flip
            if random.random() < 0.5:
                lr_img = F.vflip(lr_img)
                hr_img = F.vflip(hr_img)
                dem_img = F.vflip(dem_img)
                landcover_img = F.vflip(landcover_img)
                s1_img = F.vflip(s1_img)

        elif self.split == 'eval':
            lr_img = self.resize(lr_img)
            hr_img = self.resize(hr_img)
            dem_img = self.load_dem(self.dem_list[idx])  # (1, H, W)
            landcover_img = self.load_landcover(self.landcover_list[idx])  # (1, H, W)
            s1_img = self.load_s1(self.s1_list[idx])    

        elif self.split == 'test':
            lr_img = self.resize(lr_img)
            hr_img = self.resize(hr_img)
            dem_img = self.load_dem(self.dem_list[idx])  # (1, H, W)
            landcover_img = self.load_landcover(self.landcover_list[idx])  # (1, H, W)
            s1_img = self.load_s1(self.s1_list[idx])    
    

        return {
            "conditioning_pixel_values": lr_img,    # ([6, 192, 192])
            "output_pixel_values": hr_img,          # ([6, 192, 192])
            "ir_lr_values": lr_img[[3,4,5], :, :],  # ([3, 192, 192])
            "ir_hr_values": hr_img[[3,4,5], :, :],  # ([3, 192, 192])
            "dem_values": dem_img,              # ([1, 192, 192])
            "landcover_values": landcover_img,  # ([1, 192, 192])
            "s1_values": s1_img,  
            "neg_prompt": self.args.neg_prompt_csd,
            "null_prompt": "",
            "base_name": os.path.basename(self.lr_list[idx]),
            "month_index": month_index.squeeze(0),
            "location_center": self.get_center_location(self.lr_list[idx])
        }
    
    def get_center_location(self, path):
        with rasterio.open(path) as src:
            bounds = src.bounds
            lon_center = (bounds.left + bounds.right) / 2
            lat_center = (bounds.top + bounds.bottom) / 2

            
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(lon_center, lat_center)

        return torch.tensor([lat, lon], dtype=torch.float32)  # [lat, lon]
        

    def load_dem(self, path):
        with rasterio.open(path) as src:
            dem = src.read(1).astype(np.float32)  # (H, W)
            dem = np.nan_to_num(dem, nan=0.0)

        # Tensor (1, H, W)
        dem = torch.from_numpy(dem).unsqueeze(0).float()

        # [0, 1]
        dem = torch.clamp(dem, 0, 300) / 300.0

        # [-1, 1]
        dem = (dem - 0.5) / 0.5

        return dem
    
    def load_landcover(self, path):
        with rasterio.open(path) as src:
            lc = src.read(1).astype(np.uint8)  # (H, W)
            lc = np.nan_to_num(lc, nan=0)
        lc = torch.from_numpy(lc).unsqueeze(0).long()  # (1, H, W)
        return lc

    def load_s1(self, path):
        with rasterio.open(path) as src:
            vv = src.read(1, masked=True).filled(0).astype(np.float32)
            vh = src.read(2, masked=True).filled(0).astype(np.float32)
            # print("VV Unique values:", np.unique(vv))

            # dB
            vv = np.power(10, vv / 10)
            vh = np.power(10, vh / 10)
            # print("VV Unique values:", np.unique(vv))

            # [0,1] 
            vv_max = float(np.nanmax(vv))
            vh_max = float(np.nanmax(vh))
            vv = vv / max(vv_max, 1e-6)
            vh = vh / max(vh_max, 1e-6)
            # print("VV Unique values:", np.unique(vv))

            # NAN → 0
            vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
            vh = np.nan_to_num(vh, nan=0.0, posinf=0.0, neginf=0.0)
            # print("VV Unique values:", np.unique(vv))

        s1 = np.stack([vv, vh], axis=0).astype(np.float32)  # (2,H,W) [VV,VH]
        s1 = torch.from_numpy(s1)

        # [-1,1]
        s1 = (s1 - 0.5) / 0.5
        return s1



    def load_tiff(self, path):
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
            img = torch.from_numpy(img).float()

        # NaN -> 0
        img = torch.nan_to_num(img, nan=0.0)

        # sensor
        sensor_type = self.detect_sensor_type(path)

        # band
        img = self.select_matched_bands(img, sensor_type)
        
        # normalize
        img = torch.clamp(img, 0, 1)
        
        img = (img - 0.5) / 0.5  # Normalize [-1, 1]

        return img

    def select_matched_bands(self, img, sensor_type):
        """
        img: Tensor (C, H, W) after load
        sensor_type: "sentinel2" or "landsat8"
        """
        if sensor_type == "sentinel2":
            # (B2, B3, B4, B8, B11, B12)
            band_indices = [1, 2, 3, 7, 10, 11]
        elif sensor_type == "landsat8":
            # (B2, B3, B4, B5, B6, B7)
            band_indices = [1, 2, 3, 4, 5, 6]
        else:
            raise ValueError(f"Unknown sensor type {sensor_type}")

        img = img[band_indices, :, :]
        return img

    def detect_sensor_type(self, path):
            basename = os.path.basename(path).lower()
            if '_l8' in basename:
                return 'landsat8'
            elif '_s2' in basename:
                return 'sentinel2'
            else:
                raise ValueError(f"Cannot detect sensor type from file name: {basename}")

    def parse_month_index_from_filename(self, filename):
        basename = os.path.basename(filename)
        date_str = basename.split('_')[1]  # '20240608'
        month = int(date_str[4:6])  # '06' -> 6

        month_to_index = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        if month in month_to_index:
            return month_to_index[month]
        else:
            raise ValueError(f"Month {month} not in supported range 5-9 in file: {filename}")




class RandomBrightnessGamma:
    def __init__(self, brightness_range=(1.2, 1.6), gamma_range=(0.6, 0.9), p=0.8):
        self.brightness_range = brightness_range
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            brightness = random.uniform(*self.brightness_range)
            gamma = random.uniform(*self.gamma_range)

            # Apply per channel
            img = F.adjust_brightness(img.permute(1, 2, 0), brightness)
            img = F.adjust_gamma(img, gamma)
            img = img.permute(2, 0, 1)
        return img


def save_rgb_composite(img_tensor, save_path, apply_brightness_enhance=True):
    """
    img_tensor: Tensor (C, H, W), normalized to [-1, 1]
    apply_brightness_enhance: if True, will scale image to target mean
    """
    img = (img_tensor * 0.5) + 0.5  # [-1,1] → [0,1]
    img = img.clamp(0, 1)

    r = img[2]  # Band 4 (Red)
    g = img[1]  # Band 3 (Green)
    b = img[0]  # Band 2 (Blue)

    rgb = torch.stack([r, g, b], dim=0)  # (3, H, W)

    if apply_brightness_enhance:
        mean_val = rgb.mean()
        target_mean = 0.35
        scale = target_mean / (mean_val + 1e-6)
        rgb = rgb * scale
        rgb = rgb.clamp(0, 1)

    rgb = (rgb * 255).byte().permute(1, 2, 0).cpu().numpy()
    img_pil = Image.fromarray(rgb)
    img_pil.save(save_path)


def save_single_band_images(img_tensor, save_dir, prefix="sample"):
    """
    img_tensor: Tensor (C=6, H, W), normalized [-1,1]
    save_dir: folder to save bands
    prefix: filename prefix
    """

    img = (img_tensor * 0.5) + 0.5  # [-1,1] → [0,1]
    img = img.clamp(0, 1)

    band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

    os.makedirs(save_dir, exist_ok=True)

    for idx, band_name in enumerate(band_names):
        band = img[idx]  # (H, W)
        band = (band * 255).byte().cpu().numpy()

        band_img = Image.fromarray(band)
        band_img.save(os.path.join(save_dir, f"{prefix}_{band_name}.png"))


def save_nir_swir_composite(img_tensor, save_path, apply_brightness_enhance=True):
    """
    Save NIR-SWIR composite from a normalized [-1, 1] image tensor.
    - img_tensor: Tensor (C, H, W), normalized to [-1, 1]
    - save_path: Output path
    - apply_brightness_enhance: If True, adjusts brightness only for visualization
    """
    img = (img_tensor * 0.5) + 0.5  # Convert from [-1,1] to [0,1]
    img = img.clamp(0, 1)

    nir = img[3]    # NIR
    swir1 = img[4]  # SWIR1
    swir2 = img[5]  # SWIR2

    composite = torch.stack([nir, swir1, swir2], dim=0)  # (3, H, W)

    if apply_brightness_enhance:
        mean_val = composite.mean()
        target_mean = 0.35
        scale = target_mean / (mean_val + 1e-6)
        composite = composite * scale
        composite = composite.clamp(0, 1)

    composite = (composite * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img_pil = Image.fromarray(composite)
    img_pil.save(save_path)


def save_ndvi_map(sr_img, save_path, red_index=2, nir_index=3):
    """
    Save NDVI map from normalized [-1, 1] image tensor.
    Output: RGB pseudocolor NDVI image using RdYlGn colormap, no borders.
    """
    import matplotlib.pyplot as plt
    img = (sr_img * 0.5) + 0.5  # [-1,1] → [0,1]
    img = img.clamp(0, 1)

    red = img[red_index]
    nir = img[nir_index]
    ndvi = (nir - red) / (nir + red + 1e-6)  # (H, W)
    ndvi = ndvi.clamp(-1, 1).cpu().numpy()

    ndvi_normalized = (ndvi + 1) / 2.0  # [-1,1] → [0,1]
    colormap = plt.get_cmap('RdYlGn')
    ndvi_colored = colormap(ndvi_normalized)[:, :, :3]  # (H, W, 3)
    ndvi_rgb = (ndvi_colored * 255).astype(np.uint8)

    Image.fromarray(ndvi_rgb).save(save_path)

def save_landcover_image(lc_tensor, save_path):
        # legend
        palette = [
            '#419bdf',  # Water
            '#397d49',  # Trees
            '#88b053',  # Grass
            '#7a87c6',  # Flooded vegetation
            '#e49635',  # Crops
            '#dfc35a',  # Shrub and scrub
            '#c4281b',  # Built
            '#a59b8f',  # Bare
            '#b39fe1'   # Snow and ice
        ]
        cmap = mcolors.ListedColormap(palette)

        lc = lc_tensor.squeeze().cpu().numpy().astype(np.uint8)  # (H, W)

        plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.imshow(lc, cmap=cmap, vmin=0, vmax=8)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def save_s1_gray(s1_tensor, prefix="sample"):
    x01 = (s1_tensor * 0.5 + 0.5).clamp(0,1)    # [-1,1]→[0,1]
    x01 = torch.pow(x01, 0.35)  # gamma
    names = ["VV", "VH"]
    for i, name in enumerate(names):
        band = (x01[i] * 255).byte().cpu().numpy()
        Image.fromarray(band).save(f"{prefix}_S1_{name}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_txt_paths", type=str, default="preset/gt_path_tif_test_v2.txt")
    parser.add_argument("--highquality_dataset_txt_paths", type=str, default="preset/gt_selected_path_tif_test_v2.txt")
    parser.add_argument("--dem_dataset_txt_paths", type=str, default="preset/dem_tif_test_v2.txt")
    parser.add_argument("--landcover_dataset_txt_paths", type=str, default="preset/landcover_tif_test_v2.txt")
    parser.add_argument("--s1_dataset_txt_paths", type=str, default="preset/s1_tif_test_v2.txt")
    parser.add_argument("--resolution_ori", type=int, default=192)
    parser.add_argument("--resolution_tgt", type=int, default=192)
    parser.add_argument("--neg_prompt_csd", type=str, default="")

    args = parser.parse_args()

    dataset = PairedSROnlineTxtDataset(split='test', args=args)

    output_dir = "gt_vis_v2"
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(max(5, len(dataset))):
        example = dataset[idx]
        base_name = os.path.splitext(example["base_name"])[0]
        lr_img = example["conditioning_pixel_values"]  # (C, H, W)
        hr_img = example["output_pixel_values"]  # (C, H, W)
        s1_img = example["s1_values"]

        # Save RGB composite images
        save_rgb_composite(lr_img, os.path.join(output_dir, f"{base_name}_LR_RGB.png"))
        save_rgb_composite(hr_img, os.path.join(output_dir, f"{base_name}_HR_RGB.png"))

        # Save NIR SWIR composite images
        save_nir_swir_composite(lr_img, os.path.join(output_dir, f"{base_name}_LR_NIRSWIR.png"))
        save_nir_swir_composite(hr_img, os.path.join(output_dir, f"{base_name}_HR_NIRSWIR.png"))

        # Save NDVI maps
        save_ndvi_map(lr_img, os.path.join(output_dir, f"{base_name}_LR_NDVI.png"))
        save_ndvi_map(hr_img, os.path.join(output_dir, f"{base_name}_HR_NDVI.png"))

        # Save Sentinel1 maps
        save_s1_gray(s1_img, os.path.join(output_dir, base_name))

        # DEM visualization
        # dem = example["dem_values"].squeeze(0).cpu().numpy()  # (H, W)
        # dem_img = (dem + 1) / 2.0        # [-1,1] → [0,1]
        # dem_img = (dem_img * 255).astype(np.uint8)
        # Image.fromarray(dem_img).save(os.path.join(output_dir, f"{base_name}_DEM.png"))

        # Landcover visualization
        # save_landcover_image(example["landcover_values"], os.path.join(output_dir, f"{base_name}_LC.png"))

    print(f"Saved debug images to {output_dir}/")
