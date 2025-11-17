import os

def write_png_paths(folder_path, txt_path):
    with open(txt_path, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('_S2.tif'):
                    f.write(os.path.join(root, file) + '\n')

# Example usage:
folder_path = '/mnt/researchdrive/Agirculture/LSSR_split_v2/test'
txt_path = './gt_selected_path_tif_test_v2.txt'
write_png_paths(folder_path, txt_path)