import os
import logging
import psutil
import numpy as np
import pandas as pd
from copy import deepcopy


def flatten(xss):
    '''
    Flatten a list of lists
    
    Input:
    ------
    xss: list of lists
    
    Return: 
    ------
    flattened list
    '''
    return np.array([x for xs in xss for x in xs])

def capitalize(s):
    '''
    Capitalize a string
    
    Input:
    ------
    s: string
    
    Return:
    -------
    capitalized string
    '''
    return s[0].upper() + s[1:]

def matrix_to_edge(m, rownames, colnames):
    '''
    Convert matrix to edge list
    p.s. row for regulator, column for target
    
    
    Parameters:
    -----------
    m: matrix
    rownames: list of regulator names
    colNames: list of target names

    Return:
    -------
    edge DataFrame [TF, Target, Score]
    '''
    
    mat = deepcopy(m)
    mat = pd.DataFrame(mat)

    rownames = np.array(rownames)
    colnames = np.array(colnames)
    
    num_regs = rownames.shape[0]
    num_targets = colnames.shape[0]

    mat_indicator_all = np.zeros([num_regs, num_targets])

    mat_indicator_all[abs(mat) > 0] = 1
    idx_row, idx_col = np.where(mat_indicator_all)

    idx = list(zip(idx_row, idx_col))
    #for row, col in idx:
    #    if row == col:
    #        idx.remove((row, col))
                
    edges_df = pd.DataFrame(
        {'TF': rownames[idx_row], 'Target': colnames[idx_col], 'Score': [mat.iloc[row, col] for row, col in idx]})

    edge = edges_df.sort_values('Score', ascending=False)
    return edge


def dictys_binlinking2edge(file):
    prior_GRN = pd.read_csv(file, sep='\t', index_col=0)
    prior_GRN = prior_GRN.groupby(level=0).max()
    prior_GRN = prior_GRN.astype(int).copy()
    
    edge_list = matrix_to_edge(prior_GRN, prior_GRN.index, prior_GRN.columns)
    
    return edge_list


def log_memory_usage():
    """
    Log current memory usage
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")

def downsample_cells(adata, proportion, method="random"):
    """
    Downsample cells by a given proportion using the specified method
    
    Parameters:
        cell_selected (pd.DataFrame): The dataframe of selected cells
        adata (AnnData): The AnnData object containing gene expression data
        proportion (float): The proportion of cells to retain
        method (str): Sampling method, either "random" or "geosketch"

    Returns:
        pd.DataFrame: The downsampled cells
    """
    
    cell_index = adata.obs_names
    
    if method == "random":
        cell_sampled = np.random.choice(cell_index, int(len(cell_index) * proportion), replace=False)
    elif method == "geosketch":
        pass
    else:
        raise ValueError("Invalid sampling method {method}")

    return adata[cell_sampled]

from PIL import Image, ImageChops

# In newer versions of Pillow (10.0.0+), ANTIALIAS was deprecated and replaced
# by LANCZOS. This handles compatibility.
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.ANTIALIAS


def merge_images_vertically(image_paths, output_path, match_width=False):
    """
    Merges multiple images vertically into a single image.

    Args:
        image_paths (list): A list of file paths for the images to be merged.
        output_path (str): The file path to save the merged output image.
        match_width (bool): If True, resizes all images to match the widest one.
    """
    if not image_paths:
        print("No image paths were provided.")
        return

    # 1. Load all images into a list first
    images = []
    for path in image_paths:
        try:
            images.append(Image.open(path))
        except FileNotFoundError:
            print(f"Warning: File not found - {path}. Skipping this image.")
        except Exception as e:
            print(f"Warning: Could not load image {path}, error: {e}. Skipping this image.")
    
    if not images:
        print("Failed to load any images. Cannot merge.")
        return

    # 2. Find the max width
    max_width = 0
    for img in images:
        max_width = max(max_width, img.width)

    # 3. (Optional) Resize images if match_width is True
    if match_width:
        resized_images = []
        print(f"Enforcing same width. Resizing all images to {max_width} pixels wide.")
        for img in images:
            # If the image is already the max width, do nothing
            if img.width == max_width:
                resized_images.append(img)
            else:
                # Calculate the new height to maintain aspect ratio
                aspect_ratio = img.height / img.width
                new_height = int(max_width * aspect_ratio)
                
                # Resize the image with a high-quality downsampling filter
                resized_img = img.resize((max_width, new_height), RESAMPLE_FILTER)
                resized_images.append(resized_img)
        images = resized_images # Replace the original list with the resized images

    # 4. Calculate total height and create the new canvas
    total_height = sum(img.height for img in images)
    
    # The mode 'RGB' is chosen here. If your images have transparency,
    # you might want to use 'RGBA'.
    merged_image = Image.new('RGB', (max_width, total_height), (255, 255, 255)) # White background

    # 5. Paste each image into the merged image
    current_height = 0
    for img in images:
        # Since all images are the same width, the x_offset is always 0
        merged_image.paste(img, (0, current_height))
        current_height += img.height

    # 6. Save the final image
    try:
        merged_image.save(output_path)
        print(f"Images successfully merged and saved to: {output_path}")
        return merged_image # Return the merged PIL Image object
    except Exception as e:
        print(f"An error occurred while saving the merged image: {e}")
        return None


