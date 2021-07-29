import os
import cv2
import sys
import shutil
import random
import logging
import pathlib2
import numpy as np
import pandas as pd
import matlab.engine
from tqdm import tqdm
from src.helper import dir_is_empty, load_image_data, bootstrap_save_record, load_cifar10c_image_data
from ref.Imagenet_c_transformations import *
from .constant import CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, GAUSSIAN_NOISE, SHOT_NOISE, IMPULSE_NOISE, DEFOCUS_BLUR, GLASS_BLUR, MOTION_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION, TRANSFORMATION_LEVEL

__root__ = pathlib2.Path(__file__).absolute().parent.parent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.WARNING)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

IQA = 'vif/vifvec_release'
IQA_PATH = os.path.join(__root__, 'image-quality-tools', 'metrix_mux', 'metrix', IQA)
matlabPyrToolsPath = os.path.join(IQA_PATH, "matlabPyrTools")


def gen_bootstrap(
        num_batch: int, orig_path: str, gen_path: str, t: float, batch_size: int = 50,
        transformation=GAUSSIAN_NOISE, dataset_type: str = 'val') -> pd.DataFrame:
    logger.debug("transformation")
    gen_path = pathlib2.Path(gen_path).absolute()
    logger.info("remove existing output directory")
    if gen_path.exists() and not dir_is_empty(gen_path):
        shutil.rmtree(gen_path)
    logger.info("recreate output directory")
    gen_path.mkdir(parents=True, exist_ok=True)

    logger.info("load ground truth")
    dataset_df = load_image_data(__root__, dataset_type)

    logger.info("start matlab")
    eng = matlab.engine.start_matlab()
    eng.addpath(IQA_PATH, nargout=0)
    eng.addpath(matlabPyrToolsPath, nargout=0)
    eng.addpath(matlabPyrToolsPath + '/MEX', nargout=0)

    logger.info("start bootstrapping")
    bootstrap_data = {
        'batch_id': [],
        'within_batch_id': [],
        'filename': [],
        'original_filename': [],
        'original_path': [],
        'transformed_path': [],
        'label': [],
        'transformation': [],
        'vd_score': []
    }
    for batch_id in tqdm(range(num_batch)):
        logger.debug("batch " + str(batch_id))
        sample_batch = dataset_df.sample(n=batch_size, replace=False)
        batch_path = gen_path / f'batch_{batch_id}'
        if not batch_path.exists():
            batch_path.mkdir(parents=True, exist_ok=True)
        j = 0
        for i, row in sample_batch.iterrows():
            j += 1
            cur_row = row
            image_name = cur_row['filename']
            image_path = cur_row['path']
            # Start transformations
            if transformation in [
                    GAUSSIAN_NOISE, SHOT_NOISE, IMPULSE_NOISE, MOTION_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST,
                    JPEG_COMPRESSION]:
                img = Image.open(image_path)
            else:
                img = np.asarray(cv2.imread(image_path), dtype=np.float32)
            img_g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   # greyscale image
            while True:
                # ============= different transformation types begin =============
                if transformation == CONTRAST_G:
                    param = random.choice(contrast_params)
                    img2 = adjust_contrast(img, param)
                elif transformation == UNIFORM_NOISE:
                    param = random.choice(uniform_noise_params)
                    img2 = apply_uniform_noise(img, 0, param)
                elif transformation == LOWPASS:
                    param = random.choice(lowpass_params)
                    img2 = low_pass_filter(img, param)
                    img2 = apply_uniform_noise(img, 0, param)
                elif transformation == HIGHPASS:
                    param = random.choice(highpass_params)
                    img2 = high_pass_filter(img, param)
                    img2 = apply_uniform_noise(img, 0, param)
                elif transformation == PHASE_NOISE:
                    param = random.choice(phase_noise_params)
                    img2 = scramble_phases(img, param)
                    img2 = apply_uniform_noise(img, 0, param)
                elif transformation == GAUSSIAN_NOISE:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = gaussian_noise(img, param_index)
                elif transformation == SHOT_NOISE:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = shot_noise(img, param_index)
                elif transformation == IMPULSE_NOISE:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = impulse_noise(img, param_index)
                elif transformation == DEFOCUS_BLUR:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = defocus_blur(img, param_index)
                elif transformation == GLASS_BLUR:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = glass_blur(img, param_index)
                elif transformation == MOTION_BLUR:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = motion_blur(img, param_index)
                elif transformation == SNOW:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2 = snow(img, param_index)
                elif transformation == FROST:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, _ = frost(img, param_index)
                elif transformation == FOG:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, _ = fog(img, param_index)
                elif transformation == BRIGHTNESS:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, _ = brightness(img, param_index)
                elif transformation == CONTRAST:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, _ = contrast(img, param_index)
                elif transformation == JPEG_COMPRESSION:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, _ = jpeg_compression(img, param_index)
                    img2 = np.asarray(img2)
                    # ============= different transformation types end =============
                else:
                    raise ValueError("Invalid Transformation")
                img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                try:
                    IQA_score = eng.vifvec2_layers(
                        matlab.double(np.asarray(img_g).tolist()),
                        matlab.double(np.asarray(img2_g).tolist()))
                except:
                    logger.error("failed")
                    cur_row = dataset_df.sample(n=1).iloc[0]
                    continue
                if 1-IQA_score < t:
                    # path to save transformed image
                    new_name = f'{transformation}_{j}_{image_name}'
                    output_path = str(batch_path/new_name)

                    bootstrap_save_record(
                        batch_id, j, cur_row['filename'],
                        new_name, cur_row['path'], transformation,
                        output_path, cur_row['label'], 1-IQA_score,
                        img2, bootstrap_data)
                    break

    bootstrap_df = pd.DataFrame(data=bootstrap_data)
    return bootstrap_df


def gen_cifar10c_bootstrap(num_batch: int, gen_path: str, corruption_type: str, batch_size: int = 50) -> pd.DataFrame:
    gen_path = pathlib2.Path(gen_path).absolute()
    if gen_path.exists() and not dir_is_empty(gen_path):
        shutil.rmtree(gen_path)
    logger.info("recreate output directory")
    gen_path.mkdir(parents=True, exist_ok=True)

    logger.info("load ground truth")
    dataset_df = load_cifar10c_image_data(__root__, corruption_type)

    logger.info("start bootstrapping")
    bootstrap_data = {
        'batch_id': [],
        'within_batch_id': [],
        'filename': [],
        'original_filename': [],
        'original_path': [],
        'transformed_path': [],
        'label': [],
        'transformation': []
    }
    for batch_id in tqdm(range(num_batch)):
        logger.debug("batch " + str(batch_id))
        sample_batch = dataset_df.sample(n=batch_size, replace=False)
        batch_path = gen_path / f'batch_{batch_id}'
        if not batch_path.exists():
            batch_path.mkdir(parents=True, exist_ok=True)
        j = 0
        for i, row in sample_batch.iterrows():
            j += 1
            cur_row = row
            image_name = cur_row['filename']
            image_path = cur_row['path']
            output_path = str(batch_path/image_name)
            # shutil.copyfile(image_path, output_path)
            os.symlink(image_path, output_path)
            bootstrap_data['batch_id'].append(batch_id)
            bootstrap_data['within_batch_id'].append(j)
            bootstrap_data['original_filename'].append(image_name)
            bootstrap_data['filename'].append(image_name)
            bootstrap_data['original_path'].append(image_path)
            bootstrap_data['transformation'].append(corruption_type)
            bootstrap_data['transformed_path'].append(output_path)
            bootstrap_data['label'].append(cur_row['label'])

    bootstrap_df = pd.DataFrame(data=bootstrap_data)
    return bootstrap_df
