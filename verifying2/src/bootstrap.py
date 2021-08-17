import shutil
import pandas as pd
from abc import ABC, abstractmethod
import matlab.engine
import matlab
from tqdm import tqdm
from typing import Union
from .utils import clean_dir
from .Imagenet_c_transformations import *
from .constant import CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, GAUSSIAN_NOISE, SHOT_NOISE, \
    IMPULSE_NOISE, DEFOCUS_BLUR, GLASS_BLUR, MOTION_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION, \
    TRANSFORMATION_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s | %(lineno)d | %(message)s',
                              '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


class Bootstrapper(ABC):
    def __init__(self, num_sample_iter: int, sample_size: int, source: Union[str, pathlib2.Path],
                 destination: Union[str, pathlib2.Path]):
        self.num_sample_iter = num_sample_iter
        self.sample_size = sample_size
        self.source = pathlib2.Path(source)
        self.destination = pathlib2.Path(destination)
        self.data = None
        self.bootstrap_df = None

    @abstractmethod
    def _prepare(self):
        raise NotImplementedError


class ImagenetBootstrapper(Bootstrapper):
    def __init__(
            self, num_sample_iter: int, sample_size: int, source: Union[str, pathlib2.Path],
            destination: Union[str, pathlib2.Path],
            threshold: float, dataset_info_df: pd.DataFrame, transformation: str):
        super(ImagenetBootstrapper, self).__init__(num_sample_iter, sample_size, source, destination)
        self.threshold = threshold
        self.dataset_info_df = dataset_info_df
        self.transformation = transformation

    def _prepare(self):
        if not self.source.exists():
            raise ValueError(f"Source data {self.source} doesn't exist")
        if self.destination.exists():
            shutil.rmtree(self.destination)
        self.destination.mkdir(parents=True, exist_ok=True)

    def run(self, matlab_engine) -> pd.DataFrame:
        """run bootstrapping process to generate and save transformed images

        :param matlab_engine: matlab engine object, can be created useing matlab.start_matlab()
        :raises ValueError: Invalid transformation type
        :return: info of generated bootstrapping images
        :rtype: pd.DataFrame
        """
        self._prepare()
        logger.info("bootstrapping")
        self.data = []
        for i in tqdm(range(self.num_sample_iter)):
            sample_images = self.dataset_info_df.sample(n=self.sample_size, replace=False)
            iteration_path = self.destination / f'batch_{i}'  # output path for current bootstrap iteration
            clean_dir(iteration_path)
            k = 0
            for j, row in sample_images.iterrows():
                cur_row = row
                image_name = cur_row['original_filename']
                image_path = cur_row['original_path']
                if self.transformation in [
                        GAUSSIAN_NOISE, SHOT_NOISE, IMPULSE_NOISE, MOTION_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST,
                        JPEG_COMPRESSION]:
                    img = Image.open(image_path)
                else:
                    img = np.asarray(cv2.imread(image_path), dtype=np.float32)
                img_g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # greyscale image
                while True:
                    # ============= different transformation types begin =============
                    if self.transformation == CONTRAST_G:
                        param = random.choice(contrast_params)
                        img2 = adjust_contrast(img, param)
                    elif self.transformation == UNIFORM_NOISE:
                        param = random.choice(uniform_noise_params)
                        img2 = apply_uniform_noise(img, 0, param)
                    elif self.transformation == LOWPASS:
                        param = random.choice(lowpass_params)
                        img2 = low_pass_filter(img, param)
                    elif self.transformation == HIGHPASS:
                        param = random.choice(highpass_params)
                        img2 = high_pass_filter(img, param)
                    elif self.transformation == PHASE_NOISE:
                        param = random.choice(phase_noise_params)
                        img2 = scramble_phases(img, param)
                    elif self.transformation == GAUSSIAN_NOISE:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = gaussian_noise(img, param_index)
                    elif self.transformation == SHOT_NOISE:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = shot_noise(img, param_index)
                    elif self.transformation == IMPULSE_NOISE:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = impulse_noise(img, param_index)
                    elif self.transformation == DEFOCUS_BLUR:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = defocus_blur(img, param_index)
                    elif self.transformation == GLASS_BLUR:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = glass_blur(img, param_index)
                    elif self.transformation == MOTION_BLUR:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = motion_blur(img, param_index)
                    elif self.transformation == SNOW:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2 = snow(img, param_index)
                    elif self.transformation == FROST:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = frost(img, param_index)
                    elif self.transformation == FOG:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = fog(img, param_index)
                    elif self.transformation == BRIGHTNESS:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = brightness(img, param_index)
                    elif self.transformation == CONTRAST:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = contrast(img, param_index)
                    elif self.transformation == JPEG_COMPRESSION:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = jpeg_compression(img, param_index)
                        img2 = np.asarray(img2)
                        # ============= different transformation types end =============
                    else:
                        raise ValueError("Invalid Transformation")
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
                    try:
                        IQA_score = matlab_engine.vifvec2_layers(
                            matlab.double(np.asarray(img_g).tolist()),
                            matlab.double(np.asarray(img2_g).tolist()))
                    except Exception as e:
                        logger.error("failed")
                        cur_row = self.dataset_info_df.sample(n=1).iloc[0]
                        continue
                    if 1 - IQA_score < self.threshold:
                        # path to save transformed image
                        new_name = f'{self.transformation}_{image_name}'
                        output_path = str(iteration_path / new_name)
                        cv2.imwrite(output_path, img2)
                        self.data.append({
                            "bootstrap_iter_id": i,
                            "within_iter_id": k,
                            "original_filename": cur_row['original_filename'],
                            "new_filename": new_name,
                            "original_path": cur_row['original_path'],
                            "transformation": self.transformation,
                            "new_path": output_path,
                            "label": cur_row['label'],
                            "vd_score": 1 - IQA_score
                        })
                        break
                k += 1

        self.bootstrap_df = pd.DataFrame(data=self.data)
        return self.bootstrap_df


class Cifar10Bootstrapper(Bootstrapper):
    def __init__(
            self, num_sample_iter: int, sample_size: int, source: Union[str, pathlib2.Path],
            destination: Union[str, pathlib2.Path],
            threshold: float, dataset_info_df: pd.DataFrame, transformation: str):
        super(Cifar10Bootstrapper, self).__init__(num_sample_iter, sample_size, source, destination)
        self.threshold = threshold
        self.dataset_info_df = dataset_info_df
        self.transformation = transformation

    def _prepare(self):
        if not self.source.exists():
            raise ValueError(f"Source data {self.source} doesn't exist")
        if self.destination.exists():
            shutil.rmtree(self.destination)
        self.destination.mkdir(parents=True, exist_ok=True)

    def run(self, matlab_engine) -> pd.DataFrame:
        """run bootstrapping process to generate and save transformed images

        :param matlab_engine: matlab engine object, can be created useing matlab.start_matlab()
        :raises ValueError: Invalid transformation type
        :return: info of generated bootstrapping images
        :rtype: pd.DataFrame
        """
        self._prepare()
        logger.info("bootstrapping")
        self.data = []
        for i in tqdm(range(self.num_sample_iter)):
            sample_images = self.dataset_info_df.sample(n=self.sample_size, replace=False)
            iteration_path = self.destination / f'batch_{i}'  # output path for current bootstrap iteration
            clean_dir(iteration_path)
            k = 0
            for j, row in sample_images.iterrows():
                cur_row = row
                image_name = cur_row['original_filename']
                image_path = cur_row['original_path']
                if self.transformation in [
                        GAUSSIAN_NOISE, SHOT_NOISE, IMPULSE_NOISE, MOTION_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST,
                        JPEG_COMPRESSION]:
                    img = Image.open(image_path)
                else:
                    img = np.asarray(cv2.imread(image_path), dtype=np.float32)
                img_g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # greyscale image
                while True:
                    # ============= different transformation types begin =============
                    if self.transformation == CONTRAST_G:
                        param = random.choice(contrast_params)
                        img2 = adjust_contrast(img, param)
                    elif self.transformation == UNIFORM_NOISE:
                        param = random.choice(uniform_noise_params)
                        img2 = apply_uniform_noise(img, 0, param)
                    elif self.transformation == LOWPASS:
                        param = random.choice(lowpass_params)
                        img2 = low_pass_filter(img, param)
                    elif self.transformation == HIGHPASS:
                        param = random.choice(highpass_params)
                        img2 = high_pass_filter(img, param)
                    elif self.transformation == PHASE_NOISE:
                        param = random.choice(phase_noise_params)
                        img2 = scramble_phases(img, param)
                    elif self.transformation == GAUSSIAN_NOISE:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = gaussian_noise(img, param_index)
                    elif self.transformation == SHOT_NOISE:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = shot_noise(img, param_index)
                    elif self.transformation == IMPULSE_NOISE:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = impulse_noise(img, param_index)
                    elif self.transformation == DEFOCUS_BLUR:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = defocus_blur(img, param_index)
                    elif self.transformation == GLASS_BLUR:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = glass_blur(img, param_index)
                    elif self.transformation == MOTION_BLUR:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, param = motion_blur(img, param_index)
                    elif self.transformation == SNOW:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2 = snow(img, param_index)
                    elif self.transformation == FROST:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = frost(img, param_index)
                    elif self.transformation == FOG:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = fog(img, param_index)
                    elif self.transformation == BRIGHTNESS:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = brightness(img, param_index)
                    elif self.transformation == CONTRAST:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = contrast(img, param_index)
                    elif self.transformation == JPEG_COMPRESSION:
                        param_index = random.choice(range(TRANSFORMATION_LEVEL))
                        img2, _ = jpeg_compression(img, param_index)
                        img2 = np.asarray(img2)
                        # ============= different transformation types end =============
                    else:
                        raise ValueError("Invalid Transformation")
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        IQA_score = matlab_engine.vifvec2_layers(
                            matlab.double(np.asarray(img_g).tolist()),
                            matlab.double(np.asarray(img2_g).tolist()))
                    except Exception as e:
                        logger.error("failed")
                        cur_row = self.dataset_info_df.sample(n=1).iloc[0]
                        continue
                    if 1 - IQA_score < self.threshold:
                        # path to save transformed image
                        new_name = f'{self.transformation}_{image_name}'
                        output_path = str(iteration_path / new_name)
                        cv2.imwrite(output_path, img2)
                        self.data.append({
                            "bootstrap_iter_id": i,
                            "within_iter_id": k,
                            "original_filename": cur_row['original_filename'],
                            "new_filename": new_name,
                            "original_path": cur_row['original_path'],
                            "transformation": self.transformation,
                            "new_path": output_path,
                            "label": cur_row['label'],
                            "vd_score": 1 - IQA_score
                        })
                        break
                k += 1

        self.bootstrap_df = pd.DataFrame(data=self.data)
        return self.bootstrap_df
