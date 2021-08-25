from src import constant

if __name__ == '__main__':
    for trans in ["GAUSSIAN_NOISE",
                  "SHOT_NOISE",
                  "IMPULSE_NOISE",
                  "MOTION_BLUR",
                  "SNOW",
                  "FROST",
                  "FOG",
                  "BRIGHTNESS",
                  "CONTRAST",
                  "JPEG_COMPRESSION",
                  "ALEXNET",
                  "GOOGLENET",
                  "RESNET50",
                  "RESNEXT50",
                  "VGG_16",
                  "THRESHOLD_MAP",
                  "ROBUSTBENCH_CIFAR10_MODEL_NAMES",
                  "GENERALIZED",
                  "CONTRAST_G",
                  "UNIFORM_NOISE",
                  "LOWPASS",
                  "HIGHPASS",
                  "PHASE_NOISE",
                  "DEFOCUS_BLUR",
                  "GLASS_BLUR"]:
        if not hasattr(constant, trans):
            print(trans)
