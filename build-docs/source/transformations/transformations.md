# Transformations Collection


|No.| name | source | category | CV-HAZOP entry | note | 
|----|---|--------|----------|------------|------|
1|Gaussian Noise| imagenet-c | noise |  Observer, Quality |  | 
2|Shot Noise| imagenet-c | noise |  Observer, Quality |  |
3|Impulse Noise|imagenet-c |noise |  Observer, Quality |  | 
4|Defocus Blur|imagenet-c | blur |  Observer, Focusing (Less) |  | 
5|Frosted Glass Blur|imagenet-c | blur |  Medium, Texture |  | 
6|Motion Blur|imagenet-c| blur |  Observer, Focusing (Less) |  | 
7|Zoom Blur|imagenet-c| - | -|  Not Applicable |
8|Snow|imagenet-c| weather |  Medium, Texture |  | 
9|Frost|imagenet-c|weather |  Medium, Texture |  |
10|Fog|imagenet-c|weather |  Medium, Texture |  | 
11|Brightness|imagenet-c| contrast/brightness |  Medium, Transparency |  | 
12|Contrast|imagenet-c| contrast/brightness |  Light Sources, Intensity |  | 
13|Elastic|imagenet-c| - | -| Not Applicable |
14|Pixelate|imagenet-c| - | -| Not Applicable |
15|JPEG|imagenet-c|  digital categories |  Observer, Quantization/Sampling |  | 
16|Blur|albumentations|blur |  Observer, Focusing (Less) |  | 
17|CLAHE|albumentations| - | - | Not in any category | 
18|ChannelDropout|albumentations| - | -  | no continuous range |
19|ChannelShuffle|albumentations| - | -  | no continuous range |
20|ColorJitter|albumentations|contrast/brightness | Light Sources, Intensity + Spectrum; Medium, Transparency | Combination of transformations| 
21|Downscale|albumentations| digital categories | Observer, Quantization/Sampling |  | 
22|Emboss|albumentations| - | - |  Not in any category | 
23|Equalize|albumentations|- | - |  no continuous range |
24|FDA|albumentations|- | -|  Not Applicable (uses reference image) |
25|FancyPCA|albumentations|- | -|  Not relevant (a transformation for NN training) |
26|FromFloat|albumentations| - | - |  no continuous range |
27|GaussNoise|albumentations|- | - |  same as No.1 |
28|GaussianBlur|albumentations| blur |  Observer, Focusing (Less) |  | 
29|GlassBlur|albumentations| - |-| same as No.5 |
30|HistogramMatching|albumentations|  - | -| Not Applicable (uses reference image) |
31|HueSaturationValue |albumentations| contrast/brightness |  Light Sources, Spectrum| Combination of transformatioms | 
32|ISONoise|albumentations| noise | Observer, Quality |  | 
33|ImageCompression|albumentations| digital categories |  Observer, Quantization/Sampling |  | 
34|InvertImg|albumentations| - | - |  no continuous range |
35|MedianBlur|albumentations| blur |  Observer, Focusing (Less) |  | 
36|MotionBlur|albumentations| - |-| same as No.6 |
37|MultiplicativeNoise|albumentations| - |-| | 
38|Normalize|albumentations| - | - |  no continuous range |
39|Posterize|albumentations| - | - |  no continuous range |
40|RGBShift|albumentations| contrast/brightness |  Light Sources, Spectrum| Combination of transformatioms | 
41|RandomBrightnessContrast|albumentations| contrast/brightness | Light Sources, Intensity; Medium, Transparency| Combination of transformatioms | 
42|RandomFog|albumentations| - | - | same as No.10 | 
43|RandomGamma|albumentations| contrast/brightness |  Light Sources, Intensity |  | 
44|RandomShadow|albumentations|- | - |  Not realistic| 
45|RandomRain|albumentations| weather | Medium, Transparency/Light Sources, Intensity |  | 
46|RandomSnow|albumentations| - | - |  Not realistic| 
47|RandomSunFlare|albumentations| weather |  Medium, Transparency/Light Sources, Intensity |  | 
49|Sharpen|albumentations| - | - |  Not in any category | 
50|Solarize|albumentations| - | - |  Not relevant | 
51|Superpixels|albumentations| - | - |  Not relevant | 
52|ToFloat|albumentations| - | - |  no continuous range |
53|ToGray|albumentations| - | - |  no continuous range |
54|ToSepia|albumentations|- | - |  Not relevant | 


+ In total: 54 transformations; 49 unique ones; 45 relevant (correspond to CV-HAZOP); 40 applicable ones; 31 with continuous range; 29 realistic



