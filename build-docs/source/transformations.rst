***************
Transformations
***************

In this page, we identify a class of *safety-related* image transformations
We say that a transformation is *safety-related* if its negative effect can occur in a real-world machine vision scenario.
To assess this systematically, we utilize the CV-HAZOP checklist [CV_HAZOP]_ which comprehensively identifies the negative impacts of different modes of interference in the vision process.
A transformation that can produce such impacts is considered safety-related.
In the following table, we consider the transformations provided by the state-of-the-art library albumentations [albumentation]_ and ML robustness benchmark imagenet-c~\footnote{hendrycks2019robustness}, which consist of 50 unique transformations.
From these, we identified the ones that are *safety-related* (50 to 45), omitting the ones not applicable to our approach and the ones that can not produce a continuous range of transformed images (45 to 31).

.. toctree::
   :maxdepth: 2
   :caption: Transformations

   transformations/transformation-collection.rst

.. [CV_HAZOP]

   author={O. {Zendel} and others}
   
   booktitle={ICCV'15}
   
   title={{CV-HAZOP: Introducing Test Data Validation for Computer Vision}}
   
   year={2015}
   
   pages={2066-2074}

.. [albumentation]
   AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
   
   TITLE = {{Albumentations: Fast and Flexible Image Augmentations}},
   
   JOURNAL = {Information},
   
   VOLUME = {11},
   
   YEAR = {2020},
   
   NUMBER = {2},
   
   ARTICLE-NUMBER = {125},
   
   URL = {https://www.mdpi.com/2078-2489/11/2/125},
   
   ISSN = {2078-2489},
   
   DOI = {10.3390/info11020125}