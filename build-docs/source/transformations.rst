***************
Visual changes
***************
In this page, we first present the metric :math:`\Delta_v`, which measures human visual changes in images caused by transformations.
Then, we identify a class of *safety-related* image transformations.

Visual Change (:math:`\Delta_v`)
================================


Safety-Related Transformation
=============================

We say that a transformation is *safety-related* if its negative effect can occur in a real-world machine vision scenario.
To assess this systematically, we utilize the CV-HAZOP checklist [CV_HAZOP]_ which comprehensively identifies the negative impacts of different modes of interference in the vision process.
Since the scope of CV-HAZOP is broader than this image transformation assessment task, we remove those hazard scenarios entries that are not image-related from the checklist.
To determine whether a given image transformation belongs to our safety-related class, one should first identify the location in the machine vision pipeline to which the transformation corresponds; then the property of the location that the transformation is affecting (CV-HAZOP parameters); and finally, how the transformation is changing the property (CV-HAZOP guide words). For example, defocus blur is changing the focus of the observer (CV-HAZOP entry No.1018), i.e., camera, and therefore belongs in our class. Find `here <_static/Our_Annotaed_Verison_of_CV-HAZOP_Checklist.pdf>`_ the full list of CV-HAZOP safety-related entries.

In the following table, we consider the transformations provided by the state-of-the-art library albumentations [albumentation]_ and ML robustness benchmark imagenet-c~\footnote{hendrycks2019robustness}, which consist of 50 unique transformations.
From these, we identified the ones that are *safety-related* (50 to 45), omitting the ones not applicable to our approach and the ones that can not produce a continuous range of transformed images (45 to 31).

.. toctree::
   :maxdepth: 2

   transformations/transformation-collection.rst

.. [CV_HAZOP]

   Author: O. Zendel and others
   
   Title: CV-HAZOP: Introducing Test Data Validation for Computer Vision
   
   Year: 2015
   
.. [albumentation]
   Author: Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.
   
   Title: Albumentations: Fast and Flexible Image Augmentations
   
   Year: 2020
   
   URL: https://www.mdpi.com/2078-2489/11/2/125
