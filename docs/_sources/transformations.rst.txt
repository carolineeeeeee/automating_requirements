.. _transformation:

########################
Visual changes in images
########################

In this page, we first present the metric :math:`\Delta_v`, which measures human visual changes in images caused by transformations.
Then, we identify a class of *safety-related* image transformations.

Measuring Visual Change (:math:`\Delta_v`)
==========================================

Let an image :math:`x`, an applicable transformation :math:`T_X` with a parameter domain :math:`C` and a parameter :math:`c\in C`, s.t. :math:`x' = T_X(x,c)` be given.

:math:`\Delta_v(x,x')` is a function defined as follows:


.. image:: imgs/visual-changes.png
  :alt: visual-changes
  
The VSNR [Chandler_Hemami_07]_ and VIF [Sheikh_Bovik_06]_ implementation can be found through their papers or from this Github `link <https://github.com/sattarab/image-quality-tools>`_ and the additional required python library matlabPyrTools is available on this `page <http://www.cns.nyu.edu/~lcv/software.php>`_.


Safety-Related Transformation
=============================

We say that a transformation is *safety-related* if its negative effect can occur in a real-world machine vision scenario.
To assess this systematically, we utilize the CV-HAZOP checklist [CV_HAZOP]_ which comprehensively identifies the negative impacts of different modes of interference in the vision pipeline.

machine vision pipeline (taken from [CV_HAZOP]_)
""""""""""""""""""""""""""""""""""""""""""""""""

.. image:: imgs/vision-pipeline.png
  :alt: machine vision pipeline

CV-HAZOP checklist provides a list of properties, i.e., parameters that can change at each pipeline location.  
Please find below, the changeable properties taken from [CV_HAZOP] sorted according to their applicability to our visual change metric :math:`\Delta_v`. Columns represent the [CV_HAZOP]_ parameters’ for the perception pipeline. Rows  correspond to the parameters’ applicability in our approach

Changeable CV-HAZOP properties that are applicable to :math:`\Delta_v`
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. image:: imgs/properties.png
  :alt: changeable properties
  
  
Each checklist entry consists of a location, a parameter and a guide word indicating where, what and how much change occurs. 
It also provides the meaning, consequence and possible risk of this entry, as determined by domain experts, for example the example shown below:

Example transformations and corresponding CV-HAZOP entries
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. image:: imgs/entries-example.png
  :alt: example entries
  

Risk number 184 in the checklist has a location *Medium*, a parameter *Transparency* and a guide word *less (less of, lower)*. 
This entry means that the medium is optically thicker, thus reducing contrast, which could potentially lead to mistakes.  
  

Since the scope of CV-HAZOP is broader than this image transformation assessment task, we remove those hazard scenarios entries that are not image-related from the checklist.
To determine whether a given image transformation belongs to our safety-related class, one should first identify the location in the machine vision pipeline to which the transformation corresponds; then the property of the location that the transformation is affecting (CV-HAZOP parameters); and finally, how the transformation is changing the property (CV-HAZOP guide words). For example, defocus blur is changing the focus of the observer (CV-HAZOP entry No.1018), i.e., camera, and therefore belongs in our class. 

Find `here <_static/Our_Annotaed_Verison_of_CV-HAZOP_Checklist.pdf>`_ the full list of CV-HAZOP safety-related entries.

In the following table, we consider the transformations provided by the state-of-the-art library albumentations [albumentation]_ and ML robustness benchmark imagenet-c~\footnote{hendrycks2019robustness}, which consist of 50 unique transformations.
From these, we identified the ones that are *safety-related* (50 to 45), omitting the ones not applicable to our approach and the ones that can not produce a continuous range of transformed images (45 to 31).

.. toctree::
   :maxdepth: 2

   transformations/transformation-collection.rst


.. [Chandler_Hemami_07]

   Author: D. Chandler and S. Hemami
   
   Title: VSNR: A Wavelet-Based Visual Signal-to-Noise Ratio for Natural Images
   
   Year: 2007
   
.. [Sheikh_Bovik_06]

   Author: H. Sheikh and A. Bovik
   
   Title: Image Information and Visual Quality
   
   Year: 2006
   
.. [CV_HAZOP]

   Author: O. Zendel and others
   
   Title: CV-HAZOP: Introducing Test Data Validation for Computer Vision
   
   Year: 2015
   
.. [albumentation]
   Author: Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.
   
   Title: Albumentations: Fast and Flexible Image Augmentations
   
   Year: 2020
   
   URL: https://www.mdpi.com/2078-2489/11/2/125
