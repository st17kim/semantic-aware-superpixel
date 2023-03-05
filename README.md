# semantic-aware-superpixel
AAAI 2023, "Semantic-aware Superpixel for Weakly Supervised Semantic Segmentation"
-----------------------------------
![supercomp](https://user-images.githubusercontent.com/105955670/221941089-693b1595-cac6-44d9-9d91-4da2e0bf0157.png)

Requirements
-----------------------------
Our code requires Tensorflow 2.x

Preparation
-------------------------
1. Download the models and place them in "model/" directory.

>For the first trial, try with DINO-16-small (which is default in code)

  + DINO-16-small https://drive.google.com/file/d/1J0G1qkX_bMnwmORnTfjyeqyzkHaRZsBA/view?usp=sharing

  + DINO-16-base https://drive.google.com/file/d/1Jdti4Q4-ehEYDECv_-9D3MUGEZBR3lD8/view?usp=sharing

  + DINO-8-small https://drive.google.com/file/d/1sZVzm7zP4g9a_DpjdmgRDSB3C37OQGho/view?usp=sharing

  + DINO-8-base https://drive.google.com/file/d/1pAh5oQPHK7zkyAVJO-VNP_TSdxv9MUY3/view?usp=sharing

2. install pydensecrf (Optional but recommended)
>Raw superpixel is very noisy. To use CRF for post-processing, pydensecrf is required.

>You may install pydensecrf using: 
```conda install -c conda-forge pydensecrf```

Superpixel generation
-----------
>Running "superpixel.py" may generate superpixels for "eg.jpg" in "superpixel/".

  + You can try to use different tau to control the number of superpixel. (0~0.5 is recommended)
  + You may use ```use_crf=True``` after installing pydensecrf to post-process and obtain high-quality superpixels
  + You may use different DINO model to generate superpixels. 
    + '8-xx' model can generate better superpixel but may not work with large input image.
    + Generally, '8-xx' is better than '16-xx' and '-base' is better than '-small'.

Precomputed Superpixels
--------------
>We provide superpixels generated from PASCAL VOC 2012 (10,582 images) and MS-COCO 2017 (118,287 images) segmentation datasets. 
+ We used DINO-8-base and set tau to 0.3.
+ Open superpixels using ``` from PIL import Image; superpixel = np.array(Image.open("img_name.png"))```. You will get np array size of [height, width] whose value indicate the id of superpixel (ranging from 1 to the number of superpixels). 
+ COCO train 2017 dataset (118,287 images) includes COCO train 2014 (82,783 images).
+ Pascal superpixels: https://drive.google.com/file/d/147Ev_SkHd_M8c6scco336KWFNNV5LlXv/view?usp=sharing
+ COCO superpixels: https://drive.google.com/file/d/1wc-yuhQ2zgd5PRNqBNC-qAEECaL7PN_M/view?usp=sharing
