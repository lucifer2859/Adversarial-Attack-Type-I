## Supplementary Materials (SM) for the article: Adversarial Attack Type I: Cheat Classifiers by Significant Changes

### Description
  This is a brief description of the supplemental information of the paper: Adversarial Attack Type I: Cheat Classifiers by Significant Changes, 10.1109/TPAMI.2019.2936378.
  The supplemntary materials mainly contain two parts:
    1. visualized images and videos of the Type I adversarial examples;
    2. source codes for the implement of the proposed SVAE struture and the borrowed & modified StyleGAN for generating Type I adversarial examples.
  There are several 'readme.md' files in the source code part, which explains how to conduct the Type I adversarial attack in both SVAE and the StyleGAN structure. The experiments results in the paper should be easily reproduced with the help of this supplementary materials.

### SIZE
    51.3MB

### PLAYER INFORMATION
    Adobe Reader
    Sublime Text 2 (or 3)
    Windows Media Player


### PACKING LIST
    - SM_Type_I_Attack
        - Readme.md (this file)
        - codes
            - attack_code
                - styleGAN-TypeI
                    - Readme.md
                    - requirements.txt
                    - type1.py
                    - dnnlib (please refer to https://github.com/NVlabs/stylegan for detail)
                - SVAE-TypeI
            - defense_code
                - __init__.py
                - adv_training_TypeI.py
                - adv_training_TypeII.py
                - compare.py
                - facenet.py
                - facenet_adv.py
                - lfw.py
                - test.py
                - validate_on_lfw.py
            - facenet (please refer to https://github.com/davidsandberg/facenet/tree/master/src/models for detail)
            - moviepy (3rd-party lib for generating videos from image sequences, https://github.com/Zulko/moviepy)
        - examples
            - face (Generated Type I adversarial face images by SVAE)
            - face StyleGAN (Generated High resolution Type I adversarial face images by StyleGAN)
            - process video (Videos for illustrating the Type I attacking process)

### CONTACT INFORMATION
  Co-author1:
      Sanli Tang
      University of Shanghai Jiao Tong University
      Shanghai, China
      E-mail: tangsanli@sjtu.edu.cn

  Co-author2:
      Xiaolin Huang, 
      University of Shanghai Jiao Tong University
      Shanghai, China
      xiaolinhuang@sjtu.edu.cn