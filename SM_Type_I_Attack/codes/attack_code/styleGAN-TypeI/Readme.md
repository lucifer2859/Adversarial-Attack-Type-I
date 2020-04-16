# Function
* Generate Type I adversarial samples according to Type II Attack method in Constructing Unrestricted Adversarial Examples with Generative Models

# Structure
* karras2019stylegan-ffhq-1024x1024.pkl ---pretrained StyleGAN model, can be downloaded from https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ
* requirements.txt                      ---python environment
* type1.py                              ---main file
* facenet/face_recognition.py           ---file to build facenet from inception resnet v1
* facenet/facenet                       ---pretrained facenet model
* dnnlib                                ---files needed to load styleGAN

# reference
* StyleGAN code https://github.com/NVlabs/stylegan
* StyleGAN paper http://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf
* Constructing Unrestricted Adversarial Examples with Generative Models http://papers.nips.cc/paper/8052-constructing-unrestricted-adversarial-examples-with-generative-models.pdf