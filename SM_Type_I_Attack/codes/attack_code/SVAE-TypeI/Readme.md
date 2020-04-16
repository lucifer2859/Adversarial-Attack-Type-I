# Simple guidelines for the codes of the paper:
# Adversarial Attack Type I: Generating False Positives.

# Intrdoction
#    This is a simple guideline of the codes for image class transition and Type I attack on MNIST dataset. The codes for CelebA dataset are also provided in this 
#  directory, but the guideline will be provided after reviewing since we convert the celebA dataset into *.npy format with some preprocessing operations such as 
#  central cropping, which will be provided in the future. All the comments of the codes also will be provided after reviewing. 

# step 1: train the supervised variational autoencoder (SVAE)
python run_mnist.py --phase train --dim_z 32

# step 2: test the image class transition based on the trained model in step 1
python run_mnist.py --phase transition --dim_z 32 --test_index 0 --target_label 0

# step 3: train the MLP as the weak classifier to be attacked
python run_mnist.py --phase trainMLP

# step 4: conduct Type I attack for the MLP
python run_mnist.py --phase attack --test_index 0 --target_label 8

# all the output files in the test case will be located in ./results, which is created automatically.