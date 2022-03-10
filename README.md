# FourierMasks 
---

Code is an implementation of the paper titled "Understanding robustness and generalization 
of artificial neural networks through Fourier masks" (Under review).

---

This is an algorithm that allows us to learn modulatory masks highlighting the essential 
input frequencies needed for preserving a trained network's performance.

We achieve this by imposing invariance in the loss with respect to such modulations 
in the input frequencies. 

We first use our method to test the low-frequency preference 
hypothesis of adversarially trained or data-augmented networks. 

Our results suggest that adversarially robust networks indeed exhibit a low-frequency bias 
but we find this bias is also dependent on directions in frequency space. 

However, this is not necessarily true for other types of data augmentation. 
Our results also indicate that the essential frequencies in question are effectively the ones 
used to achieve generalization in the first place. Images seen through these modulatory masks 
are not recognizable and resemble texture-like patterns.

---

1. We train a neural net (VGG11 or ResNet18) on 5 classes of ImageNet (https://www.image-net.org/)
considering various kinds of data augmentation.

2. We then train Fourier masks as described in the article and explore robustness and generalization
properties of the pretrained neural net.

---

Replicating the results found in Notebooks 1-3:

1. choose architecture = 'vgg' or 'resnet' and a folder where the trained models will be saved, 
e.g., model_save_folder = 'trained_models'. Then run

python3 train_models.py --architecture='vgg --model_save_folder='trained_models'

2. Once the models are trained, we train the masks for each model by running (in the same fashion)

python3 train_masks.py --architecture='vgg --pretrained_model_folder='trained_models' --mask_save_folder='trained_masks'

3. We can also train single img masks for images in our validation set by running

python3 train_single_img_masks.py --architecture='vgg' --mask_save_folder='trained_single_img_masks'
