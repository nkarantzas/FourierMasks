# FourierMasks 

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
