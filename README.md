# FourierMasks
Despite the enormous success of artificial neural networks (ANNs) in many disciplines, 
the characterization of their computations and the origin of key properties such as 
generalization and robustness remain open questions. Recent literature suggests that 
robust networks with good generalization properties tend to be biased towards processing 
low frequencies in images. To explore the frequency bias hypothesis further, we develop 
an algorithm that allows us to learn \emph{modulatory masks} highlighting the 
\emph{essential input frequencies} needed for preserving a trained network's performance. 
We achieve this by imposing \emph{invariance} in the loss with respect to such modulations 
in the input frequencies. We first use our method to test the low-frequency preference 
hypothesis of adversarially trained or data-augmented networks. Our results suggest that 
adversarially robust networks indeed exhibit a low-frequency bias but we find this bias is 
also dependent on directions in frequency space. However, this is not necessarily true for 
other types of data augmentation. Our results also indicate that the essential frequencies 
in question are effectively the ones used to achieve generalization in the first place. 
Surprisingly, images seen through these modulatory masks are not recognizable and resemble 
texture-like patterns.
