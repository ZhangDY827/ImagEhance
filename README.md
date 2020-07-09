# ImagEhance
This is a pytorch implementation repository for low light image ehancement in self-supervised manner. The idea of the code is from the paper
Self-supervised Image Enhancement Network: Training With Low Light Images Only https://arxiv.org/abs/2002.11300d.

The model in this code is totally self-supervised, so that, only one single low light image is enough for pleasing results. 

# Dataset

We use the LOL dataset which contains 500 low/normal light image pairs, 485 of which are used for training and images size are 400 ∗ 600. 
Note that during the training process, I only use one low light image from eval dataset containing 15 images. The LOL dataset can be found in 
the homepage https://daooshee.github.io/BMVC2018website/.
