# Fast-Learnable-Sparse-Radon-Transform-Net
Python code for a fast learnable sparse Radon transform.


FLSRT_Net            ----> for our neural network

model                ----> a trained neural network

ma_nor & ma_ls_nor   ----> a test case

Y. Xue, H. Shen, M. Jiang, L. Feng, M. Guo and Z. Wang, "A Fast Sparse Hyperbolic Radon Transform Based on Convolutional Neural Network and Its Demultiple Application," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 7507905, doi: 10.1109/LGRS.2022.3223929.

Abstract: The hyperbolic Radon transform (RT) is a widely used demultiple method in the seismic data processing. However, this transformation faces two major defects. The limited aperture of acquisition leads to the scissor-like diffusion in the Radon domain, which introduces separation difficulties between primaries and multiples. In addition, the computation of large matrices inversion involved in hyperbolic RT reduces the processing efficiency. In this letter, a specific convolutional neural network (CNN) is designed to conduct a fast sparse hyperbolic RT (FSHRT). Two techniques are incorporated into CNN to find the sparse solution. One is the coding–decoding structure, which captures the sparse feature of Radon parameters. The other is the soft threshold activation function followed by the end of neural networks, which suppresses the small parameters and further improves the sparsity. Thus, the network realizes the direct mapping between the adjoint solution and the sparse solution. Furthermore, synthetic and field demultiple experiments are carried out to demonstrate the rapidity and effectiveness of the proposed method.

URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9957048&isnumber=9651998
