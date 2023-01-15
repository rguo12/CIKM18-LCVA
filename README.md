# CIKM18-Linked Causal Variational Autoencoder
Code for research paper:

[*Linked Causal Variational Autoencoder for Inferring Paired Spillover Effects*.](https://arxiv.org/abs/1808.03333)

Please cite us via this bibtex if you use this code for further development or as a baseline method in your work:
```
@inproceedings{rakesh2018linked,
  title={Linked Causal Variational Autoencoder for Inferring Paired Spillover Effects},
  author={Rakesh, Vineeth and Guo, Ruocheng and Moraffah, Raha and Agarwal, Nitin and Liu, Huan},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={1679--1682},
  year={2018},
  organization={ACM}
}
```

In this work, we consider spill-over effect between instances for learning causal effects from data.

## Amazon dataset

For the Amazon dataset we processed and used for the paper, please check out:
[Download Amazon Dataset Here](https://www.dropbox.com/s/d69tj0xnu3zc9bo/Amazon-CIKM18.zip?dl=0)

We will use the positive review dataset as an example. You will load data from **new_product_graph_pos.npz** and **AmazonItmFeatures_pos.csv**.

If you run the following code to load the item-item co-purchase graph **new_product_graph_pos.npz**
```
import pandas as pd
import numpy as np
import json
from scipy.sparse import csr_matrix

prod_G = np.load("new_product_graph_pos.npz")
print(prod_G.files)
data = prod_G['data']
indices = prod_G['indices']
indptr = prod_G['indptr']
shape = prod_G['shape']

csr_mat = csr_matrix((data, indices, indptr), dtype=int)

arr = csr_mat.toarray()
```
you will find there is a dimension mismatch: arr.shape is (96132, 42135) while the positive review dataset only has 42135 instances. You will only need the first 42135 rows of the matrix since the rest is just 0s. So it is safe to ignore the rest.

For the file **AmazonItmFeatures_pos.csv**, you can load it as a pd.Dataframe with the dimension (43135, 305). The last 300 columns are the doc2vec features. The first, second and third columns are the treatment, treated outcome and the controlled outcome. The forth and fifth columns can be ignored.

You can refer to [this link](https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr) for how to handle the csr matrices with indptr.

For the job training dataset, please refer to the paper.

Feel free to email me ruocheng.guo at cityu dot edu dot hk for any question and collaboration.


## Acknowledgement

The code is developed based on the code released by authors of the Neurips 2017 paper:
```
Louizos, Christos, Uri Shalit, Joris M. Mooij, David Sontag, Richard Zemel, and Max Welling. "Causal effect inference with deep latent-variable models." In Advances in Neural Information Processing Systems, pp. 6446-6456. 2017.
```
