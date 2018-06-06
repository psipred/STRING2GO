# STRING2GO

This is a Keras implementation of STRING2GO method reported in a submitted paper:

STRING2GO: Learning Functional Representations Directly from STRING Network Topology with Deep Maxout Neural Networks for Protein Function Prediction

Cen Wan, Domenico Cozzetto, Rui Fa and David T. Jones

University College London

---------------------------------------------------------------
# Requirements

- Python 3.6 
- Numpy 
- Keras (Theano backend) 
- Scikit-learn

---------------------------------------------------------------
# Running 

- Step 1. Generating network-embeddings of STRING network using Mashup [1] or node2vec [2] methods. The generated embeddings can be found in the data folder.
  - [1] Cho et al., (2016) Compact Integration of Multi-Network Topology for Functional Analysis of Genes, Cell Systems, 3
, 540–548.
  - [2] Grover A. and Leskovec, J., (2016) node2vec: Scalable Feature Learning for Networks, ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). 

- Step 2. Learning functional representations using STRING2GO. 
