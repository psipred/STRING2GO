# STRING2GO

This is a Keras implementation of STRING2GO method reported in a published paper:

Wan, C. Cozzetto, D. Fa, R. and Jones, D.T. (2019) Using Deep Maxout Neural Networks to Improve the Accuracy of Function Prediction from Protein Interaction Networks. PLoS One, 14(7): e0209958.

---------------------------------------------------------------
# Requirements

- Python 3.6 
- Numpy 
- Keras (Theano backend) 
- Scikit-learn

---------------------------------------------------------------
# Running 

- Step 1. Generating network-embeddings of STRING network using Mashup [1] or node2vec [2] methods. The generated embeddings can be found in the `./data` folder.
  - [1] Cho et al., (2016) Compact Integration of Multi-Network Topology for Functional Analysis of Genes, Cell Systems, 3
, 540–548.
  - [2] Grover A. and Leskovec, J., (2016) node2vec: Scalable Feature Learning for Networks, ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). 

- Step 2. Learning functional representations using `./src/STRING2GO_Functional_Representation_Learning.py`. 

- Step 3. Training support vector machine library for predicting protein function using `./src/STRING2GO_Functional_Representation_SVM.py`.
