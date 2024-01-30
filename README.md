## Nearest neighbor search with cosine similarity

### Approach towards the problem

The model used for solving for the topK ONET classes for a given record
is the K nearest neighbor search. The [sentence transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model was used to generate embeddings or high dimensional representations for 
a given input sequence. The data provided has title and body present for the 
corresponding job description. Three different set of embeddings were generated using 1) title 2) body 3) concatenation of title and body of a given job description record.
Cosine similarity was used to compare between a query embedding (user input sequence or test set datapoint) and the records present in the training set. 
Generating the similarity matrix can be computationally intensive in terms of time and memory. [FAISS](https://github.com/facebookresearch/faiss)
library was used to index the embeddings/vectors which led to querying at low latency speeds for topK vectors similar to the query vector.
For evaluation, weighted macro averaged precision, recall, F1 score and accuracy were computed. ***The best model turned out be embeddings generated using only the title of the record.***
Results obtained by the code at seed 42 for a 95-5 train test split where topk is 1 are below:

````
Generate scores only with title embedding....
prec: 0.711186067654829 recall: 0.6948 f1: 0.6916637380850659 accuracy: 0.6948

Generate scores only with body embedding....
prec: 0.4339160799944686 recall: 0.4152 f1: 0.4120255851643017 accuracy: 0.4152

Generate scores with both title and body embedding.....
prec: 0.48470778771532297 recall: 0.4756 f1: 0.46729399432674107 accuracy: 0.4756
````


### Intructions to run code

Place *data.csv* in the folder to run *nearest_embedding.py*. Uncomment
L120-126 to regenerate embeddings. By default, the code will use the 
embeddings placed inside the *embeddings* folder. 
