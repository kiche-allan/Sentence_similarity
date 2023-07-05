import torch #This line imports the torch library, which is a popular open-source machine learning framework used for tasks like deep learning.
from sentence_transformers import SentenceTransformer #This line imports the SentenceTransformer class from the sentence_transformers library. SentenceTransformer is a library that provides sentence and text embeddings, which are numerical representations of sentences or texts that capture their semantic meaning.

# Load the pre-trained model
model = SentenceTransformer('bert-base-nli-mean-tokens') #This line creates an instance of the SentenceTransformer class and assigns it to the variable model. The 'bert-base-nli-mean-tokens' argument specifies the pre-trained model to be used. In this case, it is using the BERT model with a mean pooling strategy to generate sentence embeddings.

# Encode sentences into embeddings
def encode_sentences(sentences):
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings

# Example usage
sentences = ['I love pizza', 'Pizza is my favorite food']
embeddings = encode_sentences(sentences)
print(embeddings)
