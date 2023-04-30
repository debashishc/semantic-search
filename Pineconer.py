import pinecone
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import Chroma, Pinecone
import torch

from dotenv import load_dotenv
import os

load_dotenv() #TODO: must be a better way to do this

PINECONE_API = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

# Set up Pinecone client with your API key
pinecone.init(api_key=PINECONE_API,
              environment=PINECONE_API_ENV)

# Set up the HuggingFace tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Define a function to encode text using the HuggingFace model
# def encode(text):
#     input_ids = tokenizer.encode(text, return_tensors="pt")[0]
#     with torch.no_grad():
#         embedding = model(input_ids).last_hidden_state.mean(dim=1)
#     return embedding.numpy()


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def encode(sentences):
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])
    return sentence_embeddings


# Define your Pinecone index name
index_name = "langchainbook"
index = pinecone.Index(index_name)

# Create a Pinecone index
# pinecone.create_index(index_name, dimension=128)

# Add some text data to the index
texts = ["hello world", "pinecone is cool", "I love HuggingFace"]
embeddings = [encode(text) for text in texts]
embeddings_list = [tensor.tolist()
                   for tensor in embeddings]  # turn tensor into list
index.upsert(vectors=(zip(texts, embeddings_list)))  # must be tuples

# TODO must check embedding dimensions so it matches vector database that was created.

# Clean up the Pinecone client
pinecone.deinit()
