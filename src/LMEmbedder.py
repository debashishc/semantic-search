import os
import pinecone
import torch
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain.vectorstores import Chroma, Pinecone
# from openai import OpenAIAPI
from transformers import AutoModel, AutoTokenizer
from weaviate import Client


load_dotenv()


class AbstractEmbedder(ABC):

    @abstractmethod
    def encode(self, sentences):
        pass


class MiniLMEmbedder(AbstractEmbedder):
    """
    A class to encode sentences using the MiniLM model from HuggingFace.
    """

    def __init__(self):
        #TODO add model and tokenizer to constructor
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    
    def create_pinecone_index(self, api_key, environment, index_name="langchainbook"):
        """
        Create a Pinecone index for the given API key and environment.
        :param api_key: The API key for the Pinecone service.
        :param environment: The environment for the Pinecone service.
        :param index_name: The name of the index to create.
        """

        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)


    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on the model output using the attention mask.
        :param model_output: Model output from HuggingFace model.
        :param attention_mask: Attention mask for the input tokens.
        :return: Mean-pooled sentence embeddings.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, sentences):
        """
        Encode the given sentences into fixed-size embeddings using the MiniLM model.
        :param sentences: A list of sentences to encode.
        :return: Sentence embeddings as a tensor.
        """
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def upsert_to_index(self, texts):
        """
        Upsert the given texts along with their embeddings to the Pinecone index.
        :param texts: A list of texts to upsert to the Pinecone index.
        """
        embeddings = [self.encode(text) for text in texts]
        embeddings_list = [tensor.tolist() for tensor in embeddings]
        self.index.upsert(vectors=(zip(texts, embeddings_list)))

    # def __del__(self):
    #     pinecone.deinit()    


# class OpenAIEmbedder(AbstractEmbedder):
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.openai = OpenAIAPI(api_key=self.api_key)
#         # Add additional setup for the OpenAI model and tokenizer, if needed.

#     def encode(self, sentences):
#         """
#         Encode the given sentences into fixed-size embeddings using the OpenAI model.
#         :param sentences: A list of sentences to encode.
#         :return: Sentence embeddings as a tensor.
#         """
#         # Implement encoding using the OpenAI model.
#         pass


def create_embedder(model_type, api_key=None, environment=None):
    """
    Factory function to create an embedder based on the requested model type.
    :param model_type: The type of model to use, either 'huggingface' or 'openai'.
    :param api_key: The API key for the model service.
    :param environment: The environment for the model service.
    :return: An instance of the appropriate embedder.
    """
    if model_type.lower() == 'huggingface':
        return MiniLMEmbedder()
    # elif model_type.lower() == 'openai':
    #     return OpenAIEmbedder(api_key, environment)
    else:
        raise ValueError("Invalid model_type. Expected 'huggingface' or 'openai'.")



class AbstractVectorStore(ABC):

    @abstractmethod
    def upsert(self, texts, embeddings):
        pass


class PineconeVectorStore(AbstractVectorStore):

    def __init__(self, api_key, environment):
        self.api_key = api_key
        self.environment = environment

        pinecone.init(api_key=self.api_key, environment=self.environment)

        self.index_name = "langchainbook"
        self.index = pinecone.Index(self.index_name)

    def upsert(self, texts, embeddings):
        embeddings_list = [tensor.tolist() for tensor in embeddings]
        self.index.upsert(vectors=(zip(texts, embeddings_list)))

    # def __del__(self):
    #     pinecone.deinit()


class WeaviateVectorStore(AbstractVectorStore):

    def __init__(self, weaviate_url, weaviate_auth):
        self.client = Client(weaviate_url)
        self.client.set_auth(weaviate_auth)

    def upsert(self, texts, embeddings):
        # Implement the upsert logic for Weaviate.
        pass

    def __del__(self):
        # Clean up resources if needed.
        pass


def create_vector_store(store_type, api_key, environment):
    if store_type.lower() == 'pinecone':
        return PineconeVectorStore(api_key, environment)
    elif store_type.lower() == 'weaviate':
        # Set Weaviate URL and authentication credentials.
        weaviate_url = 'http://localhost:8080'
        weaviate_auth = (os.getenv("WEAVIATE_API_KEY"), os.getenv("WEAVIATE_API_SECRET"))
        return WeaviateVectorStore(weaviate_url, weaviate_auth)
    else:
        raise ValueError("Invalid store_type. Expected 'pinecone' or 'weaviate'.")




# if __name__ == "__main__":
    # api_key = os.getenv("PINECONE_API_KEY")
    # environment = os.getenv("PINECONE_API_ENV")

    # embedder = MiniLMEmbedder(api_key, environment)
    # texts = ["hello world", "pinecone is cool", "I love HuggingFace"]
    # embedder.upsert_to_index(texts)


if __name__ == "__main__":
    from DocumentReader import DocumentReader

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_API_ENV")

    model_type = 'huggingface'  # or 'openai'
    embedder = create_embedder(model_type, api_key, environment)

    # Read the text from a file using the DocumentReader class
    from pathlib import Path
    file_path = Path("~/Downloads/PS1.pdf").expanduser()  # Replace this with your file path
    document_reader = DocumentReader(file_path=file_path)
    text = document_reader.read()

    # Split the text into sentences or paragraphs (customize as needed)
    sentences = text.split("\n")

    # Encode the text using MiniLMEmbedder
    embeddings = embedder.encode(sentences)

    # Upsert the encoded text into the vector store of your choice
    store_type = 'pinecone'  # or 'weaviate'
    vector_store = create_vector_store(store_type, api_key, environment)
    vector_store.upsert(sentences, embeddings)
