import requests
import json
import numpy as np
import random
import string

class vectoro_client:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def save(self):
        response = requests.post(f"{self.base_url}/save/")

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to save: {response.text}")
        
    def add_vector(self, vector, text, date, cat, cat2, cat3):
        payload = {
            "vector": vector,
            "text": text,
            "date": date,
            "cat": cat,
            "cat2": cat2,
            "cat3": cat3
        }
        response = requests.post(f"{self.base_url}/vectors/", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to add vector: {response.text}")

    def search(self, query: str = "SUMMARY", top_k=5):
        payload = query
        response = requests.post(f"{self.base_url}/search/?top_k={top_k}", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Search failed: {response.text}")

def random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Example usage
if __name__ == "__main__":
    client = vectoro_client()

    # Adding vectors to the server
    for i in range(10):
        random_vec = [float(x) for x in np.random.randn(128)]
        random_txt = random_string(10)
        response = client.add_vector(random_vec, random_txt)
        print(f"Added vector {i + 1}: {response}")

    # Searching for similar vectors
    query_vec = [float(x) for x in np.random.randn(128)]
    results = client.search(query_vec, top_k=3)
    print("\nTop 3 search results:")
    for result in results:
        print(f"Text: {result['text']}, Similarity: {result['similarity']}")
