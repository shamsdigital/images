# main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

# Initialize FastAPI app
app = FastAPI(title="Video/Image Search API")

# Load environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')

if not SUPABASE_URL or not SUPABASE_API_KEY:
    raise ValueError("Supabase credentials are not set in environment variables.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Load CLIP model and processor once at startup
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class SearchRequest(BaseModel):
    image_url: str
    threshold: float = 0.8  # Default threshold


# Define allowed origins (replace with your frontend URL)
origins = [
    "https://your-frontend-domain.com",
    "http://localhost:3000",  # Add your development URLs
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class SearchResponse(BaseModel):
    matching_images: list

def get_image_embedding(image_url: str):
    """
    Downloads an image from the given URL and returns its embedding vector.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading or processing image: {e}")
        return None

    # Process the image and get embeddings
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Convert to CPU and detach from the computation graph
    embedding = outputs[0].cpu().numpy().tolist()
    return embedding

def search_image(input_image_url: str, threshold: float = 0.8):
    """
    Searches for images in the 'images' table that are similar to the input image.
    
    Parameters:
    input_image_url (str): The URL of the image to search.
    threshold (float): The cosine similarity threshold to consider a match.
    
    Returns:
    list: A list of matching image URLs.
    """
    # Generate embedding for the input image
    input_embedding = get_image_embedding(input_image_url)
    if input_embedding is None:
        raise HTTPException(status_code=400, detail="Failed to generate embedding for the input image.")

    # Fetch all images and their embeddings from Supabase
    try:
        response = supabase.table("images").select("image_url, embedding").execute()
        images_data = response.data
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching data from database.")

    if not images_data:
        return []

    # Prepare embeddings for similarity computation
    stored_embeddings = []
    valid_images = []
    for image in images_data:
        embedding_str = image.get('embedding')
        image_url = image.get('image_url')
        if not embedding_str or not image_url:
            continue
        try:
            # Safely evaluate the string to a Python list
            embedding = ast.literal_eval(embedding_str)
            stored_embeddings.append(embedding)
            valid_images.append(image_url)
        except Exception as e:
            print(f"Error parsing embedding for image {image_url}: {e}")
            continue  # Skip this image if embedding parsing fails

    if not stored_embeddings:
        return []

    stored_embeddings = np.array(stored_embeddings)
    input_embedding_np = np.array(input_embedding).reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(input_embedding_np, stored_embeddings)[0]

    # Find indices where similarity exceeds the threshold
    matching_indices = np.where(similarities >= threshold)[0]

    if len(matching_indices) == 0:
        return []

    # Retrieve matching image URLs
    matching_images = [valid_images[i] for i in matching_indices]

    return matching_images

@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    """
    POST endpoint to search for similar images in the database.
    """
    matching_images = search_image(request.image_url, request.threshold)
    return SearchResponse(matching_images=matching_images)
