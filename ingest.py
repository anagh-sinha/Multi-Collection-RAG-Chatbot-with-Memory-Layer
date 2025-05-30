import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data(data_dir):
    """
    Load all JSON files from the given directory and return their contents in a dict.
    Keys will be the filename, values are the parsed JSON content.
    """
    data = {}
    # List of expected data files
    files = ["wearable_data.json", "chat_history.json", "user_profile.json", 
             "location_data.json", "custom_collection.json"]
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f: 
                try:
                    data[file_name] = json.load(f) # Load the JSON file into a dictionary
                except json.JSONDecodeError as e: # Handle JSON decoding errors 
                    print(f"Error reading {file_name}: {e}")
        else:
            print(f"Warning: {file_name} not found in {data_dir}")
    return data

def create_embeddings(data):
    """
    Create embeddings for text content in the provided data collections.
    Returns a dictionary with documents, their embeddings, and source labels.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2') # Load the SentenceTransformer model
    docs = [] # List to store the text descriptions of the data collections 
    sources = [] # List to store the source labels for each document
    embeddings = [] # List to store the embeddings of the documents
    # Process wearable data (list of records -> text descriptions)
    if "wearable_data.json" in data:
        for record in data["wearable_data.json"]:
            date = record.get("date", "unknown date") # Get the date from the record        
            duration = record.get("sleep_duration") or f"{record.get('hours', '')} hours" # Get the sleep duration from the record
            score = record.get("sleep_score", "") # Get the sleep score from the record
            # Compose a descriptive text for each record
            text = f"Wearable data on {date}: sleep duration {duration}, sleep score {score}."
            docs.append(text)
            sources.append("wearable_data")
    # Process user profile (dict -> one summary text)
    if "user_profile.json" in data:
        profile = data["user_profile.json"]
        profile_text = "User profile: " # Initialize the profile text
        if "name" in profile:
            profile_text += f"Name is {profile['name']}. " # Add the name to the profile text
        if "age" in profile:
            profile_text += f"Age {profile['age']}. " # Add the age to the profile text
        if "sleep_issues" in profile:
            profile_text += f"Sleep issues: {profile['sleep_issues']}. " # Add the sleep issues to the profile text
        if "preferences" in profile:
            profile_text += f"Preferences: {profile['preferences']}. " # Add the preferences to the profile text    
        docs.append(profile_text.strip()) # Add the profile text to the list of documents
        sources.append("user_profile") # Add the source label to the list of sources
    # Process location data (dict -> one summary text)
    if "location_data.json" in data:
        loc = data["location_data.json"]
        loc_text = "Location info: "
        for key, val in loc.items():
            loc_text += f"{key}: {val}, "
        loc_text = loc_text.strip().rstrip(',')
        docs.append(loc_text)
        sources.append("location_data")
    # Process custom collection (list of items -> each item as text)
    if "custom_collection.json" in data:
        for item in data["custom_collection.json"]: # Iterate through the items in the custom collection
            if isinstance(item, dict): # Check if the item is a dictionary
                title = item.get("title", "") # Get the title from the item
                content = item.get("content", "") # Get the content from the item
                text = f"{title}: {content}" if title else str(content) # Compose a text for the item
            else:
                text = str(item)
            docs.append(text)
            sources.append("custom_collection")
    # Note: chat_history is not embedded here; it's handled by the memory component.
    # Compute embeddings for each collected doc
    for text in docs:
        emb = model.encode(text)
        # Normalize the embedding to unit length (for cosine similarity)
        emb = emb / np.linalg.norm(emb) # linalg.norm() computes the Euclidean norm of the array. Normalizing the embedding to unit length ensures that the vectors are of equal scale, which is important for accurate similarity calculations.
        embeddings.append(emb.tolist())  # convert numpy array to list for JSON serialization. JSON serialization is the process of converting Python objects into a JSON-compatible format.
    return {"documents": docs, "embeddings": embeddings, "sources": sources}

def save_index(index_data, index_path): 
    """Save the index data (embeddings and documents) to a JSON file."""
    with open(index_path, 'w') as f:
        json.dump(index_data, f) # Write the index data to the JSON file        
    print(f"Saved index to {index_path} (Total documents indexed: {len(index_data['documents'])})") # Print a message indicating that the index has been saved and the total number of documents indexed

def main():
    # Determine data directory relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    index_path = os.path.join(data_dir, "index.json")
    data = load_data(data_dir)
    index_data = create_embeddings(data) # Create embeddings for the data collections
    save_index(index_data, index_path)

if __name__ == "__main__":
    main()
