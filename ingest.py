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
                    data[file_name] = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_name}: {e}")
        else:
            print(f"Warning: {file_name} not found in {data_dir}")
    return data

def create_embeddings(data):
    """
    Create embeddings for text content in the provided data collections.
    Returns a dictionary with documents, their embeddings, and source labels.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    docs = []
    sources = []
    embeddings = []
    # Process wearable data (list of records -> text descriptions)
    if "wearable_data.json" in data:
        for record in data["wearable_data.json"]:
            date = record.get("date", "unknown date")
            duration = record.get("sleep_duration") or f"{record.get('hours', '')} hours"
            score = record.get("sleep_score", "")
            # Compose a descriptive text for each record
            text = f"Wearable data on {date}: sleep duration {duration}, sleep score {score}."
            docs.append(text)
            sources.append("wearable_data")
    # Process user profile (dict -> one summary text)
    if "user_profile.json" in data:
        profile = data["user_profile.json"]
        profile_text = "User profile: "
        if "name" in profile:
            profile_text += f"Name is {profile['name']}. "
        if "age" in profile:
            profile_text += f"Age {profile['age']}. "
        if "sleep_issues" in profile:
            profile_text += f"Sleep issues: {profile['sleep_issues']}. "
        if "preferences" in profile:
            profile_text += f"Preferences: {profile['preferences']}. "
        docs.append(profile_text.strip())
        sources.append("user_profile")
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
        for item in data["custom_collection.json"]:
            if isinstance(item, dict):
                title = item.get("title", "")
                content = item.get("content", "")
                text = f"{title}: {content}" if title else str(content)
            else:
                text = str(item)
            docs.append(text)
            sources.append("custom_collection")
    # Note: chat_history is not embedded here; it's handled by the memory component.
    # Compute embeddings for each collected doc
    for text in docs:
        emb = model.encode(text)
        # Normalize the embedding to unit length (for cosine similarity)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb.tolist())  # convert numpy array to list for JSON serialization
    return {"documents": docs, "embeddings": embeddings, "sources": sources}

def save_index(index_data, index_path):
    """Save the index data (embeddings and documents) to a JSON file."""
    with open(index_path, 'w') as f:
        json.dump(index_data, f)
    print(f"Saved index to {index_path} (Total documents indexed: {len(index_data['documents'])})")

def main():
    # Determine data directory relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    index_path = os.path.join(data_dir, "index.json")
    data = load_data(data_dir)
    index_data = create_embeddings(data)
    save_index(index_data, index_path)

if __name__ == "__main__":
    main()
