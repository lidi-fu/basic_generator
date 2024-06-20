import requests
import os
import time

# Function to fetch object IDs for portrait artworks
def fetch_object_ids(query="portraits", max_objects=6000, batch_size=100):
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={query}&isOnView=true&hasImages=true"
    object_ids = []
    offset = 0
    while len(object_ids) < max_objects:
        url_with_offset = f"{url}&offset={offset}"
        print(f"Fetching object IDs from: {url_with_offset}")
        try:
            response = requests.get(url_with_offset)
            response.raise_for_status()  # Raise exception for bad responses
            data = response.json()
            new_object_ids = data.get("objectIDs", [])
            if not new_object_ids:
                print("No more object IDs found.")
                break
            object_ids.extend(new_object_ids)
            offset += len(new_object_ids)
            print(f"Fetched {len(object_ids)} object IDs so far.")
            time.sleep(1)  # Add a small delay to respect API rate limits
        except requests.exceptions.RequestException as e:
            print(f"Error fetching object IDs: {e}")
            break
    return object_ids[:max_objects]

# Function to fetch image URL for a given object ID
def fetch_image_url(object_id):
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    print(f"Fetching image URL for object ID: {object_id}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad responses
        data = response.json()
        primary_image = data.get("primaryImage")
        return primary_image, data.get("title"), data.get("artistDisplayName")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image URL for object ID {object_id}: {e}")
        return None, None, None

# Function to sanitize filenames
def sanitize_filename(filename):
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

# Function to download images
def download_images(image_data, output_dir="data/met_portraits", resume=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    downloaded_files = set()
    checkpoint_file = os.path.join(output_dir, "downloaded_files.txt")

    # Read already downloaded files
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            downloaded_files = {line.strip() for line in f.readlines()}

    new_downloads = 0
    total_images = len(image_data)
    for idx, (url, title, artist) in enumerate(image_data):
        if url:
            sanitized_title = sanitize_filename(title)
            sanitized_artist = sanitize_filename(artist)
            file_name = f"portrait_{idx}_{sanitized_title}_{sanitized_artist}.jpg"

            # Check if file already downloaded
            if file_name in downloaded_files:
                print(f"Skipping already downloaded: {file_name}")
                continue

            print(f"Downloading: {file_name} ({idx + 1}/{total_images})")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(os.path.join(output_dir, file_name), "wb") as f:
                    f.write(response.content)
                downloaded_files.add(file_name)
                with open(checkpoint_file, "a") as f:
                    f.write(f"{file_name}\n")
                print(f"Downloaded: {file_name}")
                new_downloads += 1
            except (OSError, requests.exceptions.RequestException) as e:
                print(f"Failed to download {file_name}: {e}")
    
    print(f"Downloaded {new_downloads} new images.")

# Fetch object IDs in chunks and download images incrementally
def fetch_and_download(query="portraits", max_objects=6000, batch_size=100):
    object_ids = fetch_object_ids(query=query, max_objects=max_objects, batch_size=batch_size)
    print(f"Total object IDs fetched: {len(object_ids)}")
    
    # Process and download images in batches
    for start_idx in range(0, len(object_ids), batch_size):
        end_idx = min(start_idx + batch_size, len(object_ids))
        batch_object_ids = object_ids[start_idx:end_idx]
        print(f"Processing batch {start_idx} to {end_idx}")
        
        image_data = [fetch_image_url(object_id) for object_id in batch_object_ids]
        download_images(image_data)

    print("Download process completed.")

# Run the script
fetch_and_download(query="portraits", max_objects=10000, batch_size=100)
