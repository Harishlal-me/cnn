import os
import shutil
import urllib.request
import zipfile

def download_hatexplain(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, "dataset.json")
    if os.path.exists(dest_file):
        print("HateXplain already exists.")
        return

    print("Downloading HateXplain...")
    url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
    try:
        urllib.request.urlretrieve(url, dest_file)
        print("HateXplain downloaded successfully.")
    except Exception as e:
        print(f"Failed to download HateXplain: {e}")

if __name__ == "__main__":
    base_dir = r"D:\multimodal-taskgate\datasets"
    
    # Download HateXplain
    download_hatexplain(os.path.join(base_dir, "hatexplain"))
