# script to download and extract the data from UCI ML Repo
# it creates a subfolder called /data
# should be run from the command line by the user as first step for recreating the project

import requests
import os
import zipfile

def download_and_extract(url, extract_to='./data'):
    # Ensure directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    zip_path = os.path.join(extract_to, 'dataset.zip')

    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()

        with open(zip_path, 'wb') as file:
            file.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"Success: Dataset downloaded and extracted to '{extract_to}'.")

        # Remove zip file after extraction
        os.remove(zip_path)

    except requests.RequestException as e:
        print(f"Failed to download dataset: {e}")

    except zipfile.BadZipFile as e:
        print(f"Failed to unzip file: {e}")

if __name__ == '__main__':
    dataset_url = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
    download_and_extract(dataset_url)

