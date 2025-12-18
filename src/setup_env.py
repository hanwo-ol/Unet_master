import os
import zipfile
import urllib.request
import shutil

HUGO_URL = "https://github.com/gohugoio/hugo/releases/download/v0.121.1/hugo_extended_0.121.1_windows-amd64.zip"
BIN_DIR = "bin"
ZIP_PATH = "bin/hugo.zip"
EXE_PATH = "bin/hugo.exe"

def setup_hugo():
    if os.path.exists(EXE_PATH):
        print(f"Hugo already exists at {EXE_PATH}")
        return

    os.makedirs(BIN_DIR, exist_ok=True)
    
    print(f"Downloading Hugo from {HUGO_URL}...")
    try:
        urllib.request.urlretrieve(HUGO_URL, ZIP_PATH)
        print("Download complete.")
        
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # zip contains hugo.exe at root usually
            zip_ref.extract("hugo.exe", BIN_DIR)
            
        print(f"Hugo installed to {EXE_PATH}")
        
        # Cleanup
        os.remove(ZIP_PATH)
        
    except Exception as e:
        print(f"Failed to setup Hugo: {e}")

if __name__ == "__main__":
    setup_hugo()
