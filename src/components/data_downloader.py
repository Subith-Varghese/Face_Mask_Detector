import opendatasets as od
import os

class DataDownloader:
    def __init__(self, url, download_dir="data/"):
        self.url = url
        self.download_dir = download_dir

    def download(self):
        os.makedirs(self.download_dir, exist_ok=True)
        print(f"📥 Downloading dataset from {self.url} ...")
        od.download(self.url, data_dir=self.download_dir)
        print(f"✅ Dataset downloaded to {self.download_dir}")
