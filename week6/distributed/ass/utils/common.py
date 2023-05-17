import gdown

def download_from_driver(data_driver_path, location_path):
    print(f"begin download....: {data_driver_path}")
    gdown.download(data_driver_path, location_path, quiet=False, fuzzy=True)