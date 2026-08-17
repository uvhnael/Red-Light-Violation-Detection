import kagglehub

# Download latest version
path = kagglehub.dataset_download("duydieunguyen/licenseplates")

print("Path to dataset files:", path)