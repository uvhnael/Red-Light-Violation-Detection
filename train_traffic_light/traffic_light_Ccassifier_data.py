import kagglehub

# Download latest version
path = kagglehub.dataset_download("chandanakuntala/cropped-lisa-traffic-light-dataset")

print("Path to dataset files:", path)