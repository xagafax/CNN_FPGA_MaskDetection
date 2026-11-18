import os, shutil, kagglehub
path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
base = os.path.join(path, "Face Mask Dataset")
for s in ["Train", "Validation", "Test"]:
    for c in ["WithMask", "WithoutMask"]:
        os.makedirs(f"data_raw/{c}", exist_ok=True)
        src = os.path.join(base, s, c)
        if os.path.exists(src):
            for fn in os.listdir(src): shutil.copy(os.path.join(src, fn), f"data_raw/{c}/{s}_{fn}")
