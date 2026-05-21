import kagglehub
import os
import shutil
print("Descargando dataset...")
path = kagglehub.dataset_download("feyzazkefe/trashnet")
print("Ruta de los archivos del dataset:", path)
target_dir = "./dataset"
content = os.listdir(path)
print("Contenido de la ruta descargada:", content)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
print(f"Copiando archivos a {target_dir}...")
source_path = path
if 'dataset-original' in content:
    source_path = os.path.join(path, 'dataset-original')
elif 'trashnet' in content:
     source_path = os.path.join(path, 'trashnet')
for item in os.listdir(source_path):
    s = os.path.join(source_path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)
print("El dataset está listo en ./dataset")
