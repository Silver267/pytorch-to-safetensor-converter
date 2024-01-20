import json
import os
import shutil
import sys
import torch
from collections import defaultdict
from safetensors.torch import load_file, save_file

def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    return [names for names in ptrs.values() if len(names) > 1]

def check_file_size(sf_filename, pt_filename):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"File size difference exceeds 1% between {sf_filename} and {pt_filename}")

def convert_file(pt_filename, sf_filename, copy_add_data=True):
    source_folder = os.path.dirname(pt_filename)
    dest_folder = os.path.dirname(sf_filename)
    loaded = torch.load(pt_filename, map_location="cpu")
    loaded = loaded.get("state_dict", loaded)
    shared = shared_pointers(loaded)

    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    loaded = {k: v.contiguous().half() for k, v in loaded.items()}

    os.makedirs(dest_folder, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    if copy_add_data:
        copy_additional_files(source_folder, dest_folder)

    reloaded = load_file(sf_filename)
    for k, v in loaded.items():
        if not torch.equal(v, reloaded[k]):
            raise RuntimeError(f"Mismatch in tensors for key {k}.")

def rename(pt_filename):
    return pt_filename.replace("pytorch_model", "model").replace(".bin", ".safetensors")

def copy_additional_files(source_folder, dest_folder):
    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)
        if os.path.isfile(file_path) and not (file.endswith('.bin') or file.endswith('.py')):
            shutil.copy(file_path, dest_folder)

def find_index_file(source_folder):
    for file in os.listdir(source_folder):
        if file.endswith('.bin.index.json'):
            return file
    return None

def convert_files(source_folder, dest_folder, delete_old):
    index_file = find_index_file(source_folder)
    if not index_file:
        raise RuntimeError("Index file not found. Please ensure the correct folder is specified.")

    index_file = os.path.join(source_folder, index_file)
    with open(index_file) as f:
        index_data = json.load(f)

    copy_additional_files(source_folder, dest_folder)
    for pt_filename in set(index_data["weight_map"].values()):
        full_pt_filename = os.path.join(source_folder, pt_filename)
        sf_filename = os.path.join(dest_folder, rename(pt_filename))
        convert_file(full_pt_filename, sf_filename, copy_add_data=False)
        if delete_old:
            os.remove(full_pt_filename)
    
    index_path = os.path.join(dest_folder, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        new_map = {k: rename(v) for k, v in index_data["weight_map"].items()}
        json.dump({**index_data, "weight_map": new_map}, f, indent=4)

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    source_folder = input("Source folder for PyTorch files (leave blank for script's directory): ").strip() or script_dir
    dest_folder = input("Destination folder for SafeTensors files (leave blank for default): ").strip()

    if not dest_folder:
        model_name = os.path.basename(os.path.normpath(source_folder))
        dest_folder = os.path.join(source_folder, model_name + "_safetensors")

    delete_old = input("Delete old PyTorch files? (Y/N): ").strip().upper() == 'Y'

    if "pytorch_model.bin" in os.listdir(source_folder):
        convert_file(os.path.join(source_folder, "pytorch_model.bin"), os.path.join(dest_folder, "model.safetensors"), copy_add_data=True)
        if delete_old:
            os.remove(os.path.join(source_folder, "pytorch_model.bin"))
    else:
        convert_files(source_folder, dest_folder, delete_old)

if __name__ == "__main__":
    main()
