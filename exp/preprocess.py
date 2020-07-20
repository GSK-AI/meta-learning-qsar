"""Combine chunks of compressed featurized data for ChEMBL20 to a single file"""

import logging
import os
import pickle
import shutil
import tarfile

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%m/%d/%Y %H:%M:%S",
)


def main():
    # Loading data
    path = "data/chembl20"
    tarfiles = [os.path.join(path, f) for f in os.listdir(path) if "tar.gz" in f]
    tarfiles = sorted(tarfiles)
    logging.info(f"Found files for extracting and combining: {tarfiles}")

    data = []
    for f in tqdm(tarfiles):
        tar = tarfile.open(f, "r:gz")
        data_extract = [pickle.load(tar.extractfile(m)) for m in tar.getmembers()]
        data.extend(data_extract)
    # Unroll List[dict] -> dict
    data = {k: np.concatenate([d[k] for d in data]) for k in data[0].keys()}

    # Save and copy to exp2 folder
    save_path = os.path.join(path, "featurized_data.pkl")
    pickle.dump(data, open(save_path, "wb"))

    exp2_save_path = "data/chembl20_exp2/featurized_data.pkl"
    shutil.copy(save_path, exp2_save_path)
    logging.info(f"Data saved to {save_path} and {exp2_save_path}")


if __name__ == "__main__":
    main()
