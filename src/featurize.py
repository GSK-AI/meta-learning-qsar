"""Featurize SMILES to chemical graphs

Usage:
    python src/featurize.py --data <csv file> \\
        --smiles_col <name of SMILES column> \\
        --output_col <name of output columns> \\
        --output_path <folder to store featurized data>

Note:
    - Data is assumed to be CSV and path should not contain "." except for extension (.csv)
    - When mutiple output columns are used, names are separated by ",". For example --output col "task1,task2,task3".
    - When no output_path is provided, the file is stored to the parent directory of --data
    - Two files will be saved to output_path: <output_path>/featurized_data.pkl and <output_path>/metadata.json
"""

import json
import os
import pickle

import pandas as pd
from absl import app, flags, logging

from src.featurizer import smiles2graph as s2g

FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "Input CSV, assumed to be have format <directory>/<filename>.csv")
flags.DEFINE_string("smiles_col", None, "SMILES column name")
flags.DEFINE_string("output_col", None, "Output column names separated by ','")
flags.DEFINE_string("output_path", None, "Output directory for featurized data and metadata")
flags.mark_flag_as_required("data")
flags.mark_flag_as_required("smiles_col")
flags.mark_flag_as_required("output_col")


def featurize(argv):
    # Load data
    logging.info(f"Loading data from {FLAGS.data}")
    df = pd.read_csv(FLAGS.data)
    output_col = FLAGS.output_col.split(",")

    # Featurize from smiles to graph
    logging.info(f"Featurizing {len(df)} with output columns {output_col}")
    featurized_data = s2g.featurize_df(
        data_df=df, smiles_col=FLAGS.smiles_col, output_col=output_col
    )

    # Saving featurized data and metadata
    output_path_base = "/".join(FLAGS.data.split("/")[:-1])
    if FLAGS.output_path:
        output_path_base = FLAGS.output_path
    os.makedirs(output_path_base, exist_ok=True)

    output_path = os.path.join(output_path_base, "featurized_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(featurized_data, f)

    metadata = {}
    metadata["data"] = FLAGS.data
    metadata["smiles_col"] = FLAGS.smiles_col
    metadata["output_col"] = output_col
    metadata_path = os.path.join(output_path_base, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Featurized data and metadata saved to {output_path} and {metadata_path}")


if __name__ == "__main__":
    app.run(featurize)
