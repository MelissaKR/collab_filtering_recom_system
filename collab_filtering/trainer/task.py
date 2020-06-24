import argparse
import json
import os

from . import model

import tensorflow as tf
#from tensorflow.contrib.learn.python.learn import learn_runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        help = "Path to data, either local or on GCS. We will append /stores_for_item or /items_for_store",
        required = True
    )
    parser.add_argument(
        "--output_dir",
        help = "GCS location to write checkpoints and export models",
        required = True
    )
    parser.add_argument(
        "--num_epochs",
        help = "Number of times to iterate over the training dataset. This can be fractional",
        type = float,
        default = 5
    )
    parser.add_argument(
        "--batch_size",
        help = "Matrix factorization happens in chunks. Specify chunk size here.",
        type = int,
        default = 512
    )
    parser.add_argument(
        "--n_embeds",
        help = "Number of dimensions to use for the embedding dimension",
        type = int,
        default = 10
    )
    parser.add_argument(
        "--nstores",
        help = "Total number of stores. WALS expects storeId to be indexed 0,1,2,... ",
        type = int,
        required = True
    )
    parser.add_argument(
        "--nitems",
        help = "Total number of items. WALS expects itemId to be indexed 0,1,2,... ",
        type = int,
        required = True
    )
    parser.add_argument(
        "--topk",
        help = "In batch prediction, how many top items should we predict for each store?",
        type = int,
        default = 3
    )
    parser.add_argument(
        "--job-dir",
        help = "this model ignores this field, but it is required by gcloud",
        default = "junk"
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop("job_dir", None)
    arguments.pop("job-dir", None)

    arguments["output_dir"] = os.path.join(
        arguments["output_dir"],
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
            ).get("task", {}).get("trial", "")
        )

    # Run the training job
    model.train_and_evaluate(arguments)
