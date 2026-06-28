# ml-workspace

Personal machine learning workspace for TensorFlow experiments, coursework, and notebooks. Jupyter runs in Docker so you get a consistent TensorFlow environment without installing it locally.

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- Network access on first run (to pull the image from Docker Hub if it is not cached locally)

## Quick start

From the project root:

```bash
./start_tf_jupyter.sh
```

The script:

1. Checks whether `tensorflow/tensorflow:latest-jupyter` exists locally
2. Pulls it from Docker Hub if missing
3. Starts a Jupyter container with this workspace mounted at `/tf/notebooks/ml-workspace`

Open the URL printed in the terminal (typically `http://127.0.0.1:8888/...`) and use the token shown in the output to log in.

The container runs as your host user (`$(id -u):$(id -g)`) and maps port `8888`. Notebooks you create or edit are written directly to your filesystem under `/home/dev`.

## Project layout

```
ml-workspace/
├── start_tf_jupyter.sh
├── introduction_to_tensorflow/   # TensorFlow course labs and solutions
│   ├── labs/                     # Exercise notebooks
│   ├── solutions/                # Completed reference notebooks
│   ├── data/                     # Datasets for exercises
│   ├── toy_data/
│   └── images/
├── rnn/                          # Recurrent neural network notebooks
│   └── text_generation.ipynb
└── autoencoder/                  # 1D conv autoencoder anomaly-detection POC
    ├── autoencoder_anomaly_detection.py
    ├── requirements.txt
    ├── tests/
    └── README.md
```

## Contents

### `introduction_to_tensorflow/`

Course-style notebooks covering core TensorFlow and Keras topics: tensors and variables, `tf.data`, sequential and functional APIs, custom layers, preprocessing, time series, imbalanced data, TFRecords, and more. Some notebooks also touch Google Cloud (Vertex AI, BigQuery).

- **`labs/`** — start here; work through exercises without peeking at answers
- **`solutions/`** — completed versions for comparison
- **`labs/taxifare/`** and **`solutions/taxifare/`** — training-at-scale exercise code (`trainer/`, test data)

### `rnn/`

Recurrent neural network experiments.

- **`text_generation.ipynb`** — character-level RNN text generation on Shakespeare using Keras subclassing

### `autoencoder/`

Standalone proof-of-concept for multichannel sensor anomaly detection with a 1D convolutional autoencoder. See [autoencoder/README.md](autoencoder/README.md) for Python setup, CLI options, and how to run tests outside Docker.

## Notes

- Large outputs (checkpoints, plots, logs, local datasets) are listed in `.gitignore` and are not committed by default.
- The Docker volume in `start_tf_jupyter.sh` mounts `/home/dev` to `/tf/notebooks`. Adjust the path in the script if your home directory differs.
- Cloud-focused notebooks (Vertex AI, BigQuery) require GCP credentials and resources beyond what the local Docker setup provides.

## License

MIT — see [LICENSE](LICENSE).
