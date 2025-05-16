# 🛠️ Tech Stack

The flicksync project leverages a modern, Python-based tech stack for deep learning, similarity search, and interactive web applications. Below are the major components of the stack, organized by function:
## Backend / Core Libraries

- **Python 3.8+:** Primary programming language for all scripts and applications

- **PyTorch:** Deep learning framework for loading and running transformer models

- **Hugging Face Transformers:** Provides access to state-of-the-art video transformer models (Timesformer) for embedding extraction

- **FAISS:** High-performance similarity search library for indexing and querying video embeddings, with optional GPU acceleration

- **OpenCV:** Video processing and frame extraction.

- **PyAV:** Efficient video decoding and reading.

- **imageio:** For creating GIF previews of video results.

- **NumPy, pandas:** Data manipulation and numerical operations.

## Frontend

- **Streamlit:** Python-based web framework for building interactive user interfaces and visualizing video search results in real time

## Utilities & Workflow

- **tqdm:** Progress bars for data processing scripts.

- **Jupyter Notebook:** For exploratory development, embedding generation, and reproducibility.

- **CUDA (NVIDIA GPU):** Recommended for accelerating model inference and FAISS indexing.
## Data & Storage

- **Local File System**
    Stores raw videos, processed embeddings, and FAISS index files.

- **Pickle**
    Serialization of ID maps and metadata.

