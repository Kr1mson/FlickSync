# FlickSync

## Reverse Video Search using Timesformer & FAISS

This project implements a **video similarity search system** using the [Timesformer](https://arxiv.org/abs/2102.05095) transformer model (pretrained on Kinetics-400) to generate video embeddings, and [FAISS](https://github.com/facebookresearch/faiss) for efficient nearest neighbor search. Users can upload a video, and the app will return visually similar videos from the UCF101 dataset using precomputed embedding indexes.

---

## Features

- **Video Embedding:** Uses Timesformer to extract powerful video representations.
- **Similarity Search:** Efficiently retrieves similar videos using FAISS vector search.
- **Interactive Frontend:** Built with Streamlit for easy video upload and result visualization.
- **Multiple Embedding Types:** Supports mean pooling, max pooling, and CLS token embeddings.
- **GIF Previews:** Generates GIF previews for both uploaded and retrieved videos.

---

## Project Structure

```
team-name/
├── README.md
├── Project_Requirements_doc.md
├── src/
|   ├── embedder.ipynb # Notebook for embedding generation & FAISS indexing
|   └── frontend.py # Streamlit frontend app
├── docs/
│   ├── architecture_diagram.png
│   └── tech_stack.md
├── demo_folder/
|   ├── demo/
|   │   ├── demo_embeddings/ # demo embeddings generated on first 20 classes of UCF101
|   |   ├── demo_data_subsets/
|   │   └── demo.py
│   └── screenshots/
└── team_info.doc
```

---

## 🧠 Models

-Timesformer (default, Hugging Face)

-Easily extensible to other video transformer models

-Supports multiple pooling strategies (mean, max, CLS token) for flexible embeddings

## Demo Setup

-Install all libraries using requirements.txt

-Run demo.py in demo_src

-Download the UCF101 dataset and utilise any of the first 20 classes for demo testing

## 📊 Results

Retrieves and displays the top-k most similar videos to a given query using transformer-based embeddings and FAISS.

Supports and compares different pooling strategies (mean, max, CLS).

Visual previews (GIFs) make it easy to assess retrieval quality.
