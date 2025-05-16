# Project Requirements
This document lists all the requirements for running, developing, and experimenting with the forvideo video similarity search project.
## 🖥️ System Requirements

- **Operating System**: Linux, macOS, or Windows

- **Python Version**: 3.8 or higher

- **RAM**: Minimum 8GB (16GB+ recommended for large datasets)

- **GPU**: Recommended for faster embedding extraction (CUDA-compatible for PyTorch)

## 📦 Python Dependencies

Install all dependencies using the provided requirements.txt:

## 📚 Dataset

- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) or your own video dataset

- For UCF101, download from UCF101 [Official Site](https://www.crcv.ucf.edu/data/UCF101.php) and extract to src/UCF101/.

## ⚙️ Additional Requirements

- Jupyter Notebook (for running embedder.ipynb)

- Internet Connection (for downloading pretrained models from Hugging Face)

- Hugging Face Account (optional, for uploading/sharing models)

## 📝 Notes

- If running on Windows, you may need additional packages like pywin32 for video processing.

- For large-scale datasets, ensure you have enough disk space for video files and generated embeddings.

- GPU is highly recommended for embedding extraction but not required for running the Streamlit frontend.
