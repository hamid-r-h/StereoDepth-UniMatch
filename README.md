
# Depth Estimation with Enhanced UniMatch

This repository provides an enhanced implementation of [UniMatch](https://github.com/autonomousvision/unimatch), originally developed by the Autonomous Vision Group. It primarily focuses on improving **stereo depth estimation** through various optimization techniques such as visual analysis, grayscale conversion, and image filtering. Users can experiment with stereo image pairs and evaluate resulting disparity maps using improved visualization.

## 🚀 Features

- Optimized runtime performance of `app.py`.
- Integration with UniMatch architecture for advanced stereo matching.
- Supports custom stereo image pairs.
- Preprocessing capabilities including grayscale conversion and median filtering.
- Disparity map generation and visualization.
- Support for multiple image formats (e.g., `.tif`, `.png`).
- Compatibility with aerial and ground-level images.
- Enhanced cropping and pre-filtering for better results.

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/depth-estimation-unimatch.git
cd depth-estimation-unimatch

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for notebook
pip install gradio opencv-python tifffile matplotlib rasterio
```

> ⚠️ **Prerequisites**: Ensure [PyTorch](https://pytorch.org/) with CUDA is installed for GPU acceleration.

## 📁 Directory Structure

```
depth-estimation-unimatch/
├── output/
├── app.py
├── depth_estimation.ipynb
├── requirements.txt
└── README.md
```

## 📷 Input/Output

- **Input**: Stereo image or video pairs (left and right).
- **Output**: Generated disparity map saved in the `output/` directory.

## 📓 Usage

Run the notebook:

```bash
jupyter notebook depth_estimation.ipynb
```

Alternatively, convert and adapt notebook functionalities into standalone scripts as needed.

## 🧠 Workflow Logic

The implemented workflow includes:

1. Loading and optionally cropping stereo images.
2. Converting images to grayscale and filtering for noise reduction.
3. Computing disparity using the integrated `get_disp()` function (UniMatch).
4. Visualizing and saving the generated disparity maps for evaluation.

## 🖼️ Visualization

Visualization includes detailed disparity maps and original images displayed via `matplotlib`, featuring colorbars to illustrate depth intensities clearly.

## 📌 Important Notes

- The disparity maps are normalized and scaled as needed (divided by 100 in some cases).
- Adjust image paths and parameters according to your specific dataset.

## 📃 Credits

- Built upon [UniMatch](https://github.com/autonomousvision/unimatch) by the Autonomous Vision Group.
