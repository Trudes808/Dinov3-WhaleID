# Setup Commands

Commands executed to install and launch the Voxel51/FiftyOne quickstart:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fiftyone jupyter
jupyter notebook voxel51_getting_started.ipynb

# Additional installs for DINOv3 notebook imports
pip install git+https://github.com/huggingface/transformers
pip install huggingface_hub
pip install torch pillow scikit-learn numpy
```
