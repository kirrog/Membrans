# install with conda
- conda create -n tf-gpu
- conda activate tf-gpu
- conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
- python3 -m pip install tensorflow==2.9.2
- python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
- python3 -m pip install albumentations==1.1.0 matplotlib==3.4.3 keras==2.9.0 tqdm==4.64.1