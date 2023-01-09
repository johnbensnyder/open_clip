conda create -n open_clip_env --clone pytorch
source activate open_clip_env
pip install -r open_clip/requirements-training.txt
git clone -b datatype https://github.com/johnbensnyder/sagemaker-debugger
pip install -v -e ./sagemaker-debugger
