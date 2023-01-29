FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker

RUN git clone -b datatype https://github.com/johnbensnyder/sagemaker-debugger/ \
    && pip uninstall -y smdebug \
    && pip install ./sagemaker-debugger
  
# XFormers failing on pip install. Need to investigate.
# RUN pip install -v git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
    
RUN git clone -b debugger https://github.com/johnbensnyder/open_clip/ \
    && pip install -r open_clip/requirements-training.txt
