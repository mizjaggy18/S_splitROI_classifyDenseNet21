FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
CMD nvidia-smi

FROM cytomine/software-python3-base:v2.2.0

#INSTALL
RUN pip install numpy
RUN pip install shapely
RUN pip install torch
RUN pip install torchvision


#ADD FILES
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD splitroi_densenet21.py /app/splitroi_densenet21.py
ADD 66k-4classnpc_densenet21adam_best_model_100ep.pth /app/66k-4classnpc_densenet21adam_best_model_100ep.pth
RUN chmod 444 /app/66k-4classnpc_densenet21adam_best_model_100ep.pth


ENTRYPOINT ["python3", "/app/splitroi_densenet21.py"]
