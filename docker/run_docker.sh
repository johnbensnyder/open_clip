CONTAINER_NAME=open-clip-sm
REPO=${1-ecr-pt-repo}
TAG=open-clip-sm
REGION=${2-us-west-2}
AWS_ACCOUNT=`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`
IMAGE_NAME=${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}

docker run -d -it --rm --gpus all --name ${CONTAINER_NAME} \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    -w /opt/ml/code \
    -v /home/ubuntu/open_clip:/opt/ml/code \
    ${IMAGE_NAME}