docker run --name segmentation_learning_env_$1 \
    -it \
    --rm \
    --gpus all \
    --shm-size 8G \
    --mount type=bind,source="$(pwd)",target=/root/src \
    --dns 8.8.8.8 \
    segmentation_learning:latest
