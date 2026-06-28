#!/bin/bash

IMAGE="tensorflow/tensorflow:latest-jupyter"

if ! docker image inspect "$IMAGE" &>/dev/null; then
  echo "Image $IMAGE not found locally. Pulling from Docker Hub..."
  docker pull "$IMAGE"
fi

docker run -it --rm -u "$(id -u):$(id -g)" -v /home/dev:/tf/notebooks -p 8888:8888 "$IMAGE"

