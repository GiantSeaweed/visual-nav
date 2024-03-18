xhost +local:

docker run --rm -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${PWD}:/visnav \
    --name visnav2 \
    visual-nav \
    /bin/bash -c "source ros_entrypoint.sh && cd visnav/ && bash"
