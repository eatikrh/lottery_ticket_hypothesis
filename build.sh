# podman machine init
# podman machine start
podman build -t double-descent .
podman run --rm -v $(pwd):/app double-descent
