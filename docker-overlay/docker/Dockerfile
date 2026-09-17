# Empty image: every runtime file comes from read-only bind mounts of the board (see run_overlay.sh).
FROM scratch
ENV PATH=/usr/bin:/bin HOME=/tmp XDG_CACHE_HOME=/tmp/.cache
WORKDIR /opt/neat-app
ENTRYPOINT ["/opt/neat-app/overlay-detector"]
