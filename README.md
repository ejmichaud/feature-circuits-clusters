# feature-circuits-clusters

## How to launch the service on the cloud machine (for our internal use)

Once you've connected via the digital ocean console, cd into `/root/quanta-clusters-dev` (this is the old name for this repository, located in the root user home directory). From there you can pull any recent changes or make changes yourself. The server process is running in a tmux session. You can attach to it with `tmux a`. If you want to restart the server, you can just kill the script running in that session and re-run the script with `source launch.sh`.
