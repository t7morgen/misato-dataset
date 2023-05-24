
## Pull the existing image 
We recommend to pull our misato image from Dockerhub. 

## Create your own image
You can create a docker image using the given Dockerfile. Before you build the image please download ambertools from https://ambermd.org/GetAmber.php#ambertools and place the tar.bz2 file into this folder. In case you don't use AmberTools22 please change the Dockerfile accordingly.


```
sudo docker build -t misato .
```

To build a singularity container from the docker image run the following:

```
sudo docker build -t local/misato:latest .
sudo singularity build pyg.sif docker-daemon://local/misato:latest
```