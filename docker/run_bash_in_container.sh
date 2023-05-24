#!/bin/bash
USERMAPPING="--user $(id -u $USER):$(id -g $USER)"

docker run --network=host $USERMAPPING --rm -e USER -e HOME -v $HOME/MiSaTo-dataset:$HOME/MiSaTo-dataset -it sab148/misato-dataset bash "$@"
