# FeatureDifferentiableGraphKernelConvolution

## Installation
The project folder is compatible with Visual Studio Code dev containers. The
Dockerfile inside will install Python 3.8.12 to a Cuda base image complete with
the necessary packages to run the code. The packages are inside
`requirements.txt` and `requirements2.txt`. The first will be installed after
the compilation of Python; the second after the installation of pytorch and
pytorch geometric packages that are installed using the pip command.

#### *Prerequisites*
You have to install Docker in your target system and CUDA if you want GPU
computation.

### With VS Code
If you use Visual Studio Code, it will recognize the devcontainer, install it
and then you will be able to run `python3 ./main.py` directly from the
integrated console.

### Standalone container
If you want to run the container by itself, follow this procedure:
1. you have to start the compose file  with `docker compose up -d`;
2. once the container is up you can access it with the
command `docker exec -it graph_kernel_convolution /bin/bash`;
3. run python script with `python3 ./main.py`.

You can directly execute the python script merging steps 2 and 3 running the
command `docker exec -it graph_kernel_convolution python3 ./main.py`.

### Grid search
To perform a grid search you have to launch the `launch_grid.py`. It will read
a json file containing an array of dictionaries consisting the arguments of the
argparser in `main.py`. Each argument must be an array. Example:
```
[{
    "group": "wandb group",
    "project": "wandb project",
    "grid": {
            "fold": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "lr": [1e-3],
            "hidden": [4, 8, 16, 32, 64],
            "nodes": [6, 8, 16, 32, 64],
            "jsd_weight": [0],
            "hops": [1, 2, 3],
            "layers": [1, 3, 5, 7, 9, 11],
            "dropout": [0, 0.1, 0.5],
            "dataset": ["MUTAG"],
            "temp": [true]
    }
}]
```
