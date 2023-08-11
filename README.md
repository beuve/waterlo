# Dependences

## Using VSCode (Recommended)

Using the **remote container** extension of VS Code, one can easily compile the dockerfile and use VS Code within a container.

Upon installing the remote container extension in VS Code (search for **ms-vscode-remote.remote-containers**), open the project in vscode and click on the icon in the bottom left corner. Int the dialog that appears, select **Reopen in Container**.  VS Code will then compile the Dockerfile and mount the project folder in the container with all necessary dependencies. 

> **Note**: If you want to use a different folder to store your data, simply uncomment and modify the `mounts` option in *.devcontainer/devcontainer.json*. This will mount an additional folder under */workspaces/data*.

## Using Docker

If you are not a VS Code user, you can also compile the dockerfile yourself. Then you can slightly modify the scripts to run the programms withing the container using `docker run`.  You can also mount the project folder into the container and start a bash session within the container by running: 

``` bash
docker run -v ${PWD}:/workspaces/watermark -it $IMAGE_NAME bash
```

> **Note**: If you want to use a different folder to store your data, simply add a new mounting point using the -v option of Docker.

## Using local installation

If neither of the two options above meet your workflow, you can install locally the python dependencise (torch, torchvision, and all packages listed in the *requirements.txt* file). We only tested with cuda 11.3 and can't confirm this will work with other versions. 

# Training

Training is done by running the training script (located at *./scripts/train.sh*). You need to specify the path to the input as well as a folder to store the output.

The training dataset shall be organised as follow: 

```
dataset
├── train
│   ├── train_img01.png
│   ╰── ...
╰── valid
    ├── valid_img01.png
    ╰── ...
```

You can also change the value of alpha, lambda, the loss used (either `mse` or `ssim`), or whether the compression module is enabled or not (uncomment `--compression` to enable it). 

# Testing

Once the model has been trained, you can test it in 3 steps: 

- First, the watermark is applied using the corresponding script (located at ./scripts/apply-watermarks.sh). Some variables need to be set before running the script.
- Then, a deepfake is produced. For this step, we used external scripts. 
    - FaceSwap: https://github.com/wuhuikai/FaceSwap
    - FaceShifter: https://github.com/richarduuz/Research_Project (model B)
- Finally, the modified regions can be detected using the watermark detection script (located at ./scripts/detect-watermarks.sh). Again, some variables need to be set beforehand.

> **OPTIONAL**: To get the deepfake detection accuracy, you can use the deepfake detection script (located at ./scripts/detect-deepfakes.sh). This requires a dataset of watermarked images alongside the face detection boundaries in the same folder, organized as follow:
> 
> ```
> dataset
> ├── img01.png
> ├── ...
> ╰── facial_landmarks.txt
> ```
> 
> With the facial_landmarks file organized as:
> 
> ```
> img01	x	y	w	h	
> img02	x	y	w	h	
> ... 
> ```
> Where img01 correspond to the name of the first image without its extension (i.e. `img01`, not `img01.png`), (x,y) correspond to the top left point of the face bounding box (it is assumed there is only one face in the image. if there are more, just add the coordinates of the deepfake), and (w,h) respectively the width and height of the face bounding box.

# Quality assesment

You can check the quality (PSNR) of the watermarked image using the quality script (located at ./scripts/quality.sh). 