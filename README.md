# MSc-pfi

It is important to note that all the directories refer to the remote server in DTU due to resources limitations. Data are also stored in the remote server.

In the script ''mainscript.py", the training options are initialiazed along with the definition of the directories. The dataloaders (included in the ''torchdataloaders.py" script -functions required from PyTorch" - are called in this script. Following, the selected model is loaded and its training phase is happening.

The script ''testing1.py'' performs the inference of the model to new data, to evaluate its performance for predicting the sea ice type.

