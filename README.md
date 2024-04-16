
## Set up

Install virtualenv to keep all your dependencies in one place.
```bash
sudo apt-get install virtualenv
```

After that, create a virtual environment
```bash
virtualenv venv --python=python3
```

This should create a venv folder. Activate it by running
```bash
source venv/bin/activate
```

Install the dependencies
```bash
pip install -r requirements.txt
```

Once you are all done, get out of the virtual environment by running
```bash
deactivate
```

## Using TensorBoard

First make sure you have installed TensorBoard on your virtual environment.
(Run the "Install Dependencies" step).

The default directory where we are saving our TensorBoard logs is in `/tmp/pf-logs/[timestamp]`
So, when you finish training the Autoencoder run something like:
```bash
tensorboard --logdir /tmp/pf-logs/20200925-123514
```
This will run the tensorboard server which you can visit in `http://localhost:6006/`

## Autoencoder and PCA
This project will allow you to use and autoencoder and pca modules to create experiments with Amazon Rekognition.
Amazon Rekognition is a state-of-the-art Facial Recognition System.

Our proposition is that, by manipulating the principal components of the latent representation of an image
(which will be obtained by an autoencoder) we could deceive Amazon Rekognition on recognizing one of us.

Keep in mind that if you want to reproduce this experiments, you will need an Amazon Rekognition account.
If you are a student, you can get a student license that will be enough to run some experiments.

If you want to reproduce our results, you can ask for our training dataset (and additional information)
by sending an email to us:

Marina Fuster: **mfuster@itba.edu.ar** <br/>
Ignacio Vidaurreta: **ividaurreta@itba.edu.ar**

### Autoencoder Training
All the autoencoder configuration is set at `modules/autoencoder.py`. If there is anything you would like
to change from its parameters, you can do it there. Training epochs and batch size are defined in 
`/core/training.py`.

In order to train your autoencoder make sure you have a `models/` folder and run:
```bash
python training.py
```

Once the process is finished, you should have, in your `models/` folder the following files:<br/>
`encoder.json`, `ecoder_weights.h5`, `decoder.json`, `deoder_weights.h5`

### Compute PCA utilities
In order to create experiments using principal components you will need two standard scalars and a principal
components' module. 

In order to create all these models make sure you have a `models/` folder and run
```bash
python compute_pca_utilities.py
```

### Experimentation
In order to transform an image you can use the next transformation flow as an example:
```bash
python transformation_flow.py -i ./data/marina0.jpg
```

This will store a `.csv` file in `components/`

You can make necessary changes in that file. Once you are done, run
```bash
python transformation_flow.py -c ./components/marina0.csv
```

This will reconstruct the image from modified components and store that in `components/`

Once you did this, run the bash scripts to connect to AWS Rekognition and get the JSON.

If you have finished, you can process your comparisons running

```bash
python json_processing.py -s [path_to_file]
```
if you want to process a single file or

```bash
python json_processing.py -b [path_to_folder]
```
if you want to process a batch of files.
