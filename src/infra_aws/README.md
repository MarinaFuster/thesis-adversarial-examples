## AWS Rekognition scripting module

For this to work you will need a .env file and valid aws credentials in your `$HOME`

Run `main.py` with whatever command you want to do with the following syntax
```bash
python main.py (MODE) [...extra args]
```

## Execution modes

* **DESCRIBE**: Returns information of the Collection that Rekognition uses
    * Example: `python main.py DESCRIBE`


* **ADD_S3**: Upload given images to the S3 bucket
    * Example: `python main.py ADD_S3 ../data/marina19.jpg ../data/nachito32.jpg`


* **ADD_REK**: Add images to the AWS Rekognition Collection.
    * Important: The filenames should correspond to an existing S3 file, not to a file in the local filesystem
    * Example: `python main.py ADD_REK marina18.jpg nachito65.jpg`


* **LIST_S3**: List the content of the current S3 Bucket
    * Example: `python main.py LIST_S3`


* **LIST_REK**: List the faces saved in the current AWS Rekognition Collection along with their corresponding FaceID
    * Example: `python main.py LIST_REK`


* **DELETE_REK**: Delete the given faces from the AWS Rekognition collection
    * Important: The filenames should correspond to an existing S3 file, not to a file in the local filesystem
    * Example: `python main.py DELETE_REK marina19.jpg nachito32.jpg`


* **DELETE_S3**: Remove given images from the current S3 Bucket
    * Important: The filenames should correspond to an existing S3 file, not to a file in the local filesystem
    * Example: `python main.py DELETE_S3 marina19.jpg nachito32.jpg`


* **CREATE_COLLECTION**: Create an AWS Rekognition Collection with a given name
    * Example: `python main.py CREATE_COLLECTION training_set2`


* **DELETE_COLLECTION**: Delete the AWS Rekognition Collection that corresponds to the given name
    * Example: `python main.py DELETE_COLLECTION training_set2`


* **COMPARE_FACES_COLLECTION**: Compare the given image to the faces in the AWS Rekognition Collection and save the results
    * Important: it will upload the image from your filesystem to S3 (overwriting it if it exists) and use said image for comparison
    * Example: `python main.py COMPARE_FACES_COLLECTION ../data/marina19.jpg ../data/nachito32.jpg`


* **DETECT_FACE**: Detect the faces on a given image (showing it as a rectangle) and return the confidence score and predicted age
    * Important: The filename should correspond to an image in your filesystem
    * Example: `python main.py DETECT_FACE ../data/marina18.jpg ../data/nachito65.jpg`
    