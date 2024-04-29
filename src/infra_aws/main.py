import sys
from infra_aws.rekognition import Rekognition
from infra_aws.s3 import S3


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print(f"Usage: python {args[0].split('/')[-1]} (ADD_S3|ADD_REK|COMPARE|DESCRIBE|DELETE) [...images]")

    rekognition = Rekognition()
    s3 = S3()

    if not rekognition.collection_id or not s3.bucket_name:
        print("An error occurred... Aborting...")
        exit(1)

    print(f"Collection ID: {rekognition.collection_id}\nBucket Name: {s3.bucket_name}\n"
          f"Region: {rekognition.client.meta.region_name}\n")
    DATA_DIR = "../data/"
    action = args[1]

    if action == "DESCRIBE":
        rekognition.describe_collection()
    elif action == "ADD_S3":
        images = [DATA_DIR + x for x in args[2:]]
        s3.upload_images(images)
    elif action == "ADD_REK":  # TODO not working <:(
        rekognition.upload_images(s3.bucket_name, args[2:])
    elif action == "LIST_S3":
        s3.list_images()
    elif action == "LIST_REK":
        rekognition.list_images()
    elif action == "DELETE_REK":
        rekognition.remove_images(args[2:])
    elif action == "DELETE_S3":
        s3.remove_images(args[2:])
    elif action == "DELETE_COLLECTION":
        rekognition.remove_collection()
    elif action == "CREATE_COLLECTION":
        rekognition.create_collection(args[2])
    elif action == "COMPARE_FACES_COLLECTION":
        s3.upload_images([args[2]])
        rekognition.compare_faces_to_rek_collection_from_s3(s3, args[2])
    elif action == "DETECT_FACE":
        rekognition.detect_face(args[2])
