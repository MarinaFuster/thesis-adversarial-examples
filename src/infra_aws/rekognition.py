import boto3
from botocore.exceptions import ClientError, ParamValidationError
from PIL import Image, ImageDraw


class Rekognition:
    def __init__(self):
        self.client = boto3.client('rekognition')
        self.collection_id = self._get_collection()

    def _get_collection(self):
        collections = self.client.list_collections()["CollectionIds"]
        if not collections:
            print("There are no created collections! :(")
            return ""

        return collections[0]

    def create_collection(self, collection_id):
        """
        Create an AWS Rekognition collection named `collection_id`
        """
        print(f"Creating collection {collection_id}")
        res = self.client.create_collection(
            CollectionId=collection_id,
            Tags={
                "CreatedBy": "boto3 script",
                "Description": "Faces of Maru and Nachito"
            }
        )
        print("Successfully created collection")
        print(res)

    def describe_collection(self):
        """
        Returns information of the Collection that Rekognition uses
        """
        collection = self.client.describe_collection(CollectionId=self.collection_id)
        timestamp = collection["CreationTimestamp"]
        date = timestamp.date()
        date = f"{date.day}/{date.month}/{date.year}"
        print(f"COLLECTION {self.collection_id}:\n\t- Face Count: {collection['FaceCount']}\n\t- Face Model Version:"
              f" {collection['FaceModelVersion']}\n\t- Collection ARN: {collection['CollectionARN']}\n\t"
              f"- Creation Time: {date} - {timestamp.time()}")

    def remove_collection(self):
        """
        Delete the AWS Rekognition Collection named `collection_id`
        """
        ans = input(f"You are about to remove the collection {self.collection_id} "
                    "from Amazon Rekognition. Are you sure? (y/n)\n>")

        if ans.lower() == "y":
            print(f"Removing {self.collection_id} from Rekognition")
            res = self.client.delete_collection(CollectionId=self.collection_id)
            print(f"Removed {self.collection_id} from Rekognition successfully")
        elif ans.lower() == "n":
            print("Aborting...")
            res = None
        else:
            print(f"\"{ans}\" is not a recognized answer. Exiting now...")
            res = None

        return res

    def _get_face_id_dict(self):
        objects = self.client.list_faces(CollectionId=self.collection_id)

        faces = dict()
        for face in objects["Faces"]:
            faces[face["ExternalImageId"]] = face["FaceId"]

        return faces

    def remove_images(self, remote_img_names):
        """
        Delete the given faces from the AWS Rekognition collection
        """
        faces_dict = self._get_face_id_dict()
        faces_ids = []
        for image_name in remote_img_names:
            print(f"Deleting {image_name} from {self.collection_id} ")
            faces_ids.append(faces_dict[image_name])

        res = self.client.delete_faces(CollectionId=self.collection_id, FaceIds=faces_ids)
        print(f"Successfully deleted {len(remote_img_names)} images")
        print(res)
        return res

    def list_images(self):
        """
        List the faces saved in the current AWS Rekognition Collection along with their corresponding FaceID
        """
        print(f"Listing faces saved in the {self.collection_id} collection (Rekognition)")
        faces = self._get_face_id_dict()
        for face_name in sorted(faces):
            print(f"Face: {face_name} - FaceId: {faces[face_name]}")

        return faces

    def upload_images(self, bucket_name, images):
        """
        Add images to the AWS Rekognition Collection
        """
        for filename in images:
            print(f"Uploading {filename} to collection {self.collection_id}")
            img_binary = open(filename, "rb")
            res = self.client.index_faces(
                CollectionId=self.collection_id,
                ExternalImageId=filename,
                Image={
                    'Bytes': img_binary.read()
                },
            )

            print("Upload successful!")
            print(res)

    def compare_faces_to_rek_collection_from_s3(self, s3_client, image):
        """
        Compare the given image to the faces in the AWS Rekognition Collection and save the results
        """
        print(f"Comparing {image} to faces in rekognition collection")
        res = self.client.search_faces_by_image(
            CollectionId=self.collection_id,
            Image={
                'S3Object': {
                    'Bucket': s3_client.bucket_name,
                    'Name': image
                }
            },
            FaceMatchThreshold=0.7
        )

        output_file = open(f"comparisons/{s3_client.bucket_name}-{image.split('.')[0]}-comparison.info")
        output_file.write(res)
        return res

    def compare_faces_to_rek_collection_from_local(self, s3_client, images, face_threshold=0.7):
        """
        [DEPRECATED]
        Compare the given image to the faces in the AWS Rekognition Collection and save the results
        """

        for image in images:
            image_binary = open(image, "rb")
            print(f"Comparing {image} to faces in rekognition collection")
            res = self.client.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={
                    'Bytes': image_binary.read()
                },
                FaceMatchThreshold=face_threshold
            )

            output_file = open(f"comparisons/{s3_client.bucket_name}-{image.split('.')[0]}-comparison.info", "w")
            output_file.write(str(res))

    def compare_face_to_rek_collection_from_local(self, image, face_threshold=0.0):
        """
        Compare the given image to the faces in the AWS Rekognition Collection and save the results
        """

        image_binary = open(image, "rb")
        print(f"Comparing {image} to faces in rekognition collection")
        res = self.client.search_faces_by_image(
            CollectionId=self.collection_id,
            Image={
                'Bytes': image_binary.read()
            },
            FaceMatchThreshold=face_threshold
        )
        image_binary.close()

        return res

    def detect_face(self, image):
        """
        Detect the faces on a given image (showing it as a rectangle) and return the confidence score and predicted age
        """
        img_binary = open(image, "rb")
        return self.client.detect_faces(Image={'Bytes': img_binary.read()}, Attributes=['ALL'])

    def detect_face_and_draw(self, image):
        img_binary = open(image, "rb")
        response =  self.client.detect_faces(Image={'Bytes': img_binary.read()}, Attributes=['ALL'])

        image = Image.open(image)
        imgWidth, imgHeight = image.size
        draw = ImageDraw.Draw(image)

        for face_detail in response['FaceDetails']:
            print(f"The detected face is between {str(face_detail['AgeRange']['Low'])}"
                  f" and {str(face_detail['AgeRange']['High'])} years old")

            print(f"Confidence value is: {face_detail['Confidence']}")
            box = face_detail['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            print('Left: ' + '{0:.0f}'.format(left))
            print('Top: ' + '{0:.0f}'.format(top))
            print('Face Width: ' + "{0:.0f}".format(width))
            print('Face Height: ' + "{0:.0f}".format(height))

            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top)

            )
            draw.line(points, fill='#06d6a0', width=2)

        image.show()

    def compare_face(self, source_image, target_image):
        """
        Compare the given source image with the target image
        """

        try:
            response = self.client.compare_faces(
                SourceImage={
                    "Bytes": open(source_image, "rb").read()
                },
                TargetImage={
                    "Bytes": open(target_image, "rb").read()
                },
                SimilarityThreshold=0.0
            )
        except ParamValidationError as error:
            print(f"PARAM ERROR: {error}")
            return []
        except ClientError as error:
            print(f"CLIENT ERROR!: {error}")
            return []
        for match in response["FaceMatches"]:
            print(f" SIMILARITY: {match['Similarity']}")
            quality = match["Face"]["Quality"]
            print(f" Quality_Brightness: {quality['Brightness']}")
            print(f" Quality_Sharpness: {quality['Sharpness']}")

        return response["FaceMatches"]
