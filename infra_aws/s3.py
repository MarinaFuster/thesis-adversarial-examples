import boto3


class S3:
    def __init__(self):
        self.client = boto3.client('s3')
        self.bucket_name = "pf-photo-dump"  # self._get_bucket_name()

    def _get_bucket_name(self):
        buckets = self.client.list_buckets()["Buckets"]

        if not buckets:
            print("There are no buckets created! :(")
            return ""

        return buckets[0]["Name"]

    def list_images(self):
        """
        List the content of the current S3 Bucket
        """
        objects = self.client.list_objects_v2(Bucket=self.bucket_name)
        print(f"Listing images in {self.bucket_name}")
        for image in objects["Contents"]:
            print(image["Key"])

        return objects["Contents"]

    def upload_images(self, images):
        """
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.put_object
        Upload given images to the S3 bucket
        """
        if not images:
            print("ERROR! No images given")

        for filename in images:
            print(f"Uploading {filename} to S3 bucket {self.bucket_name}")

            res = self.client.put_object(
                ACL="private",
                Body=filename,
                Key=filename.split("/")[-1],
                Bucket=self.bucket_name,
                ServerSideEncryption='AES256',
                StorageClass="STANDARD",
            )
            print(f"Upload successful")
            print(res)

    def remove_images(self, remote_img_names):
        """
        Remove given images from the current S3 Bucket
        """
        objects = [{'Key': image} for image in remote_img_names]
        print(f"Deleting {len(objects)} images")
        res = self.client.delete_objects(
            Bucket=self.bucket_name,
            Delete={
                "Objects": objects
            }
        )
        print(f"Successfully delete {len(objects)} images from S3")
        print(res)
        return res