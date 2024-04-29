#!/bin/bash

set -e # On error, we want to exit

function init_config(){
    if [ -f ".env" ]; then
        set -o allexport; source .env; set +o allexport # Read .env files
    else
        echo "Missing .env file!"
        exit 1
    fi

    if [ ! -d "$HOME/.aws" ]; then
        echo "Mising .aws file! You won't be able to connect to aws services"
        exit 1
    fi
}

function check_arguments(){
    EXPECTED_ARG_NUM=$1
    REAL_ARG_NUM=$2
    if (( "$REAL_ARG_NUM" < "$EXPECTED_ARG_NUM" )); then
        echo "Invalid argument number for $mode. Expected $EXPECTED_ARG_NUM got $REAL_ARG_NUM"
        exit 1
    fi
}

# Since we are passing a full path, we need to parse it to remove the path until the last element
# /foo/bar.jpg turns into bar.jpg
function parse_filename(){
    filename=(${1//\// })
    last_index="${#filename[@]}"
    parsed_name=${filename[last_index - 1]}
}

function save_face_to_rekognition(){
    file="$1"
    image="{\"S3Object\":{\"Bucket\":\"$BUCKET_NAME\",\"Name\":\"$file\"}}"
    aws rekognition index-faces \
    --image "$image" \
    --collection-id "$COLLECTION_ID" \
    --max-faces 1 \
    --quality-filter "AUTO" \
    --detection-attributes "ALL" \
    --external-image-id "$file" > "$BUCKET_NAME-${file%.*}.info"

    echo "Face information saved in $BUCKET_NAME-${file%.*}.info"
}

function compare_face_with_database(){
    echo "Comparing $parsed_name to rekognition database"
    aws rekognition search-faces-by-image \
    --image "{\"S3Object\":{\"Bucket\":\"$BUCKET_NAME\",\"Name\":\"$parsed_name\"}}" \
    --collection-id "$COLLECTION_ID" 
	  --face-match-treshold 80 > "$2/$BUCKET_NAME-${parsed_name%.*}-comparison.info"

    echo "Face comparison saved in $2/$BUCKET_NAME-${parsed_name%.*}-comparison.info"
}

function describe_rekognition_collections(){
    aws rekognition describe-collection --collection-id "$COLLECTION_ID"
}

# https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html
function upload_image_to_s3(){
   echo "Uploading $1 to bucket $BUCKET_NAME"
   aws s3 cp "$1" s3://"$BUCKET_NAME" --acl private --sse AES256 --storage-class STANDARD
   echo "Done! $1 is now in $BUCKET_NAME"

}

function delete_image_from_s3(){
    echo "Deleting image $1 from bucket $BUCKET_NAME"
    OBJECT_URI="s3://$BUCKET_NAME/$1"
    aws s3 rm "$OBJECT_URI"
}

init_config

USAGE_MSG="Usage: $0 (ADD|COMPARE|DESCRIBE|UPLOAD|DELETE) [...images]"
COMPARISON_DIR="comparisons"

if [ $# -lt 1 ]; then
    echo "$USAGE_MSG"
    exit 1
fi

if [ ! -d "$COMPARISON_DIR" ]; then
  mkdir "$COMPARISON_DIR"
fi
mode=$1
shift

ARG_NUM=1;
case $mode in
    "ADD")
        check_arguments "$ARG_NUM" "$#"
        for filename in "$@"; do
            echo "Adding face $filename to rekognition"
            save_face_to_rekognition "$filename"
        done
        ;;
    "COMPARE")
        check_arguments "$ARG_NUM" "$#"
        upload_image_to_s3 "$1"

        parse_filename "$1"
        echo "Parsed filename: $parsed_name"

        compare_face_with_database "$parsed_name" "$COMPARISON_DIR"
        delete_image_from_s3 "$parsed_name"
        ;;

    "DESCRIBE")
        describe_rekognition_collections
        ;;
    "UPLOAD")
        check_arguments "$ARG_NUM" "$#"
        for filename in "$@"; do
            echo "Uploading $filename to s3 bucket $BUCKET_NAME"
            upload_image_to_s3 "$filename"
        done
        ;;
    "DELETE")
        check_arguments "$ARG_NUM" "$#"
        delete_image_from_s3 "$@";
        ;;
    *)
        echo "Oh no, something went wrong"
        echo "$USAGE_MSG"

esac

