import sys
import os

from modules.config import Config


def json_config_exists(json_path):
    if not os.path.exists(json_path):
        print(f"{json_path} is not a valid configuration file")
        return False
    return True


def check_arg_count(arguments):
    if len(arguments) != 2:  # args includes filename
        print("Usage: ./main.py [path_to_json_config]")
        return False
    return True


if __name__ == "__main__":
    args = sys.argv
    if not (check_arg_count(args) and json_config_exists(args[1])):
        exit(1)

    config = Config(args[1])
    # Do stuff
    print("Done!")
