import unittest
from os import listdir

import pytest

from modules.config import Config

class ConfigModuleTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_tests(self):
        self.config = Config("models/config.json")
        yield

    def test_global_match(self):
        train, test = self.config.load_data()
        file_count = len(listdir("data"))
        self.assertEqual(file_count, len(train))
        self.assertEqual(file_count, len(test))

    def test_particular_match(self):
        self.config.image_prefix = "marina"
        train, test = self.config.load_data()
        file_count = 10
        self.assertEqual(file_count, len(train))
        self.assertEqual(file_count, len(test))

    def test_invalid_prefix_throws_exception(self):
        self.config.image_prefix = "thisisntreal"
        with self.assertRaises(FileNotFoundError):
            self.config.load_data()
    
    def test_invalid_folder(self):
        self.config.train_folder = "thisdoesntexist"
        with self.assertRaises(FileNotFoundError):
            self.config.load_data()

    def test_read_valid_json_file(self):
        maybe_json = Config.read_configuration("models/config.json")
        self.assertIsNotNone(maybe_json)

    def test_read_invalid_json_raises_exception(self):
        with self.assertRaises(FileNotFoundError):
            Config.read_configuration("idontexist")