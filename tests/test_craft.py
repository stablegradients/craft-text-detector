import unittest

from numpy import imag
from craft_text_detector import Craft
import cv2 
import numpy as np

class TestCraftTextDetector(unittest.TestCase):
    image_path = "figures/auto4.jpeg"

    def test_init(self):
        craft = Craft(
            output_dir=None,
            rectify=True,
            export_extra=False,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=720,
            refiner=False,
            crop_type="poly",
        )
        self.assertTrue(craft)

    def test_load_craftnet_model(self):
        # init craft
        craft = Craft(
            output_dir=None,
            rectify=True,
            export_extra=False,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=720,
            refiner=False,
            crop_type="poly",
        )
        # remove craftnet model
        craft.craft_net = None
        # load craftnet model
        craft.load_craftnet_model()
        self.assertTrue(craft.craft_net)

    def test_load_refinenet_model(self):
        # init craft
        craft = Craft(
            output_dir=None,
            rectify=True,
            export_extra=False,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=720,
            refiner=False,
            crop_type="poly",
        )
        # remove refinenet model
        craft.refine_net = None
        # load refinenet model
        craft.load_refinenet_model()
        self.assertTrue(craft.refine_net)

    def test_detect_text(self):
        # init craft
        craft = Craft(
            output_dir=None,
            rectify=True,
            export_extra=False,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=720,
            refiner=False,
            crop_type="poly",
        )
        # detect text
        #prediction_result = craft.detect_text(image_path=self.image_path)
        '''
        self.assertEqual(len(prediction_result["boxes"]), 52)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][0][0]), 115)
        '''
        # init craft
        craft = Craft(
            output_dir=None,
            rectify=True,
            export_extra=False,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=720,
            refiner=True,
            crop_type="poly",
        )
        # detect text
        #prediction_result = craft.detect_text(image_path=self.image_path)
        '''
        self.assertEqual(len(prediction_result["boxes"]), 19)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][2][0]), 661)
        '''
        # init craft
        craft = Craft(
            output_dir=None,
            rectify=False,
            export_extra=False,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=720,
            refiner=False,
            crop_type="box",
        )
        # detect text
        #prediction_result = craft.detect_text(image_path=self.image_path)
        '''
        self.assertEqual(len(prediction_result["boxes"]), 52)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][2][0]), 244)
        '''
        # init craft
        craft = Craft(
            output_dir=None,
            rectify=False,
            export_extra=False,
            text_threshold=0.4,
            link_threshold=0.2,
            low_text=0.4,
            cuda=False,
            long_size=720,
            refiner=True,
            crop_type="box",
        )
        # detect text
        print("initiating")
        prediction_result = craft.detect_text(image_path=self.image_path)
        im = cv2.imread(self.image_path)
        #print(im.shape)
        '''
        self.assertEqual(len(prediction_result["boxes"]), 19)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][2][0]), 661)
        '''
        
        image = cv2.imread(self.image_path)
        for i, img in enumerate(prediction_result["text_crops"]):
            cv2.imshow("image", img)
            cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()