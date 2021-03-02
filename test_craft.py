import unittest

from numpy import imag
from craft_text_detector import Craft
import cv2 
import numpy as np

class TestCraftTextDetector(unittest.TestCase):
    image_path = "figures/auto6.jpeg"

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
        print(im.shape)
        '''
        self.assertEqual(len(prediction_result["boxes"]), 19)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][2][0]), 661)
        '''
        '''
        image = cv2.imread(self.image_path)
        for i, box in enumerate(prediction_result["boxes"]):
            xmin, ymin = int(min(box[:,0])), int(min(box[:,1]))
            xmax, ymax = int(max(box[:,0])), int(max(box[:,1]))
            cv2.imshow("image"+str(i), image[ymin:ymax, xmin:xmax])
        cv2.waitKey(0)
        '''
        img = cv2.imread(self.image_path)
        '''
        for image in prediction_result["text_crops"]:
            cv2.imshow("word", image)
            cv2.waitKey(0)
        '''
        
        def combine_images(img1, img2):
            h_min = max(img1.shape[0], img2.shape[0]) 
        
            # image resizing 
            img_list = [img1, img2]  
            im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation = cv2.INTER_CUBIC)  for img in img_list]
            return cv2.hconcat(im_list_resize) 

        def warp_image(box):
            box = box.astype(np.float32)
            w, h = (int(np.linalg.norm(box[0] - box[1]) + 1),int(np.linalg.norm(box[1] - box[2]) + 1))
            tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            M = cv2.getPerspectiveTransform(box, tar)
            word_crop = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST)
            return word_crop
        boxes, num_characters = prediction_result["boxes"], prediction_result["num_characters"] 
        word_id = prediction_result["word_id"]
        text_crops = []
        num_combined_characters = []
        for i, (box, num_char)  in enumerate(zip(boxes, num_characters)):
            word_crop = warp_image(box)
            if word_id[i] == -1:
                text_crops.append(word_crop)
                num_combined_characters.append(num_char)
                cv2.imshow("warp", word_crop)
                cv2.waitKey(0)
            else:
                box_ = boxes[word_id[i]]
                word_crop_ = warp_image(box_)
                text_crops.append(combine_images(word_crop, word_crop_))
                cv2.imshow("warp",combine_images(word_crop, word_crop_))
                cv2.waitKey(0)
                num_char_ = num_characters[i]
                num_combined_characters.append(num_char+num_char_)
            
            print(prediction_result["word_id"])
            print(prediction_result["num_combined_characters"])
            print(prediction_result["num_characters"])

if __name__ == "__main__":
    unittest.main()