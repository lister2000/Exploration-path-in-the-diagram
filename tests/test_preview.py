import unittest
import numpy as np
import cv2
import sys
import os

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import GRID

class TestSamplePreview(unittest.TestCase):
    def test_show_sample_preview_returns_canvas(self):
        # create a synthetic color image (400x400)
        img = np.full((400, 400, 3), 120, dtype=np.uint8)
        # draw a darker patch to simulate real data
        cv2.rectangle(img, (50, 60), (70, 83), (80, 80, 80), -1)

        sample_rect = (50, 60, 70, 83)
        filter_threshold = 150

        # call function in headless mode (return_canvas=True)
        canvas = GRID.show_sample_preview(img, sample_rect, filter_threshold, return_canvas=True)

        # verify canvas shape and dtype
        self.assertIsNotNone(canvas)
        self.assertEqual(canvas.shape, (GRID.SAMPLE_PREVIEW_SIZE, GRID.SAMPLE_PREVIEW_SIZE, 3))
        self.assertEqual(canvas.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()
