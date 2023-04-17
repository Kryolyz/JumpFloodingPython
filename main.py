
import cv2
import numpy as np
import argparse
import time


class JFA():
    """
    This class implements the Jump Flooding Algorithm (JFA) for finding 2D distance fields from images.
    """
    def __init__(self, path, steps):
        """
        Initializes the JFA object with the given image and number of steps.
        
        :param path: The input image path to process.
        :type img: string
        :param steps: The number of steps to perform in the JFA algorithm.
        :type steps: int
        """
        
        image = cv2.imread(path)
        image_bitwise = cv2.bitwise_not(image)
        
        self.img = image_bitwise
        self.width, self.height = image_bitwise.shape[:2]
        # Find edges in bitmap
        self.edges = self.findEdges()
        # Initialize by replacing the edge pixels with values representing their own position
        self.JFA = self.JFAInitialize()
        
        # Perform JFA per step
        for i in np.arange(steps-1, -1, -1):
            start_time = time.time()
            self.JFA = self.JFAStep(2**i)
            end_time = time.time()
            print(f"Step {i} took {end_time - start_time} seconds") # get a sense for the time each step takes
            
        # replace the positions encoded in each pixel by a greyscale representing its distance to the closest edge
        self.decrypted = self.JFADecrypt()
        
        # Optional resizing for visibility, comment out if undesirable
        self.img = cv2.resize(self.img, (128 * 4, 128 * 4), interpolation=cv2.INTER_AREA)
        self.edges = cv2.resize(self.edges, (128 * 4, 128 * 4), interpolation=cv2.INTER_AREA)
        self.decrypted = cv2.resize(self.decrypted, (128 * 4, 128 * 4), interpolation=cv2.INTER_AREA)
        
        # Show the original bitmap, the edges and the distance field
        cv2.imshow("Image", self.img)
        cv2.imshow("edges", self.edges)
        cv2.imshow("JFA", self.decrypted)
        
        cv2.waitKey()
        cv2.destroyAllWindows()

    def findEdges(self):
        """
        Finds the edges in the given image.
        
        :return: A binary image where edge pixels are kept and non-edge pixels are set to 0.
        :rtype: numpy.ndarray
        """
        im = np.array(self.img)
        res = np.zeros(shape=(self.width, self.height), dtype=np.float32)
        for x in np.arange(0, self.width):
            for y in np.arange(0, self.height):
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        if 0 <= (x+dx) < self.width and 0 <= (y+dy) < self.height:
                            if im[x,y, 0] != im[x+dx, y+dy, 0]:
                                res[x, y] = im[x, y, 0]
        return res
    
    def JFAInitialize(self):
        """
        Initializes the JFA algorithm by replacing edge pixels with values representing their own position.
    
        :param im: The input binary edge image to process.
        :type im: numpy.ndarray
        :return: An image where edge pixels are replaced with values representing their own position.
        :rtype: numpy.ndarray
        """
        im = np.array(self.edges)
        width, height = im.shape[:2]
        res = np.zeros(shape=(self.width, self.height, 2), dtype=np.float32)
        for x in np.arange(0, self.width):
            for y in np.arange(0, self.height):
                if im[x, y] > 0.0:
                    res[x, y] = [x, y]
                else:
                    res[x, y] = [-1, -1]
        return res
    
    def JFAStep(self, step):
        """
        This function performs one step of the Jump Flooding Algorithm (JFA) on an image.
        
        :param step: The current step size for the JFA.
        :type step: int
        :return: The resulting image after performing one JFA step.
        :rtype: numpy.ndarray
        """
        im = np.array(self.JFA)
        width, height = im.shape[:2]
        res = im.copy()
        for x in np.arange(0, self.width):
            for y in np.arange(0, self.height):
                position = [x, y]
                # perform JFA per pixel
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= (x+dx*step) < self.width and 0 <= (y+dy*step) < self.height:
                            if im[x+dx*step, y+dy*step, 0] == -1:
                                continue;
                            seed = res[x, y]
                            thisSeed = res[x+dx*step, y+dy*step]
                            if seed.any() < 0.:
                                res[x, y] = thisSeed
                            if np.linalg.norm(position - seed) > np.linalg.norm(position - thisSeed):
                                res[x, y] = thisSeed
        return res
    
    def JFADecrypt(self):
        """
        This function decrypts an image using the Jump Flooding Algorithm (JFA).
        
        :param im: The image to be decrypted.
        :type im: numpy.ndarray
        :return: The decrypted image.
        :rtype: numpy.ndarray
        """
        im = self.JFA
        res = np.zeros(shape=(im.shape[0], im.shape[1], 3), dtype=np.float32)
        for x in np.arange(0, self.width):
            for y in np.arange(0, self.height):
                if im[x, y, 0] == -1 or np.linalg.norm(im[x, y] - [x, y]) == 0:
                    res[x, y] = np.array([0.0, 0.0, 0.0])
                else:
                    res[x, y] = np.array([1.0, 1.0, 1.0]) / np.linalg.norm(im[x, y] - [x, y])
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('steps', help='Number of steps to do the JFA', type= int)
    args = parser.parse_args()
    JFA(args.image_path, args.steps)
