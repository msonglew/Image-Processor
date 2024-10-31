"""
DSC 20 Project
Name(s): Mena Song-Lew
PID(s):  A17515597
Sources: dict .get() method https://www.w3schools.com/python/ref_dictionary_get.asp
        list .sort() method https://www.w3schools.com/python/ref_list_sort.asp
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2

        # PERSONAL DOCTESTS
        >>> pixels = 3
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = []
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [3]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [
        ...             []
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [
        ...             [3]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [
        ...             [[1, 2]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        >>> pixels = [
        ...              [[-1, 255, 255], [0, 0, 0]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        ValueError
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        if not isinstance(pixels, list):
            raise TypeError()
        if len(pixels) < 1:
            raise TypeError()

        if not all([isinstance(row, list) for row in pixels]):
            raise TypeError()
        if all([len(row) < 1 for row in pixels]):
            raise TypeError()

        if not all([len(pixels[0]) == len(pixels[i]) \
            for i in range(len(pixels))]):
            raise TypeError()

        if not all([isinstance(pix, list) for row in pixels for pix in row]):
            raise TypeError()

        if all([len(pix) != NUM_CHANNELS for row in pixels for pix in row]):
            raise TypeError()

        if not all([0<=val<=255 for row in pixels for pix in row \
            for val in pix]):
            raise ValueError()

        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)

        # PERSONAL DOCTESTS
        >>> pixels = [
        ...              [[255, 255, 255]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 1)

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]],
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (2, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[val for val in pix] for pix in row] for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)

        # PERSONAL DOCTEST
        >>> img.get_pixel(0, 2)
        Traceback (most recent call last):
        ...
        ValueError

        >>> img.get_pixel("", 0)
        Traceback (most recent call last):
        ...
        TypeError

        >>> img.get_pixel(0, "")
        Traceback (most recent call last):
        ...
        TypeError
        """

        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if (row < 0) or (self.num_rows <= row) or (col < 0) or \
        (self.num_cols <= col):
            raise ValueError()

        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """

        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if (row < 0) or (self.num_rows <= row) or (col < 0) or \
        (self.num_cols <= col):
            raise ValueError()

        if not isinstance(new_color, tuple):
            raise TypeError()
        if len(new_color) != NUM_CHANNELS:
            raise TypeError
        if not all([isinstance(val, int) for val in new_color]):
            raise TypeError()

        if not all([val <= 255 for val in new_color]):
            raise ValueError()

        for c in range(len(new_color)):
            if new_color[c] < 0:
                continue
            self.pixels[row][col][c] = new_color[c]



# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """

        neg_val = [[[255 - val for val in pix] for pix in row] for row in image.get_pixels()]

        return RGBImage(neg_val)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #
        gray = [[[sum(pix)//NUM_CHANNELS for val in pix] for pix in row] for row in image.get_pixels()]
        return RGBImage(gray)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """

        rot_180 = [row[::-1] for row in image.get_pixels()][::-1]

        return RGBImage(rot_180)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        return sum([sum([sum(pix)//len(pix) for pix in row]) for row in image.get_pixels()]) // (image.num_rows * image.num_rows)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        # YOUR CODE GOES HERE #

        if not isinstance(intensity, int):
            raise TypeError()
        if intensity > 255 or intensity < -255:
            raise ValueError()

        brighter = [[[0 if val + intensity < 0 # max and min \
        else 255 if val + intensity > 255 \
        else val + intensity for val in pix] \
        for pix in row] for row in image.get_pixels()]

        return RGBImage(brighter)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """

        blurred = image.get_pixels()

        for i in range(image.num_rows):
            for j in range(image.num_cols):
                r, g, b = 0, 0, 0
                count = 0

                for near_i in range(-1, 2):
                    for near_j in range(-1, 2):
                        if (near_i + i >= 0) and (near_i + i < image.num_rows) and (near_j + j >=0) and (near_j + j < image.num_cols):
                            r += image.get_pixel(near_i + i, near_j + j)[0]
                            g += image.get_pixel(near_i + i, near_j + j)[1]
                            b += image.get_pixel(near_i + i, near_j + j)[2]
                            count += 1

                avg_r = r // count
                avg_g = g // count
                avg_b = b // count

                blurred[i][j] = [avg_r, avg_g, avg_b]

        return RGBImage(blurred)



# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        super().__init__()
        self.cost = 0
        self.coupon = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #
        if self.coupon == 0:
            self.cost += 5
        else:
            self.coupon -= 1
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        # YOUR CODE GOES HERE #
        if self.coupon == 0:
            self.cost += 6
        else:
            self.coupon -= 1
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        # YOUR CODE GOES HERE #
        if self.coupon == 0:
            self.cost += 10
        else:
            self.coupon -= 1
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        # YOUR CODE GOES HERE #
        if self.coupon == 0:
            self.cost += 1
        else:
            self.coupon -= 1
        return adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        # YOUR CODE GOES HERE #
        if self.coupon == 0:
            self.cost += 5
        else:
            self.coupon -= 1
        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE #
        if not isinstance(amount, int):
            raise TypeError()
        if amount <= 0:
            raise ValueError()

        self.coupon += amount



# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        super().__init__()
        self.cost = 50

    def tile(self, image, new_width, new_height):
        """
        Returns a new image with size new_width x new_height where the
        given image is tiled to fill the new space

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> new_width, new_height = 70, 70
        >>> img_exp = img_read_helper('img/exp/square_32x32_tile.png')
        >>> img_tile = img_proc.tile(img_in, new_width, new_height)
        >>> img_tile.pixels == img_exp.pixels # Check tile output
        True
        >>> img_save_helper('img/out/square_32x32_tile.png', img_tile)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(image, RGBImage):
            raise TypeError()
        if not isinstance(new_width, int) or not isinstance(new_height, int):
            raise TypeError()
        if new_width <= image.num_cols or new_width <= 0 or new_height <= image.num_cols or new_height <= 0:
            raise ValueError()

        tiled = image.get_pixels()

        for i in range(new_height):
            if i >= image.num_rows:
                tiled.append(tiled[i%image.num_rows])
                continue
            for j in range(new_width):
                if j >= image.num_cols:
                    tiled[i].append(tiled[i][j%image.num_cols])       

        return RGBImage(tiled)

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(sticker_image, RGBImage) or not isinstance(background_image, RGBImage):
            raise TypeError()
        if sticker_image.num_cols > background_image.num_cols or sticker_image.num_rows > background_image.num_rows:
            raise ValueError()
        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()
        if x_pos < 0 or y_pos < 0:
            raise ValueError()
        if sticker_image.num_cols + x_pos > background_image.num_cols or sticker_image.num_rows + y_pos > background_image.num_rows:
            raise ValueError()

        with_sticker = background_image.get_pixels()
        sticker = sticker_image.get_pixels()

        for i in range(sticker_image.num_rows):
            for j in range(sticker_image.num_cols):
                with_sticker[y_pos + i][x_pos + j] = sticker[i][j]

        return RGBImage(with_sticker)

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """

        kernel = [
        [-1, -1, -1], 
        [-1,  8, -1], 
        [-1, -1, -1]
        ]

        values = [[(sum(image.get_pixel(i, j)) // NUM_CHANNELS) for j in range(image.num_cols)] for i in range(image.num_rows)]

        edge = [[j for j in row]for row in values]

        for i in range(image.num_rows):
            for j in range(image.num_cols):

                masked_value = 0

                for kernel_i in range(len(kernel)):
                    for kernel_j in range(len(kernel[0])):
                        if (kernel_i + i -1 >= 0) and (kernel_i + i -1 < image.num_rows) and (kernel_j + j -1 >=0) and (kernel_j + j -1 < image.num_cols):
                            masked_value += values[i + kernel_i -1][j + kernel_j -1] * kernel[kernel_i][kernel_j]
        
                edge[i][j] = masked_value

                if masked_value > 255:
                    edge[i][j] = [255, 255, 255]
                elif masked_value < 0:
                    edge[i][j] = [0, 0, 0]
                else:
                    edge[i][j] = [masked_value, masked_value, masked_value]

        return RGBImage(edge)



# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        
        self.k_neighbors = k_neighbors
        self.data = []

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        # YOUR CODE GOES HERE #
        if len(data) < self.k_neighbors:
            raise ValueError()

        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        # YOUR CODE GOES HERE #


        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
        if image1.num_cols != image2.num_cols or image1.num_rows != image2.num_rows:
            raise ValueError()

        return (
            sum([(image1.get_pixel(i, j)[c] - image2.get_pixel(i, j)[c]) ** 2 \
            for i in range(image1.num_rows) \
            for j in range(image1.num_cols) \
            for c in range(NUM_CHANNELS)])
            ) ** (1/2)

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        labels = {}

        max_count, max_label = 0, ''

        for candidate in candidates:
            labels[candidate] = labels.get(candidate, 0) + 1
            if labels[candidate] >= max_count:
                max_count, max_label = labels[candidate], candidate

        return max_label

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        # YOUR CODE GOES HERE #
        if len(self.data) == 0:
            raise ValueError()

        # sorted_data: (dist, label)
        sorted_data = [(self.distance(self.data[img][0], image), self.data[img][1]) for img in range(len(self.data))] 
        
        sorted_data.sort(key=lambda x: x[0])
        return self.vote(sorted_data[:self.k_neighbors])[1]



def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
