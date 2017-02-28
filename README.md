#**CarND-LaneLines-P1** 
Finding Lane Lines on the Road

<img src="./output_images/solidYellowCurve2_8_origWithFoundLanes.jpg" width="500">

####The goal of this project is to make a pipeline that finds lane lines on the road. Either images or video can be input to test the project. The project is done in [Python with OpenCV](https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python) library and can be opened in [Jupyter Notebook](https://pypi.python.org/pypi/opencv-python).

##1. Pipeline description
My pipeline consists of 10 steps:</br>
1. [Reading image or video frame](#reading-image-or-video-frame)</br>
2. [Filtering white and yellow colors](#filtering-white-and-yellow-colors)</br>
3. [Conversion to gray scale](#conversion-to-gray-scale)</br>
4. [Gaussian blurring](#gaussian-blurring)</br>
5. [Edge detection](#edge-detection)</br>
6. [Region of interest definition](#region-of-interest-definition)</br>
7. [Hough lines detection](#gaussian-blurring)</br>
8. [Filtering Hough lines](#filtering-hough-lines)</br>
9. [Averaging line segments](#averaging-line-segments)</br>
10. [Applying running average on final lines](#applying-running-average-on-final-lines)</br>

###Reading image or video frame
The main method processing the image takes its path as argument. The image is loaded using *matplotlib.image*. 
```python
def draw_lanes_image(imageName):
    image = mpimg.imread(imageName)
```
Below, there are 3 examples of loaded images. Later, after each step, intermediate results will be shown for these images. The third one is the most demanding to process as there are shadows and not so big contrasts between yellow line and the road.</br>

<img src="./output_images/solidWhiteCurve_1_original.jpg" height="160">
<img src="./output_images/solidYellowCurve2_1_original.jpg" height="160">
<img src="./output_images/challengeSnap3_1_original.jpg" height="160">

###Filtering white and yellow colors
This step wouldn't be necessary for first two, easier images. In the third example, after gray scale conversion, color difference between the yellow line and the road is very small. Thus the idea of filtering only 2 key colors for the purpose of this task. Firstly, the image is converted to [HSL color space](https://en.wikipedia.org/wiki/HSL_and_HSV). HSL (Hue, Saturation, Lightness) color space concept is based on human vision color perception. This is  why it's easier to differenciate required colors (yellow and white) even if there are shadows on the image. The code below is inspired by similar [project](https://github.com/naokishibuya/car-finding-lane-lines).
```python
 def mask_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    whiteYellowImage = cv2.bitwise_and(image, image, mask = mask)
    return whiteYellowImage
```
<img src="./output_images/solidWhiteCurve_2_whiteYellowImage.jpg" height="160">
<img src="./output_images/solidYellowCurve2_2_whiteYellowImage.jpg" height="160">
<img src="./output_images/challengeSnap3_2_whiteYellowImage.jpg" height="160">

###Conversion to gray scale

<img src="./output_images/solidWhiteCurve_3_grayImage.jpg" height="160">
<img src="./output_images/solidYellowCurve2_3_grayImage.jpg" height="160">
<img src="./output_images/challengeSnap3_3_grayImage.jpg" height="160">

###Gaussian blurring

<img src="./output_images/solidWhiteCurve_4_blurredImage.jpg" height="160">
<img src="./output_images/solidYellowCurve2_4_blurredImage.jpg" height="160">
<img src="./output_images/challengeSnap3_4_blurredImage.jpg" height="160">

###Edge detection
<img src="./output_images/solidWhiteCurve_5_maskedImage.jpg" height="160">
<img src="./output_images/solidYellowCurve2_5_maskedImage.jpg" height="160">
<img src="./output_images/challengeSnap3_5_maskedImage.jpg" height="160">

###Region of interest definition

###Hough lines detection

<img src="./output_images/solidWhiteCurve_6_origWithHoughLines.jpg" height="160">
<img src="./output_images/solidYellowCurve2_6_origWithHoughLines.jpg" height="160">
<img src="./output_images/challengeSnap3_6_origWithHoughLines.jpg" height="160">

###Filtering Hough lines

<img src="./output_images/solidWhiteCurve_7_origWithHoughLinesFiltered.jpg" height="160">
<img src="./output_images/solidYellowCurve2_7_origWithHoughLinesFiltered.jpg" height="160">
<img src="./output_images/challengeSnap3_7_origWithHoughLinesFiltered.jpg" height="160">

###Averaging line segments

<img src="./output_images/solidWhiteCurve_8_origWithFoundLanes.jpg" height="160">
<img src="./output_images/solidYellowCurve2_8_origWithFoundLanes.jpg" height="160">
<img src="./output_images/challengeSnap3_8_origWithFoundLanes.jpg" height="160">

###Applying running average on final lines



whiteYellowImage.jpg
grayImage.jpg
blurredImage.jpg
maskedImage.jpg
origWithHoughLines.jpg
origWithHoughLinesFiltered.jpg
origWithFoundLanes.jpg


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...




