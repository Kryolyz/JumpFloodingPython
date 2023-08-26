# Jump Flooding Algorithm in Python
An example implementation of the jump-flooding-algorithm for finding 2D distance fields from images.

### Usage
First, install the dependencies with 
```
pip install -r requirements.txt
```
Then you can use main.py with arguments:
```
python main.py <path to image file> <Number of steps to perform>
````
The output will be several windows showing the image, the detected edges, and the resulting distance field.
Note that the algorithm turns your image file into a bitmap and finds those edges, meaning "normal" images don't work

### Caveats
This example is meant to be easily understood above anything else. I made it for myself because I wanted to implement JFA in unity with a compute shader and had to make sure that I understood the algorithm fully beforehand.
Consequently, the performance is terrible and the code does not use things like numpy vectorization for speed ups at all because I wanted everything to be simple and clean, such that the logic is easy to follow. 

### Reproducing examples
The screenshots folder contains result for the two example bitmaps in this repo. Both were made with 4 steps in total.
Open your command line and move to the repo folder. Then use
````
python main.py example1.bmp 4
````
or
````
python main.py example2.bmp 4
````
to reproduce the examples.
