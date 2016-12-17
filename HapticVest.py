"""CameraParser - designed to parse data from a camera such as the Microsoft
Kinect, in order to translate the data into intuitively acceptable input,
such as haptic feedback, for blind individuals.

The ideal use case is as such:
1) We have a vest with haptic feedback motors. We have a Kinect, or some
    camera capable of giving us meaningful data for blind individuals, e.g.
    depth.
2) We parse data from the Kinect into subdivided segments - one
    for every motor. For example, a vest with 4 motors would mean
    to subdivide our image into 4 squares (likely in a 2x2 formation).
3) We extract features from every subdivision (such as depth and change
    in depth over time), and map these features onto different motors.

Written by Tushar Singal.
"""

import freenect
import numpy as np
from threading import Thread, RLock
import wiringpi as wp
import datetime


class CameraParser:
    """Designed to parse output from the camera to various subdivided segments.

    A use case involves haptic feedback vests, in which we can subdivide
    the camera's output into distinct segments, which we can then map onto
    vibration motors throughout the vest.
    """

    def __init__(self, x, y, subdivFunc=False, cameraFunc=False):
        """Initializes the parser with a defined function and output type.

        Subdivides output from the camera into various distinct segments with
        resolution (x, y). Applies a function to every subdivision, and maps
        results into an array of specified a type.

        Args:
            x: Desired number of subdivisions along the x-axis.
            y: Desired number of subdivisions along the y-axis.
            subdivFunc: The function to apply to every subdivided array.
                Should return a tuple. (default: {self.decayingAverage})
            cameraFunc: The function that obtains our snapshot from a camera.
                Should return a 2D array. (default: {self._getKinectDepth})
        """
        self.pinBase = 100
        self.numPins = 32
        self.pwmMinRange = 0
        self.pwmMaxRange = 500
        self.minFreqRange = 0
        self.maxFreqRange = 15

        self.state = np.zeros(self.numPins)
        self.per = np.zeros(self.numPins)
        self.startTime = np.zeros(self.numPins, dtype=datetime.datetime)
        self.updated = np.zeros(self.numPins)

        wp.wiringPiSetup()
        wp.sr595Setup(self.pinBase, self.numPins, 28, 26, 27)

        for pin in range(self.pinBase, self.pinBase + self.numPins):
            wp.digitalWrite(pin, 0)

        for i in range(self.pinBase, self.pinBase + self.numPins):
            wp.softPwmCreate(i, self.pwmMinRange, self.pwmMaxRange)

        self.x, self.y = x, y
        if subdivFunc is False:
            self.subdivFunc = self.decayingAverage
        else:
            self.subdivFunc = subdivFunc

        if cameraFunc is False:
            self.cameraFunc = self._getKinectDepth
        else:
            self.cameraFunc = cameraFunc

        self.subdivisions = np.zeros((y, x), dtype=tuple)
        # Because numpy only allows you to initialize with a scalar...
        for x, y in np.ndindex(self.subdivisions.shape):
            self.subdivisions[x, y] = (255, 0)
        self.run()

    def getValue(self, x, y):
        """Gets computed value after subdividing and applying ```self.subdivFunc```.

        Args:
            x: Width index of desired value.
            y: Height index of desired value.

        Returns:
            Values returned by ```self.subdivFunc``` when applied
            to subdivision.
            [tuple]
        """
        self.lock.acquire()
        retval = self.subdivisions[y, x]
        self.lock.release()
        return retval

    def getValues(self):
        """Gets latest snapshot of entire matrix of computed values.

        Allows you to obtain the full matrix of values for the entire image
        once subdivided and processed with ```self.subdivFunc```.

        Returns:
            A matrix of tuples, where the tuples represent the result of
            ```self.subdivFunc``` applied to every subdivision of the image
            output from the camera.
            [numpy array]
        """
        self.lock.acquire()
        retval = self.subdivisions
        self.lock.release()
        return retval

    def _getKinectDepth(self):
        """Obtains a snapshot of depth data from the Microsoft Kinect.

        Returns:
            Array of depth data values, ranging from [0-255], where
            255 is close to the Kinect, and 0 is far.
            [Numpy array]
        """
        array, _ = freenect.sync_get_depth()
        array = array.astype(np.uint8)
        return array

    def _setValues(self, image):
        """Subdivide and update values with input from camera.

        Subdivides the image into ```self.x``` by ```self.y``` segments,
        and applies ```self.subdivFunc``` to each subdivision, storing the
        resulting tuple in a matrix which can be obtained by getter
        functions.

        Args:
            image: Two-dimensional array of image to parse.
        """
        # print(image)
        heightStrides = image.shape[0] // self.y
        widthStrides = image.shape[1] // self.x
        for yi in range(self.y):
            yWindow = image[heightStrides * yi: heightStrides * (yi + 1), :]
            for xi in range(self.x):
                xyWindow = yWindow[:, widthStrides *
                                   xi: widthStrides * (xi + 1)]
                self.lock.acquire()
                self.subdivisions[yi, xi] = self.subdivFunc(xyWindow, xi, yi)
                self.lock.release()

    def decayingAverage(self, xyWindow, x, y, numSamples=4):
        """ Subdivision function. Returns depth, and difference in depth over time.

        Performs a decay-based low-pass filter over depth measurements.
        Calculates the difference in depth from the old average. This results
        in two-dimensional data: "depth" and "depth derivative", which can be
        mapped to, say, "motor intensity" and "motor frequency" in our
        haptic-feedback vest example.

        Args:
            xyWindow: The subdivision we are calculating over.
            x: The width index of the current subdivision.
            y: The height index of the current subdivision.
            numSamples: Strength of the low-pass filter in number of frames.
                (default: {4})
        """
        if x is 4 and y is 2:
            print(np.array_str(xyWindow))
        windowAverage = int(np.average(xyWindow))
        oldDistance = self.getValue(x, y)[0]
        newDistChange = windowAverage - oldDistance
        newDistValue = windowAverage * \
            (1 / numSamples) + oldDistance * ((numSamples - 1) / numSamples)
        return (int(newDistValue), int(newDistChange))

    def actuate(self, values):
        for i in range(values.shape[1]):
            for j in range(values.shape[0]):
                pinNum = j + i * values.shape[0]
                self.actuatePin(pinNum, values[j, i][0], values[j, i][1])

    def actuatePin(self, i, PVal, DVal):
        if self.state[i] == 0:
            if self.updated[i] == 0:
                self.startTime[i] = datetime.datetime.now()
                self.per[i] = 1.0 / self.DFunc(DVal)
                self.updated[i] = 1
            output = 0
            if (datetime.datetime.now() - self.startTime[i]).total_seconds() >= self.per[i]:
                self.state[i] = 1
        elif self.state[i] == 1:
            if self.updated[i] == 1:
                self.startTime[i] = datetime.datetime.now()
                self.updated[i] = 0
            output = self.PFunc(PVal)
            if (datetime.datetime.now() - self.startTime[i]).total_seconds() >= self.per[i]:
                self.state[i] = 0

        wp.softPwmWrite(self.pinBase + i, output)

    def PFunc(self, x):
        k = (self.pwmMaxRange - self.pwmMinRange) / 255

        v = 125
        y = max(self.pwmMinRange, min(v, self.pwmMaxRange))
        return int(y)

    def DFunc(self, x):
        return 2

    def run(self):
        """Threads the parser.

        Starts continually accepting and parsing frames from the camera
        in a separate thread.
        Thread-safe implementation; calling getter functions will ensure
        only the latest snapshot of data is captured.
        """
        self.lock = RLock()
        def startParsing():
            while True:
                self._setValues(self.cameraFunc())
        Thread(target=startParsing).start()


import time


def main():
    parser = CameraParser(8, 4)
    while True:
        v = parser.getValues()
        parser.actuate(v)
        print(parser.getValue(4, 2))
        print("\n\n\n\n")
        time.sleep(1)


if __name__ == '__main__':
    main()