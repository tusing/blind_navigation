"""NavigationSensors - designed to parse data from devices such as the Microsoft
Kinect and ultrasonic sensors, in order to translate the data into 
intuitively acceptable input (such as haptic feedback) for blind individuals.

The ideal use case is as such:
1) We have a vest with haptic feedback motors. We have a Kinect and an
    ultrasonic sensor.
2) We parse data from the Kinect into subdivided segments - one
    for every motor. For example, a vest with 4 motors would mean
    to subdivide our image into 4 squares (likely in a 2x2 formation).
3) We use the ultrasonic sensor to account for the Kinect's minimum range
    of 3 feet. The ultrasonic sensor reduces that to 6 cm.
4) We extract features from the Kinect and ultrasonic sensor (such as depth 
    and change in depth over time), and map these features onto 
    different vibration motors.
Written by Tushar Singal.
"""

import numpy as np
import freenect
import wiringpi as wp
import datetime
from skimage.util.shape import view_as_blocks
from copy import deepcopy
from threading import Thread, RLock


class Kinect:
    """Designed to parse output from the Kinect to various subdivided segments.

    A use case involves haptic feedback vests, in which we can subdivide
    the Kinect's output into distinct segments, which we can then map onto
    vibration motors throughout the vest.

    The Kinect's minimum range is about 3 feet. Sensor fusion is recommended
    for nominal operation.
    """

    def __init__(self, rows, cols, startStreaming=True, procFunc=None):
        """Processes input from the Kinect and returns parsed data.

        Subdivides images from the Kinect into various sub-images, to which
        a processing function ```procFunc``` is applied. Streams resultant
        output, which can be acquired with ```getValues``` (for all row by col
        processed subimages) or ```getValue``` (for a single subimage).

        Args:
            rows: The number of rows to divide the image into and process.
            cols: The number of columns to divide the image into and process.
            startStreaming: Whether to start (multithreaded, thread-safe) streaming
                and processing upon object initialization. (default: {True})
            procFunc: The post-processing function to apply to a sub-image.
                Should take arguments of from (subImage, rowIndex, colIndex).
                (default: {self._decayingAverage})
        """
        self.processed = np.zeros((rows, cols), dtype=tuple)
        self.cols = cols
        self.rows = rows
        self.lock = RLock()

        if procFunc is None:
            self.procFunc = self._decayingAverage
        else:
            self.procFunc = procFunc

        if startStreaming:
            Thread(target=self.stream).start()

    def stream(self):
        """Start streaming data from the Kinect.

        Begins recording from the Kinect and placing processed values in
        self.processed, from which data can be retreived in a thread-safe
        value using ```self.getValues``` or ```self.getValue```.
        """
        while True:
            self._processDepth(self._getRawDepth)

    def getValues(self):
        """Retreives the entire matrix of processed values for the last frame.

        Returns:
            A two-dimensional array of tuples for which every tuple is the value
            returned by ```self.procFunc``` when applied to a specific sub-image.
            [np.array, dtype=tuple]
        """
        self.lock.acquire()
        copied = np.copy(self.processed)
        self.lock.release()
        return copied

    def getValue(self, row, col):
        """Retreives a single processed value from the last frame.

        Args:
            row: The row-index of the sub-image for which processed values are
                to be accessed.
            col: The column-index of the sub-image for which processed values are
                to be accessed.

        Returns:
            The tuple returned by the computation of ```self.procFunc``` on the subimage
            located indexed ```row, col```.
            [tuple]
        """
        self.lock.acquire()
        copied = deepcopy(self.processed[row, col])
        self.lock.release()
        return copied

    def _processDepth(self, depthImage):
        """Processes depth with the provided function.

        This function will take the raw depth image output from
        ```_getRawDepth``` and subdivide the image into ```self.rows``` by
        ```self.cols``` sub-images.
        It then applies ```procFunc``` to each subdivision, and places
        the returned tuple of values in ```self.processed```.

        Args:
            depthImage: The depth-image to process.
            procFunc: The post-processing function to apply to a subdivision.
                Should return a tuple.
        """
        subdivided = view_as_blocks(
            depthImage, block_shape=(self.rows, self.cols))
        for row in range(self.rows):
            for col in range(self.cols):
                self.lock.acquire()
                self.processed[row, col] = self.procFunc(
                    subdivided[row, col], row, col)
                self.lock.release()

    def _getRawDepth(self):
        """Obtains a snapshot of depth data from the Microsoft Kinect.

        Returns:
            Array of depth data values, ranging from [0-255], where
            255 is far from Kinect, and 0 is close. Minimum range of 3 feet.
            [Numpy array]
        """
        array, _ = freenect.sync_get_depth()
        array = array.astype(np.uint8)
        return array

    def _decayingAverage(self, subImage, row, col, numSamples=4):
        """ Returns depth, and difference in depth over time.

        Performs a decay-based low-pass filter over depth measurements.
        Calculates the difference in depth from the old average. This results
        in two-dimensional data: "depth" and "depth derivative", which can be
        mapped to, say, "motor intensity" and "motor frequency" in our
        haptic-feedback vest example.

        Args:
            subImage: The subdivision we are calculating over.
            row: The width index of the current subdivision.
            col: The height index of the current subdivision.
            numSamples: Strength of the low-pass filter in number of frames.
                (default: {4})
        """
        windowAverage = int(np.average(subImage))
        oldDistance = self.processed[row, col]
        return (decayingAverage(windowAverage, oldDistance, numSamples),
                oldDistance - windowAverage)


class Ultrasonic:
    def __init__(self, trigPin=4, echoPin=4, numSamples=25, startStreaming=True):
        """Initializes recordings from an ultrasonic sensor.

        Allows user to initialize an ultrasonic sensor which has trig and
        echo pins. Averages measurements over ```numSamples```.

        Args:
            trigPin: The WiringPi pin trig is connected to. (default: {4})
            echoPin: The WiringPi pin echo is connected to. (default: {4})
            numSamples: The number of samples to perform a decaying
                average over. (default: {4})
            startStreaming: Whether to initialize streaming from the ultrasonic
                sensor upon object initialization. (default: {True})
        """
        self.trigPin = trigPin
        self.echoPin = echoPin
        self.numSamples = numSamples
        self.lock = RLock()
        self.averagedDepth = 90

        def setup():
            wp.wiringPiSetup()
            wp.pinMode(self.trigPin, wp.OUTPUT)
            wp.pinMode(self.echoPin, wp.INPUT)
            wp.digitalWrite(self.trigPin, wp.LOW)
            wp.delay(30)
        setup()

        if startStreaming():
            Thread(target=self.stream).start()

    def stream(self):
        """Start streaming data from the ultrasonic sensor.

        Begins recording from the ultrasonic sensor and placing
        retreived values in ```self.averagedDepth```, which can
        be retrieved in a thread-safe manner with ```self.getDepth```.
        """
        while True:
            self._setDepth()

    def getDepth(self):
        """Returns the last retreived depth in centimeters.

        Returns:
            The last-measured depth value in cm.
            [int]
        """
        self.lock.acquire()
        copy = self.averagedDepth
        self.lock.release()
        return copy

    def _setDepth(self):
        """Records depth from the ultrasonic sensor in centimeters.

        Much credit to
        https://ninedof.wordpress.com/2013/07/16/rpi-hc-sr04-ultrasonic-sensor-mini-project/
        """
        wp.digitalWrite(self.trigPin, wp.HIGH)
        wp.delayMicroseconds(20)
        wp.digitalWrite(self.trigPin, wp.LOW)
        while(wp.digitalRead(self.echoPin) == wp.LOW):
            pass
        startime = datetime.datetime.now()
        while wp.digitalRead(self.echoPin) == wp.HIGH:
            pass
        traveltime = (datetime.datetime.now() - startime).total_seconds()
        distance = 1000000 * traveltime / 58.0
        self.lock.acquire()
        self.averagedDepth = decayingAverage(
            distance, self.averagedDepth, self.numSamples)
        self.lock.release()


def decayingAverage(newVal, oldVal, numSamples):
    """Averages oldVal with newVal over numSamples.

    Introduces newVal into oldVal, decaying oldVal by
    a single sample and introducing newVal by a single sample.

    Args:
        newVal: The new value to average.
        oldVal: The value to decay.
        numSamples: The number of samples to decay over.

    Returns:
        Decayed average.
    """
    newAvg = newVal * (1 / numSamples) + oldVal * \
        ((numSamples - 1) / numSamples)
    return newAvg
