import cv2
import math
import numpy.typing as npt 

FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 1
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 2
FONT_LINE_RENDER = cv2.LINE_AA

class ImageOperations:
    @classmethod
    def cut_image(cls, frame: npt.NDArray, roi: tuple[int, int, int, int]) -> npt.NDArray:
        """Cut image with roi size

        Args:
            frame (npt.NDArray): frame to cut
            roi (tuple[int, int, int, int]): coords to cut

        Returns:
            npt.NDArray: frame cuted
        """
        x1, y1, x2, y2 = roi
        
        return frame[y1 : y2, x1 : x2]
        
    @classmethod
    def image_rgb(cls, frame: npt.NDArray) -> npt.NDArray:
        """ Transfomr frame to RGB

        Args:
            frame (npt.NDArray): frame to convert

        Returns:
            npt.NDArray: frame converted
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    @classmethod
    def resize(cls, frame: npt.NDArray, dsize: tuple[int,int]) -> npt.NDArray:
        """ Resize frame

        Args:
            frame (npt.NDArray): frame to resize
            dsize (tuple[int,int]): dsize to resize

        Returns:
            npt.NDArray: frame resized
        """
        return cv2.resize(frame, dsize, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    @classmethod
    def euclidean_distance(cls,point1: tuple[int,int], point2: tuple[int,int]) -> float:
        """ Mesure the euclidean distance of two points

        Args:
            point1 (tuple[int,int]): point 1
            point2 (tuple[int,int]): point 2

        Returns:
            float: Distance
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt(math.pow((x2-x1),2) + math.pow((y2-y1),2))
    
    @classmethod
    def put_text(cls, frame: npt.NDArray, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> npt.NDArray:
        """ Write text in the frame

        Args:
            frame (npt.NDArray): frame to write
            text (str): text to put in frame
            position (tuple[int, int]): position
            color (tuple[int, int, int]): color

        Returns:
            npt.NDArray: frame with the text
        """
        cv2.putText(frame, text, position, FONT_TYPE, FONT_SIZE, color, FONT_THICKNESS, FONT_LINE_RENDER)
        return frame