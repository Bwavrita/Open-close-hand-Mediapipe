import mediapipe as mp
import numpy.typing as npt
from src.utils.image_operations import ImageOperations

class HandDetection():
    mp_hands = mp.solutions.hands
    def __init__(self,
                static_image_mode: bool, 
                max_num_hands: int, 
                min_detection_confidence: float, 
                min_tracking_confidence: float,
                roi: tuple[int,int,int],
                hand_threshold: float
        ):
        """Hand detection class to define is closed ou opened

        Args:
            static_image_mode (bool): if ist photo or video
            max_num_hands (int): max hand in the frame
            min_detection_confidence (float): min detection confidence to detect hand
            min_tracking_confidence (float): min tracking confindence to track
            roi (tuple[int,int,int]): roi where the hand can stay
            hand_threshold (float): threshold to determine is closed ou opened ( depends of frame size )
        """
        self.hands = HandDetection.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,               
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.roi = roi
        self.hand_threshold = hand_threshold
        self.min_fingers = 3
        self.cont_fingers = 0
        self.hand = False
        self.list_distance = {
            0:[0,0],
            4:[0,0],
            8:[0,0],
            12:[0,0],
            16:[0,0],
            20:[0,0]
        }

    def process_frame(self, frame: npt.NDArray, dsize: tuple[int, int]) -> npt.NDArray:
        """Process fram before verify hand

        Args:
            frame (npt.NDArray): frame
            dsize (tuple[int, int]): dsize to cut

        Returns:
            _type_: frame processed
        """
        frame = ImageOperations.cut_image(frame, self.roi)
        frame_rgb = ImageOperations.image_rgb(frame)
        resized = ImageOperations.resize(frame_rgb, dsize)
        return resized
    
    def process_hand(self,frame: npt.NDArray, dsize: tuple[int, int]):
        """verify if the hand is opened or not

        Args:
            frame (npt.NDArray): frame
            dsize (tuple[int, int]): dsize to cut frame
        """
        frame_rgb = ImageOperations.image_rgb(frame)
        result = self.hands.process(frame_rgb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                bbox = self.get_square_roi_from_landmarks(hand_landmark, frame.shape)
                x1, y1, x2, y2 = bbox
                ImageOperations.draw_rectangle(frame,(x1,y1,x2,y2),(255,0,0))
                for id, coord in enumerate(hand_landmark.landmark):
                    if id in [0,4,8,12,16,20]:
                        h, w, _ = frame.shape
                        cx , cy = int(coord.x * w), int(coord.y * h)
                        self.list_distance[id][0] = cx
                        self.list_distance[id][1] = cy
                        
        list = self.list_distance
        for i in [4,8,12,16,20]:
            euclidean_distance = ImageOperations.euclidean_distance(list[0],list[i])
            
            euc = euclidean_distance/100
            
            if euc <= self.hand_threshold:
                self.cont_fingers +=1
                if self.cont_fingers >= self.min_fingers:
                    self.hand = True
            else:
                self.hand = False
                self.cont_fingers = 0
                    