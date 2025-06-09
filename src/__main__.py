import json
import cv2

from src.models.hand_detection import HandDetection
from src.utils.image_operations import ImageOperations

def main():
    with open("src/config/config_home.json","r") as file:
        dados = json.load(file)

    hand = HandDetection(
        static_image_mode=dados["static_image_mode"],
        max_num_hands=dados["max_num_hands"],
        min_detection_confidence=dados["min_detection_confidence"],
        min_tracking_confidence=dados["min_tracking_confidence"],
        roi=dados["roi"],
        hand_threshold=dados["hand_threshold"]
    )
    cap = cv2.VideoCapture("src/videos/home_video_2.mp4")
    ret, frame = cap.read()
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    text_position = (int(width*0.05), int(height*0.1))
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error to open the video")
            break
        
        frame = hand.process_hand(frame,(0,0))
        if hand.hand:
            ImageOperations.put_text(frame=frame, text="Closed", color=(0, 0, 255), position=text_position)
        else:
            ImageOperations.put_text(frame=frame, text="Opened", color=(0, 0, 255), position=text_position)
            
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
    