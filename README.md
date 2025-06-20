# Open/Closed Hand Gesture Counter with MediaPipe

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This is a real-time Computer Vision project built in Python that leverages the MediaPipe library to detect, classify, and count open and closed hand gestures from a webcam feed.

---

### ✨ Features

-   **Real-time Hand Detection:** Identifies the presence of one or more hands in the video stream.
-   **Landmark Extraction:** Uses the 21 landmarks from the `MediaPipe Hands` solution to map the hand's structure.
-   **Gesture Classification:** Differentiates between "Open Hand" and "Closed Hand" gestures based on the distance of fingertips to the wrist.
-   **Gesture Counting:** Increments a counter each time the hand transitions from an "Open" to a "Closed" state.
-   **Interactive Visualization:** Displays the webcam feed with the hand landmarks, current gesture state, and total count rendered on the screen.

---

###  демонстрация (Demonstration)

A short GIF demonstrating the program in action. This is highly effective for Computer Vision project READMEs!

![Project Demonstration](src/videos/demonstracao.gif)

*An example of the program identifying hand landmarks and classifying the gesture.*

---

### 🛠️ Technologies Used

-   **Python 3.9+**
-   **OpenCV-Python:** For webcam video capture and image manipulation.
-   **MediaPipe:** For hand detection and landmark extraction.

---

### 🚀 Getting Started

Follow these instructions to get a local copy of the project up and running.

#### Prerequisites

-   Python 3.9 or later installed.
-   A webcam connected to your computer.

#### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Bwavrita/Open-close-hand-Mediapipe.git](https://github.com/Bwavrita/Open-close-hand-Mediapipe.git)
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd Open-close-hand-Mediapipe
    ```

3.  **Create and activate a virtual environment (highly recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

### 🔧 Usage

After installation, run the main script to start the application:

```sh
python main.py
```

-   A window with your webcam feed will open.
-   Position your hand in front of the camera. The program will start tracking your hand and displaying the gesture state.
-   Close and open your hand to see the counter in action.
-   Press the **'q'** key with the video window selected to close the program.

---

### 💡 How It Works

The core logic for classifying the gesture is based on calculating distances between specific hand landmarks:

1.  **Video Capture:** `OpenCV` is used to capture the webcam feed frame by frame.
2.  **Hand Detection:** Each frame is processed by the `mediapipe.solutions.hands` solution, which returns the 21 landmarks for each detected hand.
3.  **Distance Calculation:** To determine if the hand is open, the script calculates the Euclidean distance between the tip of each of the four fingers (index, middle, ring, and pinky) and a fixed point at the base of the hand (the wrist, landmark 0).
4.  **Gesture Classification:**
    -   If the distance from the fingertips to the wrist is **greater** than a certain threshold, the gesture is classified as **"Opened Hand"**.
    -   Otherwise, it is classified as a **"Closed Hand"**.
5.  **Counting Logic:** To prevent multiple counts while a hand remains closed, the system uses a state variable. The counter is only incremented at the exact moment the state changes from "Opened" to "Closed".

---

### 📄 License

Distributed under the MIT License. See the `LICENSE` file for more information.