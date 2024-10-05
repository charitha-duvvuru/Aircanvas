import numpy as np
import cv2

# Default callback function for trackbars
def empty_callback(x):
    pass

# Create trackbars for adjusting color thresholds
def create_trackbars(window_name):
    cv2.createTrackbar("Upper Hue", window_name, 153, 180, empty_callback)
    cv2.createTrackbar("Upper Saturation", window_name, 255, 255, empty_callback)
    cv2.createTrackbar("Upper Value", window_name, 255, 255, empty_callback)
    cv2.createTrackbar("Lower Hue", window_name, 64, 180, empty_callback)
    cv2.createTrackbar("Lower Saturation", window_name, 72, 255, empty_callback)
    cv2.createTrackbar("Lower Value", window_name, 49, 255, empty_callback)

# Initialize drawing canvas
def initialize_canvas():
    return np.zeros((471, 636, 3), dtype=np.uint8) + 255

# Clear drawing canvas
def clear_canvas(canvas):
    canvas[67:, :, :] = 255
    return canvas

# Main function
def main():
    window_name = "Color detectors"
    cv2.namedWindow(window_name)
    create_trackbars(window_name)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    color_names = ["Blue", "Green", "Red", "Yellow"]

    canvas = initialize_canvas()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for i, color in enumerate(colors):
            cv2.rectangle(frame, (40 + i * 115, 1), (140 + i * 115, 65), color, -1)
            cv2.putText(frame, color_names[i], (45 + i * 115, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        upper_hue = cv2.getTrackbarPos("Upper Hue", window_name)
        upper_saturation = cv2.getTrackbarPos("Upper Saturation", window_name)
        upper_value = cv2.getTrackbarPos("Upper Value", window_name)
        lower_hue = cv2.getTrackbarPos("Lower Hue", window_name)
        lower_saturation = cv2.getTrackbarPos("Lower Saturation", window_name)
        lower_value = cv2.getTrackbarPos("Lower Value", window_name)

        upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
        lower_hsv = np.array([lower_hue, lower_saturation, lower_value])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
