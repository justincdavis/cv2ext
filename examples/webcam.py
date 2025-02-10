import cv2ext


with cv2ext.Display("webcam") as display:
    for _, frame in cv2ext.IterableVideo():
        if display.stopped:
            break

        display(frame)
