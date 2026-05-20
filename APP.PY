import cv2

cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    mask = object_detector.apply(frame)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 800:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Object Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()