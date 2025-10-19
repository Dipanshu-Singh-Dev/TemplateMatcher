def main():
    import cv2
    import numpy as np

    # Load the main image and the template
    main_image = cv2.imread('main_image.jpg')
    template = cv2.imread('template.jpg')

    # Convert images to grayscale
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Set a threshold for detection
    threshold = 0.8
    yloc, xloc = np.where(result >= threshold)

    # Draw rectangles around detected areas
    for (x, y) in zip(xloc, yloc):
        cv2.rectangle(main_image, (x, y), (x + template.shape[1], y + template.shape[0]), (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Detected', main_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()