import cv2


def invert(image_path,i):
    # Load the mask image as a grayscale image
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Invert the mask image
    inverted_mask = cv2.bitwise_not(mask)
    # Save the inverted mask image
    cv2.imwrite(f'inverted_{i}.png', inverted_mask)



images = [f'inverted_{i}.png' for i in range(0,4)]

i=0
for img in images:
    invert(img,i)
    i+=1