import numpy as np
import cv2

def create_mask(image_size, boxes):
    # Initialize an empty mask
    mask = np.zeros(image_size, dtype=np.uint8)

    # Loop over each box and draw a rectangle and write text on the mask
    for box in boxes:
        # Extract box coordinates and label
        points = [(box[i], box[i+1]) for i in range(0, len(box)-2, 2)]
        lang, label = box[-2:]

        # Determine color for label based on language
        if lang == 'Arabic':
            color = (255, 0, 0) # Blue for Arabic labels
        else:
            color = (255, 0, 0) # Green for Latin labels

        # Draw rectangle on mask
        pts = np.array(points, dtype=np.int32)
        # cv2.fillPoly(mask, [pts], color=color)

        # Write label text on mask
        x, y = points[0]
        cv2.putText(mask, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return mask







img_size = (266, 355)
boxes_info = [
    [197, 55, 303, 40, 313, 87, 201, 95, 'Arabic', 'فروجنا'],
    [55, 62, 185, 57, 186, 102, 54, 109, 'Arabic', 'المشوى'],
    [226, 112, 305, 111, 306, 157, 227, 158, 'Arabic', 'حلال'],
    [193, 129, 212, 128, 213, 162, 194, 162, 'Arabic', 'و'],
    [102, 123, 180, 123, 180, 173, 102, 173, 'Arabic', 'طازج'],
    [28, 194, 79, 193, 79, 218, 28, 219, 'Latin', 'Halal'],
    [86, 191, 142, 191, 139, 214, 84, 214, 'Latin', 'Flame'],
    [149, 186, 226, 182, 221, 208, 148, 213, 'Latin', 'Grilled'],
    [230, 182, 310, 180, 310, 205, 230, 205, 'Latin', 'Chicken'],
    [172, 243, 208, 250, 208, 265, 171, 265, 'Latin', '###'],
    [107, 249, 149, 248, 148, 264, 124, 265, 'Latin', '###']
]

mask = create_mask(img_size, boxes_info)
cv2.imwrite('mask.png',mask)