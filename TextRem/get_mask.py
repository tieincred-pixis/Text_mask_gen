import cv2
import numpy as np


class Image_mask:
    def __init__(self, image, coords):
        try:
            # Load the input image
            print(f'Loading the image {image}')
            self.image = cv2.imread(image[0])
            print(type(self.image))
        except Exception as e:
            print(e)
            self.image = image
        self.vertices = self.convert2poly(coords)

    def convert2poly(self, coords):
        n_vertices = []
        print(len(coords))
        for coord in coords:
            print(len(coord))
            coord = list(coord)
            # Convert the flat list of coordinates to a list of tuples of (x,y) coordinate pairs
            vertices = [(np.round(coord[i].cpu().numpy()), np.round(coord[i+1].cpu().numpy())) for i in range(0, len(coord), 2)]
            # Append the first vertex to the end to close the polygon
            n_vertices.append(vertices)
        return n_vertices

    
    def get_mask(self, out_name):
        polygon_vertices = [np.array(ver, dtype=np.int32) for ver in self.vertices]
        print(polygon_vertices[1])
        # Create a mask with the same size as the image, and fill it with zeros
        # print(type(self.image))
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        # Draw the polygons on the mask with white color (255)
        cv2.fillPoly(mask, polygon_vertices, 255)
        # Apply the mask to the image using bitwise AND
        cropped_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        # Create a black background image with the same size as the input image
        background = np.zeros_like(self.image)
        # Draw the polygons on the background with white color (255)
        cv2.fillPoly(background, polygon_vertices, 255)
        # Apply the mask to the background using bitwise AND
        background = cv2.bitwise_and(background, background, mask=mask)
        # Combine the cropped image and the background image using bitwise OR
        result = cv2.bitwise_or(cropped_image, background)
        # Save the result image
        cv2.imwrite(out_name, result)