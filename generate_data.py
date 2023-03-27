import random
import os
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import string

def create_image_with_text(image, text, font, text_x, text_y):
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    # Write the text in white color
    draw.text((text_x, text_y), text, font=font, fill='white', anchor='mm')
    return image


def generate_sentence():
    words = []
    length = random.choices(range(2, 21), weights=[5]*5 + [1]*14)[0]
    for _ in range(length):
        word_length = random.choices(range(1, 11), weights=[3]*3 + [1]*7)[0]
        word = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=word_length))
        words.append(word)
    sentence = ' '.join(words)
    return sentence

def adjust_text_size(text, font_path, box_width, box_height):
    font_size = 50
    font = ImageFont.truetype(font_path, font_size)
    while font.getsize(text)[0] > box_width or font.getsize(text)[1] > box_height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
    return font

def write_text_in_boxes(bounding_boxes, font_paths, save_name):
    # print((font_paths[1][1]))
    # generate_image
    img = generate_image(600)
    mask = Image.new('RGB', (600, 600), color='black')
    # Open image
    with img as img:
        # Create drawing context
        draw = ImageDraw.Draw(img)

        # Loop over bounding boxes and texts
        c=0
        bounding_boxes = random.sample(bounding_boxes,2)
        for box in bounding_boxes:
            # Select a random font and color for the text
            # print(c)
            font_path = 'fonts/'+font_paths[c][0]
            color = font_paths[c][1]
            c+=1
            # Load font
            font_size = random.randint(30,200)
            # print(font_size)
            text = generate_sentence()
            font = adjust_text_size(text, font_path, box[2]-box[0], box[3]-box[1])
            # font = ImageFont.truetype(font_path, size=font_size)

            # Calculate size of text
            text_width, text_height = draw.textsize(text, font=font)

            # Calculate text position to center of box
            text_x = (box[0] + box[2]) // 2
            text_y = (box[1] + box[3]) // 2

            # Draw text in box
            draw.text((text_x, text_y), text, fill=color, font=font, anchor='mm')
            mask = create_image_with_text(mask, text, font, text_x, text_y)
        # Save modified image
        mask.save(f"eval_mask/image_{save_name}.png")
        img.save(f"eval/image_{save_name}.png")


def select_fonts_and_colors(font_dir: str, num_fonts: int = 2) -> List[Tuple[str, Tuple[str, str]]]:
    # Get a list of all the font files in the directory
    font_files = [f for f in os.listdir(font_dir) if f.endswith('.ttf') or f.endswith('.otf')]
    
    # Select a random subset of the font files
    selected_fonts = random.sample(font_files, num_fonts)
    
    # Select two random colors for each font
    font_colors = [(random_color()) for _ in range(num_fonts)]
    
    # Combine the font names and colors into a list of tuples
    fonts_and_colors = [(font, colors) for font, colors in zip(selected_fonts, font_colors)]
    
    # Return the list of font names and colors
    return fonts_and_colors

def random_color() -> Tuple[str, str, str]:
    # Generate a random RGB color tuple
    r, g, b = [random.randint(0, 255) for _ in range(3)]
    color = (r, g, b)
    
    # Convert the RGB values to hexadecimal strings
    # hex_color = '#' + ''.join([format(c, '02x') for c in color])
    
    return color

def generate_image(size):
    # Create a new blank image of the given size
    img = Image.new('RGB', (size, size))
    ratios = generate_ratios()
    colors = generate_colors()
    # Compute the height of each region based on the given ratios
    height1 = int(ratios[0] * size)
    height2 = int(ratios[1] * size)
    height3 = int(ratios[2] * size)
    
    # Draw a rectangle in each region with the corresponding color
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, size, height1), fill=colors[0])
    draw.rectangle((0, height1, size, height1 + height2), fill=colors[1])
    draw.rectangle((0, height1 + height2, size, size), fill=colors[2])
    
    # Return the generated image
    return img


def generate_colors():
    # Generate random RGB values between 0 and 255
    r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    r2, g2, b2 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    r3, g3, b3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    
    # Return the three RGB tuples as a list
    return [(r1, g1, b1), (r2, g2, b2), (r3, g3, b3)]


def generate_ratios():
    # Generate random ratios for dividing the image into three parts
    ratio1 = random.uniform(0.1, 0.4)
    ratio2 = random.uniform(0.4, 0.6)
    ratio3 = random.uniform(0.6, 0.9)
    
    # Normalize the ratios so that they sum to 1
    total_ratio = ratio1 + ratio2 + ratio3
    ratio1 /= total_ratio
    ratio2 /= total_ratio
    ratio3 /= total_ratio
    
    return ratio1, ratio2, ratio3


def get_bounding_boxes(img_size, x):
    """
    Returns three bounding boxes that lie inside the given image size
    and are divided into three equal horizontal parts, with the first
    one at a distance of x pixels from the top of the image with 10px padding
    from the bottom of the first divide, the second one at a distance of 20px
    padding from all sides of the center and the second divide, and the third
    one at a distance of x pixels from the bottom of the image with 10px padding
    from the top of the third divide.

    Args:
    - img_size: an integer representing the size of the square image
    - x: an integer representing the distance in pixels of the top and
      bottom bounding boxes

    Returns:
    - a list of three tuples representing the bounding boxes in the format
      (x_min, y_min, x_max, y_max)
    """
    # Calculate the height of each third of the image
    third = img_size // 3

    # Calculate the coordinates of the top bounding box
    box_1 = (x, x, img_size - x, x + third - 10)

    # Calculate the coordinates of the middle bounding box
    box_2 = (x + 20, third + x + 20, img_size - x - 20, 2 * third + x - 20)

    # Calculate the coordinates of the bottom bounding box
    box_3 = (x, 2 * third + x + 10, img_size - x, img_size - x)

    # Return the list of bounding boxes
    return [box_1, box_2, box_3]


font_dir = 'fonts'
image_size = 512
padding = 30

for i in tqdm(range(4)):
    # Select the random fonts and colors
    bboxes = get_bounding_boxes(image_size, padding)
    fonts_pth = select_fonts_and_colors(font_dir)
    write_text_in_boxes(bboxes, fonts_pth, i)
