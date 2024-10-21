import cv2

def crop_occupancy_grid(pgm_file, x_start, y_start, width, height, output_file):
    img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("File not found.")
        return

    cropped_img = img[y_start:y_start + height, x_start:x_start + width]

    cv2.imwrite(output_file, cropped_img)
    print(f"Cropped map is saved as {output_file}")


pgm_file = '/Users/sunghjopnam/Desktop/map.pgm'
x_start = 0
y_start = 350
width = 200
height = 310
output_file = '/Users/sunghjopnam/Desktop/cropped_map.pgm'

crop_occupancy_grid(pgm_file, x_start, y_start, width, height, output_file)
