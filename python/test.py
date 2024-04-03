
import cv2
import numpy as np

def print_rust(m):
    print("[")
    for i in range(3):
        print(f"    [{m[i][0]}, {m[i][1]}, {m[i][2]}],")
    print("]\n")

# m = np.array([
#     [0.5, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0],
# ])
    
im = np.zeros((800, 800, 3), dtype=np.uint8)
for x in range(300, 500):
    im[300, x] = [255, 255, 255]
    im[500, x] = [255, 255, 255]
for y in range(300, 500):
    im[y, 300] = [255, 255, 255]
    im[y, 500] = [255, 255, 255]

# rotate 45deg
rad = np.pi/11
m = np.array([
    [np.cos(rad), -np.sin(rad), 0.0],
    [np.sin(rad), np.cos(rad), 0.0],
    [0.0, 0.0, 1.0],
])

m_inv = np.linalg.inv(m)
print_rust(m)
print_rust(m_inv)

# perform the transformation
warped = cv2.warpPerspective(im, m, (800, 800))
cv2.imwrite("rotated.png", warped)

exit(0)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Size:
    def __init__(self, w, h):
        self.w = w
        self.h = h

INPUT_SIZE = Size(800, 600)
SQUARE_HSIZE = 100
CORNER_HSIZE = 5

pts = [
    Point(INPUT_SIZE.w // 2 - SQUARE_HSIZE, INPUT_SIZE.h // 2 - SQUARE_HSIZE),
    Point(INPUT_SIZE.w // 2 + SQUARE_HSIZE, INPUT_SIZE.h // 2 - SQUARE_HSIZE),
    Point(INPUT_SIZE.w // 2 - SQUARE_HSIZE, INPUT_SIZE.h // 2 + SQUARE_HSIZE),
    Point(INPUT_SIZE.w // 2 + SQUARE_HSIZE, INPUT_SIZE.h // 2 + SQUARE_HSIZE),
]

dst_pts = [
    Point(INPUT_SIZE.w//2, INPUT_SIZE.h //2),
    Point(INPUT_SIZE.w//2 + SQUARE_HSIZE, INPUT_SIZE.h//2 - SQUARE_HSIZE),
    Point(INPUT_SIZE.w//2-SQUARE_HSIZE//2, INPUT_SIZE.h//2 + SQUARE_HSIZE),
    Point(INPUT_SIZE.w//2 + SQUARE_HSIZE//2, INPUT_SIZE.h//2 + SQUARE_HSIZE//2)

]
offset = 50
dst_pts = [Point(pt.x+offset, pt.y+offset) for pt in pts]

img = np.zeros((INPUT_SIZE.h, INPUT_SIZE.w, 3), dtype=np.uint8)
for pt in pts:
    for x in range(pt.x - CORNER_HSIZE, pt.x + CORNER_HSIZE + 1):
        for y in range(pt.y - CORNER_HSIZE, pt.y + CORNER_HSIZE + 1):
            img[y, x] = [255, 255, 255]
cv2.imwrite("pts.png", img)


img = np.zeros((INPUT_SIZE.h, INPUT_SIZE.w, 3), dtype=np.uint8)
for pt in pts:
    for x in range(pt.x - CORNER_HSIZE, pt.x + CORNER_HSIZE + 1):
        for y in range(pt.y - CORNER_HSIZE, pt.y + CORNER_HSIZE + 1):
            img[y, x] = [255, 255, 255]
cv2.imwrite("dts_pts.png", img)

img = np.zeros((INPUT_SIZE.h, INPUT_SIZE.w, 3), dtype=np.uint8)
for pt in pts:
    for x in range(pt.x - CORNER_HSIZE, pt.x + CORNER_HSIZE + 1):
        for y in range(pt.y - CORNER_HSIZE, pt.y + CORNER_HSIZE + 1):
            img[y, x] = [255, 255, 255]
for x in range(INPUT_SIZE.w // 2):
    img[INPUT_SIZE.h//2, x] = [255, 255, 255]
pts_np = np.array([(pt.x, pt.y) for pt in pts], dtype=np.float32)
dst_pts_np = np.array([(pt.x, pt.y) for pt in dst_pts], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts_np, dst_pts_np)

# create a matrix that rotates the image 45 degrees around the center
angle = 45
angle = angle*np.pi/180.
cosa = np.cos(angle)
sina = np.sin(angle)
c1 = (INPUT_SIZE.w-1)*0.5
c2 = (INPUT_SIZE.h-1)*0.5
R = np.array([
    [cosa, -sina, c1*(1-cosa)+c2*sina],
    [sina, cosa, -c1*sina+c2*(1-cosa)],
    [0, 0, 1]
])

# combine the two transformations
M = np.dot(M, R)
# M = R



warped_img = cv2.warpPerspective(img, M, (INPUT_SIZE.w, INPUT_SIZE.h))
cv2.imwrite("transformed.png", warped_img)

# print M in a rust friendly format
print("[")
for i in range(3):
    print(f"    [{M[i][0]}, {M[i][1]}, {M[i][2]}],")
print("]")

M_INV = np.linalg.inv(M)
print("[")
for i in range(3):
    print(f"    [{M_INV[i][0]}, {M_INV[i][1]}, {M_INV[i][2]}],")
print("]")