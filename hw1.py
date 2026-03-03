import cv2
import numpy as np

# Loading Inputs
img = cv2.imread('input.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = img_rgb/255

# Obtain Individual Color
red = img_rgb[:,:,0]
green = img_rgb[:,:,1]
blue = img_rgb[:,:,2]

# Converting RGB to HSV
Max_rgb = np.max(img_rgb, axis=2)
min_rgb = np.min(img_rgb, axis=2)
rows, cols = Max_rgb.shape
h = np.empty((rows, cols))
s = np.empty((rows, cols))
l = np.empty((rows, cols))

for i in range(rows):
    for j in range(cols):
        if Max_rgb[i,j] == min_rgb[i,j]:
            h[i,j] = 0
        elif (Max_rgb[i,j] == red[i,j]) & (green[i,j] >= blue[i,j]):
            h[i,j] = 60 * (green[i,j] - blue[i,j])/(Max_rgb[i,j] - min_rgb[i,j])
        elif (Max_rgb[i,j] == red[i,j]) & (green[i,j] < blue[i,j]):
            h[i,j] = 60 * (green[i,j] - blue[i,j])/(Max_rgb[i,j] - min_rgb[i,j]) + 360
        elif Max_rgb[i,j] == green[i,j]:
            h[i,j] = 60 * (blue[i,j] - red[i,j])/(Max_rgb[i,j] - min_rgb[i,j]) + 120
        elif Max_rgb[i,j] == blue[i,j]:
            h[i,j] = 60 * (red[i,j] - green[i,j])/(Max_rgb[i,j]-min_rgb[i,j]) + 240
        l[i,j] = (Max_rgb[i,j] + min_rgb[i,j]) / 2
        if (Max_rgb[i,j] == min_rgb[i,j]) | (l[i,j] == 0):
            s[i,j] = 0
        elif (l[i,j] > 0) & (l[i,j] <= 0.5):
            s[i,j] = (Max_rgb[i,j]-min_rgb[i,j])/(2 * l[i,j])
        elif l[i,j] < 0.5:
            s[i,j] = (Max_rgb[i,j]-min_rgb[i,j])/(2 - 2 * l[i,j])

print(s.dtype)
