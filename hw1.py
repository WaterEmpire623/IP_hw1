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

# Converting RGB to HSL
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
        elif l[i,j] > 0.5:
            s[i,j] = (Max_rgb[i,j]-min_rgb[i,j])/(2 - 2 * l[i,j])

# Histogram Equalization
l_8bit = (l*255).astype(np.uint8)
hist, _ = np.histogram(l_8bit.flatten(), bins=256, range=(0, 256))
cdf = hist.cumsum() # calculates cumulative sum
cdf_m = np.ma.masked_equal(cdf, 0) # masks all zero
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) # scaling formula
cdf = np.ma.filled(cdf_m, 0).astype(np.uint8) # unmasks all zero
l_equalized = cdf[l_8bit] # LUP mapping
l_final = l_equalized.astype(float) / 255.0

# Converting HSL to RGB
red_out = np.empty((rows, cols))
green_out = np.empty((rows, cols))
blue_out = np.empty((rows, cols))
for i in range(rows):
    for j in range(cols):
        if h[i,j] < 120:
            blue_out[i,j] = l_final[i,j] * (1 - s[i,j])
            red_out[i,j] = l_final[i,j] * (1 + s[i,j]*np.cos(np.deg2rad(h[i,j]))/np.cos(np.deg2rad(60-h[i,j])))
            green_out[i,j] = 1 - (blue_out[i,j] + red_out[i,j])
        elif (h[i,j] >= 120) & (h[i,j] < 240):
            h[i,j] = h[i,j] - 120
            red_out[i,j] = l_final[i,j] * (1 - s[i,j])
            green_out[i,j] = l_final[i,j] * (1 + s[i,j]*np.cos(np.deg2rad(h[i,j]))/np.cos(np.deg2rad(60-h[i,j])))
            blue_out[i,j] = 1 - (green_out[i,j] + red_out[i,j])
        elif (h[i,j] >= 240) & (h[i,j] <= 360):
            h[i,j] = h[i,j] - 240
            green_out[i,j] = l_final[i,j] * (1 - s[i,j])
            blue_out[i,j] = l_final[i,j] * (1 + s[i,j]*np.cos(np.deg2rad(h[i,j]))/np.cos(np.deg2rad(60-h[i,j])))
            red_out[i,j] = 1 - (green_out[i,j] + blue_out[i,j])

# Outputting Image
red_bgr = (red_out * 255).astype(np.uint8)
green_bgr = (green_out * 255).astype(np.uint8)
blue_bgr = (blue_out * 255).astype(np.uint8)
output_img = cv2.merge([blue_bgr, green_bgr, red_bgr])
cv2.imwrite('output.png', output_img)
