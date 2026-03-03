import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading Inputs
img = cv2.imread('example.png')
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
v = np.empty((rows, cols))
for i in range(rows):
    for j in range(cols):
        if Max_rgb[i,j] == min_rgb[i,j]:
            h[i,j] = 0
        elif (Max_rgb[i,j] == red[i,j]) & (green[i,j] >= blue[i,j]):
            h[i,j] = 60 * (green[i,j] - blue[i,j]) / (Max_rgb[i,j] - min_rgb[i,j])
        elif (Max_rgb[i,j] == red[i,j]) & (green[i,j] < blue[i,j]):
            h[i,j] = 60 * (green[i,j] - blue[i,j]) / (Max_rgb[i,j] - min_rgb[i,j]) + 360
        elif Max_rgb[i,j] == green[i,j]:
            h[i,j] = 60 * (blue[i,j] - red[i,j]) / (Max_rgb[i,j] - min_rgb[i,j]) + 120
        elif Max_rgb[i,j] == blue[i,j]:
            h[i,j] = 60 * (red[i,j] - green[i,j]) / (Max_rgb[i,j]-min_rgb[i,j]) + 240
        v[i,j] = Max_rgb[i,j]
        if (Max_rgb[i,j] == 0):
            s[i,j] = 0
        else:
            s[i,j] = (Max_rgb[i,j] - min_rgb[i,j]) / Max_rgb[i,j]

# Histogram Equalization
v_8bit = (v*255).astype(np.uint8)
hist, _ = np.histogram(v_8bit.flatten(), bins=256, range=(0, 256))
cdf = hist.cumsum() # calculates cumulative sum
cdf_m = np.ma.masked_equal(cdf, 0) # masks all zero
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) # scaling formula
cdf = np.ma.filled(cdf_m, 0).astype(np.uint8) # unmasks all zero
v_equalized = cdf[v_8bit] # LUP mapping
v_final = v_equalized.astype(float) / 255.0

# Converting HSV to RGB
h_i = np.empty((rows, cols))
f = np.empty((rows, cols))
p = np.empty((rows, cols))
q = np.empty((rows, cols))
t = np.empty((rows, cols))
red_out = np.empty((rows, cols))
green_out = np.empty((rows, cols))
blue_out = np.empty((rows, cols))
for i in range(rows):
    for j in range(cols):
        h_i[i,j] = np.floor(h[i,j] / 60)
        f[i,j] = h[i,j] / 60 - h_i[i,j]
        p[i,j] = v_final[i,j] * (1 - s[i,j])
        q[i,j] = v_final[i,j] * (1 - f[i,j] * s[i,j])
        t[i,j] = v_final[i,j] * (1 - (1 - f[i,j]) * s[i,j])
        if(h_i[i,j] == 0):
            red_out[i,j] = v_final[i,j]
            green_out[i,j] = t[i,j]
            blue_out[i,j] = p[i,j]
        elif(h_i[i,j] == 1):
            red_out[i,j] = q[i,j]
            green_out[i,j] = v_final[i,j]
            blue_out[i,j] = p[i,j]
        elif(h_i[i,j] == 2):
            red_out[i,j] = p[i,j]
            green_out[i,j] = v_final[i,j]
            blue_out[i,j] = t[i,j]
        elif(h_i[i,j] == 3):
            red_out[i,j] = p[i,j]
            green_out[i,j] = q[i,j]
            blue_out[i,j] = v_final[i,j]
        elif(h_i[i,j] == 4):
            red_out[i,j] = t[i,j]
            green_out[i,j] = p[i,j]
            blue_out[i,j] = v_final[i,j]
        elif(h_i[i,j] == 5):
            red_out[i,j] = v_final[i,j]
            green_out[i,j] = p[i,j]
            blue_out[i,j] = q[i,j]

# Outputting Image
red_bgr = (red_out * 255).astype(np.uint8)
green_bgr = (green_out * 255).astype(np.uint8)
blue_bgr = (blue_out * 255).astype(np.uint8)
output_img = cv2.merge([blue_bgr, green_bgr, red_bgr])
cv2.imwrite('output.png', output_img)

# Plotting 2D Histogram of Hue, Saturation
h_flatten = h.flatten()
s_flatten = s.flatten() * 255
plt.figure(figsize=(8, 6))
plt.hist2d(h_flatten, s_flatten, bins=[180, 256], range=[[0, 180], [0, 256]], cmap='inferno')
plt.title('2D Histogram of H and S Channels')
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.colorbar(label='Pixel Count')
plt.savefig('2d_histogram.png')
plt.show()

# Plotting PDF of Input and Output
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(img)
axs[0, 0].set_title('Original Image')
axs[1, 0].imshow(output_img)
axs[1, 0].set_title('Equalized Image')
hist_input, _ = np.histogram(img.flatten(), 256, [0, 256])
pdf_input = hist_input / hist_input.sum()
hist_output, _ = np.histogram(output_img.flatten(), 256, [0, 256])
pdf_output = hist_output / hist_output.sum()
axs[0, 1].plot(pdf_input)
axs[0, 1].set_title('Original PDF')
axs[1, 1].plot(pdf_output)
axs[1, 1].set_title('Equalized PDF')
plt.savefig('result.png')
plt.show()