import cv2
import numpy as np
import matplotlib.pyplot as plt

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
cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min()) # scaling formula
v_equalized = cdf[v_8bit] # LUP mapping
v_final = v_equalized.astype(float) / 255.0

# User-Specified Transformation
v_8bit_user = (v*255).astype(np.uint8)
hist_user, _ = np.histogram(v_8bit.flatten(), bins=256, range=(0, 256))
cdf_user = hist.cumsum()
gamma = 0.4  # lower value makes the image brighter
v_gamma = np.power(cdf_user / cdf_user.max(), gamma) # formula
cdf_user_final = (v_gamma * 255).astype(np.uint8) # rounds gray scale value
v_user_equalized = cdf_user_final[v_8bit]
v_user_final = v_user_equalized.astype(float) / 255.0

# Converting HSV to RGB
def hsv_to_bgr(v_final):
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
    return cv2.merge([
        (blue_out * 255).astype(np.uint8),
        (green_out * 255).astype(np.uint8),
        (red_out * 255).astype(np.uint8)
    ])

# Outputting Image
output_img = hsv_to_bgr(v_final)
output_user_img = hsv_to_bgr(v_user_final)
cv2.imwrite('output_equalized.png', output_img)
cv2.imwrite('output_user.png', output_user_img)

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
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
axs[0, 0].imshow(img)
axs[0, 0].set_title('Original Image')
output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
axs[0, 1].imshow(output_img_rgb)
axs[0, 1].set_title('Equalized Image')
output_user_img_rgb = cv2.cvtColor(output_user_img, cv2.COLOR_BGR2RGB)
axs[0, 2].imshow(output_user_img_rgb)
axs[0, 2].set_title('User-Specified Image')
hist_input, _ = np.histogram(img.flatten(), 256, [0, 256])
pdf_input = hist_input / hist_input.sum()
hist_output, _ = np.histogram(output_img.flatten(), 256, [0, 256])
pdf_output = hist_output / hist_output.sum()
hist_user_output, _ = np.histogram(output_user_img.flatten(), 256, [0, 256])
pdf_user_output = hist_user_output / hist_user_output.sum()
axs[1, 0].plot(pdf_input)
axs[1, 0].set_title('Original PDF')
axs[1, 1].plot(pdf_output)
axs[1, 1].set_title('Equalized PDF')
axs[1, 2].plot(pdf_user_output)
axs[1, 2].set_title('User-Specified PDF')
plt.savefig('result.png')
plt.show()