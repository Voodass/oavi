import cv2
import numpy as np

def extract_rgb_components(image):

    b_channel, g_channel, r_channel = cv2.split(image)
    
    
    r_image = np.zeros_like(image)
    g_image = np.zeros_like(image)
    b_image = np.zeros_like(image)
    
  
    r_image[:, :, 2] = r_channel
    g_image[:, :, 1] = g_channel
    b_image[:, :, 0] = b_channel
    
    return r_image, g_image, b_image

def convert_to_hsi(image):
 
    image_float = image.astype(np.float32) / 255.0
    
   
    b, g, r = cv2.split(image_float)
    
    
    intensity = (r + g + b) / 3.0
    
   
    intensity_mask = intensity > 0
    
   
    saturation = np.zeros_like(intensity)
    hue = np.zeros_like(intensity)
    
    
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation[intensity_mask] = 1 - 3 * min_rgb[intensity_mask] / (r[intensity_mask] + g[intensity_mask] + b[intensity_mask])
    

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g)**2 + (r - b) * (g - b)) + 1e-10  
    theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
    
    
    hue[b > g] = 2 * np.pi - theta[b > g]
    hue[b <= g] = theta[b <= g]
    
   
    hue = hue / (2 * np.pi)
    
 
    h_channel = (hue * 255).astype(np.uint8)
    s_channel = (saturation * 255).astype(np.uint8)
    i_channel = (intensity * 255).astype(np.uint8)
    
  
    h_display = np.zeros_like(image)
    s_display = np.zeros_like(image)
    i_display = cv2.merge([i_channel, i_channel, i_channel])
    
   
    h_display[:, :, 0] = h_channel  
    h_display[:, :, 1] = s_channel  
    h_display[:, :, 2] = i_channel  
    
    
    s_display = cv2.merge([s_channel, s_channel, s_channel])
    
    return h_display, s_display, i_display

def invert_brightness(image):
 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    
    hsv_image[:, :, 2] = 255 - hsv_image[:, :, 2]
    
    
    inverted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return inverted_image
