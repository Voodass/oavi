import cv2
import numpy as np

def enlarge_image(image, factor):
   
    if factor == 1.0:
        return image
    
   
    height, width = image.shape[:2]
    new_height, new_width = int(height * factor), int(width * factor)
    
    
    enlarged = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return enlarged

def decimate_image(image, factor):
    
    if factor == 1.0:
        return image
    

    height, width = image.shape[:2]
    new_height, new_width = int(height / factor), int(width / factor)
    
    
    decimated = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return decimated

def two_pass_rediscretization(image, m_factor, n_factor):

    enlarged = enlarge_image(image, m_factor)
    
   
    result = decimate_image(enlarged, n_factor)
    
    return result

def one_pass_rediscretization(image, k_factor):
   
    
    height, width = image.shape[:2]
    new_height, new_width = int(height * k_factor), int(width * k_factor)
    
   
    result = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return result
