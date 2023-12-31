# https://wenku.csdn.net/answer/ce7e0650e86b11edbcb5fa163eeb3507#:~:text=python%E5%A6%82%E4%BD%95%E5%88%A9%E7%94%A8%E5%BD%92%E4%B8%80%E5%8C%96%E5%82%85%E9%87%8C%E5%8F%B6%E6%8F%8F%E8%BF%B0%E5%AD%90%E7%89%B9%E5%BE%81%E5%AE%9E%E7%8E%B0%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%201%20%E5%AF%B9%E7%9B%AE%E6%A0%87%E5%9B%BE%E5%83%8F%E8%BF%9B%E8%A1%8C%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B%EF%BC%8C%E5%BE%97%E5%88%B0%E7%9B%AE%E6%A0%87%E7%9A%84%E8%BD%AE%E5%BB%93%E3%80%82%202%20%E5%AF%B9%E8%BD%AE%E5%BB%93%E8%BF%9B%E8%A1%8C%E9%87%87%E6%A0%B7%EF%BC%8C%E5%BE%97%E5%88%B0%E4%B8%80%E7%BB%84%E5%9D%90%E6%A0%87%E7%82%B9%E3%80%82%20%E8%BF%99%E4%BA%9B%E5%9D%90%E6%A0%87%E7%82%B9%E5%8F%AF%E4%BB%A5%E7%94%A8%E6%9D%A5%E8%A1%A8%E7%A4%BA%E7%9B%AE%E6%A0%87%E7%9A%84%E5%BD%A2%E7%8A%B6%E3%80%82%203%20%E5%AF%B9%E5%9D%90%E6%A0%87%E7%82%B9%E8%BF%9B%E8%A1%8C%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2%EF%BC%8C%E5%BE%97%E5%88%B0%E9%A2%91%E5%9F%9F%E7%9A%84%E7%B3%BB%E6%95%B0%E3%80%82,NFD%20%E7%89%B9%E5%BE%81%E3%80%82%206%20%E5%88%A9%E7%94%A8%E6%A8%A1%E6%9D%BF%E5%8C%B9%E9%85%8D%E6%88%96%E8%80%85%E5%88%86%E7%B1%BB%E5%99%A8%E7%AD%89%E6%96%B9%E6%B3%95%EF%BC%8C%E5%AF%B9%E5%BE%85%E8%AF%86%E5%88%AB%E5%9B%BE%E5%83%8F%E7%9A%84%20NFD%20%E7%89%B9%E5%BE%81%E4%B8%8E%E5%B7%B2%E7%9F%A5%E7%9B%AE%E6%A0%87%E7%9A%84%20NFD%20%E7%89%B9%E5%BE%81%E8%BF%9B%E8%A1%8C%E6%AF%94%E8%BE%83%EF%BC%8C%E4%BB%8E%E8%80%8C%E5%AE%9E%E7%8E%B0%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%82
import cv2 as cv
import numpy as np

img = cv.imread('main.png')

ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

edges = cv.Canny(thresh, 100, 200)
# print(edges)
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cnt = contours[0]
print(cnt.shape)
fourier_desc = cv.dft(np.float32(cnt[:, 0, :]), flags=cv.DFT_COMPLEX_OUTPUT)
fourier_desc = np.fft.fftshift(fourier_desc)

fourier_desc_norm = np.abs(fourier_desc) / np.abs(fourier_desc[0])

print(fourier_desc_norm.shape)