import ctypes
import os
import time
from environment import imageGrab
import MTM
import cv2
import mss
import pygetwindow
import win32con
import win32gui
from matplotlib import pyplot as plt


def loadDigits(itemName):
    root = os.path.join(os.path.dirname(os.getcwd()), "digits", itemName)
    digitsList = []
    for i in range(len(os.listdir(root)) + 1):
        if os.path.isfile(os.path.join(root, str(i) + ".jpg")):
            img = cv2.imread(os.path.join(root, str(i) + ".jpg"))
            digitsList.append((str(i), img))
    return digitsList

def countLife(img, templates):
    hits = MTM.matchTemplates(templates,
                                  img,
                                  method=cv2.TM_CCOEFF_NORMED,
                                  N_object=float("inf"),
                                  score_threshold=0.9,
                                  maxOverlap=0,
                                  searchBox=None)

    if len(hits['TemplateName']) == 0:
        return -1

    return int(hits['TemplateName'].iloc[0])

if 1 == 1:
    ctypes.windll.user32.SetProcessDPIAware()
    hwnd = win32gui.FindWindow(None, 'Brawlhalla')
    dimensions = win32gui.GetWindowRect(hwnd)


    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                          win32con.SWP_SHOWWINDOW | win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    windows = pygetwindow.getWindowsWithTitle('Brawlhalla')
    win = None

    for window in windows:
        if window.title == 'Brawlhalla':
            win = window

    width = 640
    height = 480

    win.size = (width, height)
    win.moveTo(0, 0)
    win.activate()


    sct = mss.mss()
    time.sleep(1)


    full_screen_all = imageGrab(x=0, w=width, y=0, h=height, grabber=sct)
    full_screen = full_screen_all[:, :, :3]
            # print(full_screen)
            # print(full_screen.shape)

    my_stock = full_screen[55:55+10, 548:548+10]
    enemy_stock = full_screen[55:55+10, 587:587+10]

    plt.subplot(1, 1, 1), plt.imshow(my_stock, 'gray', vmin=0, vmax=255)
    plt.show()

