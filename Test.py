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


def findOffset(image):
    root = os.path.join(os.getcwd(), "offsetGear.png")
    offsetTemplate = cv2.imread(root)
    offsetTemplate = offsetTemplate[:,:,1]

    searchImage = image[0:100, 400:640, 1]
    hits = MTM.matchTemplates([("Offset", offsetTemplate)],
                              searchImage,
                              method=cv2.TM_CCOEFF_NORMED,
                              N_object=float("inf"),
                              score_threshold=0.8,
                              maxOverlap=0,
                              searchBox=None)

    if len(hits['TemplateName']) == 0:
        print("Gear Icon Used for Template not found")
        sys.exit()

    return hits['BBox'].iloc[0]







def loadDigits():
    root = os.path.join(os.getcwd(), "digits")
    print(os.getcwd())
    print(root)
    digitsList = []

    for i in range(len(os.listdir(root))):
        print(os.path.join(root, str(i) + ".png"))
        if os.path.isfile(os.path.join(root, str(i) + ".png")):
            img = cv2.imread(os.path.join(root, str(i) + ".png"))[:,:,1]
            digitsList.append((str(i), img))
    return digitsList

def countLife(img, templates):
    hits = MTM.matchTemplates(templates,
                                  img,
                                  method=cv2.TM_CCOEFF_NORMED,
                                  N_object=float("inf"),
                                  score_threshold=0.8,
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


    print(findOffset(full_screen))
    # my_stock = full_screen[63:63+12, 547:547+10]
    # enemy_stock = full_screen[63:63+12, 585:585+10]
    # # print(my_stock[:,:,1])
    #
    #
    # template = loadDigits()
    # print(countLife(my_stock[:,:,1],template))

