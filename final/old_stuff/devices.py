import device
import cv2


def main():
    # print OpenCV version
    print("OpenCV version: " + cv2.__version__)

    # Get camera list
    device_list = device.getDeviceList()
    index = 0

    for camera in device_list:
        print(str(index) + ': ' + camera[0] + ' ' + str(camera[1]))
        index += 1

    last_index = index - 1

    if last_index < 0:
        print("No device is connected")
        return

if __name__=="__main__":
    main()
