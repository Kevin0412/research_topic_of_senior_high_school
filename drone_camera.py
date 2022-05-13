from robomaster import robot
import cv2
import numpy as np
import time

if __name__ == '__main__':
    tl_drone = robot.Drone()
    tl_drone.initialize()
    tl_flight = tl_drone.flight
    tl_flight.takeoff().wait_for_completed()

    tl_camera = tl_drone.camera
    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("high")
    tl_camera.set_resolution("high")
    tl_camera.set_bitrate(6)
    t0=time.time()
    while(True):
        img = tl_camera.read_cv2_image()
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([175,32,45]),np.array([179,108,198]))
        #cv2.line(img,(480,355),(480,365),(0,0,255),1)
        #cv2.line(img,(475,360),(485,360),(0,0,255),1)
        cv2.imshow("Drone", img)
        cv2.imshow("hsv", hsv)
        cv2.imshow("LED", mask)
        key=cv2.waitKey(1) & 0xFF
        print(time.time()-t0)
        if key==27:
            break
        elif key==ord('q'):
            t=time.time()
            cv2.imwrite('frames/drone_'+str(t)+'.png',img)
            cv2.imwrite('frames/hsv_'+str(t)+'.png',hsv)
            cv2.imwrite('frames/led_'+str(t)+'.png',mask)
    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()
    tl_flight.land().wait_for_completed()
    tl_drone.close()