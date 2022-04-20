from simulated_robot import createRobotController
from klampt.control.interop import RobotInterfacetoVis
from klampt.control import StepContext
from klampt.math import so3,se3,vectorops
from klampt.model import sensing
from klampt.io import resource
from klampt import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TARGET_LOCATION = [0,0.2,0]

if __name__ == '__main__':
    controller = createRobotController()
    rgbd_sensor = controller.robotModel().sensor('rgbd_camera')
    Tcamera_world = sensing.get_sensor_xform(rgbd_sensor)  #transform of the camera in the world frame (which is also the robot base frame)

    controllerVis = RobotInterfacetoVis(controller.arm)
    plotShown = False
    im = None
    
    def initVis():
        vis.add("world",controller.world)
        vis.addAction(controller.toggleVacuum,'Toggle vacuum','v')
        
    def loopVis():
        global plotShown,im
        with StepContext(controller):

            #print the flow sensor if the vacuum is on
            if controller.getVacuumCommand() > 0:
                print("Flow:",controller.getVacuumFlow())

            #update the Matplotlib window if the sensor is working
            rgb,depth = controller.rgbdImages()
            if rgb is not None:
                #funky stuff to make sure that the image window updates quickly
                if not plotShown:
                    im = plt.imshow(rgb)
                    plt.show(block=False)
                    plotShown = True
                else:
                    im.set_array(rgb)
                    plt.pause(0.01)

            #TODO: fill me out to perform image-based pick and place planning to create as much of a stack as you can at TARGET_LOCATION
            #
            #You are NOT allowed to cheat and access controller.sim or controller.world.
            #
            #You will want to implement a state machine...
            #
            controller.arm.moveToCartesianPosition((so3.identity(),[0.25,0,0.03]))

            controllerVis.update()

            
    def closeVis():
        controller.close()

    #maximum compability with Mac
    vis.loop(initVis,loopVis,closeVis)
