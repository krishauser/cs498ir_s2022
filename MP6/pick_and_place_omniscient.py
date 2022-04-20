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
    
    def initVis():
        vis.add("world",controller.world)
        vis.addAction(controller.toggleVacuum,'Toggle vacuum','v')
        
    def loopVis():
        with StepContext(controller):

            #Fake sensor that produces ground truth object positions and orientations
            block_transforms = []
            for i in range(controller.world.numRigidObjects()):
                body = controller.sim.body(controller.world.rigidObject(i))
                block_transforms.append(body.getTransform())

            #TODO: fill me out to perform pick and place planning to create as much of a stack as you can at TARGET_LOCATION
            #
            #You will want to implement a state machine...
            #
            controller.arm.moveToCartesianPosition((block_transforms[0][0],vectorops.add(block_transforms[0][1],[0,0,0.01])))

            controllerVis.update()

            
    def closeVis():
        controller.close()

    #maximum compability with Mac
    vis.loop(initVis,loopVis,closeVis)
    