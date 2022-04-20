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

DATASET_FOLDER = 'generated_data'

def randomize_world(world : WorldModel, sim : Simulator):
    """Helper function to help reset the world state. """
    for i in range(world.numRigidObjects()):
        obj = world.rigidObject(i)
        #TODO: sample object positions
        #Bad things will happen to the sim if the objects are colliding!
        T = obj.getTransform()
        obj.setTransform(T[0],vectorops.add(T[1],[0.01,0,0]))

    #reset the sim bodies -- this code doesn't need to be changed
    for i in range(world.numRigidObjects()):
        model = world.rigidObject(i)
        body = sim.body(model)
        body.setVelocity([0,0,0],[0,0,0])
        body.setObjectTransform(*model.getTransform())


if __name__ == '__main__':
    controller = createRobotController()
    rgbd_sensor = controller.robotModel().sensor('rgbd_camera')
    Tcamera_world = sensing.get_sensor_xform(rgbd_sensor)  #transform of the camera in the world frame (which is also the robot base frame)
    q_out_of_the_way = resource.get('out_of_the_way.config')

    controllerVis = RobotInterfacetoVis(controller.arm)
    plotShown = False
    im = None
    numExamples = 0
    state = 'move_out_of_way'

    def initVis():
        vis.add("world",controller.world)
        vis.addAction(controller.toggleVacuum,'Toggle vacuum','v')
        vis.addAction(lambda : randomize_world(controller.world,controller.sim),"Randomize world",'r')
        
    def loopVis():
        global plotShown,im,numExamples
        global state
        with StepContext(controller):
            #TODO: fill me out to perform self-supervised data generation -- will want to generate a
            #target, try grasping, and try lifting. Then use your sensors to determine whether you
            #have grasped the object, and then save the image and grasp location.
            #
            #You will want to implement a state machine...
            #
            if state == 'move_out_of_way':
                controller.setArmPosition(q_out_of_the_way)
                state = 'move_out_of_way_wait'
            elif state == 'move_out_of_way_wait':
                if controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                    state = 'move_target'
            elif state == 'move_target':
                controller.arm.moveToCartesianPosition((so3.identity(),[0.25,0,0.03]))
                state == 'move_target_wait'
            elif state == 'move_target_wait':
                if controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                    state = 'move_out_of_way'

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

            controllerVis.update()

    def closeVis():
        controller.close()

    #maximum compability with Mac
    vis.loop(initVis,loopVis,closeVis)
    