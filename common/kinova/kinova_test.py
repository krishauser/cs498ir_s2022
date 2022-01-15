#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2

from klampt import WorldModel
from klampt import vis
import math

# Actuator speed (deg/s)
SPEED = 20.0

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check


def example_send_joint_speeds(base):

    joint_speeds = Base_pb2.JointSpeeds()

    actuator_count = base.GetActuatorCount().count
    # The 7DOF robot will spin in the same direction for 10 seconds
    if actuator_count == 7:
        speeds = [SPEED, 0, -SPEED, 0, SPEED, 0, -SPEED]
        i = 0
        for speed in speeds:
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i 
            joint_speed.value = speed
            joint_speed.duration = 0
            i = i + 1
        print ("Sending the joint speeds for 10 seconds...")
        base.SendJointSpeedsCommand(joint_speeds)
        time.sleep(10)
    # The 6 DOF robot will alternate between 4 spins, each for 2.5 seconds
    if actuator_count == 6:
        print ("Sending the joint speeds for 10 seconds...")
        for times in range(4):
            del joint_speeds.joint_speeds[:]
            if times % 2:
                speeds = [-SPEED, 0.0, 0.0, SPEED, 0.0, 0.0]
            else:
                speeds = [SPEED, 0.0, 0.0, -SPEED, 0.0, 0.0]
            i = 0
            for speed in speeds:
                joint_speed = joint_speeds.joint_speeds.add()
                joint_speed.joint_identifier = i 
                joint_speed.value = speed
                joint_speed.duration = 0
                i = i + 1
            
            base.SendJointSpeedsCommand(joint_speeds)
            time.sleep(2.5)

    print ("Stopping the robot")
    base.Stop()

    return True


def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common/kinova"))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    world = WorldModel()
    world.readFile("data/robots/kinova_with_robotiq_85.urdf")
    robot = world.robot(0)
    vis.add("world",world)
    vis.show()

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
        base = BaseClient(router)

        # Example core
        while vis.shown():
            t0 = time.time()
            #read
            angle_list = [0]*base.GetActuatorCount().count
            joint_angles = base.GetMeasuredJointAngles()
            for ja in joint_angles.joint_angles:
                angle_list[ja.joint_identifier] = ja.value
            req = Base_pb2.GripperRequest()
            req.mode = 3  #position
            gripper_values = base.GetMeasuredGripperMovement(req)
            assert len(gripper_values.finger)==1,"Couldn't read finger?"
            
            #update vis
            vis.lock()
            for i,v in enumerate(angle_list):
                robot.driver(i).setValue(math.radians(v))
            robot.driver(7).setValue(gripper_values.finger[0].value)
            robot.setConfig(robot.getConfig())
            vis.unlock()

            #sleep 50Hz
            t1 = time.time()
            time.sleep(max(0,0.02-(t1-t0)))
        vis.kill()

        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
