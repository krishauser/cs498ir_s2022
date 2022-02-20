import time
from klampt import *
from klampt import vis
from klampt.vis.visualization import _backend
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory,SE3Trajectory
from klampt.model import ik
from klampt.model.robotinfo import GripperInfo
from klampt.model.subrobot import SubRobotModel
from klampt.io import numpy_convert,loader
from klampt.io import resource
import numpy as np
import os
import sys
sys.path.append('../common')
import known_grippers
from antipodal_grasp import *

######################### Problem 1 code goes here #################################

def make_grasp_approach(gripper_info,T_grasp,finger_width,distance=None):
    """Given a grasp transform and desired finger width, create a grasp
    approach trajectory that
    1. starts with the fingers open at `distance` units away,
    2. moves toward T_grasp in a straight line along the local approach direction
      `gripper_info.primaryAxis`, and then
    3. closes the fingers to the appropriate width.
    
    The result should place the fingers to be properly aligned the grasp.
    
    Args:
        gripper_info (GripperInfo)
        T_grasp (klampt se3 element): the gripper's ultimate transform
        finger_width (float): the desired width between the fingers. Assume the
            gripper linearly interpolates between `gripper_info.minimumSpan` and
            `gripper_info.maximumSpan`.
        distance (float, optional): the amount the gripper should start away from 
            the object. If None, should be set to `gripper_info.fingerLength`
    
    Returns:
        SE3Trajectory,Trajectory: A pair (gripper_traj,finger_traj) describing
        a grasp approach trajectory. gripper_traj gives the Cartesian movement
        of the gripper base, synchronized with the trajectory of the fingers.
        
        The timing of the trajectories can be arbitrary, but they should stay
        synchronized.
    """
    if distance is None:
        distance = gripper_info.fingerLength
    fopen = gripper_info.partwayOpenConfig(1)
    fclose = gripper_info.partwayOpenConfig(0)
    #TODO: construct the trajectories
    finger_trajectory = Trajectory([0,1],[fopen,fclose])
    gripper_trajectory = SE3Trajectory([0,1],[T_grasp,T_grasp])
    return gripper_trajectory,finger_trajectory

####################################################################################


def sample_grasp_approach(gripper_info,grasp_local,obj,distance=None):
    """Given an object-centric AntipodalGrasp, sample a possible grasp
    approach trajectory that
    1. starts with the fingers open at `distance` units away,
    2. moves toward T_grasp in a straight line along the local approach direction
      `gripper_info.primary_axis`, and then
    3. closes the fingers to the appropriate width.
    
    Args:
        gripper_info (GripperInfo)
        grasp_local (AntipodalGrasp): the desired grasp, given in local coordinates.
        obj (RigidObjectModel):
        distance (float, optional): the amount the gripper should start away from 
            the object. If None, should be set to `gripper_info.finger_length`
    
    Returns:
        SE3Trajectory,Trajectory: A pair (gripper_traj,finger_traj) describing
        a grasp approach trajectory. gripper_traj gives the Cartesian movement
        of the gripper base, synchronized with the trajectory of the fingers.
        
        The timing of the trajectories can be arbitrary, but they should stay
        synchronized.
    """
    grasp_world = grasp_local.get_transformed(obj.getTransform())
    Tgrasp = grasp_world.get_grasp_transform(gripper_info)
    return make_grasp_approach(gripper_info,Tgrasp,grasp_local.finger_width,distance)


def problem_1():
    world = WorldModel()
    obj = world.makeRigidObject("object1")
    #obj.geometry().loadFile("../data/objects/ycb-select/002_master_chef_can/nontextured.ply")
    #obj.geometry().loadFile("../data/objects/ycb-select/003_cracker_box/nontextured.ply")
    #obj.geometry().loadFile("../data/objects/ycb-select/011_banana/nontextured.ply"); 
    obj.geometry().loadFile("../data/objects/ycb-select/048_hammer/nontextured.ply")
    #obj.geometry().loadFile("../data/objects/cube.off"); obj.geometry().scale(0.2)
    #obj.geometry().loadFile("../data/objects/cylinder.off")

    #make sure this grasp database is named appropriately for the object loaded
    grasp_db = load_antipodal_grasp_database('048_hammer.json')

    #this will perform a reasonable center of mass / inertia estimate
    m = obj.getMass()
    m.estimate(obj.geometry(),mass=0.454,surfaceFraction=0.2)
    obj.setMass(m)

    gripper = known_grippers.robotiq_140
    res = world.readFile(os.path.join('../data/gripperinfo',gripper.klamptModel))
    if not res:
        raise ValueError("Couldn't read model",gripper.klamptModel)
    gripper_robot = world.robot(0)
    gripper_geom = gripper.getGeometry(gripper_robot,type='TriangleMesh')

    #to achieve cross-platform animations for Macs and for IPython, need to 
    #explicitly code the animation loop
    loop_data = {'gripper_traj':None,'finger_traj':None,'tstart':None}
    
    def loop(data=loop_data):
        gripper_traj,finger_traj,tstart = data['gripper_traj'],data['finger_traj'],data['tstart']
        if gripper_traj is None:
            return
        t = vis.animationTime() - tstart
        #if _backend == 'IPython':
        #    #If animation doesn't work (i.e., patch_a_pip_install.py can't be run), use this workaround
        #    data['tstart'] -= 1.0/30.0
        T = gripper_traj.eval(t)
        qf = finger_traj.eval(t)
        gripper_robot.link(0).setParentTransform(*T)
        qrobot = gripper_robot.getConfig()
        qrobot_f = gripper.setFingerConfig(qrobot,qf)
        gripper_robot.setConfig(qrobot_f)
        vis.update()
        if t > gripper_traj.duration() and t > finger_traj.duration():
            #done
            data['gripper_traj'] = None
            data['finger_traj'] = None
            data['tstart'] = None
    
    def animate(gripper_traj,finger_traj):
        for t,m in zip(gripper_traj.times,gripper_traj.milestones):
            vis.add("gripper traj "+str(t),gripper_traj.to_se3(m))
        if gripper_traj.startTime() != finger_traj.startTime():
            print("Gripper and finger trajectories not time-aligned")
            return
        if gripper_traj.endTime() != finger_traj.endTime():
            print("Gripper and finger trajectories not time-aligned")
            return
        loop_data['gripper_traj'] = gripper_traj
        loop_data['finger_traj'] = finger_traj
        loop_data['tstart'] = vis.animationTime()

    def do_wide_grasp():    
        T_test = se3.identity()
        width = 0.1
        gripper_traj,finger_traj = make_grasp_approach(gripper,T_test,width)
        #On a local install, you can uncomment these lines to inspect your outputs
        #more carefully:
        #finger_robot = SubRobotModel(gripper_robot,gripper.fingerLinks)
        #resource.edit("finger trajectory",finger_robot.tofull(finger_traj),world=world)
        #resource.edit("gripper_trajectory",gripper_traj,world=world)
        animate(gripper_traj,finger_traj)

    def do_feas_grasp():
        print("Sampling approach for grasp with score",grasp_db[0].score)
        gripper_traj,finger_traj = sample_grasp_approach(gripper,grasp_db[0],obj)
        animate(gripper_traj,finger_traj)
        grasp_db.pop(0)
        
    #make the object transparent yellow
    obj.appearance().setColor(0.8,0.8,0.2,0.5)
    #draw center of mass
    vis.add("world",world)
    vis.add("COM",m.getCom(),color=(1,0,0,1),size=0.01)
    vis.addAction(do_wide_grasp,"Test make_grasp_approach")
    vis.addAction(do_feas_grasp,"Test sample_grasp_approach")
    vis.loop(callback=loop)

if __name__ == '__main__':
    problem_1()
