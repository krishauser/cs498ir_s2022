import time
from klampt import *
from klampt import vis
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory,SE3Trajectory
from klampt.model import ik
from klampt.model.robotinfo import GripperInfo
from klampt.io import resource
import numpy as np
import os
import sys
sys.path.append('../common')
import known_grippers
from antipodal_grasp import *

#need this from problem 1
from problem1 import sample_grasp_approach

######################### Problem 2 code goes here #################################

def solve_robot_ik(robot,gripper,Tgripper):
    """Given a robot, a gripper, and a desired gripper transform,
    solve the IK problem to place the gripper at the desired transform.
    
    Note: do not modify DOFs 0-5.
    
    Returns:
        list or None: Returns None if no solution was found, and
        returns an IK-solving configuration q otherwise.
    
    This function may modify robot.
    
    Args:
        robot (RobotModel)
        gripper (GripperInfo)
        Tgripper (klampt se3 object)
    """
    #TODO: solve the IK problem
    return None

def sample_grasp_ik(robot,gripper,grasp_local,obj):
    """Given a robot, a gripper, a desired antipodal grasp
    (in local coordinates), and an object, solve the IK
    problem to place the gripper at the desired grasp.
    
    Don't forget to set the finger configurations!
    
    Note: do not modify DOFs 0-5.
    
    Returns:
        tuple (q,T): q is None if no solution was found, and
        is an IK-solving configuration q. T is the resulting
        transform of the gripper link.
    
    This function may modify robot.
    
    Args:
        robot (RobotModel)
        gripper (GripperInfo)
        grasp_local (AntipodalGrasp): given in object local coordinates
        obj (RigidObjectModel): the object
    """
    #TODO: solve the IK problem
    return None,None

def solve_grasp_ik(robot,gripper,grasp_local,obj):
    #TODO: fill me in find a grasp & IK configuration that avoids collisions
    return sample_grasp_ik(robot,gripper,grasp_local,obj)

def solve_grasp_problem(robot,gripper,obj,grasp_db,obstacles):
    """Returns a triple (qrobot,Tgripper,grasp) solving the
    grasping problem given a object-centric grasp database.
    
    Self-collisions should be avoided, as well as collisions with obj
    and environmental collisions with the Geometry3D's in obstacles.
    """
    #TODO: fill me in to pick a high-quality and reachable grasp & IK configuration
    if len(grasp_db)==0:
        return None,None,None
    qrobot, Tgripper = sample_grasp_ik(robot,gripper,grasp_db[0],obj)
    if qrobot is None:
        return None,None,None
    return qrobot,Tgripper,grasp_db[0]

###################################################################################

def problem_2():
    #load the object and grasp DB
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
    grasp_db_orig = copy.copy(grasp_db)

    #this will perform a reasonable center of mass / inertia estimate
    m = obj.getMass()
    m.estimate(obj.geometry(),mass=0.454,surfaceFraction=0.2)
    obj.setMass(m)
    
    #move the object to a floating position in front of the robot
    obj.setTransform(so3.identity(),[1,0,0.8])
    obj.appearance().setColor(0.3,0.2,0.05)
    
    #load the robot
    gripper = known_grippers.robotiq_140_trina_left
    res = world.readFile(os.path.join('../data/gripperinfo',gripper.klamptModel))
    if not res:
        raise ValueError("Couldn't read model",gripper.klamptModel)
    robot = world.robot(0)

    #set a nice configuration
    resource.setDirectory("../data/resources/TRINA")
    qhome = resource.get("home.config")
    robot.setConfig(qhome)
    
    def sample_transform_and_solve_ik():
        if len(grasp_db)==0:
            print("Out of grasps to try")
            for g in grasp_db_orig:
                grasp_db.append(g)
        gripper_traj,finger_traj = sample_grasp_approach(gripper,grasp_db[0],obj)
        Ttarget = gripper_traj.to_se3(gripper_traj.milestones[-1])
        vis.add("target xform",Ttarget)
        qrob = solve_robot_ik(robot,gripper,Ttarget)
        if qrob is not None:
            #inject the finger configuration
            qrob = gripper.set_finger_config(qrob,finger_traj.milestones[-1])
            #set the robot configuration
            robot.setConfig(qrob)
            vis.setColor(vis.getItemName(robot.link(gripper.baseLink)),0,1,0)
        else:
            vis.setColor(vis.getItemName(robot.link(gripper.baseLink)),1,0,1,1)
        vis.update()
        grasp_db.pop(0)

    def sample_ik_direct():
        if len(grasp_db)==0:
            print("Out of grasps to try")
            for g in grasp_db_orig:
                grasp_db.append(g)
        q_robot,Tgripper = sample_grasp_ik(robot,gripper,grasp_db[0],obj)
        if q_robot is not None:
            vis.add("target xform",Tgripper)
            #set the robot configuration
            robot.setConfig(q_robot)
            vis.setColor(vis.getItemName(robot.link(gripper.baseLink)),0,1,0)
        else:
            vis.setColor(vis.getItemName(robot.link(gripper.baseLink)),1,0,1,1)
        vis.update()
        grasp_db.pop(0)
    
    def sample_ik_collision_free():
        if len(grasp_db)==0:
            print("Out of grasps to try")
            for g in grasp_db_orig:
                grasp_db.append(g)
        q_robot,Tgripper = solve_grasp_ik(robot,gripper,grasp_db[0],obj)
        if q_robot is not None:
            vis.add("target xform",Tgripper)
            #set the robot configuration
            robot.setConfig(q_robot)
            vis.setColor(vis.getItemName(robot.link(gripper.baseLink)),0,1,0)
        else:
            vis.setColor(vis.getItemName(robot.link(gripper.baseLink)),1,0,1,1)
        vis.update()
        grasp_db.pop(0)
    
    obstacle_world = WorldModel()
    obstacles = []
    def load_obstacles():
        #populates the obstacles list
        obstacle_world.readFile("table_and_box.xml")
        for i in range(obstacle_world.numTerrains()):
            obstacles.append(obstacle_world.terrain(i).geometry())
            terr = obstacle_world.terrain(i)
            vis.add(terr.getName(),terr.geometry(),color=terr.appearance().getColor())
        print("Using",len(obstacles),"obstacles")
    
    def solve_grasp():
        if len(obstacles)==0:
            load_obstacles()
        t0 = time.time()
        q_robot,Tgripper = solve_grasp_problem(robot,gripper,obj,grasp_db,obstacles)
        t1 = time.time()
        print("Solved for grasp in %.3fs"%(t1-t0))
        if q_robot is not None:
            vis.add("target xform",Tgripper)
            #set the robot configuration
            robot.setConfig(q_robot)
            vis.setColor(vis.getItemName(robot.link(gripper.baseLink)),0,1,0)
        else:
            vis.setColor(vis.getItemName(robot.link(gripper.baseLink)),1,0,1,1)
        vis.update()
        grasp_db.pop(0)

    vis.addAction(sample_transform_and_solve_ik,"2.A: Sample transform + IK",'a')
    vis.addAction(sample_ik_direct,"2.B: Sample IK",'b')
    vis.addAction(sample_ik_collision_free,"2.C: Sample IK, collision-free",'c')
    vis.addAction(solve_grasp,"2.D: Solve grasping problem",'d')
    vis.addAction(load_obstacles,"Load obstacles",'o')
    vis.add("world",world)
    def setup():
        vis.edit(("world","object1"))
        pass
    vis.loop(setup)

if __name__ == '__main__':
    problem_2()
