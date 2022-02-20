import time
from klampt import *
from klampt import vis
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory,SE3Trajectory,RobotTrajectory
from klampt.plan import robotplanning
from klampt.plan.cspace import MotionPlan
from klampt.model import ik
from klampt.model.robotinfo import GripperInfo
from klampt.model.subrobot import SubRobotModel
from klampt.io import resource
import numpy as np
import os
import sys
sys.path.append('../common')
import known_grippers
from antipodal_grasp import *

#need this from problem 1
from problem1 import make_grasp_approach

#need this from problem 2
from problem2 import solve_grasp_problem

###########################################################################

def feasible_plan(world,robot,qtarget):
    """Plans for some number of iterations from the robot's current configuration to
    configuration qtarget.  Returns the first path found.

    Returns None if no path was found, otherwise returns the plan.
    """
    t0 = time.time()
    qstart = robot.getConfig()

    #these are the indices of the moving joints
    moving_joints = [11,12,13,14,15,16]
    space = robotplanning.makeSpace(world=world,robot=robot,edgeCheckResolution=1e-2,movingSubset=moving_joints)
    plan = MotionPlan(space,type='prm')
    #TODO: complete the planner setup, including endpoints, etc.
    
    #Possible hint:
    #maybe consider using planToConfig instead of setting up the space by hand?

    #Possible hint:
    #since the C-space only includes the moving joints, the SubRobotModel is a good
    #way to convert between the moving joints and the full robot model.
    #
    #SubRobotModel.tofull(qsubrobot) => qrobot
    #SubRobotModel.fromfull(qrobot) => qsubrobot
    #SubRobotModel.tofull(subrobot_path) => robot_path
    #SubRobotModel.fromfull(robot_path) => subrobot_path
    moving_robot = SubRobotModel(robot,moving_joints)
    qstart_lower = moving_robot.fromfull(qstart)
    qtarget_lower = moving_robot.fromfull(qtarget)
    
    #to be nice to the C++ module, do this to free up memory
    plan.space.close()
    plan.close()
    #this just moves in a straight line in 1 second
    return RobotTrajectory(robot,[0,1],[robot.getConfig(),qtarget])


def optimizing_plan(world,robot,qtarget):
    """Plans for some number of iterations from the robot's current configuration to
    configuration qtarget.

    Returns None if no path was found, otherwise returns the best plan found.
    """
    #TODO: copy what's in feasible_plan, but change the way in which you to terminate
    return feasible_plan(world,robot,qtarget)

def debug_plan_results(plan,robot,world):
    """Potentially useful for debugging planning results..."""
    assert isinstance(plan,MotionPlan)
    #this code just gives some debugging information. it may get expensive
    V,E = plan.getRoadmap()
    print(len(V),"feasible milestones sampled,",len(E),"edges connected")

    print("Plan stats:")
    pstats = plan.getStats()
    for k in sorted(pstats.keys()):
        print("  ",k,":",pstats[k])

    print("CSpace stats:")
    sstats = plan.space.getStats()
    for k in sorted(sstats.keys()):
        print("  ",k,":",sstats[k])
    """
    print("  Joint limit failures:")
    for i in range(robot.numLinks()):
        print("     ",robot.link(i).getName(),":",plan.space.ambientspace.joint_limit_failures[i])
    """

    path = plan.getPath()
    if path is None or len(path)==0:
        print("Failed to plan path between configuration")
        #debug some sampled configurations
        numconfigs = min(10,len(V))
        #vis.debug("some milestones",V[2:numconfigs],world=world)
        pts = []
        for i,q in enumerate(V):
            robot.setConfig(q)
            pt = robot.link(robot.numLinks()-1).getTransform()[1]
            pts.append(pt)
        for i,q in enumerate(V):
            vis.add("pt"+str(i),pts[i],hide_label=True,color=(1,1,0,0.75))
        for (a,b) in E:
            vis.add("edge_{}_{}".ormat(a,b),Trajectory(milestones=[pts[a],pts[b]]),color=(1,0.5,0,0.5),width=1,pointSize=0,hide_label=True)
        return None

    print("Planned path with length",trajectory.RobotTrajectory(robot,milestones=path).length())

def plan_grasping_motion(world,robot,gripper,obj,grasp_db):
    """Returns a (RobotTrajectory,AntipodalGrasp) pair giving the
    robot's grasping motion as well as the final grasp obtained.
    
    Returns (None,None) on failure.
    """
    qstart = robot.getConfig()
    
    obstacles = []
    
    #find a collision-free configuration that solves the IK problem and meets a grasp
    qgrasp,Tgripper,grasp = solve_grasp_problem(robot,gripper,obj,grasp_db,obstacles)
    if qgrasp is None:
        print("Grasp problem is not solvable")
        return None,None
    #You can debug things using this call
    #vis.debug(qgrasp,Tgripper,world=world)

    #determine the approach trajectory for the gripper
    approach_dist = 0.05
    gripper_traj, finger_traj = make_grasp_approach(gripper,Tgripper,grasp.finger_width,approach_dist)
    
    #Now plan a robot trajectory that reaches the approach trajectory
    #TODO:
    
    #Now convert the finger / gripper approach to a robot trajectory
    #TODO:
    return None,None


###########################################################################

def setup_world():
    #load the object and grasp DB
    world = WorldModel()
    world.readFile("table_and_box.xml")

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
    obj.setTransform(so3.rotation((0,0,1),0.5),[0.7,0.1,0.65])
    obj.appearance().setColor(0.3,0.2,0.05)
    
    #load the robot
    gripper = known_grippers.robotiq_140_trina_left
    res = world.readFile(os.path.join('../data/gripperinfo',gripper.klamptModel))
    if not res:
        raise ValueError("Couldn't read model",gripper.klamptModel)
    robot = world.robot(0)
    #lots of arm joint limits are too large
    qmin,qmax = robot.getJointLimits()
    for i in range(6,robot.numLinks()):
        if qmax[i] > 6 and qmin[i] < -6:
            qmax[i] = math.pi
            qmin[i] = -math.pi
    robot.setJointLimits(qmin,qmax)

    #set a nice configuration
    resource.setDirectory("../data/resources/TRINA")
    qhome = resource.get("home.config")
    robot.setConfig(qhome)
    return world,robot,gripper,obj,grasp_db

def problem_3ab():
    world, robot, gripper, obj, grasp_db = setup_world()
    
    #the planner's world is a temporary model, which should not be visualized
    planning_world = world.copy()
    planning_robot = planning_world.robot(0)
    
    plan_data = {'path':RobotTrajectory(robot,[0,1],[robot.getConfig(),robot.getConfig()])}
    def plan_to(data=plan_data,optimizing=False):
        qtgt = vis.getItemConfig("qtgt")
        qstart = plan_data['path'].milestones[-1]
        #update robot's start position
        planning_robot.setConfig(qstart)
        if not optimizing:
            path = feasible_plan(planning_world,planning_robot,qtgt)
        else:
            path = optimizing_plan(planning_world,planning_robot,qtgt)
        
        plan_data['path'] = path
        if path is not None:
            assert isinstance(path,RobotTrajectory)
            assert len(path.milestones) == len(path.times)
            for m in path.milestones:
                assert len(m) == robot.numLinks()
            print("Planned path with",len(path.milestones),"milestones, length",path.length())
            #plan successful, add it to the visualizer and animate the robot
            if vis.visualization._backend != 'IPython':
                vis.add("plan",path,color=(1,0.5,0,1),endeffectors=[16])
            vis.animate(("world",robot.getName()),path)
        else:
            print("Planning failed")
            #plan unsuccessful, don't show it
            robot.setConfig(qstart)
            vis.animate(("world",robot.getName()),None)
            plan_data['path'] = RobotTrajectory(robot,[0,1],[qstart,qstart])
            try:
                vis.remove("plan")
            except Exception:
                pass
    
    vis.addAction(plan_to,"Plan to target",'a')
    vis.addAction(lambda: plan_to(optimizing=True),"Plan optimizing plan to target",'b')
    vis.add("world",world)

    #clunky but cross-platform visualization setup
    def setup():
        vis.add("qtgt",robot.getConfig(),color=(1,0,0,0.5))
        vis.edit("qtgt")
        vis.animate(("world",robot.getName()),plan_data['path'])
    vis.loop(setup=setup)

def problem_3c():
    world, robot, gripper, obj, grasp_db = setup_world()
    
    #the planner's world is a temporary model, which should not be visualized
    planning_world = world.copy()
    planning_robot = planning_world.robot(0)
    
    qstart = robot.getConfig()
    
    def plan_grasp():
        planning_robot.setConfig(qstart)
        robot_traj,grasp = plan_grasping_motion(planning_world,planning_robot,gripper,obj,grasp_db)
        if robot_traj is not None:
            assert isinstance(grasp,AntipodalGrasp)
            assert isinstance(robot_traj,RobotTrajectory)
            assert robot_traj.milestones[0] == qstart
            assert len(robot_traj.milestones) == len(robot_traj.times)
            for m in robot_traj.milestones:
                assert len(m) == robot.numLinks()
            print("Generated plan to grasp with score",grasp.score)
            print("Path has",len(robot_traj.milestones),"milestones, length",robot_traj.length())
            vis.animate(("world",robot.getName()),robot_traj)
            if vis.visualization._backend != 'IPython':
                vis.add("plan",robot_traj,color=(1,0.5,0,1),endeffectors=[16])
            #grasp.add_to_vis("solved_grasp")
        else:
            print("Planning failed")
            vis.animate(("world",robot.getName()),None)
    
    vis.addAction(plan_grasp,"Plan grasp",'g')
    vis.add("world",world)
    
    vis.loop()

if __name__ == '__main__':
    #problem_3ab()
    problem_3c()