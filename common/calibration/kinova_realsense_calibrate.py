from calibrate import *
from klampt import *
from klampt.io import loader
from klampt.math import vectorops,so3,se3
import time

world = WorldModel()
world.readFile("data/robots/kinova_with_robotiq_85.urdf")
robot = world.robot(0)

camera_link = 'EndEffector_Link'
robot_configs = []
link_xforms = []
marker_xforms = []
num_images = 10
for i in range(1,num_images+1):
    robot_configs.append(loader.load('Config','extrinsic_calibration/kinova_%04d.config'%(i,)))
    marker_xforms.append(loader.load('RigidTransform','extrinsic_calibration/aruco_%04d.xform'%(i,)))
    robot.setConfig(robot_configs[-1])
    link_xforms.append(robot.link(camera_link).getTransform())

#let user edit guesses
vis.add('world',world)
vis.add('configs',robot_configs,color=(1,1,0,0.25))
robot.setConfig(robot_configs[0])
cam_xform = resource.get("realsense.xform",default=se3.identity())
cam_xform = se3.mul(link_xforms[0],cam_xform)
marker_xform = resource.get("marker.xform",default=se3.identity())
vis.add('camera',cam_xform)
vis.add('marker',marker_xform)
vis.edit('camera')
vis.edit('marker')
cam_to_link = se3.mul(se3.inv(link_xforms[0]),cam_xform)
for i,m in enumerate(marker_xforms):
    vis.add('marker_viewed_'+str(i),se3.mul(link_xforms[i],se3.mul(cam_to_link,m)))
vis.show()
while vis.shown():
    vis.lock()
    cam_xform = vis.getItemConfig('camera')
    cam_xform = (cam_xform[:9],cam_xform[9:])
    marker_xform = vis.getItemConfig('marker')
    marker_xform = (marker_xform[:9],marker_xform[9:])
    cam_to_link = se3.mul(se3.inv(link_xforms[0]),cam_xform)
    for i,m in enumerate(marker_xforms):
        vis.add('marker_viewed_'+str(i),se3.mul(link_xforms[i],se3.mul(cam_to_link,m)))
    vis.unlock()
    time.sleep(0.05)

cam_to_link = se3.mul(se3.inv(link_xforms[0]),cam_xform)
resource.set("realsense.xform",cam_to_link,type='RigidTransform')
resource.set("marker.xform",marker_xform,type='RigidTransform')

vis.clear()
obserr = [0.001,0.001,0.004,0.01,0.01,0.01]
(err,Tc,marker_dict) = calibrate_robot_camera(robot,camera_link,robot_configs,
                                marker_xforms,[0]*len(marker_xforms),{0:0},
                                observation_relative_errors = [obserr]*len(marker_xforms),
                                camera_initial_guess=cam_to_link,marker_initial_guess={0:marker_xform})
resource.set("realsense.xform",Tc,type='RigidTransform')
resource.set("marker.xform",marker_dict[0],type='RigidTransform')


print("RMSE reconstruction error:",err/math.sqrt(len(marker_xform)))
for i in range(len(marker_xforms)):
    marker_world_i = se3.mul(link_xforms[i],se3.mul(Tc,marker_xforms[i]))
    print("Observation",i,"marker translation error:",vectorops.distance(marker_world_i[1],marker_dict[0][1]))
    print("Observation",i,"marker rotation error:",so3.distance(marker_world_i[0],marker_dict[0][0]))