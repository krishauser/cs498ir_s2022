"""Defines AntipodalGrasp and the grasp database saving/loading functions.
You probably don't need to edit this file.
"""

from klampt import IKObjective
from klampt import vis
from klampt.math import vectorops,so3,se3
from klampt.model.robotinfo import GripperInfo
from klampt.io import loader
from klampt.model.contact import ContactPoint
import json
import copy
import math
import random
import numpy as np
import sys
sys.path.append('../common')
from grasp import Grasp

#settings used for drawing only
finger_radius = 0.02
contact_point_radius = 0.01

def numpy_to_list_recurse(obj):
    if isinstance(obj,dict):
        return dict((k,numpy_to_list_recurse(v)) for k,v in obj.items())
    elif isinstance(obj,(list,tuple)):
        return [numpy_to_list_recurse(v) for v in obj]
    elif isinstance(obj,np.ndarray):
        return obj.tolist()
    return obj

class AntipodalGrasp:
    """A structure containing information about antipodal grasps.
    
    Attributes:
        center (3-vector): the center of the fingers (object coordinates).
        axis (3-vector): the direction of the line through the
            fingers (object coordinates).
        approach (3-vector, optional): the direction that the fingers
            should move forward to acquire the grasp.  For example, [0,0,-1]
            indicates a top-down grasp.
        finger_width (float, optional): the width that the gripper should
            open between the fingers.
        contact1 (ContactPoint, optional): a point of contact on the
            object.
        contact2 (ContactPoint, optional): another point of contact on the
            object.
        score (float, optional): a score for the grasp
    """
    def __init__(self,center,axis):
        self.center = center
        self.axis = axis
        self.approach = None
        self.finger_width = None
        self.contact1 = None
        self.contact2 = None
        self.score = None

    def add_to_vis(self,name,color='auto',score_ref=None):
        if color == 'auto':
            if self.score is None:
                color = (0,0,0,1)
            else:
                s = self.score - score_ref if score_ref is not None else self.score
                u = math.exp(-s*2)
                color = (1-u,u,0,1)
        if self.finger_width == None:
            w = 0.05
        else:
            w = self.finger_width*0.5+finger_radius
        a = vectorops.madd(self.center,self.axis,w)
        b = vectorops.madd(self.center,self.axis,-w)
        vis.add(name,[a,b],color=color)
        if self.approach is not None:
            vis.add(name+"_approach",[self.center,vectorops.madd(self.center,self.approach,0.05)],color=(1,0.5,0,1))
        normallen = 0.05
        if self.contact1 is not None:
            vis.add(name+"cp1",self.contact1.x,color=(1,1,0,1),size=contact_point_radius)
            vis.add(name+"cp1_normal",[self.contact1.x,vectorops.madd(self.contact1.x,self.contact1.n,normallen)],color=(1,1,0,1))
        if self.contact2 is not None:
            vis.add(name+"cp2",self.contact2.x,color=(1,1,0,1),size=contact_point_radius)
            vis.add(name+"cp2_normal",[self.contact2.x,vectorops.madd(self.contact2.x,self.contact2.n,normallen)],color=(1,1,0,1))

    def transform(self,T):
        """Transforms this grasp (in-place) with the se3 element T."""
        self.center = se3.apply(T,self.center)
        self.axis = so3.apply(T[0],self.axis)
        if self.approach is not None:
            self.approach = so3.apply(T[0],self.approach)
        if self.contact1 is not None:
            self.contact1.x = se3.apply(T,self.contact1.x)
            self.contact1.n = so3.apply(T[0],self.contact1.n)
        if self.contact2 is not None:
            self.contact2.x = se3.apply(T,self.contact2.x)
            self.contact2.n = so3.apply(T[0],self.contact2.n)

    def get_transformed(self,T):
        """Returns an AntipodalGrasp transformed by the se3 element T."""
        res = copy.deepcopy(self)
        res.transform(T)
        return res

    def get_grasp_transform(self, gripper_info : GripperInfo, approach_angle='sample', axis_orientation='sample'):
        """
        Generates a compatible grasp transform for a given gripper.
        
        Returns:
            (R,t): a Klampt se3 element describing the maching gripper transform
        """
        axis = self.axis
        if axis_orientation == 'sample':
            if random.getrandbits(1):
                axis = vectorops.mul(self.axis,-1)
        elif axis_orientation == -1:
            axis = vectorops.mul(self.axis,-1)
        R = so3.align(gripper_info.secondaryAxis,axis)
        if self.approach is None:
            #no approach direction specified
            if approach_angle == 'sample':
                approach_angle = random.uniform(0,math.pi*2) 
            R_rand = so3.rotation(self.axis,approach_angle)
            R = so3.mul(R_rand,R)
        else:
            #find R to match secondaryAxis to grasp.axis, primaryAxis to grasp.approach
            rotated_approach = so3.apply(R,gripper_info.primaryAxis)
            Rapp = so3.align(rotated_approach,self.approach)
            R = so3.mul(Rapp,R)
        finger_tip_radius = gripper_info.fingerDepth*0.5
        finger_pos = vectorops.madd(gripper_info.center,gripper_info.primaryAxis,gripper_info.fingerLength-finger_tip_radius)
        t = vectorops.sub(self.center,so3.apply(R,finger_pos))
        return (R,t)
            
    def to_json(self):
        """Converts to a JSON-compatible structure."""
        res = {}
        res['type'] = 'AntipodalGrasp'
        res['center'] = self.center
        res['axis'] = self.axis
        if self.approach is not None:
            res['approach'] = self.approach
        if self.finger_width is not None:
            res['finger_width'] = self.finger_width
        if self.contact1 is not None:
            res['contact1'] = loader.to_json(self.contact1)
        if self.contact2 is not None:
            res['contact2'] = loader.to_json(self.contact2)
        if self.score is not None:
            res['score'] = self.score
        return numpy_to_list_recurse(res)
    
    def from_json(self,jsonobj):
        """Converts from a JSON-compatible structure returned by to_json."""
        if jsonobj.get('type',None) != 'AntipodalGrasp':
            raise ValueError("JSON object is not an AntipodalGrasp")
        self.center = jsonobj['center']
        self.axis = jsonobj['axis']
        self.approach = jsonobj.get('approach',None)
        self.finger_width = jsonobj.get('finger_width',None)
        self.score = jsonobj.get('score',None)
        if 'contact1' in jsonobj:
            self.contact1 = loader.from_json(jsonobj['contact1'])
        if 'contact2' in jsonobj:
            self.contact2 = loader.from_json(jsonobj['contact2'])
        return
    
    def to_grasp(self,gripper_info : GripperInfo) -> Grasp:
        """Returns a Grasp corresponding to this AntipodalGrasp."""
        res = Grasp()
        res.ik_constraint = IKObjective()
        res.ik_constraint.setLinks(gripper_info.baseLink)
        finger_tip_radius = gripperInfo.fingerDepth*0.5
        finger_pos = vectorops.madd(gripper_info.center,gripper_info.primaryAxis,gripper_info.fingerLength-finger_tip_radius)
        res.ik_constraint.setFixedPoint(finger_pos,self.center)
        if self.approach is None:
            res.ik_constraint.setAxialRotConstraint(gripper_info.secondaryAxis,self.axis)
        else:
            T = self.get_grasp_transform(gripper_info)
            res.ik_constraint.setFixedRotConstraint(T[0])
        res.finger_links = gripper_info.fingerLinks
        if self.finger_width is None:
            res.finger_config = gripper_info.partwayOpenConfig(0)
        else:
            res.finger_config = gripper_info.partwayOpenConfig(gripper_info.widthToOpening(self.finger_width))
        return res
    
            
def save_antipodal_grasp_database(grasps, fn : str):
    json_grasps = []
    for g in grasps:
        gjson = g.to_json()
        json_grasps.append(gjson)
    with open(fn,'w') as f:
        json.dump({'type':'AntipodalGraspDatabase','grasps':json_grasps},f)

def load_antipodal_grasp_database(fn : str):
    with open(fn,'r') as f:
        jsondb = json.load(f)
    if jsondb.get('type',None) != 'AntipodalGraspDatabase':
        raise IOError("Not an AntipodalGraspDatabase")
    grasps = []
    for jsongrasp in jsondb['grasps']:
        g = AntipodalGrasp(None,None)
        g.from_json(jsongrasp)
        grasps.append(g)
    return grasps

def sort_grasp_database(grasps,ascending=True):
    return [g for g in sorted(grasps, key=lambda g:(g.score if ascending else -g.score))]

###########################################################################################
