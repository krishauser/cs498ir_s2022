"""Registers all known grippers.

If you run this as the main file with no arguments, it will display all the
known grippers.

If you run this with one or more arguments, it will display the named grippers.
"""

from klampt.model.robotinfo import GripperInfo
import os
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
robotiq_85 = GripperInfo.load(os.path.join(data_dir,'gripperinfo','robotiq_85.json'))
robotiq_140 = GripperInfo.load(os.path.join(data_dir,'gripperinfo','robotiq_140.json'))

robotiq_85_kinova_gen3 = GripperInfo.mounted(robotiq_85,os.path.join(data_dir,"robots/kinova_with_robotiq_85.urdf"),"gripper:Link_0","robotiq_85-kinova_gen3")

robotiq_140_trina_left = GripperInfo.mounted(robotiq_140,os.path.join(data_dir,"robots/TRINA.urdf"),"left_gripper:base_link","robotiq_140-trina-left")
robotiq_140_trina_right = GripperInfo.mounted(robotiq_140,os.path.join(data_dir,"robots/TRINA.urdf"),"right_gripper:base_link","robotiq_140-trina-right")

if __name__ == '__main__':
    from klampt import vis
    import sys
    if len(sys.argv) == 1:
        grippers = [i for i in GripperInfo.all_grippers]
        print("ALL GRIPPERS",grippers)
    else:
        grippers = sys.argv[1:]

    for i in grippers:
        g = GripperInfo.get(i)
        print("SHOWING GRIPPER",i)
        g.addToVis()
        vis.setWindowTitle(i)
        def setup():
            vis.show()
        def cleanup():
            vis.show(False)
            vis.clear()
        vis.loop(setup=setup,cleanup=cleanup)
