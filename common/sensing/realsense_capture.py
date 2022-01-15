from realsense import RealsenseRGBDCapture
import klampt
from klampt import vis
from klampt.io import numpy_convert
from klampt.model import geometry
import time
from PIL import Image
import json

world = klampt.WorldModel()
obj = world.makeRigidObject("pc")
obj.geometry().setPointCloud(klampt.PointCloud())
obj.appearance().setSilhouette(0)  #otherwise dynamic meshed geometry for structured PC is super slow!
pc_klampt = klampt.PointCloud()
pc_klampt.addProperty('rgb')

capture = None
snapshot_counter = 0
def snapshot():
    global snapshot_counter,capture
    snapshot_counter += 1    
    color_fn = 'color_%04d.png'%snapshot_counter
    depth_fn = 'depth_%04d.png'%snapshot_counter
    depth_aligned_fn = 'depth_aligned_%04d.png'%snapshot_counter
    if snapshot_counter==1:
        c_json,d_json = capture.get_intrinsics_json()
        cintrinsics_fn = 'color_intrinsics.json'
        dintrinsics_fn = 'depth_intrinsics.json'
        with open(cintrinsics_fn,'w') as f:
            json.dump(c_json,f)
        with open(dintrinsics_fn,'w') as f:
            json.dump(d_json,f)
    print("Saved",color_fn,depth_fn)
    print("Depth image type",capture.depth_image.dtype)
    Image.fromarray(capture.depth_image).save(depth_fn)
    Image.fromarray(capture.color_image).save(color_fn)
    Image.fromarray(capture.depth_image_aligned).save(depth_aligned_fn)

bg_show = 'color'
show_points = False

def show_color():
    global bg_show
    bg_show = 'color'

def show_depth():
    global bg_show
    bg_show = 'depth'

def toggle_show_pc():
    global show_points
    show_points = not show_points
    if not show_points:
        obj.geometry().setPointCloud(klampt.PointCloud())

vis.addAction(snapshot,"Take snapshot",'s')
vis.addAction(show_color,"Show color",'c')
vis.addAction(show_depth,"Show depth",'d')
vis.addAction(toggle_show_pc,"Toggle points",'p')
vis.add("world",world)
vis.show()

try:
    capture = RealsenseRGBDCapture()
    while vis.shown():
        t0 = time.time()
        
        res = capture.get(show_points)
        if res is None:
            time.sleep(0.01)
            continue

        color,depth = res[0],res[1]
        if bg_show=='color':
            vis.scene().setBackgroundImage(color)
        else:
            vis.scene().setBackgroundImage(depth)
        t1 = time.time()

        if show_points and len(res)==3:
            (points,point_colors) = res[2]
            pc_klampt.setPoints(points.shape[0],points.flatten())
            pc_klampt.setProperties(0,[float(v) for v in point_colors])
            pc_klampt.setSetting("width",str(640))
            pc_klampt.setSetting("height",str(480))
        t3 = time.time()
        obj.geometry().setPointCloud(pc_klampt)

    exit(0)
#except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)
except Exception as e:
    import traceback
    print(e)
    traceback.print_exc()
    pass