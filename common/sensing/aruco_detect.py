import os
import sys
import cv2
import numpy as np
import aruco
import pkg_resources  # part of setuptools
import glob
from PIL import Image
from klampt.io import loader
from klampt.math import se3

if __name__ == '__main__':
    # load board and camera parameters
    camparam = aruco.CameraParameters()
    camparam.readFromXMLFile(os.path.join(os.path.dirname(__file__), "realsense_color_640x480.yml"))

    # create detector and get parameters
    detector = aruco.MarkerDetector()
    detector.setDictionary("fourbyfour.dict")
    params = detector.getParameters()

    # load video
    for fn in glob.glob("extrinsic_calibration/color_*.png"):
        print("Loading image",fn)
        img = Image.open( fn )
        img.load()
        frame = np.asarray( img)
        markers = detector.detect(frame)
        print("Detected",len(markers),"Markers")

        ind = fn.rfind('.')
        count = fn[ind-4:ind]
        outfn = 'extrinsic_calibration/aruco_{}.xform'.format(count)
        with open(outfn,'w') as f:
            for marker in markers:
                # print marker ID and point positions
                print("Marker: {:d}".format(marker.id))
                for i, point in enumerate(marker):
                    print("\t{:d} {}".format(i, str(point)))
                marker.draw(frame, np.array([255, 255, 255]), 2)
                print("center: {}".format(marker.getCenter()))
                # print("contour: {}".format(marker.contourPoints))
                points3d = marker.get3DPoints()
                print("3D points: {:}".format(points3d))
                # calculate marker extrinsics for marker size of 15cm
                marker.calculateExtrinsics(0.15, camparam)
                mtx = marker.getTransformMatrix()
                print("M: {}".format(mtx))
                #print("R: {}".format(marker.Rvec))
                #print("T: {}".format(marker.Tvec))
                # print("Marker extrinsics:\n{}\n{}".format(marker.Tvec, marker.Rvec))
                aruco.CvDrawingUtils.draw3dAxis(frame, camparam, marker.Rvec, marker.Tvec, .1)
                print("detected ids: {}".format(", ".join(str(m.id) for m in markers)))
                f.write(loader.write(se3.from_homogeneous(mtx),'RigidTransform'))

        # add aruco version to frame
        y, x, c = frame.shape
        text = "aruco {}".format(pkg_resources.require("aruco")[0].version)
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 2
        thickness = 2
        cv2.putText(frame, text, (15, y - 15), font, font_scale, (255, 255, 255), thickness,
                    cv2.LINE_AA)

        # show frame
        if 'DISPLAY' in os.environ.keys():
            cv2.imshow("frame", frame)
            cv2.waitKey(100)
        else:
            print("No display!")
