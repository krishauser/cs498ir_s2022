# First import the library
import pyrealsense2 as rs
import numpy as np

class RealsenseRGBDCapture:
    """Simplifies depth capture from Realsense cameras"""
    def __init__(self,depth_aligned_to_color=True):
        # Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()

        # Configure streams
        self.config = rs.config()
            
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        if depth_aligned_to_color:
            self.alignment = rs.align(rs.stream.color)
        else:
            self.alignment = rs.align(rs.stream.depth)
        self.pc = rs.pointcloud()
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth scale",self.depth_scale,"(inv",1.0/self.depth_scale,")")

    def get_intrinsics_json(self):
        """Returns intrinsics (color_intrinsics,depth_intrinsics) as JSON-compatible
        dictionaries.
        """
        depth_profile = self.profile.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        color_profile = self.profile.get_stream(rs.stream.color) # Fetch stream profile for depth stream
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        cintrinsics_json = dict((k,getattr(color_intrinsics,k)) for k in ['width','height','fx','fy','ppx','ppy','model','coeffs'])
        dintrinsics_json = dict((k,getattr(depth_intrinsics,k)) for k in ['width','height','fx','fy','ppx','ppy','model','coeffs'])
        for k,v in cintrinsics_json.items():
            if not isinstance(v,(str,float,int,bool,list)):
                cintrinsics_json[k] = str(v)
        for k,v in dintrinsics_json.items():
            if not isinstance(v,(str,float,int,bool,list)):
                dintrinsics_json[k] = str(v)
        return cintrinsics_json,dintrinsics_json

    def get(self,want_pc=False):
        """Returns the (color,depth) pair.  If want_pc=True, returns the triple
        (color,depth,pc) where pc is a point cloud.  pc is a pair (positions,colors)
        with positions an Nx3 array and colors a length N array of packed integer RGB
        values.
        """
        frames = self.pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color: return

        self.depth_image = np.asanyarray(depth.get_data())
        self.color_image = np.asanyarray(color.get_data())

        aligned_frames = self.alignment.process(frames)
        color_aligned = aligned_frames.get_color_frame()
        depth_aligned = aligned_frames.get_depth_frame()
        self.color_image_aligned = np.asanyarray(color_aligned.get_data())
        self.depth_image_aligned = np.asanyarray(depth_aligned.get_data())

        if want_pc:
            self.pc.map_to(color_aligned)
            # Generate the pointcloud and texture mappings
            points = self.pc.calculate(depth_aligned)
            vtx = np.asarray(points.get_vertices())
            pure_point_cloud = np.zeros((640*480, 3))
            pure_point_cloud[:, 0] = -vtx['f0']
            pure_point_cloud[:, 1] = -vtx['f1']
            pure_point_cloud[:, 2] = -vtx['f2']
            color_channels = self.color_image_aligned.reshape(640*480, 3).astype(np.uint32)
            rgb = np.bitwise_or.reduce((np.left_shift(color_channels[:,0],16),np.left_shift(color_channels[:,1],8),color_channels[:,2]))
            return self.color_image_aligned,self.depth_image_aligned,(pure_point_cloud,rgb)
        return self.color_image_aligned,self.depth_image_aligned

