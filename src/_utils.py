
def tartanair_settings(orb_feats):
    settings_config = f"""#--------------------------------------------------------------------------------------------
# Viewer Parameters (TARTANAIR)
#--------------------------------------------------------------------------------------------
# Viewer.on: 1 is ON, 0 is OFF
Viewer.on: 1

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 1
Viewer.LineSize: 1
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500

Viewer.ViewpointYtop: -4.0


#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: 320.0
Camera.fy: 320.0
Camera.cx: 320.0
Camera.cy: 240.0

Camera.k1: 0
Camera.k2: 0
Camera.p1: 0
Camera.p2: 0
Camera.k3: 0

Camera.width: 640
Camera.height: 480

# Camera frames per second 
Camera.fps: 25.0

# IR projector baseline times fx (aprox.)
Camera.bf: 80.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 0

# Close/Far threshold. Baseline times.
ThDepth: 40.0

# Deptmap values factor 
DepthMapFactor: 1.0

#--------------------------------------------------------------------------------------------
# Features extraction 
#--------------------------------------------------------------------------------------------

FeatureTrackerConfig.nFeatures: {orb_feats}
"""

    return settings_config
#--------------------------------------------------------------------------------------------
def tartanair_config(dataset, settings):
    name = dataset.split("_")[-1]
    slam_config = f"""DATASET:
  type: TARTANAIR_DATASET

TARTANAIR_DATASET:
  type: tartanair
  sensor_type: mono # Here, 'sensor_type' can be 'mono', 'stereo' or 'rgbd'
  base_path: {dataset}
  name: {name}
  settings: {settings}
  groundtruth_file: auto
  environment_type: outdoor

SYSTEM_STATE: 
  load_state: False                # flag to enable SLAM state reloading (map state + loop closing state) and relocalization
  folder_path: /home/isaac/dev/pyslam/results/slam_state  # default folder path (relative to root of this repository) where the system state is saved or reloaded


SAVE_TRAJECTORY:
  save_trajectory: True
  format_type: kitti             # supported formats: `tum`, `kitti`, `euroc`
  output_folder: /home/isaac/dev/pyslam/results/metrics # relative to pyslam root folder 
  basename: trajectory           # basename of the trajectory saving output

GLOBAL_PARAMETERS: 
  kLogsFolder: /home/isaac/pyslam_logs       # to redirect the prints of SLAM to the set folder
"""

    return slam_config



def kitti_settings(orb_feats, dataset):
    if dataset in ['00', '01', '02']:
      settings_config = f"""#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
# Viewer.on: 1 is ON, 0 is OFF
Viewer.on: 1

Viewer.KeyFrameSize: 0.6
Viewer.KeyFrameLineWidth: 2
Viewer.GraphLineWidth: 1
Viewer.PointSize: 2
Viewer.CameraSize: 0.7
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -100
Viewer.ViewpointZ: -0.1
Viewer.ViewpointF: 2000

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: 718.856
Camera.fy: 718.856
Camera.cx: 607.1928
Camera.cy: 185.2157

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: 1241
Camera.height: 376

# Camera frames per second 
Camera.fps: 10.0

# stereo baseline times fx
Camera.bf: 386.1448

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: 35

#--------------------------------------------------------------------------------------------
# Features extraction 
#--------------------------------------------------------------------------------------------

FeatureTrackerConfig.nFeatures: {orb_feats}

#--------------------------------------------------------------------------------------------
# Matching (Data Association)
#--------------------------------------------------------------------------------------------

#Matching.farPointsThreshold: 20.0    # meters (to exclude far points from matching when we have an outdoor scene)


#--------------------------------------------------------------------------------------------
# Keyframe generation
#--------------------------------------------------------------------------------------------
# KeyFrame.useFovCentersBasedGeneration: 1  # compute 3D fov centers of camera frames by using median depth and use their distances to control keyframe generation
# KeyFrame.maxFovCentersDistance: 0.1       # max distance between fov centers in order to generate a keyframe
"""
    elif dataset in ['03']:
        settings_config = f"""#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
# Viewer.on: 1 is ON, 0 is OFF
Viewer.on: 1

Viewer.KeyFrameSize: 0.6
Viewer.KeyFrameLineWidth: 2
Viewer.GraphLineWidth: 1
Viewer.PointSize: 2
Viewer.CameraSize: 0.7
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -100
Viewer.ViewpointZ: -0.1
Viewer.ViewpointF: 2000

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: 721.5377
Camera.fy: 721.5377
Camera.cx: 609.5593
Camera.cy: 172.854

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: 1241
Camera.height: 376

# Camera frames per second 
Camera.fps: 10.0

# stereo baseline times fx
Camera.bf: 387.5744

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: 40

#--------------------------------------------------------------------------------------------
# Features extraction 
#--------------------------------------------------------------------------------------------

FeatureTrackerConfig.nFeatures: {orb_feats}

#--------------------------------------------------------------------------------------------
# Matching (Data Association)
#--------------------------------------------------------------------------------------------

#Matching.farPointsThreshold: 20.0    # meters (to exclude far points from matching when we have an outdoor scene)

#--------------------------------------------------------------------------------------------
# Keyframe generation
#--------------------------------------------------------------------------------------------
# KeyFrame.useFovCentersBasedGeneration: 1  # compute 3D fov centers of camera frames by using median depth and use their distances to control keyframe generation
# KeyFrame.maxFovCentersDistance: 0.1       # max distance between fov centers in order to generate a keyframe
"""
    else:
        settings_config = f"""#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
# Viewer.on: 1 is ON, 0 is OFF
Viewer.on: 1

Viewer.KeyFrameSize: 0.6
Viewer.KeyFrameLineWidth: 2
Viewer.GraphLineWidth: 1
Viewer.PointSize: 2
Viewer.CameraSize: 0.7
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -100
Viewer.ViewpointZ: -0.1
Viewer.ViewpointF: 2000

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: 707.0912
Camera.fy: 707.0912
Camera.cx: 601.8873
Camera.cy: 183.1104

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: 1226
Camera.height: 370

# Camera frames per second 
Camera.fps: 10.0

# stereo baseline times fx
Camera.bf: 379.8145

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: 40

#--------------------------------------------------------------------------------------------
# Features extraction 
#--------------------------------------------------------------------------------------------

FeatureTrackerConfig.nFeatures: {orb_feats}

#--------------------------------------------------------------------------------------------
# Matching (Data Association)
#--------------------------------------------------------------------------------------------

#Matching.farPointsThreshold: 20.0    # meters (to exclude far points from matching when we have an outdoor scene)

#--------------------------------------------------------------------------------------------
# Keyframe generation
#--------------------------------------------------------------------------------------------
# KeyFrame.useFovCentersBasedGeneration: 1  # compute 3D fov centers of camera frames by using median depth and use their distances to control keyframe generation
# KeyFrame.maxFovCentersDistance: 0.1       # max distance between fov centers in order to generate a keyframe
"""

    return settings_config
#--------------------------------------------------------------------------------------------
def kitti_config(dataset, settings):
    slam_config = f"""DATASET:
  type: KITTI_DATASET

KITTI_DATASET:
  type: kitti
  sensor_type: mono # Here, 'sensor_type' can be 'mono' or 'stereo'
  base_path: /home/isaac/dev/pyslam/data/KITTI
  #
  name: '{dataset}'
  settings: {settings}
  is_color: False # do you have the color images for the kitti dataset? (image2 and image3 folders)
  groundtruth_file: auto

SYSTEM_STATE: 
  load_state: False                # flag to enable SLAM state reloading (map state + loop closing state) and relocalization
  folder_path: /home/isaac/dev/pyslam/results/slam_state  # default folder path (relative to root of this repository) where the system state is saved or reloaded


SAVE_TRAJECTORY:
  save_trajectory: True
  format_type: kitti             # supported formats: `tum`, `kitti`, `euroc`
  output_folder: /home/isaac/dev/pyslam/results/metrics # relative to pyslam root folder 
  basename: trajectory           # basename of the trajectory saving output

GLOBAL_PARAMETERS: 
  kLogsFolder: /home/isaac/pyslam_logs       # to redirect the prints of SLAM to the set folder
"""

    return slam_config

def euroc_settings(orb_feats):
    settings_config = f"""#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0


# Camera calibration and distortion parameters 
Camera.fx: 458.654
Camera.fy: 457.296
Camera.cx: 367.215
Camera.cy: 248.375

# from https://lamor.fer.hr/images/50036607/2022-cvisic-calib-ras.pdf
# Camera.fx: 461.946
# Camera.fy: 460.653
# Camera.cx: 366.502
# Camera.cy: 249.262

Camera.k1: -0.28340811
Camera.k2: 0.07395907
Camera.p1: 0.00019359
Camera.p2: 1.76187114e-05

Camera.width: 752
Camera.height: 480

# Camera frames per second 
Camera.fps: 20

# stereo baseline times fx
Camera.bf: 47.90639384423901

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: 35.0

#--------------------------------------------------------------------------------------------
# Features extraction 
#--------------------------------------------------------------------------------------------

FeatureTrackerConfig.nFeatures: {orb_feats}
"""
    return settings_config

def euroc_config(dataset, settings):
    slam_config = f"""DATASET:
  type: EUROC_DATASET  

EUROC_DATASET:
  type: euroc
  sensor_type: mono # Here, sensor_type can be 'mono' or 'stereo'
  base_path: /home/isaac/dev/pyslam/data/EuRoC
  # NOTE: The name is supposed to be one of the following: MH01, MH02, MH03, MH04, MH05, V101, V102, V103, V201, V202, V203
  name: {dataset}
  # 'settings' will be used when sensor_type: : 'mono'
  settings: {settings}
  # 'settings_stereo' will be used when sensor_type: 'stereo' (if available)
  #settings_stereo: settings/EuRoC_stereo.yaml
  associations: auto
  groundtruth_file: auto
  start_frame_id: 0

SYSTEM_STATE: 
  load_state: False                # flag to enable SLAM state reloading (map state + loop closing state) and relocalization
  folder_path: /home/isaac/dev/pyslam/results/slam_state  # default folder path (relative to root of this repository) where the system state is saved or reloaded


SAVE_TRAJECTORY:
  save_trajectory: True
  format_type: kitti             # supported formats: `tum`, `kitti`, `euroc`
  output_folder: /home/isaac/dev/pyslam/results/metrics # relative to pyslam root folder 
  basename: trajectory           # basename of the trajectory saving output

GLOBAL_PARAMETERS: 
  kLogsFolder: /home/isaac/pyslam_logs       # to redirect the prints of SLAM to the set folder
"""
    return slam_config