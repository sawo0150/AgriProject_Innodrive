from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        # Launch the localization_converter.py node
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'simple_control', 'localization_converter'
            ],
            output='screen'
        ),

        # Launch the rtabmap.launch.py file with arguments
        ExecuteProcess(
            cmd=[
                'ros2', 'launch', 'simple_control', 'my_rtabmap_params.launch.py',
                'localization:=true',
                'subscribe_rgbd:=true',
                'rgbd_sync:=true',
                'approx_synch:=true',
                'rgb_topic:=/sim_camera/rgb',
                'depth_topic:=/sim_camera/depth',
                'camera_info_topic:=/sim_camera/camera_info',
                # 'imu_topic:=/sim_camera/sim_imu',
                'frame_id:=RSD455_align',
                # 'odom_frame_id:=rtabmap_odom',
                # 'map_frame_id:=rtabmap_map',
                'approx_sync:=true',
                'wait_imu_to_init:=false',
                'qos:=1',
                'Rtabmap_DetectionRate:=3.0',
                'RGBD_MaxLoopClosureDistance:=2.0',
                'RGBD_ProximityMaxGraphDepth:=5',
                # 'rviz:=true',
                'use_sim_time:=true',
            ],
            output='screen'
        )
    ])