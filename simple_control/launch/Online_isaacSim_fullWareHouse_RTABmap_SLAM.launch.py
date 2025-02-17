from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        # Launch the keyboard_controller.py node
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'simple_control', 'keyboard_controller'
            ],
            output='screen'
        ),

        # Launch the rtabmap.launch.py file with arguments
        ExecuteProcess(
            cmd=[
                'ros2', 'launch', 'simple_control', 'my_rtabmap_params.launch.py',
                'rtabmap_args:=--delete_db_on_start',
                'subscribe_rgbd:=true',
                'rgbd_sync:=true',
                'approx_synch:=true',
                'rgb_topic:=/sim_camera/rgb',
                'depth_topic:=/sim_camera/depth',
                'camera_info_topic:=/sim_camera/camera_info',
                'imu_topic:=/sim_camera/sim_imu',
                'frame_id:=RSD455',
                'approx_sync:=true',
                'wait_imu_to_init:=false',
                'use_odom_gravity:=true'
                'qos:=1',
                'rviz:=true'
            ],
            output='screen'
        )
    ])