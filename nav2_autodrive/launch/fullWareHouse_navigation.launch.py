## Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

PI = 3.141592

def generate_launch_description():

    # 현재 파이썬 파일(launch) 위치
    current_dir = os.path.dirname(os.path.realpath(__file__))

    use_sim_time = LaunchConfiguration("use_sim_time", default="True")

    map_dir = LaunchConfiguration(
        "map",
        default=os.path.join(current_dir, "..", "maps", "nav2_map.yaml"),
    )

    param_dir = LaunchConfiguration(
        "params_file",
        default=os.path.join(current_dir, "..", "params", "fullWareHouse_navigation_params.yaml"),
    )


    nav2_bringup_launch_dir = os.path.join(get_package_share_directory("nav2_bringup"), "launch")

    rviz_config_dir = os.path.join(current_dir, "..", "rviz2", "fullWareHouse_navigation.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value=map_dir, description="Full path to map file to load"),
            DeclareLaunchArgument(
                "params_file", default_value=param_dir, description="Full path to param file to load"
            ),
            DeclareLaunchArgument(
                "use_sim_time", default_value="true", description="Use simulation (Omniverse Isaac Sim) clock if true"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2_bringup_launch_dir, "rviz_launch.py")),
                launch_arguments={"namespace": "", 
                                  "use_namespace": "False", 
                                  "rviz_config": rviz_config_dir,
                                  "use_sim_time": use_sim_time
                                  }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([nav2_bringup_launch_dir, "/bringup_launch.py"]),
                launch_arguments={"map": map_dir, "use_sim_time": use_sim_time, "params_file": param_dir}.items(),
            ),

            # Launch the localization_converter.py node
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'nav2_autodrive', 'rtabmapTF_publisher'
                ],
                output='screen'
            ),

            # Launch the localization_converter.py node
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'nav2_autodrive', 'TimestampSync'
                ],
                output='screen'
            ),

            # Node(
            #     package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
            #     remappings=[('cloud_in', ['/sim_camera/depth_pcl_synched']),
            #                 ('scan', ['/scan'])],
            #     parameters=[{
            #         'target_frame': 'RSD455_align',
            #         'transform_tolerance': 0.01,
            #         'min_height': -0.4,
            #         'max_height': 1.5,
            #         'angle_min': -PI/2,  # -M_PI/2
            #         'angle_max': PI/2,  # M_PI/2
            #         'angle_increment': 0.0087,  # M_PI/360.0
            #         'scan_time': 0.3333,
            #         'range_min': 0.05,
            #         'range_max': 100.0,
            #         'use_inf': True,
            #         'inf_epsilon': 1.0,
            #         # 'concurrency_level': 1,
            #     }],
            #     name='pointcloud_to_laserscan'
            # )
        ]
    )
