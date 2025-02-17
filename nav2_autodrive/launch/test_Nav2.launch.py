import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    # 현재 파이썬 파일(launch) 위치
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(current_dir, '..', 'config')

    params_file = os.path.join(config_dir, 'nav2_test_params.yaml')  # <-- Nav2 파라미터 파일
    map_file    = os.path.join(config_dir, 'nav2_test_map.yaml')     # <-- 맵 메타파일

    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    bringup_launch = os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup_launch),
            launch_arguments={
                'map': map_file,
                'params_file': params_file,
                'use_sim_time': 'True',
                'autostart': 'True'
            }.items()
        )
    ])
