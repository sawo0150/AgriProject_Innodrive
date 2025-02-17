from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'nav2_autodrive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 3) launch 폴더 내의 모든 .launch.py
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='swpants05',
    maintainer_email='swpants05@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 명령어 = 모듈.파이썬파일:진입함수
            'rtabmapTF_publisher = nav2_autodrive.rtabmapTF_publisher:main',
            'TimestampSync = nav2_autodrive.TimestampSync:main',
        ],
    },
)
