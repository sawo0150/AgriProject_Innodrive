import os
import subprocess

# # 데이터베이스 경로 설정
db_path = os.path.expanduser("~/.ros/rtabmap.db")
output_dir = "./clouddata"  # 추출된 클라우드 데이터를 저장할 디렉토리

# # 출력 디렉토리 생성
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # rtabmap-export 명령어 실행
# command = f"rtabmap-export --cloud --output_dir {output_dir} {db_path}"
# try:
#     print("Running command:", command)
#     subprocess.run(command, shell=True, check=True)
#     print(f"Cloud data exported successfully to {output_dir}")
# except subprocess.CalledProcessError as e:
#     print("Error during cloud data export:", e)

import open3d as o3d

# 저장된 PCD 파일 경로
# cloud_file = os.path.join(output_dir, "rtabmap_cloud.ply")  # 기본 이름: rtabmap_cloud.ply
cloud_file = os.path.join(output_dir, "cloud.ply")  # 기본 이름: rtabmap_cloud.ply

# PCD 파일 로드 및 시각화
if os.path.exists(cloud_file):
    print(f"Loading point cloud from {cloud_file}")
    pcd = o3d.io.read_point_cloud(cloud_file)
    
    # 포인트 클라우드 시각화
    print("Visualizing point cloud...")
    o3d.visualization.draw_geometries([pcd], window_name="RTAB-Map Point Cloud")
else:
    print(f"Point cloud file not found: {cloud_file}")
