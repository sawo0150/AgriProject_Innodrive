#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct
import open3d as o3d
import os

class PlaneExtractorOnce(Node):
    def __init__(self):
        super().__init__('plane_extractor_once')

        # 파라미터로 저장할 파일명 받기 (기본값은 plane_projection_matrix.npz)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        default_npz_path = os.path.join(current_dir, '../src', 'plane_projection_matrix.npz')
        default_npz_path = os.path.abspath(default_npz_path)
        self.plane_matrix_file = default_npz_path

        # 구독: 딱 한 번만 받고 종료할 것이므로, latch처럼 동작
        self.sub = self.create_subscription(
            PointCloud2,
            '/sim_camera/depth_pcl',
            self.depth_callback,
            10
        )

        self.received_once = False

        self.get_logger().info("PlaneExtractorOnce node started. Waiting for one PointCloud2 message...")

    def depth_callback(self, msg: PointCloud2):
        # 이미 한 번 받았으면 더 이상 처리하지 않음
        if self.received_once:
            return
        self.received_once = True

        self.get_logger().info("Received PointCloud2. Extracting plane...")

        # 1) PointCloud2 → Nx3 array
        points_3d = self.convert_pointcloud2_to_xyz(msg)
        if len(points_3d) < 100:
            self.get_logger().warn("PointCloud size < 100. Not enough for plane extraction.")
            return

        # 2) RANSAC 평면 추정
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        distance_threshold = 0.01  # 상황에 맞게 조정
        ransac_n = 3
        num_iterations = 1000

        plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        # plane_model = [a, b, c, d], 식: aX + bY + cZ + d=0

        [a, b, c, d] = plane_model
        normal = np.array([a, b, c], dtype=np.float32)
        D = -d  # n^T x = D 형태로 쓸 때
        denom = normal.dot(normal)

        self.get_logger().info(f"Plane model: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0, inliers={len(inliers)}")

        # 3) 평면 사영행렬(4x4) 계산
        #    P = [[I - (nn^T)/(n^T n)]   [ (D/(n^T n))*n ]
        #         [       0  0  0       1                  ]]
        I3 = np.eye(3, dtype=np.float32)
        nnT = np.outer(normal, normal)
        M = I3 - (nnT / denom)
        v = (D / denom) * normal

        proj_mat = np.eye(4, dtype=np.float32)
        proj_mat[:3, :3] = M
        proj_mat[:3, 3]  = v

        # 4) 파일로 저장
        np.savez(self.plane_matrix_file, plane_matrix=proj_mat)
        self.get_logger().info(f"Saved plane projection matrix to {self.plane_matrix_file}.")

        # 5) 시각화 준비
        #   (a) 원본 pointcloud
        #   (b) inlier 점들 color
        #   (c) 임의 한 점 + 투영된 점

        # (a) 전체 PCL
        # 이미 pcd를 만들었음 (pcd.points)
        # (b) inlier vs outlier 색상 구분
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])      # 빨간색
        outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8]) # 회색

        # (c) 임의 한 점 골라보기 (inlier 중 하나 pick)
        sample_idx = inliers[len(inliers)//2]  # 중간 정도 인덱스
        # sample_point = np.array(pcd.points[sample_idx])  # shape=(3,)
        sample_point = np.array([2,-2,15])  # shape=(3,)
        print(pcd.points[sample_idx])

        # -> 사영
        homo_orig = np.array([sample_point[0], sample_point[1], sample_point[2], 1.0], dtype=np.float32)
        proj_homo = proj_mat @ homo_orig
        projected_point = proj_homo[:3] / proj_homo[3]

        self.get_logger().info(f"Sample point (original) = {sample_point}")
        self.get_logger().info(f"Projected point on plane = {projected_point}")

        # Open3D에서 점 두 개를 sphere 등으로 표시
        sphere_orig = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere_orig.compute_vertex_normals()
        sphere_orig.paint_uniform_color([0, 1, 0]) # 초록
        sphere_orig.translate(sample_point)

        sphere_proj = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere_proj.compute_vertex_normals()
        sphere_proj.paint_uniform_color([0, 0, 1]) # 파랑
        sphere_proj.translate(projected_point)

        # 6) 시각화
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, sphere_orig, sphere_proj])

        # 시각화 창 닫으면, Node 종료
        self.get_logger().info("Done. Shutting down node...")
        rclpy.shutdown()

    def convert_pointcloud2_to_xyz(self, cloud_msg: PointCloud2) -> np.ndarray:
        """
        sensor_msgs/PointCloud2 -> Nx3 float array
        """
        points = []
        point_step = cloud_msg.point_step
        data = cloud_msg.data
        # 예시로, x=offset0, y=4, z=8 (datatype float32)
        for i in range(cloud_msg.height * cloud_msg.width):
            start_i = i * point_step
            x = struct.unpack_from('f', data, start_i + 0)[0]
            y = struct.unpack_from('f', data, start_i + 4)[0]
            z = struct.unpack_from('f', data, start_i + 8)[0]
            # 유효 depth 필터 (NaN / INF 등 배제)
            if abs(x) < 1000 and abs(y) < 1000 and abs(z) < 1000:
                points.append([x, y, z])
        return np.array(points, dtype=np.float32)

def main(args=None):
    rclpy.init(args=args)
    node = PlaneExtractorOnce()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
