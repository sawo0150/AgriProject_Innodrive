
import os
import numpy as np
from scipy.spatial.transform import Rotation as R  # 쿼터니언→회전행렬

class localization_converter():
    def __init__(self):

        # 실행 중인 파일의 위치를 기반으로 경로 계산
        current_dir = os.path.dirname(os.path.realpath(__file__))
        plane_transform_file_path = os.path.join(current_dir, '../files', 'plane_transform_info.npz')
        plane_transform_file_path = os.path.abspath(plane_transform_file_path)  # 절대 경로로 변환

        # plane_params, transformation_matrix 로드
        self.plane_params, self.transformation_matrix = self.load_plane_transform_info(plane_transform_file_path)

    def load_plane_transform_info(self, file_path):
        """
        npz파일에서 plane_params, transformation_matrix 등을 읽어온다.
        예: plane_transform_info.npz 안에
            - plane_params = [a, b, c]
            - transformation_matrix = 4x4
        """
        try:
            data = np.load(file_path)
            plane_params = data['plane_params']  # shape=(3,)
            transformation_matrix = data['transformation_matrix']  # shape=(4,4)
            print(f"Loaded plane_transform_info from {file_path}.")
            return plane_params, transformation_matrix
        except Exception as e:
            print(f"Failed to load plane_transform_info.npz: {e}")
            raise

    def localization_pose_callback(self, msg):
        """Callback function for localization pose updates."""
        try:
            # 3D Pose
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w

            # 3D -> (평면 투영) -> 2D 위치 (u, v)
            u, v = self.project_position_to_2d(x, y, z)

            # 쿼터니언 -> 평면 좌표계에서의 yaw
            yaw_2d = self.compute_2d_yaw_in_plane(qx, qy, qz, qw)

            return (u, v, yaw_2d)
        except Exception as e:
            print(f"Error in localization_pose_callback: {e}")

    # ------------------------------------------------------------------------------
    # 1) 3D 위치 -> (a,b,c)평면 수직투영 -> R(회전행렬)로 평면좌표계 변환 -> (u,v)
    # ------------------------------------------------------------------------------
    def project_position_to_2d(self, x, y, z):
        """
        plane_params = [a, b, c]: 평면 식 1 = a*x + b*y + c*z
        transformation_matrix: 4x4 (상단 3x3은 world->plane 회전행렬)

        반환: (u,v)
        """
        a, b, c = self.plane_params
        # 1) 평면으로 수직투영
        denom = a*a + b*b + c*c
        t = (1.0 - (a*x + b*y + c*z)) / denom
        px, py, pz = x + a*t, y + b*t, z + c*t

        # 2) world->plane 회전(R^T) (4x4사용하되, z축 성분은 버릴 예정)
        Rmat = self.transformation_matrix[:3, :3]  # 3x3
        local_3d = Rmat.T @ np.array([px, py, pz])
        # => local_3d = (X_plane, Y_plane, Z_plane)
        # 보통 평면좌표계에서는 Z_plane ~ 0 근처

        return local_3d[0], local_3d[1]

    # ------------------------------------------------------------------------------
    # 2) 쿼터니언 -> 월드 전방벡터 -> 평면 좌표계 회전 -> 2D yaw 계산
    # ------------------------------------------------------------------------------
    def compute_2d_yaw_in_plane(self, qx, qy, qz, qw):
        """
        1) 쿼터니언 -> 회전행렬 -> 로컬 [1,0,0]을 월드 전방벡터로 변환
        2) world->plane 회전(R^T) 적용
        3) 2D에서 atan2
        """
        # (1) 로컬 +X축 -> 월드 전방벡터
        rot = R.from_quat([qx, qy, qz, qw])  # scipy는 [x,y,z,w] 순서
        forward_world = rot.apply([0.0, 0.0, -1.0])  # shape=(3,)

        # (2) plane 좌표계로 회전
        Rmat = self.transformation_matrix[:3, :3]  # world->plane
        forward_plane_3d = Rmat.T @ forward_world
        fx, fy = forward_plane_3d[0], forward_plane_3d[1]

        # (3) 2D yaw
        yaw = np.arctan2(fy, fx)
        return yaw


def main(args=None):
    node = localization_converter()

if __name__ == '__main__':
    main()
