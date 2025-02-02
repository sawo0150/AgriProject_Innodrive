import numpy as np
import math
import time
import os

# OpenCV
import cv2

# plane 변환에서 사용한 scipy (쿼터니언, 회전행렬 등)
from scipy.spatial.transform import Rotation as R

class WaypointVisualizer():
    def __init__(self):
        # ---------------------------
        # 1) plane_transform_info 로드
        # ---------------------------
        current_dir = os.path.dirname(os.path.realpath(__file__))
        default_npz_path = os.path.join(current_dir, '../files', 'plane_transform_info.npz')
        default_npz_path = os.path.abspath(default_npz_path)
        self.plane_params, self.transformation_matrix = self.load_plane_transform_info(default_npz_path)
        # plane_params = [a, b, c],   plane 식: aX + bY + cZ = 1
        # transformation_matrix: 4x4  (상단 3x3은 map->plane or plane->map 회전)

        # ---------------------------
        # 1-2) 추가로, 'plane_projection_matrix.npz' (Depth RANSAC 결과) 불러오기
        #      => 카메라 좌표계에서 "평면으로 강제 사영"하는 4x4 행렬
        # ---------------------------
        self.plane_proj_mat = None
        plane_proj_file = os.path.join(current_dir, '../files', 'plane_projection_matrix.npz')
        plane_proj_file = os.path.abspath(plane_proj_file)
        if os.path.exists(plane_proj_file):
            try:
                data2 = np.load(plane_proj_file)
                self.plane_proj_mat = data2['plane_matrix']  # shape=(4,4)
                print(f"Loaded plane_projection_matrix from {plane_proj_file}")
            except Exception as e:
                print(f"Failed to load plane_projection_matrix.npz: {e}")
        else:
            print(f"{plane_proj_file} does not exist! Plane projection will be unavailable.")

        # # ---------------------------
        # # 2) Camera Info 구독 → 내부 파라미터(K, D)
        # # ---------------------------
        # self.camera_info_sub = self.create_subscription(
        #     CameraInfo,
        #     '/sim_camera/camera_info',
        #     self.camera_info_callback,
        #     10
        # )
        self.camera_matrix = None
        self.dist_coeffs   = None
        self.img_width     = 0
        self.img_height    = 0

        # # ---------------------------
        # # 3) RGB Image 구독
        # #    - 최신 이미지를 저장해두었다가,
        # #      /rtabmap/localization_pose 콜백에서 시각화할 때 사용
        # # ---------------------------
        # self.image_sub = self.create_subscription(
        #     Image,
        #     '/sim_camera/rgb',
        #     self.image_callback,
        #     10
        # )
        self.latest_image = None  # 매 프레임 갱신

        # # ---------------------------
        # # 4) Localization Pose 구독
        # #    => 여기서 "Camera의 map 상의 Pose"를 받고,
        # #       매번 이 콜백에서 Waypoint → 영상 투영 시각화 실행
        # # ---------------------------
        # self.localization_sub = self.create_subscription(
        #     PoseWithCovarianceStamped,
        #     '/rtabmap/localization_pose',
        #     self.localization_pose_callback,
        #     10
        # )

        # # ---------------------------
        # # 5) ActionServer (NavigateWaypoints)
        # #    => Waypoint(Plane 2D 좌표) 받아서 저장
        # # ---------------------------
        # self._action_server = ActionServer(
        #     self,
        #     NavigateWaypoints,
        #     'navigate_waypoints',
        #     execute_callback=self.execute_callback,
        #     goal_callback=self.goal_callback,
        #     cancel_callback=self.cancel_callback
        # )

        # Waypoints (plane 좌표계, Pose2D[] 형태) 보관

        # "현재 카메라 pose (map 좌표)" 보관
        self.current_cam_pose = None  # (x, y, z, qx, qy, qz, qw)
        self.T_map_cam = None

    # ---------------------------
    # plane_transform_info 로드
    # ---------------------------
    def load_plane_transform_info(self, file_path):
        """
        npz 파일에서 plane_params=[a,b,c], transformation_matrix(4x4) 읽기
        """
        try:
            data = np.load(file_path)
            plane_params = data['plane_params']            # shape=(3,)
            transformation_matrix = data['transformation_matrix']  # shape=(4,4)
            print(f"Loaded plane_transform_info from {file_path}.")
            return plane_params, transformation_matrix
        except Exception as e:
            print(f"Failed to load plane_transform_info.npz: {e}")
            raise

    # =====================================================
    #    1) Camera Info 콜백 (내부 파라미터 설정)
    # =====================================================
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.img_width  = msg.width
            self.img_height = msg.height
            k = np.array(msg.k, dtype=np.float32).reshape(3,3)
            self.camera_matrix = k
            self.dist_coeffs   = np.array(msg.d, dtype=np.float32)
            print(f"Camera info received: {self.img_width}x{self.img_height}")

    # =====================================================
    #    2) Image 콜백 (최신 이미지만 저장)
    # =====================================================
    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # print.info("Camera Image received")
        # 여기서는 시각화 X, 단순 저장만

    # =====================================================
    #    3) localization_pose 콜백
    #       => 여기서 Waypoints 투영 후, 이미지 표시
    # =====================================================
    def localization_pose_callback(self, msg):
        """
        /rtabmap/localization_pose가 들어올 때마다,
        - camera의 map 상 pose 업데이트
        - latest_image에 waypoint 투영해서 표시 ( = 시각화 )
        """
        # 3-1) camera pose(map에서)
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.current_cam_pose = (p.x, p.y, p.z, o.x, o.y, o.z, o.w)



        # 3-4) Waypoint(plane 2D) → map 3D → camera 3D → image(u,v)
        self.T_map_cam = self.compute_map_to_camera_transform(self.current_cam_pose)
    
    def cal_projectedWaypoints(self, wp_x, wp_y):

        if len(wp_x) == 0 or len(wp_y) == 0:
            # waypoint 없으면 그냥 패스(이미지 표시만)
            return
        
        # 3-2) camera_matrix, latest_image, waypoints_plane_2d 모두 준비됐는지 확인
        if  self.camera_matrix is None:
            return
        
        projected_waypoints =[]
        for (x,y) in zip(wp_x, wp_y):
            # wp: Pose2D (x=..., y=..., theta=...)라고 가정
            # => plane 2D (u,v)
            plane_u = x
            plane_v = y

            # (a) plane -> map (x,y,z)
            x_map, y_map, z_map = self.plane_to_map(plane_u, plane_v)

            # (b) map -> camera
            point_map = np.array([[x_map],[y_map],[z_map],[1.0]], dtype=np.float32)
            point_cam_homo = self.T_map_cam @ point_map
            Xc, Yc, Zc, _ = point_cam_homo.flatten()

            # Zc <= 0 이면 카메라 앞이 아님
            if Zc <= 0:
                projected_waypoints.append([None, None])
                continue

            # (c) camera -> image (opencv projectPoints)
            obj_pts = np.array([[Xc, Yc, Zc]], dtype=np.float32)
            rvec = np.zeros((3,1), dtype=np.float32)
            tvec = np.zeros((3,1), dtype=np.float32)
            imgpts, _ = cv2.projectPoints(obj_pts, rvec, tvec,
                                          self.camera_matrix, self.dist_coeffs)
            (px, py) = imgpts[0][0]
            projected_waypoint =[]
            # 이미지 범위 내라면 표시
            if 0 <= int(px) < self.img_width and 0 <= int(py) < self.img_height:
                projected_waypoint.append((px, py))
            else:
                projected_waypoint.append(None)

            # -------- (2) 평면 사영(카메라 좌표에서 한 번 더 사영) → 이미지
            if self.plane_proj_mat is not None:
                point_cam_homo2 = self.plane_proj_mat @ point_cam_homo  # 4x1
                Xc_p, Yc_p, Zc_p, Wc_p = point_cam_homo2
                # normalize
                if abs(Wc_p) > 1e-9:
                    Xc_p /= Wc_p
                    Yc_p /= Wc_p
                    Zc_p /= Wc_p

                if Zc_p > 0:
                    obj_pts2 = np.array([[Xc_p, Yc_p, Zc_p]], dtype=np.float32)
                    imgpts2, _ = cv2.projectPoints(obj_pts2, rvec, tvec,
                                                   self.camera_matrix,
                                                   self.dist_coeffs)
                    (px2, py2) = imgpts2[0][0]

                    if 0 <= int(px2) < self.img_width and 0 <= int(py2) < self.img_height:
                        projected_waypoint.append((px2, py2))
            if len(projected_waypoint)<2:
                projected_waypoint.append(None)
            projected_waypoints.append(projected_waypoint)

        return projected_waypoints
        #return 되는 구조: [사영X 점 (x,y), 상영 O 점 (x,y)]

    # -----------------------------------------------------
    # plane_to_map(u,v) : 평면 좌표(plane_u, plane_v)를
    #                     map상의 (x,y,z)로 역변환
    # -----------------------------------------------------
    def plane_to_map(self, plane_u, plane_v):
        """
        <project_position_to_2d>의 반대 연산 (plane->world).
        plane_params = [a,b,c], => 평면식: aX + bY + cZ = 1
        transformation_matrix(4x4) 중 3x3은 "map->plane" or "plane->map" 회전.
        원 코드에서:
         map->plane 변환시: local_3d = R^T @ (Px,Py,Pz)
        => plane->map 시: (Px,Py,Pz) = R @ local_3d

        알고리즘:
         1) local_3d = [plane_u, plane_v, 0] (Z_plane=0 근처)
         2) (X0,Y0,Z0) = R @ local_3d
         3) t = (1 - (aX0 + bY0 + cZ0)) / (a^2 + b^2 + c^2)
         4) X = X0 + a*t,  Y = Y0 + b*t,  Z = Z0 + c*t
        => (X,Y,Z)가 실제 map 상 평면식 aX+bY+cZ=1 만족
        """
        a, b, c = self.plane_params
        Rmat = self.transformation_matrix[:3,:3]  # map->plane? or plane->map?

        # 여기서 원 코드 "project_position_to_2d"는 map->plane 시 Rmat.T를 곱함
        # => plane->map 시에는 Rmat = (Rmat^T)^T
        local_3d_plane = np.array([plane_u, plane_v, 0.0], dtype=np.float32)
        X0, Y0, Z0 = Rmat @ local_3d_plane

        denom = a*a + b*b + c*c
        t = (1.0 - (a*X0 + b*Y0 + c*Z0)) / denom
        X = X0 + a*t
        Y = Y0 + b*t
        Z = Z0 + c*t
        # print(t)
        return (X, Y, Z)

    # -----------------------------------------------------
    # map->camera 변환행렬(4x4) 계산
    #  => /rtabmap/localization_pose를 "camera가 map에서 (x,y,z,qx,qy,qz,qw)로 존재"
    #     라고 해석하면,  camera->map 행렬 T_cam_map 을 만들고, inverse 하여 map->camera
    # -----------------------------------------------------
    def compute_map_to_camera_transform(self, cam_pose):
        x, y, z, qx, qy, qz, qw = cam_pose
        # camera->map (4x4)
        T_cam_map = np.eye(4, dtype=np.float32)
        # 회전
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()  # shape=(3,3)
        T_cam_map[:3,:3] = rot
        # 병진
        T_cam_map[0,3] = x
        T_cam_map[1,3] = y
        T_cam_map[2,3] = z

        # map->camera = inverse
        T_map_cam = np.linalg.inv(T_cam_map)
        return T_map_cam


def main(args=None):
    node = WaypointVisualizer()