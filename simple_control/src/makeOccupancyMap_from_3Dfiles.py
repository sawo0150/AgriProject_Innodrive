import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R
from PIL import Image   # 추가 (이미지 저장용)

###############################################################################
# 1. Camera Pose 데이터 로드 (쿼터니언 포함)
###############################################################################
def load_camera_poses_with_quat(file_path):
    """
    poses.txt 파일 예시 구조:
    #timestamp  x   y   z   qx   qy   qz   qw
    23.950001  -0.0 -0.0 -0.0  -0.0  0.000001  0.000000  1.000000
    ...
    -> (x, y, z, qx, qy, qz, qw) 형태로 로드
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # usecols=(1,2,3,4,5,6,7): 첫번째 컬럼(#timestamp)은 스킵
    data = np.loadtxt(file_path, skiprows=0, usecols=(1, 2, 3, 4, 5, 6, 7))
    if data.shape[0] < 3:
        raise ValueError("Not enough camera poses to compute a plane. At least 3 poses are required.")

    return data  # shape: (N, 7) = [x, y, z, qx, qy, qz, qw]


###############################################################################
# 2. 평면 추출 (Least Squares)
###############################################################################
def fit_plane(points):
    """
    points: (N,3) 형태. 1 = a*x + b*y + c*z 형태의 평면을 리니어하게 맞춤
    반환: plane_params = [a,b,c]
    """
    if points.shape[0] < 3:
        raise ValueError("Not enough points to fit a plane. At least 3 points are required.")

    def plane_equation(params, pts):
        a, b, c = params
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return a*x + b*y + c*z - 1

    def error(params, pts):
        return plane_equation(params, pts)

    initial_guess = [1, 1, 1]  # 초기값
    params, success = leastsq(error, initial_guess, args=(points,))
    if not success:
        raise ValueError("Plane fitting failed. Please check the input data.")
    return params  # [a, b, c]


###############################################################################
# 3. Point Cloud 로드 & 필터링
###############################################################################
def load_point_cloud(ply_file):
    if not os.path.exists(ply_file):
        raise FileNotFoundError(f"File not found: {ply_file}")
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("Point cloud is empty.")
    return pcd, points

def filter_points_within_plane(points, plane_params, height=1.0):
    """
    1 <= distance <= 1+height 범위 내에 있는 점들만 필터링
    distance = (a*x + b*y + c*z - 1)/||(a,b,c)||
    """
    a, b, c = plane_params
    normal_len = np.sqrt(a**2 + b**2 + c**2)
    distances = (a*points[:,0] + b*points[:,1] + c*points[:,2] - 1)/normal_len
    mask = (1 <= distances) & (distances <= 1+height)
    filtered_points = points[mask]
    if filtered_points.size == 0:
        raise ValueError("No points found within the specified plane range.")
    return filtered_points

###############################################################################
# 4. 평면 Mesh 생성 (시각화용)
###############################################################################
def generate_plane_mesh(plane_params, x_range, y_range, grid_size=0.1):
    a, b, c = plane_params
    x = np.arange(x_range[0], x_range[1], grid_size)
    y = np.arange(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    # 1 = aX + bY + cZ -> Z = (1 - aX - bY)/c
    zz = (1 - a*xx - b*yy)/c
    plane_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    return plane_points

###############################################################################
# 5. 3D 시각화
###############################################################################
def visualize_with_plane_and_poses(original_pcd, filtered_points, plane_points, poses_xyz):
    """
    - original_pcd: 원본 PCL
    - filtered_points: 필터링된 PCL (numpy)
    - plane_points: 평면 mesh
    - poses_xyz: (N,3) 형태, 포즈의 x,y,z만 시각화 (쿼터니언은 반영X)
    """
    # 원래 포인트 클라우드(회색)
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # 필터링된 포인트(빨강)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.paint_uniform_color([1, 0, 0])
    # 평면(초록)
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
    plane_pcd.paint_uniform_color([0, 1, 0])
    # 포즈(파랑)
    pose_pcd = o3d.geometry.PointCloud()
    pose_pcd.points = o3d.utility.Vector3dVector(poses_xyz)
    pose_pcd.paint_uniform_color([0, 0, 1])
    # 시각화
    o3d.visualization.draw_geometries(
        [original_pcd, filtered_pcd, plane_pcd, pose_pcd],
        window_name="Point Cloud with Plane and Poses",
        width=800, height=600,
    )

###############################################################################
# 6. 평면 투영(회전행렬 생성) + 투영행렬 재사용
###############################################################################
def project_and_transform_points_to_plane(points, plane_params):
    """
    1) points를 plane_params( a, b, c ) 평면(1=ax+by+cz)에 수직 투영
    2) world->plane 회전행렬(3x3)을 구해 4x4 변환행렬 작성
    3) 투영된 점들을 plane좌표계(2D)로 변환하여 반환
    """
    a, b, c = plane_params

    # 평면 법선벡터
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # 평면 x,y축 결정 (z축 = normal)
    x_axis = np.cross(normal, np.array([0, 0, 1], dtype=np.float64))
    if np.linalg.norm(x_axis) == 0:
        x_axis = np.array([1, 0, 0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 회전행렬 R (world->plane)
    rotation_matrix = np.vstack((x_axis, y_axis, normal)).T  # 3x3
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix

    projected_points_2d = []
    for x, y, z in points:
        # (1) 평면에 수직투영
        t = (1 - (a*x + b*y + c*z)) / (a*a + b*b + c*c)
        px, py, pz = x + a*t, y + b*t, z + c*t
        # (2) world->plane 회전 (R^T)
        local_3d = rotation_matrix.T @ np.array([px, py, pz])
        projected_points_2d.append(local_3d[:2])

    return np.array(projected_points_2d), transformation_matrix


def project_points_onto_plane_using_matrix(points, plane_params, transformation_matrix):
    """
    이미 구해둔 transformation_matrix[:3,:3] (R) 재사용 + plane_params로 수직투영
    """
    a, b, c = plane_params
    Rmat = transformation_matrix[:3, :3]  # world->plane

    projected_points_2d = []
    for x, y, z in points:
        # 1) 평면 수직투영
        t = (1 - (a*x + b*y + c*z)) / (a*a + b*b + c*c)
        px, py, pz = x + a*t, y + b*t, z + c*t
        # 2) 이미 구한 R로 평면 좌표계로 회전
        local_3d = Rmat.T @ np.array([px, py, pz])
        projected_points_2d.append(local_3d[:2])

    return np.array(projected_points_2d)

###############################################################################
# 7. Occupancy Map 생성
###############################################################################
def create_occupancy_map(points_2d, grid_size=0.1, threshold=5):
    """
    points_2d: (N,2)
    grid_size: 맵 해상도
    threshold: 해당 셀에 점이 threshold 개수 이상이면 occupied
    리턴:
      - occupancy_map: 2D numpy, shape=(num_x, num_y), 값은 0(빈 공간) 또는 1(차지)
      - x_bins, y_bins
    """
    x_min, x_max = np.min(points_2d[:,0]), np.max(points_2d[:,0])
    y_min, y_max = np.min(points_2d[:,1]), np.max(points_2d[:,1])
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)

    occupancy_map = np.zeros((len(x_bins), len(y_bins)), dtype=int)
    for x, y in points_2d:
        x_idx = np.searchsorted(x_bins, x) - 1
        y_idx = np.searchsorted(y_bins, y) - 1
        if 0 <= x_idx < len(x_bins) and 0 <= y_idx < len(y_bins):
            occupancy_map[x_idx, y_idx] += 1

    occupancy_map = (occupancy_map >= threshold).astype(int)
    return occupancy_map, x_bins, y_bins

###############################################################################
# 8. 2D 맵 시각화 + 포즈 점 & 방향(화살표) 함께 표시
###############################################################################
def quaternion_to_forward_vector(qx, qy, qz, qw):
    """
    로컬 X축([1,0,0])이 전방이라고 가정했을 때,
    쿼터니언으로 회전시켜 월드좌표계 전방벡터를 구함.
    """
    rot = R.from_quat([qx, qy, qz, qw])  # scipy: 순서 [x, y, z, w]
    forward_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)  
    # ↑ 여기서 [0,0,1]이 전방이 되도록 가정했지만,
    #   로봇 모델에 따라 X 또는 Z가 전방일 수 있음. 필요 시 수정.
    forward_world = rot.apply(forward_local)
    return forward_world  # shape=(3,)

def visualize_occupancy_map_with_poses(
    map_data, x_bins, y_bins,
    plane_params, transformation_matrix,
    poses_3d_quat,
    output_file="occupancy_map.png"
):
    """
    - map_data: (num_x, num_y) 형태의 occupancy map (0 or 1)
    - x_bins, y_bins: 각 축의 bin 경계
    - plane_params, transformation_matrix: 평면 투영 정보
    - poses_3d_quat: (N,7) = [x,y,z,qx,qy,qz,qw]
    """
    plt.figure(figsize=(10,8))
    # occupancy map 시각화
    #   map_data.shape=(Xbins, Ybins), but imshow expects [row, col] = [y, x]
    #   => 따라서 transpose + origin='lower' 로 표시
    plt.imshow(
        map_data.T, origin='lower', cmap='gray',
        extent=(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1])
    )
    plt.colorbar(label="Occupancy")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Occupancy Map with 2D Poses & Directions")

    Rmat = transformation_matrix[:3, :3]

    for pose in poses_3d_quat:
        x, y, z, qx, qy, qz, qw = pose
        # 1) 위치 수직투영 -> plane 좌표계
        t = (1 - (plane_params[0]*x + plane_params[1]*y + plane_params[2]*z)) / (plane_params**2).sum()
        px = x + plane_params[0]*t
        py = y + plane_params[1]*t
        pz = z + plane_params[2]*t
        local_pos_3d = Rmat.T @ np.array([px, py, pz])
        pos_2d = local_pos_3d[:2]

        # 2) 전방벡터
        fwd_world = quaternion_to_forward_vector(qx, qy, qz, qw)
        fwd_plane_3d = Rmat.T @ fwd_world
        fwd_2d = fwd_plane_3d[:2]
        
        # 3) 2D 맵에 표시 (점 + 화살표)
        plt.scatter(pos_2d[0], pos_2d[1], c="blue", s=15)
        arrow_scale = 0.5
        plt.arrow(
            pos_2d[0], pos_2d[1],
            fwd_2d[0]*arrow_scale, fwd_2d[1]*arrow_scale,
            head_width=0.1, head_length=0.15,
            fc='red', ec='red'
        )

    plt.savefig(output_file)
    plt.show()
    print(f"Map with 2D poses saved: {output_file}")

###############################################################################
# 9. 변환 정보 등을 파일에 저장
###############################################################################
def save_occupancy_map(map_data, x_bins, y_bins, output_file="occupancy_map.npz"):
    np.savez_compressed(output_file, map_data=map_data, x_bins=x_bins, y_bins=y_bins)
    print(f"Occupancy Map saved: {output_file}")

def save_plane_transform_info(plane_params, transformation_matrix, output_file="plane_transform_info.npz"):
    """
    plane_params와 transformation_matrix 등, 투영에 필요한 정보를 한 번에 저장
    """
    np.savez_compressed(
        output_file,
        plane_params=plane_params,
        transformation_matrix=transformation_matrix
    )
    print(f"Plane transform info saved: {output_file}")


###############################################################################
# 10. Nav2 map_server 용 (이미지+YAML) 저장 함수
###############################################################################
def save_map_for_nav2(occupancy_map, x_bins, y_bins, resolution,
                      map_img_file="map.png", map_yaml_file="map.yaml"):
    """
    occupancy_map: 2D numpy (shape = (num_x, num_y)), 0=free, 1=occupied
    x_bins, y_bins: np.arange(...) 로 생성된 bin 경계
    resolution: float. 예: 0.1 (m/px)
    map_img_file: 저장할 이미지 파일명(png, pgm 등)
    map_yaml_file: 저장할 yaml 메타파일명

    - Nav2(ROS)에서 사용하기 위해:
      1) 픽셀값: 0(검정)=차지 / 255(흰색)=빈공간 (혹은 반대로)
      2) 실제 물리 좌표에서 origin은 (x_bins[0], y_bins[0])를 기준점으로 삼음
      3) .yaml에 image, resolution, origin, free_thresh, occupied_thresh 등 기입
    """
    # (1) occupancy_map을 8bit grayscale로 변환
    #     1(occupied) → 0(검정), 0(free) → 255(흰색)
    #     shape=(num_x, num_y). 
    #     주의: row,y -> num_y, col,x -> num_x이므로 transpose or flip 처리

    occ_img = np.zeros_like(occupancy_map, dtype=np.uint8)
    occ_img[occupancy_map == 0] = 255  # free
    occ_img[occupancy_map == 1] = 0    # occupied


    # nav_msgs/Map은 x→col, y→row에서 y가 위로 갈수록 row 증가.
    # 일반 이미지 좌표는 row=0이 위쪽, but ROS는 y=0이 아래쪽.
    # => flipud를 해서 "원점이 왼쪽하단"이 되도록 맞춤.
    # occ_img = np.flipud(occ_img)
    occ_img = np.rot90(occ_img, 1) #90도 회전

    # (4) YAML에서 사용할 origin 계산
    # occupancy_map의 shape:
    #   num_rows = len(x_bins) (원래 x축 방향 셀 수)
    #   num_cols = len(y_bins) (원래 y축 방향 셀 수)
    num_rows = occupancy_map.shape[0]
    num_cols = occupancy_map.shape[1]

    # 원래 occupancy map에서 (0,0) pos₂d가 위치하는 픽셀 인덱스 (실수값)
    # row = (x - x_bins[0]) / resolution, col = (y - y_bins[0]) / resolution
    row0 = -x_bins[0] / resolution   # (0,0)에 대한 row index
    col0 = -y_bins[0] / resolution   # (0,0)에 대한 col index

    # 회전(np.rot90, k=1) 후 픽셀 좌표는 다음과 같이 변환됨:
    # new_row = num_cols - 1 - (col0), new_col = row0
    new_row = num_cols - 1 - col0
    new_col = row0

    # nav2에서의 OccupancyGrid는, 
    #   world_x = origin_x + (pixel_col * resolution)
    #   world_y = origin_y + (pixel_row * resolution)
    # 이므로, (0,0) pos₂d가 회전 이미지상의 (new_col, new_row) 픽셀에 위치한다면,
    # 이미지의 (0,0) (왼쪽 아래) 픽셀의 월드 좌표는
    #   origin_x = 0 - new_col * resolution, origin_y = 0 - new_row * resolution
    origin_x = - new_col * resolution
    origin_y = - new_row * resolution

    # 위 계산을 전개하면 (resolution == grid_size):
    #   origin_x = x_bins[0]
    #   origin_y = -((num_cols - 1)*resolution + y_bins[0])
    origin = [origin_x, origin_y, 0.0]

    # (5) YAML 메타파일 작성 (Nav2 map_server 표준 필드)
    yaml_content = f"""image: {map_img_file}
resolution: {resolution}
origin: [{origin[0]}, {origin[1]}, {origin[2]}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(map_yaml_file, 'w') as f:
        f.write(yaml_content)
    print(f"Saved map yaml: {map_yaml_file}")


###############################################################################
# --------------------------- 메인 스크립트 예시 -------------------------------
###############################################################################
if __name__ == "__main__":
    output_dir = "./clouddata"
    poses_file = os.path.join(output_dir, "poses.txt")
    point_cloud_file = os.path.join(output_dir, "cloud.ply")

    # (1) 포즈 로드 (x,y,z, qx,qy,qz,qw)
    poses_3d_quat = load_camera_poses_with_quat(poses_file)
    # plane fit용으로 위치만 추출 (N,3)
    poses_xyz = poses_3d_quat[:, :3]

    # (2) 평면 추출
    plane_params = fit_plane(poses_xyz)  # [a,b,c]
    a, b, c = plane_params
    print(f"Plane equation: {a}x + {b}y + {c}z = 1")

    # (3) Point Cloud 로드 & 필터
    pcd, cloud_points = load_point_cloud(point_cloud_file)
    filtered_points = filter_points_within_plane(cloud_points, plane_params, height=1.0)

    # (4) 평면 mesh 생성
    x_min, x_max = np.min(cloud_points[:,0]), np.max(cloud_points[:,0])
    y_min, y_max = np.min(cloud_points[:,1]), np.max(cloud_points[:,1])
    plane_mesh_points = generate_plane_mesh(plane_params, (x_min, x_max), (y_min, y_max))

    # (5) 3D 시각화 (선택)
    visualize_with_plane_and_poses(pcd, filtered_points, plane_mesh_points, poses_xyz)

    # (6) 평면 투영(환경 점) + 변환행렬 구하기
    projected_env_points, transformation_matrix = project_and_transform_points_to_plane(filtered_points, plane_params)

    # (7) 포즈도 평면에 투영 (이미 구한 R 재사용)
    projected_poses_2d = project_points_onto_plane_using_matrix(poses_xyz, plane_params, transformation_matrix)

    # (8) 점유맵 생성
    #    grid_size = 해상도(m/픽셀), threshold = 각 grid안에 몇 개의 점 이상이면 occupied으로 간주
    grid_size = 0.1
    occupancy_map, x_bins, y_bins = create_occupancy_map(projected_env_points, grid_size=grid_size, threshold=3)

    # (9) 2D 맵 + 포즈(방향) 시각화 (디버깅용)
    visualize_occupancy_map_with_poses(
        occupancy_map, x_bins, y_bins,
        plane_params, transformation_matrix,
        poses_3d_quat,
        output_file="occupancy_map_with_poses.png"
    )

    # (10) 결과 저장 (numpy)
    save_occupancy_map(occupancy_map, x_bins, y_bins, output_file="occupancy_map.npz")
    save_plane_transform_info(plane_params, transformation_matrix, output_file="plane_transform_info.npz")

    # (11) Nav2 map_server 용 (이미지 + yaml) 생성
    save_map_for_nav2(
        occupancy_map, x_bins, y_bins, resolution=grid_size,
        map_img_file="nav2_map.png",
        map_yaml_file="nav2_map.yaml"
    )
    print("Done. You can use nav2_map.yaml + nav2_map.png in Nav2 map_server.")
