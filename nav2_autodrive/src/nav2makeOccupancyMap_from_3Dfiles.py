import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R
from PIL import Image   # 이미지 저장용

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
    data = np.loadtxt(file_path, skiprows=0, usecols=(1, 2, 3, 4, 5, 6, 7))
    if data.shape[0] < 3:
        raise ValueError("Not enough camera poses to compute a plane. At least 3 poses are required.")
    return data  # shape: (N, 7) = [x, y, z, qx, qy, qz, qw]

###############################################################################
# 2. 평면 추출 (Least Squares)
###############################################################################
def fit_plane(points):
    """
    points: (N,3) 형태. 평면 1 = a*x + b*y + c*z 형태로 맞춤
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
    initial_guess = [1, 1, 1]
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
    1 <= distance <= 1+height 범위 내의 점들만 필터링  
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
    # 평면 방정식: 1 = a*x + b*y + c*z  ->  z = (1 - a*x - b*y)/c
    zz = (1 - a*xx - b*yy)/c
    plane_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    return plane_points

###############################################################################
# 5. 3D 시각화
###############################################################################
def visualize_with_plane_and_poses(original_pcd, filtered_points, plane_points, poses_xyz):
    # 원래 포인트 클라우드(회색)
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # 필터링된 점(빨강)
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
    o3d.visualization.draw_geometries(
        [original_pcd, filtered_pcd, plane_pcd, pose_pcd],
        window_name="Point Cloud with Plane and Poses",
        width=800, height=600,
    )

###############################################################################
# 6. 평면 투영 및 좌표 변환 행렬 계산 (T_plane)
###############################################################################
def project_and_transform_points_to_plane(points, plane_params):
    """
    1) 각 점을 평면(1 = a*x+b*y+c*z)에 수직 투영  
    2) 평면 좌표계(평면의 x,y 축은 평면 내에서 결정; z는 법선)를 위해 회전행렬을 구함  
       → 이때, world→plane의 완전한 4x4 변환행렬 T_plane을 생성 (평면 위로 원점의 투영을 기준으로)
    3) 투영된 점들을 plane 좌표계 (2D)로 변환하여 반환
    """
    a, b, c = plane_params

    # 평면 법선 (정규화)
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # 평면 내 x축: 법선과 [0,0,1]의 외적 (만약 평행하면 [1,0,0])
    x_axis = np.cross(normal, np.array([0, 0, 1], dtype=np.float64))
    if np.linalg.norm(x_axis) == 0:
        x_axis = np.array([1, 0, 0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    # 평면 내 y축
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation_matrix = np.vstack((x_axis, y_axis, normal)).T  # 3x3

    # --- T_plane 계산 ---
    # RTABmap의 map frame 원점을 평면 위로 수직투영 (t0)
    t0 = (1 - 0) / (a*a + b*b + c*c)  # (a*0+b*0+c*0=0)
    p0 = np.array([a*t0, b*t0, c*t0])  # 원점의 투영점
    T_plane = np.eye(4)
    T_plane[:3, :3] = rotation_matrix
    T_plane[:3, 3] = - rotation_matrix @ p0  # 평면 좌표계에서 p0가 0이 되도록

    # 각 점에 대해 수직투영 후 평면 좌표계 2D 점 계산
    projected_points_2d = []
    for x, y, z in points:
        t = (1 - (a*x + b*y + c*z)) / (a*a + b*b + c*c)
        px, py, pz = x + a*t, y + b*t, z + c*t
        # world -> plane (회전만 적용; T_plane에 포함된 translation은 p0 처리용)
        local_3d = rotation_matrix.T @ np.array([px, py, pz])
        projected_points_2d.append(local_3d[:2])
    return np.array(projected_points_2d), T_plane

def project_points_onto_plane_using_matrix(points, plane_params, T_plane):
    """
    이미 구한 T_plane[:3,:3] (회전 R)을 재사용하여,
    각 점을 평면에 수직투영한 후 plane 좌표계의 2D 점으로 변환
    """
    a, b, c = plane_params
    Rmat = T_plane[:3, :3]
    projected_points_2d = []
    for x, y, z in points:
        t = (1 - (a*x + b*y + c*z)) / (a*a + b*b + c*c)
        px, py, pz = x + a*t, y + b*t, z + c*t
        local_3d = Rmat.T @ np.array([px, py, pz])
        projected_points_2d.append(local_3d[:2])
    return np.array(projected_points_2d)

###############################################################################
# 7. Occupancy Map 생성 (입력: 2D 평면상 점들)
###############################################################################
def create_occupancy_map(points_2d, grid_size=0.1, threshold=5):
    """
    points_2d: (N,2)
    grid_size: 해상도 (m/픽셀)
    threshold: 각 셀에 점이 threshold개 이상이면 occupied (1)
    반환:
      - occupancy_map: 2D numpy (0: free, 1: occupied)
      - x_bins, y_bins: 각 축의 bin 경계
    """
    x_min, x_max = np.min(points_2d[:,0]), np.max(points_2d[:,0])
    y_min, y_max = np.min(points_2d[:,1]), np.max(points_2d[:,1])
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    occupancy_map = np.zeros((len(x_bins), len(y_bins)), dtype=int)
    for x, y in points_2d:
        x_idx = np.searchsorted(x_bins, x) - 1
        y_idx = np.searchsorted(y_bins, y) - 1
        if 0 <= x_idx < len(x_bins) and 0 <= y_idx < len(y_bins):
            occupancy_map[x_idx, y_idx] += 1
    occupancy_map = (occupancy_map >= threshold).astype(int)
    return occupancy_map, x_bins, y_bins

###############################################################################
# 8. 2D 맵 시각화 + 포즈 점 및 방향(화살표) 표시
###############################################################################
def quaternion_to_forward_vector(qx, qy, qz, qw):
    """
    로컬 [0,0,1] (전방) 벡터를 쿼터니언으로 회전시켜 월드 전방 벡터 계산  
    (필요에 따라 전방벡터 정의 수정 가능)
    """
    rot = R.from_quat([qx, qy, qz, qw])
    forward_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_world = rot.apply(forward_local)
    return forward_world

def visualize_occupancy_map_with_poses(
    map_data, x_bins, y_bins,
    plane_params, T_plane,
    poses_3d_quat,
    shift=np.array([0.0, 0.0]),  # 2D 평면 상의 shift (즉, occupancy map 생성 시 빼준 center)
    output_file="occupancy_map.png"
):
    """
    - map_data: occupancy map (0 또는 1)
    - x_bins, y_bins: 각 축의 bin 경계 (shift된 좌표계)
    - plane_params, T_plane: 평면 투영 정보
    - poses_3d_quat: (N,7) = [x,y,z,qx,qy,qz,qw]
    - shift: 투영된 2D 좌표에 대해 추가로 빼줄 offset (즉, center)
    """
    plt.figure(figsize=(10,8))
    plt.imshow(
        map_data.T, origin='lower', cmap='gray',
        extent=(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1])
    )
    plt.colorbar(label="Occupancy")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Occupancy Map with 2D Poses & Directions")

    Rmat = T_plane[:3, :3]
    for pose in poses_3d_quat:
        x, y, z, qx, qy, qz, qw = pose
        # (1) 포즈 위치를 평면에 수직투영하고, T_plane을 사용해 plane 좌표계로 변환
        a, b, c = plane_params
        t = (1 - (a*x + b*y + c*z)) / (a*a + b*b + c*c)
        px, py, pz = x + a*t, y + b*t, z + c*t
        local_pos_3d = Rmat.T @ np.array([px, py, pz])
        pos_2d = local_pos_3d[:2] - shift  # shift 적용 (맵 중심이 0,0)
        # (2) 전방벡터 계산 후, 평면 좌표계로 변환하고 shift 적용
        fwd_world = quaternion_to_forward_vector(qx, qy, qz, qw)
        fwd_plane_3d = Rmat.T @ fwd_world
        fwd_2d = fwd_plane_3d[:2]
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
# 9. 변환 정보 저장 (occupancy map, 평면 투영 관련 정보)
###############################################################################
def save_occupancy_map(map_data, x_bins, y_bins, output_file="occupancy_map.npz"):
    np.savez_compressed(output_file, map_data=map_data, x_bins=x_bins, y_bins=y_bins)
    print(f"Occupancy Map saved: {output_file}")

def save_plane_transform_info(plane_params, T_plane, output_file="plane_transform_info.npz"):
    np.savez_compressed(
        output_file,
        plane_params=plane_params,
        T_plane=T_plane
    )
    print(f"Plane transform info saved: {output_file}")

def save_frame_transform_info(T_total, T_plane, T_center, output_file="frame_transform_info.npz"):
    """
    RTABmap의 map frame → Nav2 map frame (바닥 기준, 맵 중심 0,0) 변환 행렬들을 저장
    T_total = T_center · T_plane
    """
    np.savez_compressed(
        output_file,
        T_total=T_total,
        T_plane=T_plane,
        T_center=T_center
    )
    print(f"Frame transform info saved: {output_file}")

###############################################################################
# 10. Nav2 map_server 용 (이미지 + YAML) 저장 함수
###############################################################################
def save_map_for_nav2(occupancy_map, x_bins, y_bins, resolution,
                      map_img_file="nav2_map.png", map_yaml_file="nav2_map.yaml"):
    """
    occupancy_map: 2D numpy (0: free, 1: occupied)
    x_bins, y_bins: 해당 축의 bin 경계 (이미지 좌표; center가 0이라면 x_bins[0]≈-width/2 등)
    resolution: float (예: 0.1)
    map_img_file: 생성할 이미지 파일
    map_yaml_file: YAML 메타파일
    Nav2용으로, 이미지 픽셀: 0(occupied)=검정, 255(free)=흰색.
    map.yaml의 origin은 이미지 좌측 하단 좌표 (즉, [-width/2, -height/2, 0])가 되도록 함.
    """
    # occupancy_map → 8bit grayscale (occupied: 0, free: 255)
    occ_img = np.zeros_like(occupancy_map, dtype=np.uint8)
    occ_img[occupancy_map == 0] = 255
    occ_img[occupancy_map == 1] = 0
    occ_img = np.flipud(occ_img)  # 좌표 변환: ROS는 y-up
    pil_img = Image.fromarray(occ_img)
    pil_img.save(map_img_file)
    print(f"Saved map image: {map_img_file}")

    # 이미지의 실제 크기 (m)
    width_m = occupancy_map.shape[0] * resolution
    height_m = occupancy_map.shape[1] * resolution
    # 이미지 중심이 (0,0)이므로, 왼쪽 아래는 (-width/2, -height/2)
    origin_x = -width_m / 2.0
    origin_y = -height_m / 2.0
    yaml_content = f"""image: {map_img_file}
resolution: {resolution}
origin: [{origin_x}, {origin_y}, 0.0]
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

    # (1) 포즈 로드 (x,y,z,qx,qy,qz,qw)
    poses_3d_quat = load_camera_poses_with_quat(poses_file)
    poses_xyz = poses_3d_quat[:, :3]

    # (2) 평면 추출
    plane_params = fit_plane(poses_xyz)  # [a,b,c]
    a, b, c = plane_params
    print(f"Plane equation: {a}x + {b}y + {c}z = 1")

    # (3) Point Cloud 로드 & 필터링
    pcd, cloud_points = load_point_cloud(point_cloud_file)
    filtered_points = filter_points_within_plane(cloud_points, plane_params, height=1.0)

    # (4) 평면 mesh 생성 (시각화용)
    x_min, x_max = np.min(cloud_points[:,0]), np.max(cloud_points[:,0])
    y_min, y_max = np.min(cloud_points[:,1]), np.max(cloud_points[:,1])
    plane_mesh_points = generate_plane_mesh(plane_params, (x_min, x_max), (y_min, y_max))

    # (5) 3D 시각화 (선택)
    visualize_with_plane_and_poses(pcd, filtered_points, plane_mesh_points, poses_xyz)

    # (6) 평면 투영 및 T_plane 계산
    projected_env_points, T_plane = project_and_transform_points_to_plane(filtered_points, plane_params)

    # (7) 포즈도 평면에 투영 (T_plane 재사용)
    projected_poses_2d = project_points_onto_plane_using_matrix(poses_xyz, plane_params, T_plane)

    # (8) occupancy map 좌표계를 Nav2 map frame (이미지 중심 0,0)으로 만들기 위해 shift 계산  
    #     projected_env_points는 현재 평면 좌표계 상의 2D 점들 → center 계산
    center = np.array([
        (projected_env_points[:,0].min() + projected_env_points[:,0].max()) / 2.0,
        (projected_env_points[:,1].min() + projected_env_points[:,1].max()) / 2.0
    ])
    # T_center: 평면 좌표계에서 (center, 1.0857m) 만큼 보정 (1.0857m는 평면이 실제 바닥보다 높은 만큼)
    T_center = np.eye(4)
    T_center[:3, 3] = np.array([-center[0], -center[1], -1.0857])
    # 최종 변환: RTABmap map frame → Nav2 map frame
    T_total = T_center @ T_plane

    # (9) 투영된 점들 및 포즈에 shift 적용 (즉, Nav2 map frame 상에서 중심 0,0)
    shifted_env_points = projected_env_points - center
    shifted_poses_2d = projected_poses_2d - center

    # (10) occupancy map 생성 (shift된 좌표 사용)
    grid_size = 0.1  # m/px
    occupancy_map, x_bins, y_bins = create_occupancy_map(shifted_env_points, grid_size=grid_size, threshold=3)

    # (11) 2D 맵 + 포즈(방향) 시각화 (디버깅용)
    visualize_occupancy_map_with_poses(
        occupancy_map, x_bins, y_bins,
        plane_params, T_plane,
        poses_3d_quat,
        shift=center,  # 내부에서 빼줌 → (0,0) 중심
        output_file="occupancy_map_with_poses.png"
    )

    # (12) 결과 저장 (numpy 파일)
    save_occupancy_map(occupancy_map, x_bins, y_bins, output_file="occupancy_map.npz")
    save_plane_transform_info(plane_params, T_plane, output_file="plane_transform_info.npz")
    save_frame_transform_info(T_total, T_plane, T_center, output_file="frame_transform_info.npz")

    # (13) Nav2 map_server 용 map 파일 생성 (이미지 + yaml)
    save_map_for_nav2(
        occupancy_map, x_bins, y_bins, resolution=grid_size,
        map_img_file="nav2_map.png",
        map_yaml_file="nav2_map.yaml"
    )
    print("Done. You can use nav2_map.yaml & nav2_map.png in Nav2 map_server.")
