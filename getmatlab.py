import os
import glob
import numpy as np
import pyvista as pv
import nibabel as nib
from scipy.io import savemat
import mne
import cv2


def compute_and_save_fiducial_points(niiFiles, subject, subjects_dir):
    # 读取 NIfTI 文件
    nifti_file =  glob.glob(os.path.join(niiFiles, '*.gz')) + glob.glob(os.path.join(niiFiles, '*.nii'))  # 替换为你的 NIfTI 文件路径
    if not nifti_file:
        print("未找到.nii/nii.gz文件")
        return
    nifti_file = nifti_file[0]
    # 读取 .obj 或 .stl 文件
    obj_files = glob.glob(os.path.join(subjects_dir, subject, '*.obj'))
    if not obj_files:
        print("未找到.obj文件")
        return
    mesh = pv.read(obj_files[0])  # 或者使用 'path_to_your_file.stl'

    # 提取点云数据
    points = mesh.points

    # 找到与X轴的交点 (Y=0, Z=0)
    intersection_points_x = points[(abs(points[:, 1]) <= 1) & (abs(points[:, 2]) <= 1)]

    # 找到与Y轴的交点 (X=0, Z=0)
    intersection_points_y = points[(abs(points[:, 0]) <= 1) & (abs(points[:, 2]) <= 1)]

    # 找到与Z轴的交点 (X=0, Y=0)
    intersection_points_z = points[(abs(points[:, 0]) <= 1) & (abs(points[:, 1]) <= 1)]

    def add_intersection_points(intersection_points, idx=0):
        if intersection_points.size != 0:
            # 找到最外侧的点 (坐标最大和最小的点)
            min_point = intersection_points[np.argmin(intersection_points[:, idx])]
            max_point = intersection_points[np.argmax(intersection_points[:, idx])]

            print(f"与轴的交点的最小坐标的点:", min_point)
            print(f"与轴的交点的最大坐标的点:", max_point)

            # 返回最外侧的点
            return min_point, max_point
        else:
            print("没有找到与轴的交点")
            return None, None

    # 可视化X轴交点
    min_x_point, max_x_point = add_intersection_points(intersection_points_x, idx=0)

    # 可视化Y轴交点
    min_y_point, max_y_point = add_intersection_points(intersection_points_y, idx=1)

    # 可视化Z轴交点
    min_z_point, max_z_point = add_intersection_points(intersection_points_z, idx=2)
    print(intersection_points_z)

    # 读取NIfTI文件中的仿射矩阵
    img = nib.load(nifti_file)
    vox2ras = img.affine

    # 定义转换矩阵
    transform_matrix = np.linalg.pinv(np.array([[-1,0,0,128],[0,0,1,-128],[0,-1,0,128],[0,0,0,1]]))
    # transform_matrix = np.linalg.pinv(vox2ras)

    # 标准转个体
    Z_MNI = transform_matrix @  np.concatenate([np.array(max_z_point),np.ones(1)]).T
    L_MNI = transform_matrix @  np.concatenate([np.array(min_x_point),np.ones(1)]).T
    R_MNI = transform_matrix @  np.concatenate([np.array(max_x_point),np.ones(1)]).T
    N_MNI = transform_matrix @  np.concatenate([np.array(max_y_point),np.ones(1)]).T
    Z_MNI = Z_MNI[0:3]
    L_MNI = L_MNI[0:3]
    R_MNI = R_MNI[0:3]
    N_MNI = N_MNI[0:3]

    # 保存为.mat文件
    mni_coordinates = np.array([Z_MNI, L_MNI, R_MNI, N_MNI])
    savemat(os.path.join(subjects_dir, subject, 'fiducial.mat'), {'fiducial_point': mni_coordinates})

    # 设置源空间并保存
    src = mne.setup_source_space(subject, subjects_dir=subjects_dir, spacing='oct6', add_dist=False)
    src_file = os.path.join(subjects_dir, subject, 'src.fif')
    mne.write_source_spaces(src_file, src, overwrite=True)

    # 保存曲率
    lcurv_file = os.path.join(subjects_dir, subject, 'surf', 'lh.curv')
    lcurv_data = nib.freesurfer.io.read_morph_data(lcurv_file)
    rcurv_file = os.path.join(subjects_dir, subject, 'surf', 'rh.curv')
    rcurv_data = nib.freesurfer.io.read_morph_data(rcurv_file)

    curv = np.concatenate([lcurv_data[src[0]['inuse']==1],rcurv_data[src[1]['inuse']==1]]).T

    curvcurv = curv.copy()
    curvcurv[curv<0] = -1
    curvcurv[curv>0] = 1
    savemat(os.path.join(subjects_dir, subject, 'curv.mat'), {'curvature': curvcurv})
    
    # # 可视化原始点云数据和交点
    # p = pv.Plotter(off_screen=True)
    # p.add_points(points, color='black', point_size=1)
    # p.show_axes()
    # p.add_mesh(mesh, color='white')
    # if min_x_point is not None and max_x_point is not None:
    #     p.add_mesh(pv.Sphere(radius=5, center=min_x_point), color='blue')
    #     p.add_mesh(pv.Sphere(radius=5, center=max_x_point), color='blue')
    # if min_y_point is not None and max_y_point is not None:
    #     p.add_mesh(pv.Sphere(radius=5, center=min_y_point), color='green')
    #     p.add_mesh(pv.Sphere(radius=5, center=max_y_point), color='green')
    # if min_z_point is not None and max_z_point is not None:
    #     p.add_mesh(pv.Sphere(radius=5, center=min_z_point), color='red')
    #     p.add_mesh(pv.Sphere(radius=5, center=max_z_point), color='red')
    # outputDir='output_images_camera_path'
    # capture_camera_path_images(p, outputDir)
    # images_to_avi(outputDir, os.path.join(subjects_dir, subject, 'res.avi'))

def images_to_avi(folder_path, output_path, fps=30):
    """
    将指定文件夹中的图片按顺序合成一个AVI格式的视频。

    参数：
    folder_path (str): 图片文件夹的路径。
    output_path (str): 输出视频的保存路径和名称。
    fps (int): 每秒显示的图片数（帧率）。默认为30。

    返回值：
    无
    """
    # 获取文件夹中的所有图片文件
    images = [img for img in os.listdir(folder_path) if img.endswith('.jpg') or img.endswith('.png')]

    # 确保图片按顺序排列
    images.sort()

    # 检查是否有图片
    if not images:
        raise ValueError("文件夹中没有找到图片")

    # 读取第一张图片以获取视频帧的尺寸
    first_image = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = first_image.shape

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 将所有图片写入视频
    for image in images:
        img = cv2.imread(os.path.join(folder_path, image))
        video.write(img)

    # 释放资源
    video.release()
    cv2.destroyAllWindows()

def capture_camera_path_images(plotter, output_dir='output_images_camera_path'):
    """
    Capture images along a camera path and save them to the specified directory.

    Parameters:
    - plotter: The PyVista plotter object used for rendering.
    - output_dir: Directory to save the images. Defaults to 'output_images_camera_path'.
    """
    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)

    # 相机视角设置
    elevation_angles_up = np.linspace(-60, 30, num=90)  # 从-30度到30度的俯仰角
    elevation_angles_down = np.linspace(30, 0, num=50)  # 从30度回到0度的俯仰角
    azimuth_angles = np.linspace(0, 360, num=360)  # 从0到360度的方位角

    frame_counter = 0

    # 俯仰角从-30度到30度
    for elevation in elevation_angles_up:
        plotter.camera.elevation = elevation
        plotter.render()
        image_filename = os.path.join(output_dir, f'image_{frame_counter:04d}.png')
        plotter.screenshot(image_filename)
        frame_counter += 1

    # 俯仰角从30度回到0度
    for elevation in elevation_angles_down:
        plotter.camera.elevation = elevation
        plotter.render()
        image_filename = os.path.join(output_dir, f'image_{frame_counter:04d}.png')
        plotter.screenshot(image_filename)
        frame_counter += 1

    # 绕轴旋转360度
    for azimuth in azimuth_angles:
        plotter.camera.azimuth = azimuth
        plotter.render()
        image_filename = os.path.join(output_dir, f'image_{frame_counter:04d}.png')
        plotter.screenshot(image_filename)
        frame_counter += 1

if __name__ == '__main__':
    # 使用你的文件路径替换 'F:\\程序\\MEG_FT'
    compute_and_save_fiducial_points('dataInput', 'fastSurfer_web_2024-04-11-10-18-32_GCqR9', 'freesurfer')
    # dataInput 为nii的文件夹
    # fastSurfer_web_2024-04-11-10-18-32_GCqR9 是freesurfer分割的subject
    # freesurfer 是subject 的文件夹
