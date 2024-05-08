import smooth_utils as smooth
from Viewer.controller import Controller
from physics_warpper import PhysicsInfo
from bvh_loader import *
class CharacterController():
    def __init__(self, viewer, controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller: Controller = controller
        # motion
        self.motions = []
        # self.simu_flag=self.viewer.simu_flag
        # 添加motion
        self.motions.append(BVHMotion(bvh_file_name='motion_material/physics_motion/long_walk_mirror.bvh'))
        self.root_pos_offset = np.array([0, 0, 0])
        # 当前角色的参考root旋转，注意一定是一个Y轴旋转
        # 表示的也是offset，即当前角色的root旋转是通过motion中的root旋转(左）乘以这个offset得到的
        self.root_rot_offset = np.array([0, 0, 0, 1])
        # 同时，由于root——rot会影响pos的定位，因此需要一个旋转的参考root
        self.rot_pos_ref=np.array([0,0,0])
        
        # debug
        self.root_pos_save=np.array([0,0,0])
        
        # 当前角色处于正在跑的BVH的第几帧
        # 现在是假设只用一个长跑的BVH，所以只需要一个变量来记录当前在播放帧数
        self.cur_frame = 140
        
        # smooth参数
        # 用于计算时间的初始帧
        self.smooth_frame = -1
        # 记录当前的位置和旋转
        self.cur_pos = np.zeros_like(self.motions[0].joint_position[0])
        self.cur_rot = np.zeros_like(self.motions[0].joint_rotation[0])
        self.cur_pos_pre = np.zeros_like(self.motions[0].joint_position[0])
        self.cur_rot_pre = np.zeros_like(self.motions[0].joint_rotation[0])
        # 记录跳变时发生的位置速度等差值
        self.d_pos = np.zeros_like(self.motions[0].joint_position[0])
        self.d_rot = np.zeros_like(self.motions[0].joint_rotation[0])
        self.d_vel = np.zeros_like(self.motions[0].joint_vel[0])
        self.d_avel = np.zeros_like(self.motions[0].joint_avel[0])
        
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list
                     ):
        '''
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态
            当然我们只是提供了速度和角速度的输入，如果通过pos和rot已经很好选择下一个动作了，可以不必须使用速度和角速度
            desired_pos_list: 期望位置, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望位置(XoZ平面)， 期望位置可以用来拟合根节点位置
            desired_rot_list: 期望旋转, 6x4的矩阵, 四元数, 每一行对应0，20，40...帧的期望旋转(Y旋转), 期望旋转可以用来拟合根节点旋转
            desired_vel_list: 期望速度, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望速度(XoZ平面), 期望速度可以用来拟合根节点速度
            desired_avel_list: 期望角速度, 6x3的矩阵, [0, y, 0], 每一行对应0，20，40...帧的期望角速度(Y旋转), 期望角速度可以用来拟合根节点角速度
        
        Output: 输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
            为了方便物理控制，也把局部的返回了
            joint_pos 和 joint_rot
        Tips:
            1. 注意应该利用的期望位置和期望速度应该都是在XoZ平面内，期望旋转和期望角速度都是绕Y轴的旋转。其他的项没有意义

        '''
        # 以下实现的是data-driven的方法
        #print(desired_rot_list)
        #print(desired_avel_list)
        controller_pos=None
        #controller_pos = desired_pos_list[0]
            
        half_life=0.1
        motion:BVHMotion = self.motions[0]
        N=motion.num_frames
        self.cur_frame = (self.cur_frame + 1) % N
        dt=1.0/60.0
        motion.adjust_joint_name(self.viewer.joint_name)
        joint_name = motion.joint_name
        
        if   self.cur_frame % 15 != 0 :
            root_pos= R.from_quat(self.root_rot_offset).apply(motion.joint_position[self.cur_frame,0]-self.rot_pos_ref)+self.rot_pos_ref+self.root_pos_offset
            root_rot= (R.from_quat(self.root_rot_offset)*R.from_quat(motion.joint_rotation[self.cur_frame,0])).as_quat()
            
            all_pos= motion.joint_position[self.cur_frame].copy()
            all_rot= motion.joint_rotation[self.cur_frame].copy()
            all_pos[0]=root_pos
            all_rot[0]=root_rot
            
            if self.smooth_frame>0:
                offset_pos=smooth.decay_spring_implicit_damping_pos(self.d_pos,self.d_vel,half_life,(self.cur_frame-self.smooth_frame)*dt)[0]
                offset_rot=smooth.decay_spring_implicit_damping_rot(self.d_rot,self.d_avel,half_life,(self.cur_frame-self.smooth_frame)*dt)[0]
                all_pos+=offset_pos
                all_rot=R.from_rotvec(R.from_quat(all_rot).as_rotvec()+offset_rot).as_quat()
            self.cur_pos_pre=self.cur_pos
            self.cur_rot_pre=self.cur_rot
            self.cur_pos=all_pos
            self.cur_rot=all_rot
            
            all_pos=all_pos.reshape(1,-1,3)
            all_rot=all_rot.reshape(1,-1,4)
            joint_translation, joint_orientation = motion.batch_forward_kinematics(joint_position=all_pos, joint_rotation=all_rot)
            joint_translation = joint_translation[0]
            joint_orientation = joint_orientation[0]
            all_pos=all_pos[0]
            all_rot=all_rot[0]
            self.root_pos_save=root_pos
            #print(self.cur_frame)
            return joint_name, joint_translation, joint_orientation, all_pos, all_rot
        # 提前计算一下当前帧的root位置和旋转
        root_pos= R.from_quat(self.root_rot_offset).apply(motion.joint_position[self.cur_frame,0]-self.rot_pos_ref)+self.rot_pos_ref+self.root_pos_offset
        root_rot= (R.from_quat(self.root_rot_offset)*R.from_quat(motion.joint_rotation[self.cur_frame,0])).as_quat()
        # 这两个值是之后可能jump到的动作要贴合的对象
        #print("root_pos           :",root_pos)
        #print("desired_pos_list[0]:",desired_pos_list[0])
        root_pos_g=root_pos.copy()
        root_pos_g[1]=0
        
        
        # 读取目前的局部pos和rot
        cur_local_pos = motion.joint_position[self.cur_frame]
        cur_local_rot= self.root_rot_offset*motion.joint_rotation[self.cur_frame]
        cur_local_vel= motion.joint_vel[self.cur_frame]
        cur_local_avel= motion.joint_avel[self.cur_frame]
        # 计算所有帧中哪个帧最接近目标位置
        best_frame = 0
        best_cost = 1e10
        rate=1.8 # 用于调整转移代价和贴近目标代价的比例
        # 由原来的for循环改写为用numpy的broadcasting来计算
        indices = np.arange(160,N-140)
        selected_pos = motion.joint_position[indices]
        selected_rot = motion.joint_rotation[indices]
        selected_vel = motion.joint_vel[indices]
        selected_avel = motion.joint_avel[indices]
        # 转移的代价
        # 由于绝对位置和绝对朝向可以调整，转移的代价只记相对姿势导致的，即除了根节点以外的其他关节的rot和avel，以及根节点的vel和avel
        cost = shift_cost(cur_local_vel,cur_local_rot,cur_local_avel,selected_vel,selected_rot,selected_avel)

        # 贴近目标的代价
        # 将目标格式化
        idx_l=np.ix_(indices,[0,20,40,60,80,100])
        s_root_pos = motion.joint_position[idx_l[0]+idx_l[1],0]
        s_root_pos[:,:,1] =0 
        s_root_vel= motion.joint_vel[idx_l[0]+idx_l[1],0]
        s_root_vel[:,:,1] =0
        # 注意这里的rot是三维的，需要在衡量cost时转换成Y旋转，并使得0时刻的旋转与当前的旋转相同
        s_root_rot = motion.joint_rotation[idx_l[0]+idx_l[1],0]
        s_root_avel= motion.joint_avel[idx_l[0]+idx_l[1],0]
        s_root_avel[:,[0,2]] =0
        # 预处理这些数据，将其match上root_pos和root_rot
        # 这将改变s_root_pos和s_root_rot以及s_root_vel
        # 平移root_pos使得0时刻的root_pos与当前的root_pos相同，旋转root_rot使得0时刻的root_rot的指向与当前的root_rot相同
        s_root_pos[:,:]-=s_root_pos[:,0].reshape(-1,1,3)
        s_root_pos[:,:]+=root_pos
        # 计算所有帧对应的Y旋转
        root_rot_offset= -cal_rot_offset(s_root_rot[:,0],root_rot.reshape(-1,4))
        # 为了方便broadcasting，且由于Rotation库的转化只接受二维输入，将root_rot_offset复制6次转换成二维
        root_rot_offset=np.repeat(root_rot_offset[:, np.newaxis, :], 6, axis=1)
        root_rot_offset=root_rot_offset.reshape(-1,4) # (6*N,4)
        #s_root_pos_start=np.repeat(s_root_pos[:,0][:, np.newaxis, :], 6, axis=1)
        #s_root_pos_start=s_root_pos_start.reshape(-1,3) # (6*N,3)
        # 旋转
        s_root_pos[:,:]=R.from_quat(root_rot_offset).apply((s_root_pos[:,:]-s_root_pos[:,0].reshape(-1,1,3)).reshape(-1,3)).reshape(-1,6,3)+s_root_pos[:,0].reshape(-1,1,3)
        s_root_rot[:,:]=(R.from_quat(root_rot_offset)*R.from_quat(s_root_rot[:,:].reshape(-1,4))).as_quat().reshape(-1,6,4)
        s_root_vel[:,:]=R.from_quat(root_rot_offset).apply(s_root_vel[:,:].reshape(-1,3)).reshape(-1,6,3)
        root_rot_offset=root_rot_offset.reshape(-1,6,4)
        # 单独计算一下旋转后的velocity区别的cost，这很重要
        cost += 2*vel_cost(s_root_vel[:,0],cur_local_vel[0])
        # 计算desired的cost
        cost += rate*de_cost(desired_pos_list,s_root_pos,desired_vel_list,s_root_vel,\
            desired_rot_list,s_root_rot,desired_avel_list,s_root_avel,root_pos,root_rot)
        
        # 计算最合适帧
        best_index = np.argmin(cost)
        best_frame = indices[best_index]
        best_cost = cost[best_index]
        new_root_pos_offset= -motion.joint_position[best_frame,0]+root_pos
        new_root_pos_offset[1]=0
        new_root_rot_offset= root_rot_offset[best_index,0]
        new_rot_ref = motion.joint_position[best_frame,0].copy()
        new_rot_ref[1]=0
        #print("cur_frame",self.cur_frame,"best_frame",best_frame,"best_cost",best_cost)
        # 输出
        # 添加smooth
        if  (abs(best_frame-self.cur_frame)<20): # 如果帧数差距很小，就继续播放当前的动画
            root_pos= R.from_quat(self.root_rot_offset).apply(motion.joint_position[self.cur_frame,0]-self.rot_pos_ref)+self.rot_pos_ref+self.root_pos_offset
            root_rot= (R.from_quat(self.root_rot_offset)*R.from_quat(motion.joint_rotation[self.cur_frame,0])).as_quat()
            
            all_pos= motion.joint_position[self.cur_frame].copy()
            all_rot= motion.joint_rotation[self.cur_frame].copy()
            all_pos[0]=root_pos
            all_rot[0]=root_rot
            
            if self.smooth_frame>0:
                offset_pos=smooth.decay_spring_implicit_damping_pos(self.d_pos,self.d_vel,half_life,(self.cur_frame-self.smooth_frame)*dt)[0]
                offset_rot=smooth.decay_spring_implicit_damping_rot(self.d_rot,self.d_avel,half_life,(self.cur_frame-self.smooth_frame)*dt)[0]
                all_pos+=offset_pos
                all_rot=R.from_rotvec(R.from_quat(all_rot).as_rotvec()+offset_rot).as_quat()
            self.cur_pos_pre=self.cur_pos
            self.cur_rot_pre=self.cur_rot
            self.cur_pos=all_pos
            self.cur_rot=all_rot
            
            
            all_pos=all_pos.reshape(1,-1,3)
            all_rot=all_rot.reshape(1,-1,4)
            joint_translation, joint_orientation = motion.batch_forward_kinematics(joint_position=all_pos, joint_rotation=all_rot)
            joint_translation = joint_translation[0]
            joint_orientation = joint_orientation[0]
            all_pos=all_pos[0]
            all_rot=all_rot[0]
            self.root_pos_save=root_pos
            #print(self.cur_frame,"continued")
            return joint_name, joint_translation, joint_orientation, all_pos, all_rot
        else: # 否则就切换到最佳帧
            # 更新root的offset等
            self.root_pos_offset = new_root_pos_offset
            self.root_rot_offset = new_root_rot_offset
            self.rot_pos_ref = new_rot_ref
            self.cur_frame = best_frame
            root_pos= (R.from_quat(self.root_rot_offset).apply(motion.joint_position[self.cur_frame,0]-self.rot_pos_ref))+self.rot_pos_ref+self.root_pos_offset
            root_rot= (R.from_quat(self.root_rot_offset)*R.from_quat(motion.joint_rotation[self.cur_frame,0])).as_quat()
            
            all_pos= motion.joint_position[self.cur_frame].copy()
            all_rot= motion.joint_rotation[self.cur_frame].copy()
            all_pos[0]=root_pos
            all_rot[0]=root_rot
            
            # 发生了跳变，需要重新计算smooth等参数，基于新算出的pos和rot和此前保存的pos和rot
            self.smooth_frame=self.cur_frame
            self.d_pos= (self.cur_pos-all_pos)
            self.d_rot= R.from_quat(self.cur_rot).as_rotvec()-R.from_quat(all_rot).as_rotvec()
            prev_vel=(self.cur_pos-self.cur_pos_pre)/dt
            next_pos= motion.joint_position[self.cur_frame+1].copy()
            next_pos[0]= (R.from_quat(self.root_rot_offset).apply(motion.joint_position[self.cur_frame+1,0]-self.rot_pos_ref))+self.rot_pos_ref+self.root_pos_offset
            next_rot= motion.joint_rotation[self.cur_frame+1].copy()
            next_rot[0]= (R.from_quat(self.root_rot_offset)*R.from_quat(motion.joint_rotation[self.cur_frame+1,0])).as_quat()
            new_vel=(next_pos-all_pos)/dt
            self.d_vel= (prev_vel-new_vel)
            prev_avel=smooth.quat_to_avel([self.cur_rot_pre,self.cur_rot],dt)[0]
            new_avel=smooth.quat_to_avel([all_rot,next_rot],dt)[0]
            self.d_avel= (prev_avel-new_avel)
            # 继续
            if self.smooth_frame>0:
                offset_pos=smooth.decay_spring_implicit_damping_pos(self.d_pos,self.d_vel,half_life,(self.cur_frame-self.smooth_frame)*dt)[0]
                offset_rot=smooth.decay_spring_implicit_damping_rot(self.d_rot,self.d_avel,half_life,(self.cur_frame-self.smooth_frame)*dt)[0]
                all_pos+=offset_pos
                all_rot=R.from_rotvec(R.from_quat(all_rot).as_rotvec()+offset_rot).as_quat()
            
            self.cur_pos_pre=self.cur_pos
            self.cur_rot_pre=self.cur_rot
            self.cur_pos=all_pos
            self.cur_rot=all_rot
            
            
            all_pos=all_pos.reshape(1,-1,3)
            all_rot=all_rot.reshape(1,-1,4)
            joint_translation, joint_orientation = motion.batch_forward_kinematics(joint_position=all_pos, joint_rotation=all_rot)
            joint_translation = joint_translation[0]
            joint_orientation = joint_orientation[0]
            all_pos=all_pos[0]
            all_rot=all_rot[0]
            print(self.cur_frame,"jumped")
            self.root_pos_save=root_pos
            return joint_name, joint_translation, joint_orientation, all_pos, all_rot

    def sync_controller_and_character(self, character_root_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，
        这里用一个简单的方案，将手柄的位置对齐于角色我位置
        '''
        controller_pos = character_root_state
        self.controller.set_pos(controller_pos)
    
def shift_cost(cur_local_vel,cur_local_rot,cur_local_avel,selected_vel,selected_rot,selected_avel):
    """
    用于衡量转移的代价
    由于我们的匹配允许相对位置的移动和相对朝向的旋转，所以我们衡量的是其余的代价
    转移的代价只记相对姿势导致的，即除了根节点以外的其他关节的rot和avel，以及根节点的vel和avel
    """
    rate=[1,1,2,1]
    # 根关节速度和角速度
    #cost = np.linalg.norm(cur_local_vel[0]-selected_vel[:,0],axis=-1)*rate[0]
    cost = np.linalg.norm(cur_local_avel[0]-selected_avel[:,0],axis=-1)*rate[1]
    #非根关节的rot和avel
    cost += np.sum(np.linalg.norm(cur_local_rot.reshape(1,-1,4)-selected_rot,axis=-1),axis=-1)*rate[2]
    cost -= np.linalg.norm(cur_local_rot[0]-selected_rot[:,0],axis=-1)*rate[2]
    cost += np.sum(np.linalg.norm(cur_local_avel.reshape(1,-1,3)-selected_avel,axis=-1),axis=-1)*rate[3]
    return cost
    pass

def vel_cost(select_root_vel,cur_rot_vel):
    """_summary_

    Args:
        select_root_vel (N,3): _description_
        cur_rot_vel (3): _description_
    """
    cost = np.linalg.norm(select_root_vel-cur_rot_vel,axis=-1)
    return cost

def de_cost(desired_pos_list,s_root_pos,desired_vel_list,s_root_vel,\
                desired_rot_list,s_root_rot,desired_avel_list,s_root_avel,root_pos,root_rot):
    """
    输入的数据已经预处理完成
    因此在所有值上我们直接减，得到关注的6个关键帧的绝对偏差即可
    """
    # somehow 下面两个返回的是7个关键帧的速度，所以不知道什么情况，去掉第一个使其能跑
    desired_avel_list = desired_avel_list[:-1]
    desired_rot_list = desired_rot_list[:-1]
    # 经过测试，由于原来的desired_pos和vel实际上上限比较低，无法激活跑的动作，这里将其放大
    desired_pos_list[:]=1.9*(desired_pos_list[:]-desired_pos_list[0])+desired_pos_list[0]
    desired_vel_list=1.9*desired_vel_list
    # reshape
    #desired_pos_list.reshape(1,6,3)
    rate=[1,1,0.5,0.5]
    cost=np.sum(np.linalg.norm(desired_pos_list-s_root_pos,axis=-1),axis=-1)*rate[0]
    cost+=np.sum(np.linalg.norm(desired_vel_list-s_root_vel,axis=-1),axis=-1)*rate[1]
    cost+=np.sum(np.linalg.norm(desired_rot_list-s_root_rot,axis=-1),axis=-1)*rate[2]
    cost+=np.sum(np.linalg.norm(desired_avel_list-s_root_avel,axis=-1),axis=-1)*rate[3]
    cost+=np.linalg.norm(desired_avel_list-s_root_avel,axis=-1)[:,0]
    
    return cost
    pass

def cal_rot_offset(form,res):
    """计算从form到res的Y轴旋转

    Args:
        form (_type_): 三维旋转quat
        res (_type_): 三维旋转quat
        返回的是（-1，4）的数组
    """
    rot= np.array([0,0,0,1])
    form=form.reshape(-1,4)
    res=res.reshape(-1,4)
    form_in_y,_= new_decompose_rotation_with_yaxis(form)
    res_in_y,_= new_decompose_rotation_with_yaxis(res)
    res_in_y_rot = R.from_quat(res_in_y)
    form_in_y_rot = R.from_quat(form_in_y)
    rot= res_in_y_rot * form_in_y_rot.inv()
    rot = rot.as_quat()
    return rot

# 自己实现一个支持二维的broadcast的Y分解
def new_decompose_rotation_with_yaxis(rotations):
    '''
    输入: rotations 形状为 (N, 4) 的 ndarray, 包含多个四元数旋转
    输出: Ry, Rxz，分别为绕 y 轴的旋转和转轴在 xz 平面的旋转，并满足 R = Ry * Rxz
    '''
    rot = R.from_quat(rotations)

    matrices = rot.as_matrix()
    yaxis = matrices[:, :, 1].reshape(-1, 3)
    global_y = np.array([0, 1, 0]).reshape(1, 3)
    angles = np.arccos(np.sum(yaxis * global_y, axis=1))
    axes = np.cross(yaxis, global_y)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    rot_vecs = axes * angles[:, np.newaxis]

    rot_inv = R.from_rotvec(rot_vecs)
    Ry = (rot_inv * rot)
    Rxz = (Ry.inv() * rot)
    Ry = Ry.as_quat()
    Rxz = Rxz.as_quat()

    return Ry, Rxz