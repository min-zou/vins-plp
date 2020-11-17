#include "line_feature_manager.h"

LineFeatureManager::LineFeatureManager(Eigen::Matrix<double, 3, 3> *_Rs) : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void LineFeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void LineFeatureManager::clearState()
{};

//TODO:后端优化求解失败的时候应该移除掉失败线
void LineFeatureManager::removeFailures()
{
    for (auto it = line_feature.begin(), it_next = line_feature.begin();
            it != line_feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 1)
            line_feature.erase(it);
    }
}
void LineFeatureManager::removeLineOutlier(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{

    for (auto it_per_id = line_feature.begin(), it_next = line_feature.begin();
            it_per_id != line_feature.end(); it_per_id = it_next)
    {
        it_next++;
        it_per_id->used_num = it_per_id->line_feature_per_frame.size();
        if (!(it_per_id->used_num >= 2 && it_per_id->start_frame < WINDOW_SIZE - 2))
            continue;

        int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);

        Eigen::Vector3d twc = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d Rwc = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        // 计算初始帧上线段对应的3d端点
        Vector3d pc, nw, vw;
        // nc = it_per_id->line.head(3);
        // vc = it_per_id->line.tail(3);
        Utility::cvtOrthonormalToPlucker(it_per_id->line, nw, vw);
        Eigen::Vector6d line_w;
        line_w.head(3) = nw;
        line_w.tail(3) = vw;
//       double  d = nc.norm()/vc.norm();
//       if (d > 5.0)
        {
//           std::cerr <<"remove a large distant line \n";
//           line_feature.erase(it_per_id);
//           continue;
        }

        auto line_c = Utility::plk_to_pose( line_w, Rwc.transpose(), -Rwc.transpose() * twc);

        Matrix4d Lc;
        Lc << Utility::skewSymmetric(line_c.head(3)), line_c.tail(3), -line_c.tail(3).transpose(), 0;


        Eigen::Vector3d p11 = it_per_id->line_feature_per_frame[0].pts_s;
        Eigen::Vector3d p21 = it_per_id->line_feature_per_frame[0].pts_e;

        Vector2d ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
        ln = ln / ln.norm();

        Vector3d p12 = Vector3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
        Vector3d p22 = Vector3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
        Vector3d cam = Vector3d( 0, 0, 0 );

        Vector4d pi1 = Utility::pi_from_ppp(cam, p11, p12);
        Vector4d pi2 = Utility::pi_from_ppp(cam, p21, p22);

        Vector4d e1 = Lc * pi1;
        Vector4d e2 = Lc * pi2;
        e1 = e1 / e1(3);
        e2 = e2 / e2(3);

        //std::cout << "line endpoint: "<<e1 << "\n "<< e2<<"\n";
        if (e1(2) < 0 || e2(2) < 0)
        {
            line_feature.erase(it_per_id);
            continue;
        }
        if ((e1 - e2).norm() > 10)
        {
            line_feature.erase(it_per_id);
            continue;
        }

        /*
                // 点到直线的距离不能太远啊
                Vector3d Q = plucker_origin(nc,vc);
                if(Q.norm() > 5.0)
                {
                    linefeature.erase(it_per_id);
                    continue;
                }
        */
        // 并且平均投影误差不能太大啊

        int i = 0;
        double allerr = 0;
        Eigen::Vector3d tij;
        Eigen::Matrix3d Rij;
        Eigen::Vector4d obs;

        //std::cout<<"reprojection_error: \n";
        for (auto &it_per_frame : it_per_id->line_feature_per_frame)   // 遍历所有的观测， 注意 start_frame 也会被遍历
        {
            imu_j++;

            obs << it_per_frame.pts_s[0], it_per_frame.pts_s[1], it_per_frame.pts_e[0], it_per_frame.pts_e[1];
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];

            double err =  reprojection_error(obs, R1, t1, line_w);

//            if(err > 0.0000001)
//                i++;
//            allerr += err;    // 计算平均投影误差

            if (allerr < err)   // 记录最大投影误差，如果最大的投影误差比较大，那就说明有outlier
                allerr = err;
        }
//        allerr = allerr / i;
        if (allerr > 3.0 / 500.0)
        {
//            std::cout<<"remove a large error\n";
            line_feature.erase(it_per_id);
        }
    }
}
double LineFeatureManager::reprojection_error( Vector4d obs, Matrix3d Rwc, Vector3d twc, Vector6d line_w )
{

    double error = 0;

    Vector3d n_w, d_w;
    n_w = line_w.head(3);
    d_w = line_w.tail(3);

    Vector3d p1, p2;
    p1 << obs[0], obs[1], 1;
    p2 << obs[2], obs[3], 1;

    Vector6d line_c = Utility::plk_from_pose(line_w, Rwc, twc);
    Vector3d nc = line_c.head(3);
    double sql = nc.head(2).norm();
    nc /= sql;

    error += fabs( nc.dot(p1) );
    error += fabs( nc.dot(p2) );

    return error / 2.0;
}
void LineFeatureManager::addFeature(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &line_image, double td)
{
    ROS_DEBUG("input line feature: %d", (int) line_image.size());
    ROS_DEBUG("num of line feature: %d", getFeatureCount());
    for (auto &id_lines : line_image)//遍历这一帧里面所有的线段
    {
        LineFeaturePerFrame line_f_per_fra(id_lines.second.front().second, td);

        int line_feature_id = id_lines.first;
        auto it = find_if(line_feature.begin(), line_feature.end(), [line_feature_id](const LineFeaturePerId & it) {
            return it.line_feature_id == line_feature_id;
        });//在line_feature里面寻找是否有相同id的线特征存在

        if (it == line_feature.end())//如果不存在则新建一个id的线特，并将这个特征在当前帧的观测添加进去
        {
            line_feature.push_back(LineFeaturePerId(line_feature_id, frame_count));
            line_feature.back().line_feature_per_frame.push_back(line_f_per_fra);
        }
        else if (it->line_feature_id == line_feature_id)//如果存在则在相应id的线特征中添加当前帧的观测
        {
            it->line_feature_per_frame.push_back(line_f_per_fra);
        }
    }
}

//边缘化最老帧相关的特征点
void LineFeatureManager::removeBack()
{
    for (auto it = line_feature.begin(), it_next = line_feature.begin();
            it != line_feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->line_feature_per_frame.erase(it->line_feature_per_frame.begin());
            if (it->line_feature_per_frame.size() == 0)
                line_feature.erase(it);
        }
    }
}


//边缘化次新帧相关的特征点
void LineFeatureManager::removeFront(int frame_count)
{
    for (auto it = line_feature.begin(), it_next = line_feature.begin();
            it != line_feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;//从起始帧到次新帧的位置
            //如果次新帧之前已经跟踪结束就什么都不做
            if (it->endFrame() < frame_count - 1)
                continue;

            //如果次新帧仍然被跟踪，则删除feature_per_frame中次新帧对应的FeaturePerFrame
            it->line_feature_per_frame.erase(it->line_feature_per_frame.begin() + j);
            if (it->line_feature_per_frame.size() == 0)
                line_feature.erase(it);
        }
    }
}

//返回当前线特征的数量
int LineFeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : line_feature)
    {

        it.used_num = it.line_feature_per_frame.size();

        if (it.is_triangulation && it.used_num >= LINE_MIN_OBS && it.start_frame < WINDOW_SIZE - 2) //要求线特征的被观察的数量大于2,也就是有用的线特征
        {
            cnt++;
        }
    }
    return cnt;
}

//返回当前所有的世界坐标系下线特征
vector<vector<double>> LineFeatureManager::getLineVector()
{
    vector<vector<double>> lineVector;
    for (auto &it_per_id : line_feature)
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();

        if (it_per_id.is_triangulation && it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2) //要求线特征的被观察的数量大于2,也就是有用的线特征
        {
            lineVector.push_back(it_per_id.line);
        }
    }
    return lineVector;
}


void LineFeatureManager::setLineFeature(vector<vector<double>> lineVector)
{
    int feature_index = -1;
    for (auto &it_per_id : line_feature)
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();

        if (!(it_per_id.is_triangulation && it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2)) //要求线特征的被观察的数量大于2,也就是有用的线特征
        {
            continue;
        }
        it_per_id.line = lineVector[++feature_index];

        //TODO:如何仿照点一样判断是否求解成功，并移除失败线
        Eigen::Vector3d n, d;
        Utility::cvtOrthonormalToPlucker(it_per_id.line, n, d);
        Eigen::Vector3d dis = d.cross(n);//或者反过来，总之有一个是对的
        if (dis.z() < 0)
        {
            it_per_id.solve_flag = 1;//失败估计
        }
        else
        {
            it_per_id.solve_flag = 2;
        }
    }
}

void LineFeatureManager::line_triangulate(Eigen::Matrix<double, 3, 1> *Ps, Vector3d *tic, Matrix3d *ric)
{
    for (auto &it_per_id : line_feature)
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();
        if (it_per_id.is_triangulation)
        {
            continue;
        }
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // if (it_per_id.line.empty())
        //     continue;

        int s_f = it_per_id.start_frame;
        int e_f = it_per_id.start_frame + it_per_id.used_num - 1;

        Eigen::Vector3d t0 = Ps[s_f] + Rs[s_f] * tic[0];
        Eigen::Matrix3d R0 = Rs[s_f] * ric[0];

        Eigen::Vector3d t1 = Ps[e_f] + Rs[e_f] * tic[0]; //第R_1和t_1是第j帧在世界坐标系下的位姿
        Eigen::Matrix3d R1 = Rs[e_f] * ric[0];

        Eigen::Vector3d t = R0.transpose() * (t1 - t0);   // tij
        Eigen::Matrix3d R = R0.transpose() * R1;          // Rij

        Vector3d p3 = it_per_id.line_feature_per_frame.back().pts_s;
        Vector3d p4 = it_per_id.line_feature_per_frame.back().pts_e;
        p3 = R * p3 + t;
        p4 = R * p4 + t;
        Vector4d pij = Utility::pi_from_ppp(p3, p4, t);
        Eigen::Vector3d nj = pij.head(3); nj.normalize();
        auto pii = Utility::pi_from_ppp(it_per_id.line_feature_per_frame[0].pts_s, it_per_id.line_feature_per_frame[0].pts_e, Vector3d( 0, 0, 0 ));
        auto ni = pii.head(3); ni.normalize();
        double cos_theta = ni.dot(nj);
        if (cos_theta > 0.998)
        {
            continue;
        }
        //计算起始帧上直线构成的平面
        Vector3d pi_xyz_0 = it_per_id.line_feature_per_frame[0].pts_s.cross(it_per_id.line_feature_per_frame[0].pts_e);//起始点与终止点叉乘
        double pi_w_0 = pi_xyz_0.dot(t0);//pi_xyz和相机中心点成

        //计算结束帧上直线构成的平面
        Vector3d pi_xyz_1 = it_per_id.line_feature_per_frame.back().pts_s.cross(it_per_id.line_feature_per_frame.back().pts_e);//起始点与终止点叉乘
        double pi_w_1 = pi_xyz_1.dot(t1);//pi_xyz和相机中心点成

        Vector4d pi_0, pi_1;
        pi_0 << pi_xyz_0.x(), pi_xyz_0.y(), pi_xyz_0.z(), pi_w_0;//构建前后两帧pi平面
        pi_1 << pi_xyz_1.x(), pi_xyz_1.y(), pi_xyz_1.z(), pi_w_1;
        // pi_xyz_0.normalize();
        // pi_xyz_1.normalize();


        Matrix4d matrix_pu = pi_0 * pi_1.transpose() - pi_1 * pi_0.transpose();

        Vector3d pu_n, pu_d;
        pu_n = matrix_pu.block<3, 1>(0, 3);
        pu_d << -matrix_pu(1, 2), matrix_pu(0, 2), -matrix_pu(0, 1);

        it_per_id.is_triangulation = true;



        it_per_id.line.resize(5);
        Utility::cvtPluckerToOrthonormal(pu_n, pu_d, it_per_id.line);

        //TODO:在点的三角化中会对求解结果有一个限制
    }
}

void LineFeatureManager::triangulateLine(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    //std::cout<<"linefeature size: "<<linefeature.size()<<std::endl;
    for (auto &it_per_id : line_feature)        // 遍历每个特征，对新特征进行三角化
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();    // 已经有多少帧看到了这个特征
        if (!(it_per_id.used_num >= 5 && it_per_id.start_frame < WINDOW_SIZE - 2))   // 看到的帧数少于2， 或者 这个特征最近倒数第二帧才看到， 那都不三角化
            continue;

        if (it_per_id.is_triangulation)       // 如果已经三角化了
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);

        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        double d = 0, min_cos_theta = 1.0;
        Eigen::Vector3d tij;
        Eigen::Matrix3d Rij;
        Eigen::Vector4d obsi, obsj; // obs from two frame are used to do triangulation

        // plane pi from ith obs in ith camera frame
        Eigen::Vector4d pii;
        Eigen::Vector3d ni;      // normal vector of plane
        for (auto &it_per_frame : it_per_id.line_feature_per_frame)   // 遍历所有的观测， 注意 start_frame 也会被遍历
        {
            imu_j++;

            if (imu_j == imu_i)  // 第一个观测是start frame 上
            {
                obsi << it_per_frame.pts_s[0], it_per_frame.pts_s[1], it_per_frame.pts_e[0], it_per_frame.pts_e[1];
                // Eigen::Vector3d p1( it_per_frame.pts_s[0], it_per_frame.pts_s[1], 1 );
                // Eigen::Vector3d p2( it_per_frame.pts_e[0], it_per_frame.pts_e[1], 1 );
                pii = Utility::pi_from_ppp(it_per_frame.pts_s, it_per_frame.pts_e, Vector3d( 0, 0, 0 ));
                ni = pii.head(3); ni.normalize();
                continue;
            }

            // 非start frame(其他帧)上的观测
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];

            Eigen::Vector3d t = R0.transpose() * (t1 - t0);   // tij
            Eigen::Matrix3d R = R0.transpose() * R1;          // Rij

            Eigen::Vector4d obsj_tmp;
            obsj_tmp << it_per_frame.pts_s[0], it_per_frame.pts_s[1], it_per_frame.pts_e[0], it_per_frame.pts_e[1];


            // plane pi from jth obs in ith camera frame
            Vector3d p3( obsj_tmp(0), obsj_tmp(1), 1 );
            Vector3d p4( obsj_tmp(2), obsj_tmp(3), 1 );
            p3 = R * p3 + t;
            p4 = R * p4 + t;
            Vector4d pij = Utility::pi_from_ppp(p3, p4, t);
            Eigen::Vector3d nj = pij.head(3); nj.normalize();

            double cos_theta = ni.dot(nj);
            if (cos_theta < min_cos_theta)
            {
                min_cos_theta = cos_theta;
                tij = t;
                Rij = R;
                obsj = obsj_tmp;
                d = t.norm();
            }
            // if( d < t.norm() )  // 选择最远的那俩帧进行三角化
            // {
            //     d = t.norm();
            //     tij = t;
            //     Rij = R;
            //     obsj = it_per_frame.lineobs;      // 特征的图像坐标
            // }

        }

        // if the distance between two frame is lower than 0.1m or the parallax angle is lower than 15deg , do not triangulate.
        // if(d < 0.1 || min_cos_theta > 0.998)
        if (min_cos_theta > 0.998)
            // if( d < 0.2 )
            continue;

        // plane pi from jth obs in ith camera frame
        Vector3d p3( obsj(0), obsj(1), 1 );
        Vector3d p4( obsj(2), obsj(3), 1 );
        p3 = Rij * p3 + tij;
        p4 = Rij * p4 + tij;
        Vector4d pij = Utility::pi_from_ppp(p3, p4, tij);

        Vector6d plk = Utility::pipi_plk( pii, pij );
        Vector3d n = plk.head(3);
        Vector3d v = plk.tail(3);

        //Vector3d cp = plucker_origin( n, v );
        //if ( cp(2) < 0 )
        {
            //  cp = - cp;
            //  continue;
        }

        //Vector6d line;
        //line.head(3) = cp;
        //line.tail(3) = v;
        //it_per_id.line_plucker = line;

        // plk.normalize();
        it_per_id.line_plucker = plk;  // plk in camera frame
        it_per_id.is_triangulation = true;

        //  used to debug
        Vector3d pc, nc, vc;
        nc = it_per_id.line_plucker.head(3);
        vc = it_per_id.line_plucker.tail(3);


        Matrix4d Lc;
        Lc << Utility::skewSymmetric(nc), vc, -vc.transpose(), 0;

        auto obs_startframe = it_per_id.line_feature_per_frame[0];   // 第一次观测到这帧
        Vector3d p11 = obs_startframe.pts_s;
        Vector3d p21 = obs_startframe.pts_e;
        Vector2d ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
        ln = ln / ln.norm();

        Vector3d p12 = Vector3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
        Vector3d p22 = Vector3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
        Vector3d cam = Vector3d( 0, 0, 0 );

        Vector4d pi1 = Utility::pi_from_ppp(cam, p11, p12);
        Vector4d pi2 = Utility::pi_from_ppp(cam, p21, p22);

        Vector4d e1 = Lc * pi1;
        Vector4d e2 = Lc * pi2;
        e1 = e1 / e1(3);
        e2 = e2 / e2(3);

        Vector3d pts_1(e1(0), e1(1), e1(2));
        Vector3d pts_2(e2(0), e2(1), e2(2));

        Vector3d w_pts_1 =  Rs[imu_i] * (ric[0] * pts_1 + tic[0]) + Ps[imu_i];
        Vector3d w_pts_2 =  Rs[imu_i] * (ric[0] * pts_2 + tic[0]) + Ps[imu_i];
        // it_per_id.ptw1 = w_pts_1;
        // it_per_id.ptw2 = w_pts_2;

    }

//    removeLineOutlier(Ps,tic,ric);
}







