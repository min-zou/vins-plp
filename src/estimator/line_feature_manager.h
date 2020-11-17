/*
PLVIO
 */

#ifndef SRC_LINE_FEATURE_MANAGER_H
#define SRC_LINE_FEATURE_MANAGER_H

#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
using namespace std;

class LineFeaturePerFrame
{
public:
    LineFeaturePerFrame(const Eigen::Matrix<double, 4, 1> &line, double td)
    {
        pts_s.x() = line(0);
        pts_s.y() = line(1);
        pts_s.z() = 1;
        pts_e.x() = line(2);
        pts_e.y() = line(3);
        pts_e.z() = 1;
    }
    Vector3d pts_s;
    Vector3d pts_e;
};

class LineFeaturePerId
{
public:
    LineFeaturePerId(int _line_feature_id, int _start_frame)
        : start_frame(_start_frame), line_feature_id(_line_feature_id),
          used_num(0), solve_flag(0),is_triangulation(false)
    {
    }

    int endFrame()
    {
        return start_frame + line_feature_per_frame.size() - 1;
    }

    vector<LineFeaturePerFrame> line_feature_per_frame;
    int start_frame;
    int line_feature_id;
    bool is_triangulation = false;
    vector<double> line;
    int used_num;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;


    bool is_outlier;
    bool is_margin;
    Vector6d line_plucker;
    Vector4d obs_init;
    Vector4d obs_j;

    Eigen::Vector3d tj_;   // tij
    Eigen::Matrix3d Rj_;
    Eigen::Vector3d ti_;   // tij
    Eigen::Matrix3d Ri_;
    int removed_cnt;
    int all_obs_cnt;    // 总共观测多少次了？
};

class LineFeatureManager
{
public:
    LineFeatureManager(Matrix3d _Rs[]);
    void setRic(Matrix3d _ric[]);
    void clearState();
    void removeFailures();
    void removeLineOutlier(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    list<LineFeaturePerId> line_feature;
    void addFeature(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &line_image, double td);
    void removeBack();
    void removeFront(int frame_count);
    //TODO：实现直线三角化函数
    void line_triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    vector<vector<double>> getLineVector();
    int getFeatureCount();
    void setLineFeature(vector<vector<double>> lineVector);
    double reprojection_error( Vector4d obs, Matrix3d Rwc, Vector3d twc, Vector6d line_w );
    void triangulateLine(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    const Matrix3d *Rs;
    Matrix3d ric[2];
};


#endif //SRC_LINE_FEATURE_MANAGER_H
