/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/
#include<ros/ros.h>

#include "feature_tracker.h"
#include <string>
#include <cmath>
using namespace cv::ximgproc;


cv::Mat map1, map2;
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
	const int BORDER_SIZE = 1;
	int img_x = cvRound(pt.x);
	int img_y = cvRound(pt.y);
	return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}
bool FeatureTracker::lineinSquare(const cv::Point2f &pt)
{
	const int BORDER_SIZE = 6;
	int img_x = cvRound(pt.x);
	int img_y = cvRound(pt.y);
	return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
	//printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
	double dx = pt1.x - pt2.x;
	double dy = pt1.y - pt2.y;
	return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
	int j = 0;
	for (int i = 0; i < int(v.size()); i++)
		if (status[i])// 只把正常跟踪的留在向量中
			v[j++] = v[i];
	v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
	int j = 0;
	for (int i = 0; i < int(v.size()); i++)
		if (status[i])
			v[j++] = v[i];
	v.resize(j);
}

FeatureTracker::FeatureTracker()
{
	stereo_cam = 0;
	n_id = 0;
	n_kl_id = 0;
	hasPrediction = false;
	int length_threshold = 50;
	float distance_threshold = 1.41421356f;
	double canny_th1 = 35.0;
	double canny_th2 = 20.0;
	int canny_aperture_size = 7;
	bool do_merge = true;
	fld = createFastLineDetector(length_threshold,
	                             distance_threshold, canny_th1, canny_th2, canny_aperture_size,
	                             do_merge);

	lsd = cv::line_descriptor::LSDDetectorC::createLSDDetectorC();

	opts.refine       = 0;
	opts.scale        = 1.2;
	opts.sigma_scale  = 0.6;
	opts.quant        = 2.0;
	opts.ang_th       = 22.5;
	opts.log_eps      = 1.0;
	opts.density_th   = 0.6;
	opts.n_bins       = 1024;
	opts.min_length   = 50;
}

void FeatureTracker::setMask()
{
	// 一张纯白
	mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
	// line_mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
	// prefer to keep features that are tracked for long time
	// 该点的跟踪次数、该点的位置、该点的ID
	vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

	for (unsigned int i = 0; i < cur_pts.size(); i++)
		cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

	// 根据跟踪次数，按照从多到少来排序
	sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
	{
		return a.first > b.first;
	});

	cur_pts.clear();
	ids.clear();
	track_cnt.clear();

	for (auto &it : cnt_pts_id)
	{
		// ??? 会出现重复点 ???
		if (mask.at<uchar>(it.second.first) == 255)
		{
			cur_pts.push_back(it.second.first);
			ids.push_back(it.second.second);
			track_cnt.push_back(it.first);
			// 化成黑点
			cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
		}
	}
	// dbg("set line mask");
	// for (auto &l : cl_pts)
	// {
	//     // dbg(l);
	//     if (line_mask.at<uchar>(l.first) == 255 ||
	//             line_mask.at<uchar>(l.second) == 255)
	//     {
	//         cv::line(line_mask, l.first, l.second, 0, MIN_DIST);
	//         cv::circle(line_mask, l.first, MIN_DIST * 0.5, 0, -1);
	//         cv::circle(line_mask, l.second, MIN_DIST * 0.5, 0, -1);
	//     }

	// }
	// dbg("mask over");
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
	//printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
	double dx = pt1.x - pt2.x;
	double dy = pt1.y - pt2.y;
	return sqrt(dx * dx + dy * dy);
}



void FeatureTracker::lsd_ex( cv::Mat image, cv::Mat &mask, std::vector<KeyLine>& _keylines)
{


	int length_threshold = 50;
	float distance_threshold = 1.41421356f;
	double canny_th1 = 40.0;
	double canny_th2 = 40.0;
	int canny_aperture_size = 3;
	bool do_merge = true;
	cv::Ptr<FastLineDetector> fld = createFastLineDetector(length_threshold,
	                                distance_threshold, canny_th1, canny_th2, canny_aperture_size,
	                                do_merge);

	nl_pts.clear();
	fld->detect(image, nl_pts);
}

void FeatureTracker::lineTrack(const cv::Mat &prev_img, const cv::Mat &cur_img, std::vector<cv::Vec4f> &prev_lines,
                               std::vector<cv::Vec4f> &cur_lines, std::vector<int> &status)
{
	cv::Mat show_img, show_img2;
	// dbg("what");
	cur_img.copyTo(show_img);
	prev_img.copyTo(show_img2);
	cv::cvtColor(show_img, show_img, CV_GRAY2RGB);
	cv::cvtColor(show_img2, show_img2, CV_GRAY2RGB);

	// dbg("sdadalda");
	fld->detect(cur_img, cur_lines);
	fld->detect(prev_img, prev_lines);

	std::vector<int> state(prev_lines.size(), 0);
	std::vector<int> match(cur_lines.size(), -1);
	int cur_id = 0;
	// dbg("bug1");

	for (auto &cl : cur_lines)
	{
		// dbg("bug2");

		auto center = cv::Point2f(0.5 * (cl[0] + cl[2]), 0.5 * (cl[1] + cl[3]));
		auto center1 = cv::Point2f(0.5 * (center.x + cl[2]), 0.5 * (center.y + cl[3]));
		auto center2 = cv::Point2f(0.5 * (cl[0] + center.x), 0.5 * (cl[1] + center.y));
		auto center3 = cv::Point2f(0.5 * (center.x + center1.x), 0.5 * (center.y + center1.y));
		auto center4 = cv::Point2f(0.5 * (center.x + center2.x), 0.5 * (center.y + center2.y));//5 sample points
		double angle_cl = (cl[3] - cl[1]) / (cl[2] - cl[0]);
		double theta_cl = fabs(atan(angle_cl));
		if (angle_cl < 0)
		{
			theta_cl = 3.141593 - theta_cl;
		}
		// dbg(theta_cl);
		int id = -1 ;
		// dbg("bug3");
		int min = 100;
		int match_id = -1;
		for (auto &pl : prev_lines)
		{
			id++;
			if (state[id])
			{
				continue;
			}
			auto center_p = cv::Point2f(0.5 * (pl[0] + pl[2]), 0.5 * (pl[1] + pl[3]));

			double angle_pl = (pl[3] - pl[1]) / (pl[2] - pl[0]);
			double theta_pl = fabs(atan(angle_pl));
			if (angle_pl < 0)
			{
				theta_pl = 3.141593 - theta_pl;
			}
			// dbg(theta_pl);
			// dbg(center_p);
			// dbg(center);
			double center_ang = (center_p.y - center.y) / (center_p.x - center.x);
			double theta_center = fabs(atan(center_ang));

			if (center_ang < 0)
			{
				theta_center = 3.141593 - theta_center;
			}
			// dbg(theta_center);
			double delta_ = fabs(theta_pl + theta_cl - 2 * theta_center) * 180 / 3.141593;
			Eigen::Vector2d c_d(center_p.x - center.x, center_p.y - center.y);
			double delta_c = c_d.norm();
			// dbg(delta_);
			if (delta_ < 22 && delta_c <= 16)
			{
				auto center1_p = cv::Point2f(0.5 * (center_p.x + pl[2]), 0.5 * (center_p.y + pl[3]));
				auto center2_p = cv::Point2f(0.5 * (pl[0] + center_p.x), 0.5 * (pl[1] + center_p.y));
				auto center3_p = cv::Point2f(0.5 * (center_p.x + center1_p.x), 0.5 * (center_p.y + center1_p.y));
				auto center4_p = cv::Point2f(0.5 * (center_p.x + center2_p.x), 0.5 * (center_p.y + center2_p.y));//5 sample points
				// dbg("bug3");

				float z1 = ZNCC(prev_img, cur_img, center_p, center, 8);
				// dbg("bug4");

				float z21 = ZNCC(prev_img, cur_img, center1_p, center2, 8);
				float z31 = ZNCC(prev_img, cur_img, center2_p, center1, 8);

				float z22 = ZNCC(prev_img, cur_img, center1_p, center1, 8);
				float z32 = ZNCC(prev_img, cur_img, center2_p, center2, 8);

				float z41 = ZNCC(prev_img, cur_img, center3_p, center3, 8);
				float z51 = ZNCC(prev_img, cur_img, center4_p, center4, 8);

				float z42 = ZNCC(prev_img, cur_img, center3_p, center4, 8);
				float z52 = ZNCC(prev_img, cur_img, center4_p, center3, 8);

				double z_t = z1 + z21 + z22 + z31 + z32 + z41 + z42 + z51 + z52;
				// bool match_ = z1 < 1 && (z21 < 1 | z22 < 1) && (z31 < 1 | z32 < 1) && (z41 < 1 | z42 < 1) && (z51 < 1 | z52 < 1);
				if (z_t < min)
				{
					min = z_t ;
					match_id = id;
					// dbg("get match");
				}
			}
		}
		if (min < 15)
		{
			match[cur_id] = match_id;
			state[match_id] = 1;
		}


		cur_id++;
	}
	dbg(match);
	int show_id = 0;
	for (int i = 0; i < match.size(); ++i)
	{
		if (match[i] >= 0)
		{
			cv::Scalar lineColor;

			int R = ( rand() % (int) ( 255 + 1 ) );
			int G = ( rand() % (int) ( 255 + 1 ) );
			int B = ( rand() % (int) ( 255 + 1 ) );

			lineColor = cv::Scalar( R, G, B );


			/* draw line */
			cv::line( show_img, cv::Point2f(cur_lines[i][0], cur_lines[i][1]),  cv::Point2f(cur_lines[i][2], cur_lines[i][3]),
			          lineColor, 1 );
			cv::line( show_img2, cv::Point2f(prev_lines[match[i]][0], prev_lines[match[i]][1]),
			          cv::Point2f(prev_lines[match[i]][2], prev_lines[match[i]][3]),
			          lineColor, 1 );
			std::string num = std::to_string(i);

			cv::putText(show_img, num, cv::Point2f(0.5 * (cur_lines[i][0] + cur_lines[i][2]), 0.5 * (cur_lines[i][3] + cur_lines[i][1])),
			            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0));
			cv::putText(show_img2, num, 0.5 * cv::Point2f(prev_lines[match[i]][0] + prev_lines[match[i]][2], prev_lines[match[i]][3] + prev_lines[match[i]][1]),
			            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0));

		}
	}

	cv::imshow("Source Image1", show_img);

	cv::imshow("Source Image2", show_img2);

	cv::waitKey(1);

}
// void FeatureTracker::NearbyLineTracking(const vector<Line> forw_lines, const vector<Line> cur_lines,
//                                             vector<pair<int, int> > &lineMatches)
void FeatureTracker::lineTrackwithZncc(const cv::Mat &prev_img, const cv::Mat &cur_img, const std::vector<cv::Vec4f> &prev_lines,
                                       std::vector<cv::Vec4f> &cur_lines, std::vector<int> &match, std::vector<int> &no_match)
{
	cv::Mat show_img, show_img2;
	// dbg("what");
	cur_img.copyTo(show_img);
	prev_img.copyTo(show_img2);
	cv::cvtColor(show_img, show_img, CV_GRAY2RGB);
	cv::cvtColor(show_img2, show_img2, CV_GRAY2RGB);

	// dbg("sdadalda");
	// fld->detect(cur_img, cur_lines);

	lsd->detect(cur_img, cur_lines, opts.scale, 1, opts);

	std::vector<int> state(prev_lines.size(), -1);
	match.resize(cur_lines.size(), -1);
	int cur_id = 0;
	// dbg("bug1");

	for (auto &cl : cur_lines)
	{
		// dbg("bug2");

		auto center = cv::Point2f(0.5 * (cl[0] + cl[2]), 0.5 * (cl[1] + cl[3]));
		auto center1 = cv::Point2f(0.5 * (center.x + cl[2]), 0.5 * (center.y + cl[3]));
		auto center2 = cv::Point2f(0.5 * (cl[0] + center.x), 0.5 * (cl[1] + center.y));
		auto center3 = cv::Point2f(0.5 * (center.x + center1.x), 0.5 * (center.y + center1.y));
		auto center4 = cv::Point2f(0.5 * (center.x + center2.x), 0.5 * (center.y + center2.y));//5 sample points
		if (center1.x  > center2.x)
		{
			auto temp = center1;
			center1 = center2;
			center2 = center1;
		}
		if (center3.x > center4.x)
		{

			auto temp = center3;
			center3 = center4;
			center4 = center3;
		}
		// double angle_cl = (cl[3] - cl[1]) / (cl[2] - cl[0]);
		// double theta_cl = fabs(atan(angle_cl));
		// if (angle_cl < 0)
		// {
		//     theta_cl = 3.141593 - theta_cl;
		// }
		// dbg(theta_cl);
		int id = -1 ;
		// dbg("bug3");
		double min = 100;
		double center_length = 100;
		// dbg(state);
		int match_id = -1;
		for (auto &pl : prev_lines)
		{
			id++;
			// dbg(state[id]);
			if (state[id] >= 0)
			{
				// dbg("wtha");
				continue;
			}
			auto center_p = cv::Point2f(0.5 * (pl[0] + pl[2]), 0.5 * (pl[1] + pl[3]));

			// double angle_pl = (pl[3] - pl[1]) / (pl[2] - pl[0]);
			// double theta_pl = fabs(atan(angle_pl));
			// if (angle_pl < 0)
			// {
			//     theta_pl = 3.141593 - theta_pl;
			// }
			// dbg(theta_pl);
			// dbg(center_p);
			// dbg(center);
			// double center_ang = (center_p.y - center.y) / (center_p.x - center.x);
			// double theta_center = fabs(atan(center_ang));

			// if (center_ang < 0)
			// {
			//     theta_center = 3.141593 - theta_center;
			// }
			// dbg(theta_center);
			// double delta_ = fabs(theta_pl + theta_cl - 2 * theta_center) * 180 / 3.141593;
			Eigen::Vector2d c_d(center_p.x - center.x, center_p.y - center.y);
			double delta_c = c_d.norm();
			Eigen::Vector2d serr(pl[0] - cl[0], pl[1] - cl[1]);
			Eigen::Vector2d eerr(pl[2] - cl[2], pl[3] - cl[3]);
			double error = serr.transpose() * eerr;
			// dbg(delta_);
			// dbg(delta_c);
			if (/*delta_ < 22 &&*/ delta_c <= 25 && error < 0.25 * 60 * 60)
			{
				auto center1_p = cv::Point2f(0.5 * (center_p.x + pl[2]), 0.5 * (center_p.y + pl[3]));
				auto center2_p = cv::Point2f(0.5 * (pl[0] + center_p.x), 0.5 * (pl[1] + center_p.y));
				auto center3_p = cv::Point2f(0.5 * (center_p.x + center1_p.x), 0.5 * (center_p.y + center1_p.y));
				auto center4_p = cv::Point2f(0.5 * (center_p.x + center2_p.x), 0.5 * (center_p.y + center2_p.y));//5 sample points
				// dbg("bug3");
				if (center1_p.x > center2_p.x)
				{

					auto temp = center1_p;
					center1_p = center2_p;
					center2_p = center1_p;

				}
				if (center3_p.x > center4_p.x)
				{

					auto temp = center3_p;
					center3_p = center4_p;
					center4_p = center3_p;

				}

				float z1 = ZNCC(prev_img, cur_img, center_p, center, 6);
				// dx(i, j) = [I(i + 1, j) - I(i - 1, j)] / 2;
				// dy(i, j) = [I(i, j + 1) - I(i, j - 1)] / 2;
				// double g11 = 0.5 * (cur_img.at<float>(int(center.y) + 1, int(center.x)) - cur_img.at<float>(int(center.y) - 1, int(center.x)))
				//              + 0.5 * (cur_img.at<float>(int(center.y), int(center.x) + 1) - cur_img.at<float>(int(center.y), int(center.x) - 1));
				// dbg("bug4");
				// double g21 = 0.5 * (prev_img.at<float>(int(center_p.y) + 1, int(center_p.x)) - prev_img.at<float>(int(center_p.y) - 1, int(center_p.x)))
				//              + 0.5 * (prev_img.at<float>(int(center_p.y), int(center_p.x) + 1) - prev_img.at<float>(int(center_p.y), int(center_p.x) - 1));

				// float z21 = ZNCC(prev_img, cur_img, center1_p, center2, 6);
				// float z31 = ZNCC(prev_img, cur_img, center2_p, center1, 6);

				float z22 = ZNCC(prev_img, cur_img, center1_p, center1, 6);
				float z32 = ZNCC(prev_img, cur_img, center2_p, center2, 6);

				float z41 = ZNCC(prev_img, cur_img, center3_p, center3, 6);
				float z51 = ZNCC(prev_img, cur_img, center4_p, center4, 6);

				// float z42 = ZNCC(prev_img, cur_img, center3_p, center4, 8);
				// float z52 = ZNCC(prev_img, cur_img, center4_p, center3, 8);

				double z_t = z1 + z22  + z32 + z41  + z51 /*+ z21 + z31 + z42 + z52*/;
				/*				dbg("####################");
								dbg(cur_img.at<float>(int(center.y) + 1, int(center.x)));
								dbg(g11 - g21);
								dbg(id);
								dbg(cur_id);
								dbg(z_t);
								dbg(delta_c);
								dbg(fabs(min - z_t));
								dbg("####################");*/

				// bool match_ = z1 < 1 && (z21 < 1 | z22 < 1) && (z31 < 1 | z32 < 1) && (z41 < 1 | z42 < 1) && (z51 < 1 | z52 < 1);
				if (z_t < min || fabs(min - z_t) < 1.0)
				{

					if (delta_c + 5 < center_length)
					{
						min = z_t ;
						match_id = id;
						center_length = delta_c;
					}
					// else
					// {
					// min = z_t ;
					// match_id = id;
					// center_length = delta_c;
					// delta_c =
					// }

					// dbg(z21);
					// dbg(z31);
					// dbg(z22);
					// dbg(z32);
					// dbg(z41);
					// dbg(z51);
					// dbg(z42);
					// dbg(z52);

					// dbg("get match");
				}
			}
		}
		// dbg(min);
		if (min < 8)
		{
			match[cur_id] = match_id;
			state[match_id] = 1;
		}
		else
		{
			no_match.push_back(cur_id);
		}
		cur_id++;
	}
	// dbg(match);
	// int show_id = 0;
	// cv::Mat img1 = show_img.clone();
	// cv::Mat img2 = show_img2.clone();
	// int line_id = 0;
	// for (auto id : no_match)
	// {
	//     cv::line( show_img, cv::Point2f(cur_lines[id][0], cur_lines[id][1]),  cv::Point2f(cur_lines[id][2], cur_lines[id][3]),
	//               cv::Scalar( 0, 0, 255 ), 1 );
	// }
	/*	for (auto&l : cur_lines)
		{
			cv::line( show_img, cv::Point2f(l[0], l[1]),  cv::Point2f(l[2], l[3]),
			          cv::Scalar( 0, 0, 255 ), 1 );

		}
		for (auto&l : prev_lines)
		{
			cv::line( show_img2, cv::Point2f(l[0], l[1]),  cv::Point2f(l[2], l[3]),
			          cv::Scalar( 0, 0, 255 ), 1 );
		}
		cv::line( show_img2, cv::Point2f(130, 130),  cv::Point2f(155, 130),
		          cv::Scalar( 0, 255, 0 ), 1 );
		for (int i = 0; i < match.size(); ++i)
		{
			if (match[i] >= 0)
			{
				cv::Scalar lineColor;

				int R = ( rand() % (int) ( 255 + 1 ) );
				int G = ( rand() % (int) ( 255 + 1 ) );
				int B = ( rand() % (int) ( 255 + 1 ) );

				lineColor = cv::Scalar( 255, 0, 1 );


				cv::line( show_img, cv::Point2f(cur_lines[i][0], cur_lines[i][1]),  cv::Point2f(cur_lines[i][2], cur_lines[i][3]),
				          lineColor, 1 );
				cv::line( show_img2, cv::Point2f(prev_lines[match[i]][0], prev_lines[match[i]][1]),
				          cv::Point2f(prev_lines[match[i]][2], prev_lines[match[i]][3]),
				          lineColor, 1 );
				std::string num = std::to_string(i);
				std::string num1 = std::to_string(match[i]);

				cv::putText(show_img, num + ":" + num1, cv::Point2f(0.5 * (cur_lines[i][0] + cur_lines[i][2]), 0.5 * (cur_lines[i][3] + cur_lines[i][1])),
				            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0));

				cv::putText(show_img2, num1 + ":" + num, 0.5 * cv::Point2f(prev_lines[match[i]][0] + prev_lines[match[i]][2], prev_lines[match[i]][3] + prev_lines[match[i]][1]),
				            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0));
			}
		}
		cv::Mat outImg = cv::Mat::zeros( show_img.rows, show_img.cols + show_img2.cols, show_img.type() );

		cv::Mat roi_left( outImg, cv::Rect( 0, 0, show_img.cols, show_img.rows ) );
		cv::Mat roi_right( outImg, cv::Rect( show_img.cols, 0, show_img2.cols, show_img2.rows ) );
		show_img.copyTo( roi_left );
		show_img2.copyTo( roi_right );

		cv::imshow("Source Image", outImg);
		cv::waitKey(1);*/



}
float FeatureTracker::ZNCC(cv::Mat lImg, cv::Mat rImg, cv::Point2i lPoint, cv::Point2i rPoint, int size)
{
	if (lImg.empty() || lImg.channels() != 1 || rImg.empty() || rImg.channels() != 1)
	{
		cout << "Image error in ZNCC!" << endl;
		return 100;
	}
	float diff = 100;
	float l_avg = 0, r_avg = 0;
	int size2 = (2 * size + 1) * (2 * size + 1);
	if (lPoint.x - size >= 0 && lPoint.x + size < lImg.cols && lPoint.y - size >= 0 && lPoint.y + size < lImg.rows
	        && rPoint.x - size >= 0 && rPoint.x + size < rImg.cols && rPoint.y - size >= 0 && rPoint.y + size < rImg.rows)
	{
		// dbg(".===================");
		for (int i = -size; i <= size; i++)
		{
			for (int j = -size; j <= size; j++)
			{
				{
					float temp = lImg.at<float>(lPoint.y + i, lPoint.x + j);
					float v1 = lImg.ptr<uchar>(lPoint.y + i)[lPoint.x + j];
					float v2 = rImg.ptr<uchar>(rPoint.y + i)[rPoint.x + j];
					l_avg = l_avg + v1 / size2;
					r_avg = r_avg + v2 / size2;
				}
			}
		}

		float lr = 0, ll = 0, rr = 0;//这些是自相关和互相关
		for (int i = -size; i <= size; i++)
		{
			for (int j = -size; j <= size; j++)
			{
				float v1 = lImg.ptr<uchar>(lPoint.y + i)[lPoint.x + j];
				float v2 = rImg.ptr<uchar>(rPoint.y + i)[rPoint.x + j];
				lr += (v1 - l_avg) * (v2 - r_avg);
				ll += (v1 - l_avg) * (v1 - l_avg);
				rr += (v2 - r_avg) * (v2 - r_avg);
			}
		}

		diff = fabs(sqrt(ll * rr) / lr); ////////////////////
		// dbg(diff);
	}
	else
	{
		diff = 100;
	}

	return diff;
}

void FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1,
                                map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame,
                                map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &linefeatureFrame)
{
	TicToc t_r;
	cur_time = _cur_time;
	cur_img = _img;
	row = cur_img.rows;
	col = cur_img.cols;
	cv::Mat rightImg = _img1;
	// cv::Mat test;
	if (USE_FISHEYE_REMAP)
	{
		cur_img_line = _img;

		cv::remap(cur_img_line, cur_img_line, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	}
	std::vector<cv::Vec4f> pl;
	std::vector<int> t_status;


	// cv::Mat img = cur_img.clone();
	// cv::Mat gradx = cv::Mat::zeros( cur_img.rows, cur_img.cols, CV_32F);
	// cv::Mat grady = cv::Mat::zeros(cur_img.rows, cur_img.cols, CV_32F);
	// cv::Mat mag =  cv::Mat::zeros(cur_img.rows, cur_img.cols, CV_32F);

	// cv::GaussianBlur( cur_img, img, cv::Size( 3, 3 ), 0, 0 );
	// cv::Scharr(img, gradx, CV_32F, 1, 0, 1 / 32.0);
	// cv::Scharr(img, grady, CV_32F, 0, 1, 1 / 32.0);
	// cv::magnitude(gradx, grady, mag);
	// cv::imshow("test", img);
	// cv::imshow("origin", cur_img);
	// cv::imshow("undist", test);
	// cv::waitKey(1);
	/*
	{
	    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	    clahe->apply(cur_img, cur_img);
	    if(!rightImg.empty())
	        clahe->apply(rightImg, rightImg);
	}
	*/
	cur_pts.clear();

	cur_lines.clear();
	auto time1 = ros::Time::now();

	if (prev_pts.size() > 0 && prev_lines.size()) // 光流跟踪
	{
		// dbg("test1");
		std::vector<int> lines_match;
		std::vector<int> new_lines;
		std::vector<cv::Vec4f> temp_l;
		if (USE_FISHEYE_REMAP)
		{
			lineTrackwithZncc(prev_img_line, cur_img_line, prev_lines,
			                  temp_l, lines_match, new_lines);
		}
		else
		{
			lineTrackwithZncc(prev_img, cur_img, prev_lines,
			                  temp_l, lines_match, new_lines);
		}
		auto temp_ids = ids_kl;
		auto temp_cnt = kl_cnt;
		int match_id = 0;
		for (int i = 0; i < lines_match.size(); ++i)
		{
			if (lines_match[i] >= 0)
			{
				// dbg(lines_match[i]);
				kl_cnt[match_id] = temp_cnt[lines_match[i]];
				ids_kl[match_id] = temp_ids[lines_match[i]];
				cur_lines.push_back(temp_l[i]);
				match_id++;
			}
		}
		kl_cnt.resize(cur_lines.size());
		ids_kl.resize(cur_lines.size());
		for (int i = 0; i < new_lines.size(); ++i)
		{
			cur_lines.push_back(temp_l[new_lines[i]]);
			kl_cnt.push_back(0);
			ids_kl.push_back(n_kl_id++);
		}
		// dbg("dsfssfs");
		// auto temp_prev_pts = prev_pts;

		// std::vector<cv::Point2f> temp_cur_pts;

		// std::vector<cv::Vec4f> temp_nl;
		// fld->detect(cur_img, temp_nl);
		TicToc t_o;
		vector<uchar> status;
		vector<float> err;
		if (hasPrediction)
		{
			cur_pts = predict_pts;
			cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

			int succ_num = 0;
			for (size_t i = 0; i < status.size(); i++)
			{
				if (status[i])
					succ_num++;
			}
			if (succ_num < 10) // 跟踪数量太少，扩大金字塔层数 2^3 = 8
				cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
		}
		// 进入到这里
		else {
			cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
			/*            auto first1 = temp_cur_pts.end() - 3 * pl_pts.size();
			            auto last1  = temp_cur_pts.end();
			            std::vector<cv::Point2f> cut1_vector(first1, last1);
			            int line_start_ = prev_pts.size();
			            int j = 0;
			            for (int i = line_start_; i < temp_cur_pts.size(); i = i + 3)
			            {
			                if ((status[i] || status[i + 2]) && status[i + 1] && true
			                        && inBorder(temp_cur_pts[i]) && inBorder(temp_cur_pts[i + 1]) && inBorder(temp_cur_pts[i + 2]))
			                {

			                    double dx1 = temp_cur_pts[i + 2].x - temp_cur_pts[i + 1].x;
			                    double dx2 = temp_cur_pts[i + 1].x - temp_cur_pts[i].x;
			                    double dx3 = temp_cur_pts[i + 2].x - temp_cur_pts[i].x;
			                    double k1 = (temp_cur_pts[i + 2].y - temp_cur_pts[i + 1].y) / dx1;
			                    double k2 = (temp_cur_pts[i + 1].y - temp_cur_pts[i].y) / dx2;
			                    double k3 = (temp_cur_pts[i + 2].y - temp_cur_pts[i].y) / dx3;

			                    if (k1 - k2 + k3 - k2 <= 0.002)
			                    {
			                        // dbg("here");
			                        kl_cnt[j] = kl_cnt[(i - line_start_) / 3];
			                        // dbg("here1");

			                        ids_kl[j++] = ids_kl[(i - line_start_) / 3];
			                        // dbg("here2");

			                        cl_pts.emplace_back(cv::Point2f(temp_cur_pts[i].x, temp_cur_pts[i].y), cv::Point2f(temp_cur_pts[i + 2].x, temp_cur_pts[i + 2].y));
			                    }
			                    // dbg(temp_cur_pts);
			                }
			            }
			            temp_cur_pts.resize(prev_pts.size());
			            cur_pts = temp_cur_pts;
			            kl_cnt.resize(j);
			            ids_kl.resize(j);
			            dbg(cl_pts.size());
			            dbg(j);*/
		}

		// reverse check
		// 反向搜索一次
		if (FLOW_BACK)
		{
			vector<uchar> reverse_status;
			vector<cv::Point2f> reverse_pts = prev_pts;
			cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			//cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3);
			for (size_t i = 0; i < cur_pts.size(); i++)
			{
				if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
				{
					status[i] = 1;
				}
				else
					status[i] = 0;
			}
		}

		// 保证在边界框内
		for (int i = 0; i < int(cur_pts.size()); i++)
			if (status[i] && !inBorder(cur_pts[i]))
				status[i] = 0;

		reduceVector(prev_pts, status);
		reduceVector(cur_pts, status);
		reduceVector(ids, status);
		reduceVector(track_cnt, status);
		ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
		//printf("track cnt %d\n", (int)ids.size());
	}

	// 把每一个的跟踪加1
	for (auto &n : track_cnt)
		n++;
	for (auto &n : kl_cnt)
	{
		n++;
	}
	dbg("before add feature");
	if (true)
	{
		// rejectWithF();
		ROS_DEBUG("set mask begins");
		TicToc t_m;
		// 对跟踪到的特征点 按照跟踪次数降序排列（认为特征点被跟踪到的次数越多越好）
		setMask();
		// cv::imshow("line", line_mask);
		// cv::imshow("point", mask);

		// cv::waitKey(1);
		ROS_DEBUG("set mask costs %fms", t_m.toc());

		ROS_DEBUG("detect feature begins");
		TicToc t_t;
		// 最大跟踪数 - 当前帧预测的点的数量
		int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
		if (n_max_cnt > 0)
		{

			if (mask.empty())
				cout << "mask is empty " << endl;
			if (mask.type() != CV_8UC1)
				cout << "mask type wrong " << endl;

			cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
			// dbg("line1");
			// nl_pts.clear();
			// std::vector<cv::Vec4f> temp_nl;
			if (prev_lines.empty())
			{
				fld->detect(cur_img, cur_lines);
				for (auto &l : cur_lines)
				{
					// cl_pts.emplace_back(cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]));
					ids_kl.push_back(n_kl_id++);
					kl_cnt.push_back(1);
				}
			}
			// dbg("hh", temp_nl.size());
			// for (auto& l : temp_nl)
			// {
			//     if (line_mask.at<uchar>(cv::Point2f(l[0], l[1])) == 255 &&
			//             line_mask.at<uchar>(cv::Point2f(l[2], l[3])) == 255)
			//     {
			//         nl_pts.push_back(l);
			//     }
			// }
			// dbg(nl_pts.size());
			// setMask();
			// std::vector<KeyLine> kls;
			// std::vector<Eigen::Vector3d> lineVec2d;
			// std::vector<std::vector<int> >  keycompare;
			// std::vector<cv::Point3d> vps;
			// for (auto line : kls)
			// {
			//     // auto line = kls[key];
			//     n_pts.push_back(cv::Point2f(line.startPointX, line.startPointY));
			//     n_pts.push_back(cv::Point2f(line.endPointX, line.endPointY));
			// }
		}
		else {
			dbg("enough");
			n_pts.clear();
			// nl_pts.clear();
		}

		// lsd_ex( cur_img, mask, n_kls);

		ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

		for (auto &p : n_pts)
		{
			cur_pts.push_back(p);
			ids.push_back(n_id++);
			track_cnt.push_back(1);
		}


		//printf("feature cnt after add %d\n", (int)ids.size());
	}

	cur_un_pts = undistortedPts(cur_pts, m_camera[0]);//左目特征点去畸变

	pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);//左目特征点速度
	// cur_un_lines = undistortedLinePts(cl_pts, m_un_camera[0]);
	// dbg(cur_un_lines.size());
	dbg(ids_kl.size());
	//
	// if (!_img1.empty() && stereo_cam) // 左目和右目之间做光流跟踪
	// {
	//     ids_right.clear();
	//     cur_right_pts.clear();
	//     cur_un_right_pts.clear();
	//     right_pts_velocity.clear();
	//     cur_un_right_pts_map.clear();
	//     if (!cur_pts.empty())
	//     {
	//         //printf("stereo image; track feature on right image\n");
	//         vector<cv::Point2f> reverseLeftPts;
	//         vector<uchar> status, statusRightLeft;
	//         vector<float> err;
	//         // cur left ---- cur right
	//         cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
	//         // reverse check cur right ---- cur left
	//         if (FLOW_BACK) // 右目和左目之间再做一次光流，去除误匹配
	//         {
	//             cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
	//             for (size_t i = 0; i < status.size(); i++)
	//             {
	//                 if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
	//                     status[i] = 1;
	//                 else
	//                     status[i] = 0;
	//             }
	//         }

	//         ids_right = ids;
	//         reduceVector(cur_right_pts, status);
	//         reduceVector(ids_right, status);
	//         // only keep left-right pts
	//         /*
	//         reduceVector(cur_pts, status);
	//         reduceVector(ids, status);
	//         reduceVector(track_cnt, status);
	//         reduceVector(cur_un_pts, status);
	//         reduceVector(pts_velocity, status);
	//         */
	//         cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);//右目特征点去畸变
	//         right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);//右目特征点速度
	//     }
	//     prev_un_right_pts_map = cur_un_right_pts_map;
	// }
	//

	auto time2 = ros::Time::now();
	dbg(time2.toSec() - time1.toSec());
	if (SHOW_TRACK) {
		// dbg(cv::useOptimized());
		const float FLT_SCALE = 1.f / (1 << 20);
		// dbg(FLT_SCALE);
		// dbg(cv::ocl::isOpenCLActivated());
		drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
		if (!cur_lines.empty())
		{
			drawKeylines(imTrack, cur_lines, cv::Scalar::all( -1 ));
			int id = 0;
			for (auto& l_pt : cur_lines)
			{
				std::string s = std::to_string(ids_kl[id]);
				if (kl_cnt[id] > 2)
				{
					std::string num = std::to_string(kl_cnt[id]);

					cv::putText(imTrack, num, cv::Point2f(l_pt[0], l_pt[1]),
					            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0));
				}
				cv::putText(imTrack, s, 0.5 * (cv::Point2f(l_pt[0], l_pt[1]) + cv::Point2f(l_pt[2], l_pt[3])),
				            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
				id++;
			}
		}
	}

	//各种赋值操作
	prev_img = cur_img;
	if (USE_FISHEYE_REMAP)
	{
		prev_img_line = cur_img_line;
	}
	prev_pts = cur_pts;
	prev_lines = cur_lines;
	prev_un_pts = cur_un_pts;
	prev_un_pts_map = cur_un_pts_map;
	prev_time = cur_time;
	hasPrediction = false;

	prevLeftPtsMap.clear();
	for (size_t i = 0; i < cur_pts.size(); i++)
		prevLeftPtsMap[ids[i]] = cur_pts[i];

	// key = 特征点ID， value = 0(左目下全是0) + [x,y,1,u,v,V_u,V_v]
	// map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

	for (size_t i = 0; i < ids.size(); i++)
	{
		int feature_id = ids[i];
		double x, y , z;
		x = cur_un_pts[i].x;
		y = cur_un_pts[i].y;
		z = 1;
		double p_u, p_v;
		p_u = cur_pts[i].x;
		p_v = cur_pts[i].y;
		int camera_id = 0;
		double velocity_x, velocity_y;
		velocity_x = pts_velocity[i].x;
		velocity_y = pts_velocity[i].y;

		Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
		xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
		featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
	}

	for (size_t i = 0; i < ids_kl.size(); i++)
	{
		int line_feature_id = ids_kl[i];
		// dbg(line_feature_id);
		int camera_id = 0;
		Eigen::Matrix<double, 4, 1> line_pts;

		Eigen::Vector2d a1(cur_lines[i][0], cur_lines[i][1]);
		Eigen::Vector3d b1;

		Eigen::Vector2d a2(cur_lines[i][2], cur_lines[i][3]);
		Eigen::Vector3d b2;
		if (USE_FISHEYE_REMAP)
		{
			m_un_camera[0]->liftProjective(a1, b1);
			m_un_camera[0]->liftProjective(a2, b2);

		}
		else
		{
			m_camera[0]->liftProjective(a1, b1);
			m_camera[0]->liftProjective(a2, b2);
		}
		line_pts << b1[0] , b1[1] , b2[0] , b2[1];
		// line_pts << cur_lines[i][0], cur_lines[i][1], cur_lines[i][2], cur_lines[i][3];
		linefeatureFrame[line_feature_id].emplace_back(camera_id,  line_pts);
	}
	if (!_img1.empty() && stereo_cam)// 双目处理 计算在像素坐标系下的运动速度
	{
		for (size_t i = 0; i < ids_right.size(); i++)
		{
			int feature_id = ids_right[i];
			double x, y , z;
			x = cur_un_right_pts[i].x;
			y = cur_un_right_pts[i].y;
			z = 1;
			double p_u, p_v;
			p_u = cur_right_pts[i].x;
			p_v = cur_right_pts[i].y;
			int camera_id = 1;
			double velocity_x, velocity_y;
			velocity_x = right_pts_velocity[i].x;
			velocity_y = right_pts_velocity[i].y;

			Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
			xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
			featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
		}
	}

	//printf("feature track whole time %f\n", t_r.toc());
	// return featureFrame;
}
void FeatureTracker::drawKeylines( cv::Mat& inoutImage, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& keylines,
                                   const cv::Scalar& color )
{
	for (auto &l : keylines)
	{
		/* decide lines' color  */
		cv::Scalar lineColor;
		if ( color == cv::Scalar::all( -1 ) )
		{
			int R = ( rand() % (int) ( 255 + 1 ) );
			int G = ( rand() % (int) ( 255 + 1 ) );
			int B = ( rand() % (int) ( 255 + 1 ) );

			lineColor = cv::Scalar( R, G, B );
		}

		else
			lineColor = color;

		/* draw line */
		cv::line( inoutImage, l.first, l.second, lineColor, 1 );
	}
}
void FeatureTracker::drawKeylines( cv::Mat& inoutImage, const std::vector<cv::Vec4f>& keylines,
                                   const cv::Scalar& color )
{
	for (auto &l : keylines)
	{
		/* decide lines' color  */
		cv::Scalar lineColor;
		if ( color == cv::Scalar::all( -1 ) )
		{
			int R = ( rand() % (int) ( 255 + 1 ) );
			int G = ( rand() % (int) ( 255 + 1 ) );
			int B = ( rand() % (int) ( 255 + 1 ) );

			lineColor = cv::Scalar( R, G, B );
		}

		else
			lineColor = color;

		/* draw line */
		cv::line( inoutImage, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), lineColor, 1 );
	}
}
// rejectWithF
void FeatureTracker::rejectWithF()
{
	if (cur_pts.size() >= 8)
	{
		ROS_DEBUG("FM ransac begins");
		TicToc t_f;
		vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
		for (unsigned int i = 0; i < cur_pts.size(); i++)
		{
			Eigen::Vector3d tmp_p;
			m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
			tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
			tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
			un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

			m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
			tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
			tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
			un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
		}

		vector<uchar> status;
		cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
		int size_a = cur_pts.size();
		reduceVector(prev_pts, status);
		reduceVector(cur_pts, status);
		reduceVector(cur_un_pts, status);
		reduceVector(ids, status);
		reduceVector(track_cnt, status);
		ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
		ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
	}
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file, std::vector<string> un_calib)
{
	for (size_t i = 0; i < calib_file.size(); i++)
	{
		ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
		camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
		m_camera.push_back(camera);
	}
	// 1330.53261064 * 0.4, 0.4 * 1330.11728828
	if (USE_FISHEYE_REMAP)
	{
		m_camera[0]->initUndistortRectifyMap(map1, map2, 433, 433, cv::Size(0, 0), 295, 271);
		for (size_t i = 0; i < un_calib.size(); i++)
		{
			ROS_INFO("reading paramerter of camera %s", un_calib[i].c_str());
			camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(un_calib[i]);
			m_un_camera.push_back(camera);
		}
	}

	if (calib_file.size() == 2)
		stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
	cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
	vector<Eigen::Vector2d> distortedp, undistortedp;
	for (int i = 0; i < col; i++)
		for (int j = 0; j < row; j++)
		{
			Eigen::Vector2d a(i, j);
			Eigen::Vector3d b;
			m_camera[0]->liftProjective(a, b);
			distortedp.push_back(a);
			undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
			//printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
		}
	for (int i = 0; i < int(undistortedp.size()); i++)
	{
		cv::Mat pp(3, 1, CV_32FC1);
		pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
		pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
		pp.at<float>(2, 0) = 1.0;
		//cout << trackerData[0].K << endl;
		//printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
		//printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
		if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
		{
			undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
		}
		else
		{
			//ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
		}
	}
	// turn the following code on if you need
	// cv::imshow(name, undistortedImg);
	// cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
	vector<cv::Point2f> un_pts;
	for (unsigned int i = 0; i < pts.size(); i++)
	{
		Eigen::Vector2d a(pts[i].x, pts[i].y);
		Eigen::Vector3d b;
		cam->liftProjective(a, b);
		un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
	}
	return un_pts;
}
std::vector<std::pair<cv::Point2f, cv::Point2f>>
        FeatureTracker::undistortedLinePts(std::vector<std::pair<cv::Point2f, cv::Point2f>> &lines, camodocal::CameraPtr cam)
{
	std::vector<std::pair<cv::Point2f, cv::Point2f>> un_lines;

	for (auto &l : lines)
	{
		Eigen::Vector2d a1(l.first.x, l.first.y), a2(l.second.x, l.second.y);
		Eigen::Vector3d b1, b2;
		cam->liftProjective(a1, b1);
		cam->liftProjective(a2, b2);

		un_lines.emplace_back(cv::Point2f(b1.x() / b1.z(), b1.y() / b1.z()),
		                      cv::Point2f(b2.x() / b2.z(), b2.y() / b2.z()));
	}
	return un_lines;
}
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
        map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
	vector<cv::Point2f> pts_velocity;
	cur_id_pts.clear();
	// 当前帧 ID-点
	for (unsigned int i = 0; i < ids.size(); i++)
	{
		cur_id_pts.insert(make_pair(ids[i], pts[i]));
	}

	// caculate points velocity
	if (!prev_id_pts.empty())
	{
		// 当前帧和上一帧的时间差
		double dt = cur_time - prev_time;

		for (unsigned int i = 0; i < pts.size(); i++)
		{
			std::map<int, cv::Point2f>::iterator it;
			it = prev_id_pts.find(ids[i]);
			if (it != prev_id_pts.end())
			{
				double v_x = (pts[i].x - it->second.x) / dt;
				double v_y = (pts[i].y - it->second.y) / dt;
				pts_velocity.push_back(cv::Point2f(v_x, v_y));
			}
			else
				pts_velocity.push_back(cv::Point2f(0, 0));

		}
	}
	// 第一帧，速度为0
	else
	{
		for (unsigned int i = 0; i < cur_pts.size(); i++)
		{
			pts_velocity.push_back(cv::Point2f(0, 0));
		}
	}
	return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
	//int rows = imLeft.rows;
	int cols = imLeft.cols;
	if (!imRight.empty() && stereo_cam)
		cv::hconcat(imLeft, imRight, imTrack);
	else
		imTrack = imLeft.clone();
	cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

	for (size_t j = 0; j < curLeftPts.size(); j++)
	{
		double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
		cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
	}
	if (!imRight.empty() && stereo_cam)
	{
		for (size_t i = 0; i < curRightPts.size(); i++)
		{
			cv::Point2f rightPt = curRightPts[i];
			rightPt.x += cols;
			cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
			//cv::Point2f leftPt = curLeftPtsTrackRight[i];
			//cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
		}
	}

	map<int, cv::Point2f>::iterator mapIt;
	for (size_t i = 0; i < curLeftIds.size(); i++)
	{
		int id = curLeftIds[i];
		mapIt = prevLeftPtsMap.find(id);
		if (mapIt != prevLeftPtsMap.end())
		{
			cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
		}
	}

	//draw prediction
	/*
	for(size_t i = 0; i < predict_pts_debug.size(); i++)
	{
	    cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
	}
	*/
	//printf("predict pts size %d \n", (int)predict_pts_debug.size());

	//cv::Mat imCur2Compress;
	//cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
	hasPrediction = true;
	predict_pts.clear();
	predict_pts_debug.clear();
	map<int, Eigen::Vector3d>::iterator itPredict;
	for (size_t i = 0; i < ids.size(); i++)
	{
		//printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
		int id = ids[i];
		itPredict = predictPts.find(id);
		if (itPredict != predictPts.end())
		{
			Eigen::Vector2d tmp_uv;

			m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);

			predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
			predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
		}
		else
			predict_pts.push_back(prev_pts[i]);
	}
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
	std::set<int>::iterator itSet;
	vector<uchar> status;
	for (size_t i = 0; i < ids.size(); i++)
	{
		itSet = removePtsIds.find(ids[i]);
		if (itSet != removePtsIds.end())
			status.push_back(0);
		else
			status.push_back(1);
	}

	reduceVector(prev_pts, status);
	reduceVector(ids, status);
	reduceVector(track_cnt, status);
}


cv::Mat FeatureTracker::getTrackImage()
{
	return imTrack;
}