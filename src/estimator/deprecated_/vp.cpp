#include "vp.h"
#include "time.h"
#include <iostream>
#include <thread>
using namespace std;

void VPDetection::run( std::vector<cv::Vec4f> &lines, std::vector<cv::Point3d> &vps,
                       std::vector<std::vector<int> > &clusters , const Eigen::Matrix3d &rcw)
{
	lines_ = lines;
	noiseRatio = 0.5;
	rcw_ = rcw;
	getLineParams();

	std::vector<std::vector<cv::Point3d> > vpHypo;
	std::vector<std::vector<double> > sphereGrid;

	std::thread VPHypVia2Lines(&VPDetection::getVPHypVia2Lines, this, ref(vpHypo));
	std::thread SphereGrids(&VPDetection::getSphereGrids, this, ref(sphereGrid));

	VPHypVia2Lines.join();
	SphereGrids.join();

	getBestVpsHyp( sphereGrid, vpHypo, vps );
	double thAngle = 6.0 / 180.0 * CV_PI;
	lines2Vps( thAngle, vps, clusters );
	int clusteredNum = 0;
	for ( int i = 0; i < 3; ++i )
	{
		clusteredNum += clusters[i].size();
	}
}
void VPDetection::getLineParams()
{
	lineInfos.reserve(lines.size());
	for (auto &line : lines)
	{
		Eigen::Vector3d p1(line[0] , line[1], 1.0);
		Eigen::Vector3d p2(line[2], line[3], 1.0);
		Eigen::Vector3d p12 = p2 - p1;
		LineInfo line_info;
		line_info.para = p1.cross( p2 );

		double dy = p12[1];
		double dx = p12[0];
		line_info.length = p12.norm();
		double angle = atan2( dy, dx );
		if ( angle < 0 )
		{
			angle += CV_PI;
		}
		line_info.orientation = angle;

		lineInfos.push_back(line_info);
	}

}
void VPDetection::getVPHypVia2Lines( std::vector<std::vector<cv::Point3d> > &vpHypo )
{
	int num = lines.size();

	int numVp2 = 360;
	// double stepVp2 = 2.0 * CV_PI / numVp2;
	double stepVp2 = CV_PI / numVp2;

	std::vector<std::vector<cv::Point3d> > temp(numVp2, std::vector<cv::Point3d>(3) );
	vpHypo = temp;
	int count = 0;
	// get the vp1,init with imu gravity direction 
	Eigen::Vector3d vp1;
	Eigen::Matrix3d k;
	k << f , 0.0 , pp.x, 0.0, f, pp.y, 0.0, 0.0, 1.0;
	vp1 = rcw * k * Eigen::Vector3d(0, 0, 1);//首先确定z方向的灭点
	if ( vp1(2) == 0 ) { vp1(2) = 0.0011; }
	double N = vp1.norm();
	vp1 *= 1.0 / N;

	// get the vp2 and vp3
	Vector3d vp2(0.0, 0.0, 0.0);
	Vector3d vp3(0.0, 0.0, 0.0);
	for ( int j = 0; j < numVp2; ++ j )
	{
		// vp2
		double lambda = j * stepVp2;

		double k1 = vp1(0) * sin( lambda ) + vp1(1) * cos( lambda );
		double k2 = vp1(2);
		double phi = atan( - k2 / k1 );

		double Z = cos( phi );
		double X = sin( phi ) * sin( lambda );
		double Y = sin( phi ) * cos( lambda );

		vp2(0) = X;  vp2(1) = Y;  vp2(2) = Z;
		if ( vp2(2) == 0.0 ) { vp2(2) = 0.0011; }
		N = vp2.norm();
		vp2 *= 1.0 / N;
		if ( vp2(2) < 0 ) { vp2 *= -1.0; }

		// vp3
		vp3 = vp1.cross( vp2 );
		if ( vp3(2) == 0.0 ) { vp3(2) = 0.0011; }
		N = vp3.norm();
		vp3 *= 1.0 / N;
		if ( vp3(2) < 0 ) { vp3 *= -1.0; }

		//
		vpHypo[count][0] = cv::Point3d( vp1(0), vp1(1), vp1(2) );
		vpHypo[count][1] = cv::Point3d( vp2(0), vp2(1), vp2(2) );
		vpHypo[count][2] = cv::Point3d( vp3(0), vp3(1), vp3(2) );

		count ++;
	}

}


void VPDetection::getSphereGrids( std::vector<std::vector<double> > &sphereGrid )
{
	// build sphere grid with 1 degree accuracy
	double angelAccuracy = 1.0 / 180.0 * CV_PI;
	double angleSpanLA = CV_PI / 2.0;
	double angleSpanLO = CV_PI * 2.0;
	int gridLA = angleSpanLA / angelAccuracy;
	int gridLO = angleSpanLO / angelAccuracy;

	std::vector<std::vector<double> > temp( gridLA, std::vector<double>(gridLO) );
	sphereGrid = temp ;
	for ( int i = 0; i < gridLA; ++i )
	{
		for ( int j = 0; j < gridLO; ++j )
		{
			sphereGrid[i][j] = 0.0;
		}
	}

	// put intersection points into the grid
	double angelTolerance = 60.0 / 180.0 * CV_PI;
	Eigen::Vector3d ptIntersect;
	double x = 0.0, y = 0.0;
	double X = 0.0, Y = 0.0, Z = 0.0, N = 0.0;
	double latitude = 0.0, longitude = 0.0;
	int LA = 0, LO = 0;
	double angleDev = 0.0;
	for ( int i = 0; i < lines.size() - 1; ++i )
	{
		for ( int j = i + 1; j < lines.size(); ++j )
		{
			ptIntersect = lineInfos[i].para.cross( lineInfos[j].para );

			if ( ptIntersect[2] == 0 )
			{
				continue;
			}

			x = ptIntersect[0] / ptIntersect[2];
			y = ptIntersect[1] / ptIntersect[2];

			X = x - pp.x;
			Y = y - pp.y;
			Z = f;
			N = sqrt( X * X + Y * Y + Z * Z );

			latitude = acos( Z / N );
			longitude = atan2( X, Y ) + CV_PI;

			LA = int( latitude / angelAccuracy );
			if ( LA >= gridLA )
			{
				LA = gridLA - 1;
			}

			LO = int( longitude / angelAccuracy );
			if ( LO >= gridLO )
			{
				LO = gridLO - 1;
			}

			//
			angleDev = abs( lineInfos[i].orientation - lineInfos[j].orientation );
			angleDev = min( CV_PI - angleDev, angleDev );
			if ( angleDev > angelTolerance )
			{
				continue;
			}

			sphereGrid[LA][LO] += sqrt( lineInfos[i].length * lineInfos[j].length ) * ( sin( 2.0 * angleDev ) + 0.2 ); // 0.2 is much robuster
		}
	}

	//
	int halfSize = 1;
	int winSize = halfSize * 2 + 1;
	int neighNum = winSize * winSize;

	// get the weighted line length of each grid
	std::vector<std::vector<double> > sphereGridNew( gridLA, std::vector<double> (gridLO) );
	for ( int i = halfSize; i < gridLA - halfSize; ++i )
	{
		for ( int j = halfSize; j < gridLO - halfSize; ++j )
		{
			double neighborTotal = 0.0;
			for ( int m = 0; m < winSize; ++m )
			{
				for ( int n = 0; n < winSize; ++n )
				{
					neighborTotal += sphereGrid[i - halfSize + m][j - halfSize + n];
				}
			}

			sphereGridNew[i][j] = sphereGrid[i][j] + neighborTotal / neighNum;
		}
	}
	sphereGrid = sphereGridNew;
}

void VPDetection::getBestVpsHyp( std::vector<std::vector<double> > &sphereGrid, std::vector<std::vector<cv::Point3d> > &vpHypo, std::vector<cv::Point3d> &vps )
{
	int num = vpHypo.size();
	double oneDegree = 1.0 / 180.0 * CV_PI;

	// get the corresponding line length of every hypotheses
	std::vector<double> lineLength( num, 0.0 );
	for ( int i = 0; i < num; ++ i )
	{
		std::vector<cv::Point2d> vpLALO( 3 );
		for ( int j = 0; j < 3; ++ j )
		{
			if ( vpHypo[i][j].z == 0.0 )
			{
				continue;
			}

			if ( vpHypo[i][j].z > 1.0 || vpHypo[i][j].z < -1.0 )
			{
				cout << 1.0000 << endl;
			}
			double latitude = acos( vpHypo[i][j].z );
			double longitude = atan2( vpHypo[i][j].x, vpHypo[i][j].y ) + CV_PI;

			int gridLA = int( latitude / oneDegree );
			if ( gridLA == 90 )
			{
				gridLA = 89;
			}

			int gridLO = int( longitude / oneDegree );
			if ( gridLO == 360 )
			{
				gridLO = 359;
			}

			lineLength[i] += sphereGrid[gridLA][gridLO];
		}
	}

	// get the best hypotheses
	int bestIdx = 0;
	double maxLength = 0.0;
	for ( int i = 0; i < num; ++ i )
	{
		if ( lineLength[i] > maxLength )
		{
			maxLength = lineLength[i];
			bestIdx = i;
		}
	}

	vps = vpHypo[bestIdx];
}


void VPDetection::lines2Vps( double thAngle, std::vector<cv::Point3d> &vps, std::vector<std::vector<int> > &clusters )
{
	clusters.clear();
	clusters.resize( 3 );

	//get the corresponding vanish points on the image plane
	std::vector<cv::Point2d> vp2D( 3 );
	for ( int i = 0; i < 3; ++ i )
	{
		vp2D[i].x =  vps[i].x * f / vps[i].z + pp.x;
		vp2D[i].y =  vps[i].y * f / vps[i].z + pp.y;
	}

	for ( int i = 0; i < lines.size(); ++ i )
	{
		double x1 = lines[i].startPointX ;
		double y1 = lines[i].startPointY;
		double x2 = lines[i].endPointX;
		double y2 = lines[i].endPointY;
		double xm = ( x1 + x2 ) / 2.0;
		double ym = ( y1 + y2 ) / 2.0;

		double v1x = x1 - x2;
		double v1y = y1 - y2;
		double N1 = sqrt( v1x * v1x + v1y * v1y );
		v1x /= N1;   v1y /= N1;

		double minAngle = 1000.0;
		int bestIdx = 0;
		for ( int j = 0; j < 3; ++ j )
		{
			double v2x = vp2D[j].x - xm;
			double v2y = vp2D[j].y - ym;
			double N2 = sqrt( v2x * v2x + v2y * v2y );
			v2x /= N2;  v2y /= N2;

			double crossValue = v1x * v2x + v1y * v2y;
			if ( crossValue > 1.0 )
			{
				crossValue = 1.0;
			}
			if ( crossValue < -1.0 )
			{
				crossValue = -1.0;
			}
			double angle = acos( crossValue );
			angle = min( CV_PI - angle, angle );

			if ( angle < minAngle )
			{
				minAngle = angle;
				bestIdx = j;
			}
		}
		//
		if ( minAngle < thAngle )
		{
			clusters[bestIdx].push_back( i );
		}
	}
}

