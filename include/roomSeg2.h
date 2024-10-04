#ifndef _ROOMSEG2_H
#define _ROOMSEG2_H

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "utility.h"
#include <queue>

using namespace cv;
using namespace std;

#define DEBUG


struct LINEINFO
{
    std::pair<cv::Point, cv::Point> virtual_wll;

    double distance;

    // 두 LINEINFO가 동일한지 비교하는 연산자
    bool operator==(const LINEINFO &other) const
    {
      return (virtual_wll == other.virtual_wll) || 
              (virtual_wll.first == other.virtual_wll.second && virtual_wll.second == other.virtual_wll.first);    
    }
};

// 경로 데이터 구조체 정의
struct SEGDATA
{
    cv::Point centerPoint; // 기준이 되는 Point
    std::vector<cv::Point> feturePoints;
    std::vector<cv::Point> trajectoryPoints; // 경로를 저장하는 vector

    // 생성자
    SEGDATA() = default;
    SEGDATA(const cv::Point &key, const std::vector<cv::Point> &feture, const std::vector<cv::Point> &traj)
        : centerPoint(key), feturePoints(feture), trajectoryPoints(traj) {}
};


// Custom comparator for cv::Point
struct PointComparator
{
    bool operator()(const cv::Point &a, const cv::Point &b) const
    {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    }
};

bool comparePoints(const cv::Point_<int>& p1, const cv::Point_<int>& p2);

// Custom comparator for LINEINFO to use in a set
struct LineInfoComparator
{
    bool operator()(const LINEINFO &a, const LINEINFO &b) const
    {
        return (PointComparator{}(a.virtual_wll.first, b.virtual_wll.first)) ||
              (a.virtual_wll.first == b.virtual_wll.first && PointComparator{}(a.virtual_wll.second, b.virtual_wll.second));
    }
};


class ROOMSEG
{
private:
  /* data */

  cv::Mat img_raw_;    
  int rows_;
  int cols_;

  double angle_;
  int rows_rot_;
  int cols_rot_;

  cv::Mat rotationMatrix_;
  cv::Mat re_rotationMatrix_;

  cv::Mat img_wall_;
  cv::Mat img_freespace_;
  
  Mat img_grid_; 
  Mat img_grid_skeletion_;

  std::vector<cv::Point> featurePts_;  
  std::vector<cv::Point> trajectoryPts_;

  std::vector<std::vector<LINEINFO>> lineInfo_;
  int radius_ = 20;
  std::vector<LINEINFO> virtual_line_;  
  std::vector<cv::Rect> regions_box_;


  //cv::Mat img_contour_;
  // cv::Mat img_segroom_;
  // cv::Mat img_label_;

  std::vector<std::vector<cv::Point>> seg_contours_;

  cv::Vec4i findLongestLine(const std::vector<cv::Vec4i> &lines);  
  void zhangSuenThinning(const cv::Mat &src, cv::Mat &dst);
  void findConnectedComponents(const vector<Point> &contour, vector<vector<Point>> &components);
  Point calculateSnappedPoint(const Point& pixel, int gridSize);
  void gridSnapping(const Mat& inputImage, Mat& outputImage, int gridSize);


  bool isHalfOverlap(const Point &center1, int radius1, const Point &center2, int radius2); 
  bool isOverlap(const Point& center1, int radius1, const Point& center2, int radius2);

  vector<Point> addHalfOverlappingCircles(const vector<Point> &data, int radius);
  vector<Point> addNOverlappingCircles(const vector<Point>& data, int radius);

  void buildDataBase();
  void buildDataBaseTest(const cv::Mat& img_color_map);

  double pointToLineDistance(const cv::Point &p, const cv::Point &lineStart, const cv::Point &lineEnd);
  bool isPointNearLine(const cv::Point &p, const cv::Point &lineStart, 
                        const cv::Point &lineEnd, double threshold);

  std::vector<LINEINFO> checkPointsNearLineSegments(const std::vector<cv::Point> &dataA, 
                                                  const std::vector<cv::Point> &dataB, 
                                                  double distance_threshold = 5.0);

  
  std::vector<cv::Point> findPointsInRange(const std::vector<cv::Point> &points,
                                            int x_min, int x_max,
                                            int y_min, int y_max);

  SEGDATA exploreFeaturePoint(std::vector<cv::Point> &feature_points,
                              std::vector<cv::Point> &trajectory_points,
                              const cv::Point &center, int radius);


  std::vector<LINEINFO> convertToLineInfo(const std::vector<std::vector<LINEINFO>> &a);
  std::vector<LINEINFO> removeDuplicatesLines(const std::vector<LINEINFO> &lines);
  
  bool linesOverlap(const LINEINFO &line1, const LINEINFO &line2);
  std::vector<LINEINFO> fillterLine(std::vector<LINEINFO> &lines);
  
  
  bool areEqualIgnoringOrder(const LINEINFO &a, const LINEINFO &b);
  
  //cv::Point findNearestBlackPoint(const cv::Mat& image, cv::Point center);
  //void regionGrowing(const cv::Mat &binaryImage, cv::Mat &output, cv::Point seed, uchar fillColor);
  

  cv::Mat extractWallElements(uchar thread_wall_value = 64);  
  cv::Mat extractWallElements2(cv::Mat& img_src);
  cv::Mat extractWallElements3();

  cv::Mat extractFreeSpaceElements(uchar thread_space_value = 200);
  
  void makeRotationMatrix();
  cv::Mat makeRotatedImage(cv::Mat& img);
  

  void makeGridSnappingContours(int length_contours=15, int gridSize=3);  
  
  //void makeRegionToBox();
  //cv::Mat makeCorrectionRegion();
  void addGridPoints(vector<Point>& outputPoints, const Point& snappedPoint, int gridSize);


public:

  ROOMSEG(cv::Mat img);
  ~ROOMSEG();
  
  cv::Mat getImgGridSnapping()  
  {
    return img_grid_; 
  }

  std::vector<cv::Point> getFeaturePts()  
  {
    return featurePts_; 
  }

  std::vector<cv::Point> getTrajectoryPts()
  {
    return trajectoryPts_; 
  }
  

  std::vector<LINEINFO> getVirtualLines()
  {
    return virtual_line_;
  }

  std::vector<std::vector<cv::Point>> getSegContours()
  {
    return seg_contours_; 
  }

  int getRotRows()
  {
    return rows_rot_;
  }

  int getRotCols()
  {
    return cols_rot_;
  }


  void initialImageSetting();    
  void extractFeaturePts();   
  void extracTrajectorPts();
  void extractVirtualLine(double length_virtual_line = 21.0);

  cv::Mat makeRotatedReturn(cv::Mat& img_src);  
  cv::Point rotatedPoint2Point(cv::Point src);

};



#endif //_ROOMSEG_H
