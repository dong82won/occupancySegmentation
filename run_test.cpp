#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>

//#include "utility.h"
#include "roomSeg.h"

using namespace cv;
using namespace std;

int main()
{
    //1. 이미지 입력 및 회전 -------------------------------------------------------
    std::string home_path = getenv("HOME");
    // std::cout << home_path << std::endl;

    // 이미지 파일 경로
    cv::Mat img_raw = cv::imread(home_path + "/myStudyCode/RoomSegmentation/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);    
    //cv::Mat img_raw = cv::imread(home_path + "/myStudyCode/RoomSegmentation/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);

    if (img_raw.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    imshow("img_raw", img_raw);

    ROOMSEG rs(img_raw);
    rs.initialImageSetting();

 
    rs.extractFeaturePts();
    std::vector<cv::Point> featurePts = rs.getFeaturePts();

    cv::Mat img_grid = rs.getImgGridSnapping();
    cv::Mat color_img_grid;
    cv::cvtColor(img_grid, color_img_grid, COLOR_GRAY2BGR);    
    for (const auto &pt : featurePts)
    {
        cv::circle(color_img_grid, pt, 3, cv::Scalar(0, 255, 0), -1);
    }
    
    rs.extracTrajectorPts();
    std::vector<cv::Point> trajectoryPts = rs.getTrajectoryPts();    

    for (const auto &pt : trajectoryPts)
    {
        cv::circle(color_img_grid, pt, 1, cv::Scalar(255, 0, 255), -1);
    }

    double length_line = 22.0;
    rs.extractVirtualLine(length_line);
    std::vector<LINEINFO> vitual_lines = rs.getVirtualLines();

    std::cout << "vitual_lines.size(): " << vitual_lines.size() << std::endl;    
    for (const auto &line : vitual_lines)
    {
        std::cout << "Line: ("
                << line.virtual_wll.first.x << ", " << line.virtual_wll.first.y << ") to ("
                << line.virtual_wll.second.x << ", " << line.virtual_wll.second.y << ") - Distance: "
                << line.distance << std::endl;
        cv::line(color_img_grid, line.virtual_wll.first, line.virtual_wll.second, CV_RGB(255, 0, 0), 3);
    }
    
    
    rs.segmentationRoom();
    //std::vector<std::vector<cv::Point>> seg_room = rs.getRoomImage();



    /*
    //std::vector<cv::Vec3b> colors(seg_room.size());
    std::vector<cv::Scalar> colors(seg_room.size());
   // 각 라벨에 무작위 색상 할당 (외곽선을 그리기 전 사용)    
    for (int i = 1; i < seg_room.size(); i++) {
        //colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
        colors[i] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
    }


    cv::Mat img_color_raw;
    cvtColor(img_raw, img_color_raw, COLOR_GRAY2BGR);

    cout <<"seg_room.size(): " << seg_room.size() << endl;


    for (size_t p =0; p< seg_room.size(); p++)
    {
        //cv::Scalar color = randomColor();
        //cout <<"seg_room [" <<  seg_room[p] << " ]" << endl;

        for (size_t q=0; q<seg_room[p].size(); q++)
        {
            cv::Point pts = seg_room[p][q];
            cv::circle(img_color_raw, pts, 1, colors[q], -1);
        }
    }



    
    */
    cv::imshow("color_img_grid", color_img_grid); 
    cv::waitKey(0);

    return 0;
}



//  // Alpha blending을 위한 변수 설정 (투명도)
//     double alpha = 0.5;  // 첫 번째 이미지의 가중치
//     double beta = 1.0 - alpha;  // 두 번째 이미지의 가중치

//     cv::Mat blended;
//     // 두 이미지를 중첩합니다
//     cv::addWeighted(img_color_raw, alpha, img_seg, beta, 0.0, blended);

//     // 결과 이미지를 출력합니다
//     cv::imshow("Blended Image", blended);
