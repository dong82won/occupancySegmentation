#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>

//#include "utility.h"
#include "roomSeg2.h"

using namespace cv;
using namespace std;

//------------------------------------------------------------------------------
// 점 - vertex
// 선 - edge
// 면 - face
//------------------------------------------------------------------------------
int main()
{
    //1. 이미지 입력 및 회전 -------------------------------------------------------
    std::string home_path = getenv("HOME");
    // std::cout << home_path << std::endl;

    // 이미지 파일 경로
    cv::Mat img_raw = cv::imread(home_path + "/myStudyCode/occupancySegmentation/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat img_raw = cv::imread(home_path + "/myWorkCode/occupancySegmentation/imgdb/test_00.png", cv::IMREAD_GRAYSCALE);    
    //cv::Mat img_raw = cv::imread(home_path + "/myWorkCode/occupancySegmentation/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);

    if (img_raw.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    cv::Mat color_img_raw;
    cv::cvtColor(img_raw, color_img_raw, COLOR_GRAY2BGR);    

    imshow("img_raw", img_raw);

    ROOMSEG rs(img_raw);
    rs.initialImageSetting();

    rs.extractFeaturePts();
    std::vector<cv::Point> featurePts = rs.getFeaturePts();
    
    cv::Mat img_grid = rs.getImgGridSnapping(); 
    imshow("img_grid", img_grid);

    rs.extracTrajectorPts();    
    std::vector<cv::Point> trajectoryPts = rs.getTrajectoryPts();    
    
    double length_line = 23.0;
    std::vector<std::pair<cv::Point, cv::Point>> vitual_edges = rs.makeVirtualEdge(length_line);

#ifdef DEBUG    

    cv::Mat color_img_grid;    
    cv::cvtColor(img_grid, color_img_grid, COLOR_GRAY2BGR);    

    for (const auto &pt : featurePts)
    {
        cv::circle(color_img_grid, pt, 3, cv::Scalar(0, 255, 0), -1);         
    }
    std::cout << std::endl;

    for (const auto &pt : trajectoryPts)
    {
        cv::circle(color_img_grid, pt, 1, cv::Scalar(255, 0, 0), -1);
    }

    for(size_t i = 0; i < vitual_edges.size(); i++)        
    {
        cv::line(color_img_grid, vitual_edges[i].first, 
                                vitual_edges[i].second, cv::Scalar(0, 0, 255), 3); 
    }        
    cv::imshow("color_img_grid", color_img_grid);
#endif

    rs.makeFaceRegion(featurePts, vitual_edges);
    std::vector<std::vector<cv::Point>> face = rs.extractFaceContours();
    
//------------------------------------------------------------------------------
    cv::Mat img_r_grid = rs.returnRotatedImage(img_grid);
    cv::imshow("img_r_grid", img_r_grid);
    
    cv::Mat imgc_r_grid;
    cvtColor(img_r_grid, imgc_r_grid, COLOR_GRAY2BGR);

    cv::Mat img_result = cv::Mat::zeros(img_r_grid.size(), CV_8UC3);
    img_result.setTo(cv::Scalar(255, 255, 255));

    RNG rng(12345);
    for (size_t i = 0; i< face.size(); i++)
    {   
        cv::Vec3b color = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        //std::cout <<"face: " << i << std::endl;        
                
        std::vector<cv::Point> rotated_rpts;
        for (size_t j=0; j<face[i].size(); j++)        
        {
            cv::Point pts= face[i][j];            
            cv::Point rpt = rs.rotatedPoint2Point(pts); 
            rotated_rpts.push_back(rpt);            
        }

        cv::fillPoly(img_result, rotated_rpts, color);         
        // for (size_t j=0; j <rotated_rpts.size(); j++)
        // {
        //     cv::line(img_result, rotated_rpts[j], rotated_rpts[(j+1) % rotated_rpts.size()], color, 3);
        // }
    }

    std::vector<cv::Point> edge = changeMatoPoint(img_r_grid);
    for (size_t i = 0; i < edge.size(); i++)     
    {
        cv::circle(img_result, edge[i], 2, cv::Scalar(0, 0, 0), -1);
    }    

    for (size_t i = 0; i < vitual_edges.size(); i++)     
    {
        cv::Point spt = rs.rotatedPoint2Point(vitual_edges[i].first);
        cv::Point ept = rs.rotatedPoint2Point(vitual_edges[i].second);
        cv::line(img_result, spt, ept, cv::Scalar(0, 255, 0), 3); 
    }

    cv::imshow("img_result", img_result);


/*
    // 이미지 가로로 결합
    cv::Mat img_hcon_bottom;
    cv::hconcat(imgc_contours_grid, labels_region, img_hcon_bottom);
    cv::imshow("img_hcon_bottom", img_hcon_bottom);              


    cv::Mat img_color_raw;    
    cv::cvtColor(img_raw, img_color_raw, COLOR_GRAY2BGR);    


    // 이미지 가로로 결합
    cv::Mat img_hcon_top;
    cv::hconcat(img_color_raw, labels_region, img_hcon_top);
    cv::imshow("img_hcon_top", img_hcon_top);      

    // 이미지 세로로 결합
    cv::Mat img_combin;
    cv::vconcat(img_hcon_top, img_hcon_bottom, img_combin);
    cv::imshow("img_combin", img_combin);      
    cv::imwrite( home_path + "/myWorkCode/occupancySegmentation//imgdb/regRoom.png",  img_combin);
    

    cv::Mat img_combin2;
    cv::hconcat(img_hcon_top, img_hcon_bottom, img_combin2);
    cv::imshow("img_combin2", img_combin2);      
    cv::imwrite( home_path + "/myWorkCode/occupancySegmentation//imgdb/regRoom2.png",  img_combin2);

*/
    // Alpha blending을 위한 변수 설정 (투명도)
    double alpha = 0.5;  // 첫 번째 이미지의 가중치
    double beta = 1.0 - alpha;  // 두 번째 이미지의 가중치

    cv::Mat blended;
    // 두 이미지를 중첩합니다
    cv::addWeighted(color_img_raw, alpha, img_result, beta, 0.0, blended);

    // 결과 이미지를 출력합니다
    cv::namedWindow("Blended_Image", WINDOW_KEEPRATIO && WINDOW_AUTOSIZE);
    cv::imshow("Blended_Image", blended);

    cv::waitKey(0);
    return 0;
}
