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
    cv::Mat img_raw = cv::imread(home_path + "/myStudyCode/occupancySegmentation//imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);    
    //cv::Mat img_raw = cv::imread(home_path + "/myStudyCode/occupancySegmentation/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);

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

    cv::Mat inverted_grid;
    bitwise_not(img_grid, inverted_grid);  
    cv::imshow("inverted_grid", inverted_grid);

    //중요 데이터
    cv::Mat img_contours_grid = rs.makeRotatedReturn(inverted_grid); 
    cv::imshow("img_contours_grid", img_contours_grid);

    rs.extracTrajectorPts();    
    std::vector<cv::Point> trajectoryPts = rs.getTrajectoryPts();    

#ifdef DEBUG    

    cv::Mat color_img_grid;    
    cv::cvtColor(img_grid, color_img_grid, COLOR_GRAY2BGR);    

    for (const auto &pt : featurePts)
    {
        cv::circle(color_img_grid, pt, 3, cv::Scalar(0, 255, 0), -1); 
    }

    for (const auto &pt : trajectoryPts)
    {
        cv::circle(color_img_grid, pt, 1, cv::Scalar(255, 0, 255), -1);
    }
#endif

    double length_line = 23.0;
    rs.extractVirtualLine(length_line);
        
    std::vector<LINEINFO> vitual_lines = rs.getVirtualLines();
    std::cout << "vitualLines.size(): " << vitual_lines.size() << std::endl;    

    std::vector<LINEINFO> vlines;
    for (const auto &line : vitual_lines)
    {

#ifdef DEBUG
        std::cout << "Line: ("
                << line.virtual_wll.first.x << ", " << line.virtual_wll.first.y << ") to ("
                << line.virtual_wll.second.x << ", " << line.virtual_wll.second.y << ") - Distance: "
                << line.distance << std::endl;
#endif

        cv::Point spt = line.virtual_wll.first;
        cv::Point ept = line.virtual_wll.second;        

        adjustLineSlope(spt, ept, img_grid.rows, img_grid.cols);

        LINEINFO vl;
        vl.virtual_wll.first = spt;
        vl.virtual_wll.second = ept;
        vlines.push_back(vl);       
        cv::line(inverted_grid, spt, ept, cv::Scalar(0), 1); 

#ifdef DEBUG
        cv::line(color_img_grid, spt, ept, cv::Scalar(0, 0, 255), 3);         
#endif
    }    

#ifdef DEBUG
    cv::imshow("color_img_grid", color_img_grid);
#endif

    // Convex Hull 계산
    std::vector<cv::Point> hull;
    convexHull(featurePts, hull);    
    // Convex Hull 경계선을 그리기
    for (size_t i = 0; i < hull.size(); i++) {
        line(inverted_grid, hull[i], hull[(i+1) % hull.size()], Scalar(0), 1);
    }

    //----------------------------------------------------------------------------------------------
    
    cv::Mat img_region_grid = rs.makeRotatedReturn(inverted_grid); 
    cv::imshow("img_region_grid", img_region_grid);

    // 레이블링 작업을 수행합니다 (8방향 연결).
    cv::Mat labels, stats, centroids;    
    int n_labels = cv::connectedComponentsWithStats(img_region_grid, labels, stats, centroids, 4, CV_32S);
        
    cv::Mat labels_region = cv::Mat::zeros(img_region_grid.size(), CV_8UC3); 
    labels_region.setTo(cv::Scalar(255, 255, 255));

    std::cout <<"=============================" <<std::endl;
    std::cout << "n_labels: " << n_labels << std::endl;
    std::cout <<"=============================" <<std::endl;
 
    RNG rng(12345);  // 랜덤 색상을 위한 시드
    for (int label = 1; label < n_labels; ++label)
    {
        cv::Vec3b color = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        
        // 바운딩 박스 좌표
        int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        int y = stats.at<int>(label, cv::CC_STAT_TOP);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(label, CC_STAT_AREA);

        // std::cout << "  Bounding Box: x=" << x << ", y=" << y << ", width=" << width << ", height=" << height << std::endl;
        // std::cout << "  Area: " << stats.at<int>(label, CC_STAT_AREA) << std::endl;

        if (x >0 && y > 0)
        {
            if ( area > 15*15) 
            {   
                
                // // 각 레이블의 픽셀에 색을 입힘
                // for (int row = 0; row < labels.rows; row++) 
                // {
                //     for (int col = 0; col < labels.cols; col++) 
                //     {
                //         if (labels.at<int>(row, col) == label) 
                //         {
                //             img_result_color.at<cv::Vec3b>(row, col) = color;
                //         }
                //     }
                // } 
                

                // Create a binary mask for the current label
                cv::Mat mask = (labels == label);

                // Find contours in the mask
                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // Draw contours for the current label on a new image                
                cv::drawContours(labels_region, contours, -1, color, -1);

                /*
                //데이터 최종 근사화
                for (size_t p = 0; p < contours.size(); p++)
                {
                    std::cout <<"contours[ "<< p << " ]size(): " << contours[p].size() << std::endl;                    
                    std::vector<cv::Point> approx;
                    double epsilon = 0.003 * cv::arcLength(contours[p], true); // 윤곽선 길이에 따른 epsilon 설정
                    cv::approxPolyDP(contours[p], approx, epsilon, true);     // 윤곽선 근사화

                    std::cout <<"approx.size(): " << approx.size() << std::endl;
                    for (size_t n = 0; n <approx.size(); n++)
                    {                        
                        line(labels_region, approx[n], approx[(n+1) % approx.size()], Scalar(0, 255, 0), 1);
                        cv::circle(labels_region, approx[n], 3, Scalar(0, 255, 0), -1);
                    }
                }
                */  
            }   
        }            
    }
    cv::imshow("labels_region", labels_region);              



    cv::Mat imgc_contours_grid;
    cvtColor(img_contours_grid, imgc_contours_grid, COLOR_GRAY2BGR);
    cv::imshow("imgc_contours_grid", imgc_contours_grid);       


    // 이미지 가로로 결합
    cv::Mat img_hcon_bottom;
    cv::hconcat(imgc_contours_grid, labels_region, img_hcon_bottom);
    cv::imshow("img_hcon_bottom", img_hcon_bottom);              


    for (int i = 0; i < img_contours_grid.rows; i++)
    {
        for (int j=0; j< img_contours_grid.cols; j++)
        {            
            cv::Vec3b color = imgc_contours_grid.at<cv::Vec3b>(i, j); 
            if (color[0] == 0 && color[1] == 0 && color[2] == 0)
            {
                labels_region.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    for (size_t v = 0; v <vlines.size(); v++)     
    {
        cv::Point spt = rs.rotatedPoint2Point(vlines[v].virtual_wll.first);
        cv::Point ept = rs.rotatedPoint2Point(vlines[v].virtual_wll.second);
        cv::line(labels_region, spt, ept, cv::Scalar(0, 255, 255), 2); 
    }
    

    cv::imshow("labels_region2", labels_region);    


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
    cv::imwrite( home_path + "/myStudyCode/occupancySegmentation//imgdb/regRoom.png",  img_combin);
    
    // Alpha blending을 위한 변수 설정 (투명도)
    double alpha = 0.5;  // 첫 번째 이미지의 가중치
    double beta = 1.0 - alpha;  // 두 번째 이미지의 가중치

    cv::Mat blended;
    // 두 이미지를 중첩합니다
    cv::addWeighted(img_color_raw, alpha, labels_region, beta, 0.0, blended);

    // 결과 이미지를 출력합니다
    cv::namedWindow("Blended_Image", WINDOW_KEEPRATIO && WINDOW_AUTOSIZE);
    cv::imshow("Blended_Image", blended);

    cv::waitKey(0);

    return 0;
}

 
