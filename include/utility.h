#ifndef _UTILLITY_H
#define _UTILLITY_H

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
//#include <cmath>
#include <random>


// 교차점 구조체
struct Edge {
    int yMin;
    int yMax;
    double xOfYMin;
    double invSlope;
};

// // Custom comp
cv::Scalar randomColor();
std::vector<cv::Point> changeMatoPoint(cv::Mat &image);
cv::Mat changePointoMat(std::vector<cv::Point> src, int rows, int cols);

double calculateDistance(const cv::Point &p1, const cv::Point &p2);
std::vector<cv::Point> sortPoints(const std::vector<cv::Point> &points);
cv::Mat extractMat2Mat(cv::Mat &image, uchar pixel_value=128);

void removeDuplicatePoints(std::vector<cv::Point> &points);
void mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold);
void drawingSetpRectangle(cv::Mat &image, cv::Point circlesCenters, int radius);
std::vector<cv::Point> edgePointsInCircle(const cv::Point &center, int radius);
void drawingOutLineCircule(const cv::Mat &image,cv::Point circlesCenters, int radius);

double calculateSlope(const cv::Point& p1, const cv::Point& p2);
std::pair<cv::Point, cv::Point> adjustLineSlope(std::pair<cv::Point, cv::Point> lines, int* state);
void mergeCloseSegments(std::vector<std::pair<cv::Point, cv::Point>>& points, double threshold = 5.0);

void fillPolygon(cv::Mat& image, const std::vector<cv::Point>& contour, const cv::Scalar& color);
void updateEdgeTable(const cv::Point& p1, const cv::Point& p2, std::vector<std::vector<Edge>>& edgeTable);

#endif 
