#include "roomSeg.h"

ROOMSEG::ROOMSEG(cv::Mat img): img_raw_(img)
{
    cols_ = img.cols;
    rows_ = img.rows;
}

ROOMSEG::~ROOMSEG()
{
}

bool comparePoints(const cv::Point_<int>& p1, const cv::Point_<int>& p2) {
    // X 좌표를 먼저 비교하고, X 좌표가 같으면 Y 좌표로 비교
    if (p1.x != p2.x) {
        return p1.x < p2.x;  // X 좌표 오름차순
    } else {
        return p1.y < p2.y;  // Y 좌표 오름차순
    }
}

cv::Mat ROOMSEG::extractWallElements(uchar thread_wall_value)
{   
    
    cv::Mat img_wall = cv::Mat::zeros(img_raw_.size(), CV_8UC1);

    for (int i = 0; i < img_wall.rows; i++)
    {
        for (int j = 0; j < img_wall.cols; j++)
        {
            uchar pixelValue = img_raw_.at<uchar>(i, j);

            //if (pixelValue == 0) continue;
            if (pixelValue < thread_wall_value)
            {
                img_wall.at<uchar>(i, j) = 255;
            }
        }
    }
    std::cout <<"extractWallElements()..." << std::endl;   
    
    return img_wall;
}


cv::Mat ROOMSEG::extractWallElements2(cv::Mat& img_src)
{   
    cv::Mat img_wall = cv::Mat::zeros(img_raw_.size(), CV_8UC1);
    
    // 외곽선 좌표 저장할 벡터
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // 외곽선 추출 (mask에서)
    cv::findContours(img_src, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

     // 외곽선을 그리기 위한 컬러 이미지로 변환
    cv::Mat colorImg;
    cv::cvtColor(img_src, colorImg, cv::COLOR_GRAY2BGR);

    // 검출된 외곽선을 이미지에 그리기
    for (size_t i = 0; i < contours.size(); i++) {
        cout <<"contours.size(): " << contours[i].size() << endl;
        if (contours[i].size() > 9) 
        {
            cv::drawContours(colorImg, contours, (int)i, cv::Scalar(0, 255, 0), 2, cv::LINE_8, hierarchy, 0);
            cv::drawContours(img_wall, contours, (int)i, cv::Scalar(255), 2, cv::LINE_8, hierarchy, 0);
        }        
    }
    // 이미지 출력
    cv::imshow("Contours", colorImg);
    
    return img_wall;
}


cv::Mat ROOMSEG::extractFreeSpaceElements(uchar thread_space_value)
{
    cv::Mat img_freeSpace = cv::Mat::zeros(img_raw_.size(), CV_8UC1);
    for (int i = 0; i < img_raw_.rows; i++)
    {
        for (int j = 0; j < img_raw_.cols; j++)
        {
            if (img_raw_.at<uchar>(i, j) >= thread_space_value)
            {
                img_freeSpace.at<uchar>(i, j) = 255;
            }
        }
    }
    std::cout <<"extractFreeSpaceElements()..." << std::endl;
    

    return img_freeSpace;
}


// 가장 긴 직선 찾기
Vec4i ROOMSEG::findLongestLine(const vector<Vec4i> &lines)
{
    Vec4i longestLine;
    double maxLength = 0;

    for (const auto &line : lines)
    {
        double length = norm(Point(line[0], line[1]) - Point(line[2], line[3]));
        if (length > maxLength)
        {
            maxLength = length;
            longestLine = line;
        }
    }

    return longestLine;
}

cv::Mat ROOMSEG::makeRotationMatrix()
{
   // Canny 엣지 검출
    cv::Mat edges;
    cv::Canny(img_raw_, edges, 50, 200);

    // 직선 검출 (Hough 변환)
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 5);  
    Vec4i longestLine = findLongestLine(lines);

    angle_ = atan2(longestLine[3] - longestLine[1], longestLine[2] - longestLine[0]) * 180.0 / CV_PI;

    Point2f center(cols_/2.0, rows_/2.0);
    cv::Mat rotationMatrix = getRotationMatrix2D(center, angle_, 1.0);

    // 회전 각도에 따른 새로운 가로 및 세로 크기 계산
    double abs_cos = abs(rotationMatrix.at<double>(0, 0));
    double abs_sin = abs(rotationMatrix.at<double>(0, 1));
    
    cols_rot_ = int(rows_* abs_sin + cols_ * abs_cos);
    rows_rot_ = int(rows_ * abs_cos + cols_ * abs_sin);

    std::cout <<"rows x cols : " << rows_ <<",  " << cols_ << std::endl;
    std::cout <<"rot_size: " << cols_rot_ <<",  " << rows_rot_ << std::endl;

    // 변환 행렬의 이동을 수정하여 이미지가 잘리지 않도록 설정
    rotationMatrix.at<double>(0, 2) += (cols_rot_ / 2.0) - center.x;
    rotationMatrix.at<double>(1, 2) += (rows_rot_ / 2.0) - center.y;

    return rotationMatrix;
}

// 이미지 회전
cv::Mat ROOMSEG::makeRotatedImage(cv::Mat& img, cv::Mat rotationMatrix)
{       
    cv::Mat img_rotated;
    warpAffine(img, img_rotated, rotationMatrix, Size(cols_rot_, rows_rot_));
    return img_rotated;
}


cv::Mat ROOMSEG::makeRotatedReturn(cv::Mat& img_src)
{    

    cv::Point2f center(cols_rot_/2.0, rows_rot_/2.0);
    cv::Mat rotationMatrix = getRotationMatrix2D(center, -angle_, 1.0);    
    
    // 변환 행렬의 이동을 수정하여 이미지가 잘리지 않도록 설정
    rotationMatrix.at<double>(0, 2) += (cols_ / 2.0) - center.x;
    rotationMatrix.at<double>(1, 2) += (rows_ / 2.0) - center.y;
    
    cv::Mat img_dst;
    warpAffine(img_src, img_dst, rotationMatrix, Size(cols_, rows_));
    return img_dst;
}


// Zhang-Suen Thinning Algorithm
void ROOMSEG::zhangSuenThinning(const cv::Mat &src, cv::Mat &dst)
{

    cv::Mat img;
    int th = (int)cv::threshold(src, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    img /= 255;
    cv::Mat prev = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::Mat diff;

    do
    {
        cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

        for (int y = 1; y < img.rows - 1; ++y)
        {
            for (int x = 1; x < img.cols - 1; ++x)
            {
                uchar p2 = img.at<uchar>(y - 1, x);
                uchar p3 = img.at<uchar>(y - 1, x + 1);
                uchar p4 = img.at<uchar>(y, x + 1);
                uchar p5 = img.at<uchar>(y + 1, x + 1);
                uchar p6 = img.at<uchar>(y + 1, x);
                uchar p7 = img.at<uchar>(y + 1, x - 1);
                uchar p8 = img.at<uchar>(y, x - 1);
                uchar p9 = img.at<uchar>(y - 1, x - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (img.at<uchar>(y, x) == 1 && B >= 2 && B <= 6 && A == 1 &&
                    (p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0))
                {
                    marker.at<uchar>(y, x) = 1;
                }
            }
        }

        img -= marker;

        for (int y = 1; y < img.rows - 1; ++y)
        {
            for (int x = 1; x < img.cols - 1; ++x)
            {
                uchar p2 = img.at<uchar>(y - 1, x);
                uchar p3 = img.at<uchar>(y - 1, x + 1);
                uchar p4 = img.at<uchar>(y, x + 1);
                uchar p5 = img.at<uchar>(y + 1, x + 1);
                uchar p6 = img.at<uchar>(y + 1, x);
                uchar p7 = img.at<uchar>(y + 1, x - 1);
                uchar p8 = img.at<uchar>(y, x - 1);
                uchar p9 = img.at<uchar>(y - 1, x - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (img.at<uchar>(y, x) == 1 && B >= 2 && B <= 6 && A == 1 &&
                    (p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0))
                {
                    marker.at<uchar>(y, x) = 1;
                }
            }
        }

        img -= marker;
        cv::absdiff(img, prev, diff);
        img.copyTo(prev);

    } while (cv::countNonZero(diff) > 0);

    img *= 255;
    dst = img.clone();
}


// 연결된 성분을 찾는 함수
void ROOMSEG::findConnectedComponents(const vector<Point> &contour, vector<vector<Point>> &components)
{

    // 체인 코드 방향 (8방향)
    int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

    vector<bool> visited(contour.size(), false);

    // 현재 성분을 저장할 변수
    vector<Point> currentComponent;

    for (size_t i = 0; i < contour.size(); i++)
    {
        if (visited[i])
            continue; // 이미 방문한 점은 건너뜀

        // 새로운 성분 시작
        currentComponent.clear();
        vector<Point> stack; // DFS를 위한 스택
        stack.push_back(contour[i]);

        while (!stack.empty())
        {
            Point p = stack.back();
            stack.pop_back();

            // 현재 점을 성분에 추가하고 방문 처리
            currentComponent.push_back(p);
            int idx = find(contour.begin(), contour.end(), p) - contour.begin();
            visited[idx] = true;

            // 8방향으로 탐색
            for (int dir = 0; dir < 8; dir++)
            {
                Point neighbor(p.x + dx[dir], p.y + dy[dir]);

                // 인접 점이 윤곽선에 존재하고 방문하지 않았을 경우 스택에 추가
                if (find(contour.begin(), contour.end(), neighbor) != contour.end() && !visited[find(contour.begin(), contour.end(), neighbor) - contour.begin()])
                {
                    stack.push_back(neighbor);
                }
            }
        }

        // 연결된 성분을 저장
        components.push_back(currentComponent);
    }
}

// 격자 점 계산 함수
Point ROOMSEG::calculateSnappedPoint(const Point& pixel, int gridSize) {
    int snappedX = round(static_cast<double>(pixel.x) / gridSize) * gridSize;
    int snappedY = round(static_cast<double>(pixel.y) / gridSize) * gridSize;
    return Point(snappedX, snappedY);
}

void ROOMSEG::gridSnapping(const Mat& inputImage, Mat& outputImage, int gridSize) {
    outputImage = Mat::zeros(inputImage.size(), inputImage.type()); // 결과 이미지를 0으로 초기화

    // 이미지에서 모든 픽셀을 순회
    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            // 흰색 픽셀(255)인 경우
            if (inputImage.at<uchar>(y, x) == 255) {
                // 가장 가까운 격자 점 계산
                Point snappedPoint = calculateSnappedPoint(Point(x, y), gridSize);

                // 격자 점으로 픽셀 이동
                int halfGridSize = (gridSize / 2)+1;

                // 지정된 격자 크기만큼의 영역을 흰색으로 설정
                for (int dy = -halfGridSize; dy < halfGridSize; dy++) {
                    for (int dx = -halfGridSize; dx < halfGridSize; dx++) {
                        int newY = snappedPoint.y + dy;
                        int newX = snappedPoint.x + dx;

                        // 이미지 경계 내에서만 설정
                        if (newX >= 0 && newX < outputImage.cols && 
                            newY >= 0 && newY < outputImage.rows) {
                            outputImage.at<uchar>(newY, newX) = 255; // 흰색 픽셀로 설정
                        }
                    }
                }
            }
        }
    }
}

void ROOMSEG::makeGridSnappingContours(int length_contours, int gridSize)
{
    Mat img_dilate;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(img_wall_, img_dilate, MORPH_DILATE, kernel, Point(-1, -1), 1);
    //imshow("img_dilate", img_dilate);

    cv::Mat img_skeletion; 
    zhangSuenThinning(img_dilate, img_skeletion);
    //cv::imshow("img_wall_skeletion", img_skeletion);    

    std::vector<cv::Point> skeletionPts = changeMatoPoint(img_skeletion);            
    
    //연결된 성분을 저장할 벡터
    vector<vector<Point>> edgePts;
    findConnectedComponents(skeletionPts, edgePts);

    cv::Mat img_edge = cv::Mat::zeros(img_skeletion.size(), CV_8UC1);        
    
    for (size_t i = 0; i < edgePts.size(); i++)
    {       
        //contours threshold
        if (edgePts[i].size() > length_contours)
        {
            for (const auto &pt : edgePts[i])
            { 
                int y = pt.y;
                int x = pt.x;
                img_edge.at<uchar>(y, x) = 255;
            }
        }
    } 
    
    gridSnapping(img_edge, img_grid_, gridSize);    
    zhangSuenThinning(img_grid_, img_grid_skeletion_);

    std::cout <<"makeGridSnappingContours()..." << std::endl;
    cv::imshow("img_grid_", img_grid_);
    cv::imshow("img_grid_skeletion_", img_grid_skeletion_);
}


void ROOMSEG::extractFeaturePts()
{
  // 수평선 감지
    cv::Mat horizontal;
    cv::Mat horizontal_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1));
    cv::morphologyEx(img_grid_skeletion_, horizontal, cv::MORPH_OPEN, horizontal_kernel, cv::Point(-1, -1), 1);

    // 수직선 감지
    cv::Mat vertical;
    cv::Mat vertical_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));
    cv::morphologyEx(img_grid_skeletion_, vertical, cv::MORPH_OPEN, vertical_kernel, cv::Point(-1, -1), 1);

    std::vector<cv::Point> horizontalPts = changeMatoPoint(horizontal);    
    vector<vector<Point>> edgeHorizontalPts;
    findConnectedComponents(horizontalPts, edgeHorizontalPts);

    std::vector<cv::Point> verticalPts = changeMatoPoint(vertical);
    vector<vector<Point>> edgeVerticalPts;
    findConnectedComponents(verticalPts, edgeVerticalPts);

    std::vector<cv::Point> featurePts;    
    for (size_t i = 0; i < edgeHorizontalPts.size(); i++)
    {    
        cv::Point startPt = edgeHorizontalPts[i].front();
        cv::Point endPt = edgeHorizontalPts[i].back();

        featurePts.push_back(startPt);
        featurePts.push_back(endPt);
    }

    for (size_t i = 0; i < edgeVerticalPts.size(); i++)
    {    
        cv::Point startPt = edgeVerticalPts[i].front();
        cv::Point endPt = edgeVerticalPts[i].back();

        featurePts.push_back(startPt);
        featurePts.push_back(endPt);
    }

    removeDuplicatePoints(featurePts);

    featurePts_ = sortPoints(featurePts);
}

void ROOMSEG::extracTrajectorPts()
{

    cv::Mat dist_transform;
    cv::distanceTransform(img_freespace_, dist_transform, cv::DIST_L2, 3);
    normalize(dist_transform, dist_transform, 0, 255, cv::NORM_MINMAX);

    cv::Mat dist_transform_8u;
    dist_transform.convertTo(dist_transform_8u, CV_8UC1);
    //cv::imshow("distTransform", dist_transform_8u);

    cv::Mat img_dist_bin;
    threshold(dist_transform_8u, img_dist_bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    //cv::imshow("img_dist_bin", img_dist_bin);

    cv::Mat img_freeSpace_skeletion;
    zhangSuenThinning(img_dist_bin, img_freeSpace_skeletion);
    //cv::imshow("img_freeSpace_skeletion", img_freeSpace_skeletion);

    std::vector<cv::Point> trajectoryPts = changeMatoPoint(img_freeSpace_skeletion);
    
    trajectoryPts_ = sortPoints(trajectoryPts);
}


// 두 원형 영역이 반만 겹치는지 확인하는 함수
bool ROOMSEG::isHalfOverlap(const Point &center1, int radius1, const Point &center2, int radius2)
{
    double distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    return distance <= (radius1 + radius2) / 2.0;
}

// 원형 탐색 범위를 추가하는 함수
vector<Point> ROOMSEG::addHalfOverlappingCircles(const vector<Point> &data, int radius)
{
    vector<Point> circlesCenters;

    for (const auto &point : data)
    {
        bool overlap = false;
        // 새로 추가할 원형 범위가 기존의 범위와 반만 겹치는지 확인
        for (const auto &existingCenter : circlesCenters)
        {
            if (isHalfOverlap(existingCenter, radius, point, radius))
            {
                overlap = true;
                break;
            }
        }
        if (!overlap)
        {
            circlesCenters.push_back(point);
        }
    }
    return circlesCenters;
}

// 두 원형 영역이 겹치는지 확인하는 함수
bool ROOMSEG::isOverlap(const Point& center1, int radius1, const Point& center2, int radius2) {
    double distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    return distance < (radius1 + radius2);
}

vector<Point> ROOMSEG::addNOverlappingCircles(const vector<Point>& data, int radius) {
    vector<Point> circlesCenters;
    
    for (const auto& point : data) {
        bool overlap = false;
        
        // 새로 추가할 원형 범위가 기존의 범위와 반만 겹치는지 확인
        for (const auto& existingCenter : circlesCenters) {
            if (isOverlap(existingCenter, radius, point, radius)) {
                overlap = true;
                break;
            }
        }
        
        if (!overlap) {
            circlesCenters.push_back(point);
        }
    }
    
    return circlesCenters;
}



// x, y 범위 내에 포함되는 포인트를 찾는 함수
std::vector<cv::Point> ROOMSEG::findPointsInRange(const std::vector<cv::Point> &points, 
                                                int x_min, int x_max, 
                                                int y_min, int y_max)
{

    std::vector<cv::Point> filteredPoints;

    // std::copy_if를 사용하여 조건에 맞는 점들만 필터링
    std::copy_if(points.begin(), points.end(), std::back_inserter(filteredPoints),
                [x_min, x_max, y_min, y_max](const cv::Point &pt)
                {
                    return (pt.x >= x_min && pt.x <= x_max && pt.y >= y_min && pt.y <= y_max);
                });

    return filteredPoints;
}

SEGDATA ROOMSEG::exploreFeaturePoint(std::vector<cv::Point> &feature_points,
                            std::vector<cv::Point> &trajectory_points,
                            const cv::Point &center, int radius)
{
    int x_min = center.x - radius;
    int x_max = center.x + radius;
    int y_min = center.y - radius;
    int y_max = center.y + radius;

    std::vector<cv::Point> featurePts = findPointsInRange(feature_points, x_min, x_max, y_min, y_max);
    std::vector<cv::Point> trajectoryPts = findPointsInRange(trajectory_points, x_min, x_max, y_min, y_max);

    SEGDATA dst = SEGDATA(center, featurePts, trajectoryPts);

    return dst;
}


// 선분과 점 사이의 수직 거리를 계산하는 함수
double ROOMSEG::pointToLineDistance(const cv::Point &p, const cv::Point &lineStart, const cv::Point &lineEnd)
{
    double A = p.x - lineStart.x;
    double B = p.y - lineStart.y;
    double C = lineEnd.x - lineStart.x;
    double D = lineEnd.y - lineStart.y;

    double dot = A * C + B * D;
    double len_sq = C * C + D * D;
    double param = (len_sq != 0) ? dot / len_sq : -1; // 선분의 길이가 0이 아닐 때만 계산

    double xx, yy;

    if (param < 0)
    {
        xx = lineStart.x;
        yy = lineStart.y;
    }
    else if (param > 1)
    {
        xx = lineEnd.x;
        yy = lineEnd.y;
    }
    else
    {
        xx = lineStart.x + param * C;
        yy = lineStart.y + param * D;
    }

    double dx = p.x - xx;
    double dy = p.y - yy;
    return std::sqrt(dx * dx + dy * dy);
}

// 직선 세그먼트 근처에 점이 있는지 확인하는 함수
bool ROOMSEG::isPointNearLine(const cv::Point &p, const cv::Point &lineStart,
                            const cv::Point &lineEnd, double threshold)
{
    // 점이 선분에서 특정 거리 이내에 있는지 확인
    double distance = pointToLineDistance(p, lineStart, lineEnd);
    return distance <= threshold;
}


// 데이터 A와 B에 대해 직선 세그먼트에서 점을 확인하는 함수
std::vector<LINEINFO> ROOMSEG::checkPointsNearLineSegments(const std::vector<cv::Point> &dataA, 
                                                        const std::vector<cv::Point> &dataB, 
                                                        double distance_threshold)
{
    LINEINFO line;
    std::vector<LINEINFO> lines;

    for (size_t i = 0; i < dataA.size(); ++i)
    {
        for (size_t j = i + 1; j < dataA.size(); ++j)
        {
            cv::Point start = dataA[i];
            cv::Point end = dataA[j];

            std::cout << "Line segment: (" << start.x << ", " << start.y << ") -> ("
                    << end.x << ", " << end.y << ") = distance " << calculateDistance(start, end) << "\n";

            bool foundPointNearLine = false;

            for (const auto &bPoint : dataB)
            {
                if (isPointNearLine(bPoint, start, end, distance_threshold))
                {
                    std::cout << "    Point near line: (" << bPoint.x << ", " << bPoint.y << ")\n";
                    foundPointNearLine = true;
                }
            }

            if (foundPointNearLine)
            {
                line.virtual_wll = std::make_pair(start, end);
                line.distance = calculateDistance(start, end);
                lines.emplace_back(line);
            }
            else
            {
                std::cout << "    No points from dataB are near this line.\n";
            }
        }
    }
    return lines;
}


void ROOMSEG::buildDataBase()
{
    vector<Point> circlesCenters = addHalfOverlappingCircles(trajectoryPts_, radius_);            
    for (size_t i = 0; i < circlesCenters.size();)
    {
        cv::Point exploreCenterPt = circlesCenters[i];        

        SEGDATA db = exploreFeaturePoint(featurePts_, trajectoryPts_, exploreCenterPt, radius_);
        // 지정한 윈도우 안에 feture point 2개 이하 이며 탐색 제외
        if (db.feturePoints.size() < 2)
        {
            circlesCenters.erase(circlesCenters.begin() + i);
        }
        else
        {                    
            mergeClosePoints(db.feturePoints, 3);
            // 거리 계산 함수 호출            
            std::vector<LINEINFO> check_lines = checkPointsNearLineSegments(db.feturePoints, db.trajectoryPoints, 3);
            lineInfo_.push_back(check_lines);
            ++i;            
        }        
    }    
}


void ROOMSEG::buildDataBaseTest(const cv::Mat& img_color_map)
{
    //int radius = 20; // 탐색 범위 반지름
    
    std::vector<cv::Point> circlesCenters = addHalfOverlappingCircles(trajectoryPts_, radius_);
        
    for (size_t i = 0; i < circlesCenters.size();)
    {
        cv::Mat img_update = img_color_map.clone();

        cv::Point exploreCenterPt = circlesCenters[i]; 
        SEGDATA db = exploreFeaturePoint(featurePts_, trajectoryPts_, exploreCenterPt, radius_);

        // 지정한 윈도우 안에 feture point 2개 이하 이며 탐색 제외
        if (db.feturePoints.size() < 2)
        {
            circlesCenters.erase(circlesCenters.begin() + i);

            cv::drawMarker(img_update, exploreCenterPt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS);
            drawingOutLineCircule(img_update, exploreCenterPt, radius_);
            //drawingSetpRectangle(img_update, exploreCenterPt, radius);
        }
        else
        {   
            cv::drawMarker(img_update, exploreCenterPt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS); 
            drawingOutLineCircule(img_update, exploreCenterPt, radius_);
            //drawingSetpRectangle(img_update, exploreCenterPt, radius);

            std::cout << "-----------------------------------------------------" << std::endl;
            std::cout << "Center Point:(" << db.centerPoint.x << ", " << db.centerPoint.y << ")\n";

            std::cout << "TrajectoryPts:";
            for (const auto &pt : db.trajectoryPoints)
            {
                cv::circle(img_update, pt, 2, cv::Scalar(0, 255, 0), -1);
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            };
            std::cout << std::endl;            
            
            mergeClosePoints(db.feturePoints, 3);
            std::cout << "FeaturePt: ";           
            for (const auto &pt : db.feturePoints)
            {
                cv::circle(img_update, pt, 3, cv::Scalar(0, 0, 255), -1);
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            }
            std::cout << std::endl;
            
            // 거리 계산 함수 호출            
            std::vector<LINEINFO> check_lines = checkPointsNearLineSegments(db.feturePoints, db.trajectoryPoints, 3);            
            lineInfo_.push_back(check_lines);
            std::cout << std::endl;            
            ++i;            
        }
        cv::imshow("img_update", img_update);
        cv::waitKey(0);
    }    
}



// std::vector<std::vector<LINEINFO>>를 std::vector<LINEINFO>로 변환하는 함수
std::vector<LINEINFO> ROOMSEG::convertToLineInfo(const std::vector<std::vector<LINEINFO>> &a)
{
    std::vector<LINEINFO> b; // 1D 벡터를 위한 벡터
    // 2D 벡터를 순회하며 각 요소를 1D 벡터에 추가
    for (const auto &innerVector : a)
    {
        b.insert(b.end(), innerVector.begin(), innerVector.end());
    }

    return b; // 변환된 벡터 반환
}


void ROOMSEG::extractVirtualLine(double length_virtual_line)
{    
    buildDataBase();

    // cv::Mat color_img_raw_rotated;
    // cv::cvtColor(img_raw_rotated_, color_img_raw_rotated, COLOR_GRAY2BGR);
    // buildDataBaseTest(color_img_raw_rotated);    

    std::vector<LINEINFO> contvert_type = convertToLineInfo(lineInfo_);     
    std::cout << "contvert_type.size(): " << contvert_type.size() << std::endl;

    // Remove duplicates
    std::vector<LINEINFO> unique_lines = removeDuplicatesLines(contvert_type);    
    std::cout << "unique_lines.size(): " << unique_lines.size() << std::endl;

    std::vector<LINEINFO> filtered_lines = fillterLine(unique_lines);
    std::cout << "filtered_lines.size(): " << filtered_lines.size() << std::endl;

    for (const auto &line : filtered_lines)
    {
        std::cout << "Line: ("
                << line.virtual_wll.first.x << ", " << line.virtual_wll.first.y << ") to ("
                << line.virtual_wll.second.x << ", " << line.virtual_wll.second.y << ") - Distance: "
                << line.distance << std::endl;
    }

    // Output the filtered lines
    //for (const auto &line : filtered_lines)
    for (size_t i=0; i< filtered_lines.size(); i++)
    {        
        if (filtered_lines[i].distance < length_virtual_line)
        {
            virtual_line_.push_back(filtered_lines[i]);
            // cv::line(img_freespace_,filtered_lines[i].virtual_wll.first, 
            //                         filtered_lines[i].virtual_wll.second,
            //                         cv::Scalar(0), 2);                                
        }
    } 

    //cv::imshow("img_freespace_2", img_freespace_);
}



// 중복되는 점과 점은 거리가 큰 것은 제거함
//---------------------------------------------------------------------------
// Check if two lines overlap based on their endpoints
bool ROOMSEG::linesOverlap(const LINEINFO &line1, const LINEINFO &line2)
{
    return (line1.virtual_wll.first == line2.virtual_wll.first ||
            line1.virtual_wll.first == line2.virtual_wll.second ||
            line1.virtual_wll.second == line2.virtual_wll.first ||
            line1.virtual_wll.second == line2.virtual_wll.second);
}

// Filter lines based on overlapping conditions
std::vector<LINEINFO> ROOMSEG::fillterLine(std::vector<LINEINFO> &lines)
{
    std::vector<LINEINFO> result;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        bool toRemove = false;
        for (size_t j = 0; j < lines.size(); ++j)
        {
            if (i != j && linesOverlap(lines[i], lines[j]))
            {
                if (lines[i].distance > lines[j].distance)
                {
                    toRemove = true; // Mark for removal
                    break;
                }
            }
        }

        if (!toRemove)
        {
            result.push_back(lines[i]); // Keep line if it is not marked for removal
        }
    }
    return result;
}


// Function to check if two LINEINFO objects are equal regardless of the order of points
bool ROOMSEG::areEqualIgnoringOrder(const LINEINFO &a, const LINEINFO &b)
{
    return (a.virtual_wll == b.virtual_wll) ||
            (a.virtual_wll.first == b.virtual_wll.second && a.virtual_wll.second == b.virtual_wll.first);
}

// Function to remove duplicates from a vector of LINEINFO
std::vector<LINEINFO> ROOMSEG::removeDuplicatesLines(const std::vector<LINEINFO> &lines)
{
    std::set<LINEINFO, LineInfoComparator> uniqueLines;

    for (const auto &line : lines)
    {
        bool isDuplicate = false;
        for (const auto &uniqueLine : uniqueLines)
        {
            if (areEqualIgnoringOrder(line, uniqueLine))
            {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate)
        {
            uniqueLines.insert(line);
        }
    }

    // Convert back to vector
    return std::vector<LINEINFO>(uniqueLines.begin(), uniqueLines.end());
}


/*
// 이진화된 이미지에서 흰색 중심점 근처의 검은색 좌표 찾기
cv::Point ROOMSEG::findNearestBlackPoint(const cv::Mat& image, cv::Point center) 
{
    int maxRadius = std::min(image.rows, image.cols); // 탐색 가능한 최대 반경
    for (int radius = 1; radius < maxRadius; radius++) {
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                cv::Point newPoint = center + cv::Point(dx, dy);
                // 이미지 범위 확인
                if (newPoint.x >= 0 && newPoint.x < image.cols && newPoint.y >= 0 && newPoint.y < image.rows) {
                    // 검은색 픽셀(0)을 찾음
                    if (image.at<uchar>(newPoint) == 0) {
                        return newPoint; // 검은색 픽셀 좌표 반환
                    }
                }
            }
        }
    }
    return center; // 검은색을 찾지 못하면 원래 좌표 반환
}
*/

void ROOMSEG::makeRegionToBox()
{       


}

cv::Mat ROOMSEG::makeCorrectionRegion()
{  

    // cv::Mat img_contour;
    // cv::bitwise_not(img_grid_, img_contour);
    // cv::Mat img_virtual_line = img_contour.clone();

    
    for (size_t i=0; i<virtual_line_.size(); i++)
    {

        cv::Point spt = virtual_line_[i].virtual_wll.first;
        cv::Point ept =  virtual_line_[i].virtual_wll.second;

        cout <<"초기 좌표" << spt << ", " << ept << endl;
        adjustLineSlope(spt, ept);
        cout <<"수정 좌표" << spt << ", " << ept << endl;

        //cv::line(img_freespace_,cv::Point(spt.x -2, spt.y +2), 
        //                         cv::Point(ept.x +2, ept.y -2), 
        //                         cv::Scalar(0), 2);                                 
        cv::line(img_freespace_,spt, ept, cv::Scalar(0), 2); 
    }

    cv::imshow("img_freespace_", img_freespace_);

  // 3. 레이블링 작업을 수행합니다 (8방향 연결).
    cv::Mat labels, stats, centroids;    
    int n_labels = cv::connectedComponentsWithStats(img_freespace_, labels, stats, centroids, 4, CV_32S);
    
    std::cout <<"=============================" <<std::endl;
    std::cout << "n_labels: " << n_labels << std::endl;
    std::cout <<"=============================" <<std::endl;

    std::vector<cv::Vec3b> colors(n_labels);
   // 각 라벨에 무작위 색상 할당 (외곽선을 그리기 전 사용)    
    for (int i = 1; i < n_labels; i++) {
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }
    
    cv::Mat color_img_grid_;
    cvtColor(img_grid_, color_img_grid_, COLOR_GRAY2BGR);


    cv::Mat color_img_label = cv::Mat::zeros(img_grid_.size(), CV_8UC3);    
    for (int label = 1; label < n_labels; ++label)
    {
        // 바운딩 박스 좌표
        int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        int y = stats.at<int>(label, cv::CC_STAT_TOP);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        cv::Rect box(x, y, width, height);
        int box_size = width*height;
        
        if ( box_size > 9*9) 
        {
            // // 각 레이블의 픽셀에 색을 입힘
            // for (int row = 0; row < labels.rows; row++) 
            // {
            //     for (int col = 0; col < labels.cols; col++) 
            //     {
            //         if (labels.at<int>(row, col) == label) 
            //         {
            //             color_img_contour.at<cv::Vec3b>(row, col) = colors[label];
            //         }
            //     }
            // }            
            
            regions_box_.push_back(box);

            // 라벨 마스크를 생성
            cv::Mat mask = labels == label;
            
            // 외곽선 좌표 저장할 벡터
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            // 외곽선 추출 (mask에서)
            cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);



            cv::drawContours(color_img_label, contours, -1, colors[label], 1);
        }        
    }
    std::cout << "regions_box_: " << regions_box_.size() << std::endl;
    std::cout <<"=============================" <<std::endl;

    cv::imshow("color_img_label", color_img_label);

    cv::Mat test;
    add(color_img_label, color_img_grid_, test);
    cv::imshow("test", test);


    // for (int i =0; i< regions_box_.size(); i++)
    // {
    //     cv::Rect box = regions_box_[i];
    //     std::cout << "box  : " << box << std::endl;
    //     //cv::Point box_center = box.tl() + 0.5 * cv::Point(box.size());        
    //     rectangle(img_contour, box, cv::Scalar(0), 3);
    // }
    // cv::imshow("img_contour1", img_contour);

    // cv::floodFill(img_contour, cv::Point(5, 5), cv::Scalar(0));    
    // cv::imshow("img_contour2", img_contour);

    return color_img_label;
}


void ROOMSEG::initialImageSetting()
{
    
    cv::Mat img_freespace = extractFreeSpaceElements(160);
    cv::imshow("extractFreeSpaceElements()...", img_freespace);

    //cv::Mat img_wall = extractWallElements2(img_freespace);

    int thread_wall_value = 50;    
    cv::Mat img_wall = extractWallElements(thread_wall_value);
    cv::imshow("extractWallElements()...", img_wall);

    cv::Mat rotatioion_mat = makeRotationMatrix();

    img_wall_ = makeRotatedImage(img_wall, rotatioion_mat);
    img_freespace_ = makeRotatedImage(img_freespace, rotatioion_mat);

    cv::imshow("img_wall_...", img_wall_);
    cv::imshow("img_freespace_...", img_freespace_);

    int length_contours=15;
    int gridSize=3;
    makeGridSnappingContours(length_contours, gridSize);  
}


void ROOMSEG::segmentationRoom()
{
    //makeRegionToBox();    
    cv::Mat img_contour = makeCorrectionRegion();





//     cv::Mat img_seg = makeRotatedReturn(img_contour);
//     cv::imshow("img_seg", img_seg);

//     // 라벨링 결과 저장할 행렬 (labels)
//     cv::Mat labels;
//     int nLabels = cv::connectedComponents(img_seg, labels, 4, CV_32S);

//     // 라벨링된 결과에서 외곽선을 추출하기 위한 결과 이미지 (컬러)
//     cv::Mat img_label_ = cv::Mat::zeros(img_seg.size(), CV_8UC3);    
//     std::vector<cv::Vec3b> colors(nLabels);
//    // 각 라벨에 무작위 색상 할당 (외곽선을 그리기 전 사용)    
//     for (int i = 1; i < nLabels; i++) {
//         colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
//     }
    
//     std::vector<cv::Point> segPts;
//     // 각 라벨의 외곽선을 추출하여 출력하기
//     for (int label = 1; label < nLabels; label++) {
//         // 라벨 마스크를 생성
//         cv::Mat mask = labels == label;
//         // 외곽선 좌표 저장할 벡터
//         std::vector<std::vector<cv::Point>> contours;
//         std::vector<cv::Vec4i> hierarchy;

//         // 외곽선 추출 (mask에서)
//         cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

//         cout <<"Label num: " << label << endl;
//         cout <<"contours.size(): " << contours.size() << endl;
//         for (int n = 0; n < contours.size(); n++)
//         {
//             cout <<"contours[" << n << "] = " << contours[n].size() << endl;            

//             for (int m =0; m < contours[n].size(); m++)
//             {
//                 segPts.push_back(contours[n][m]);
//             }            
//         }
//         rooms_contours_.push_back(segPts);

//         // 외곽선을 색상과 함께 그리기
//         cv::drawContours(img_label_, contours, -1, colors[label], -1);
//     }

//      // 결과 이미지 출력
//     cv::imshow("Labeled Image", img_label_);

}

