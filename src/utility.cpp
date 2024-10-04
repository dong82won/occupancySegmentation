
#include "utility.h"

// Custom comp
cv::Scalar randomColor()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    return cv::Scalar(dist(gen), dist(gen), dist(gen)); // 임의의 BGR 색상 생성
}

std::vector<cv::Point> changeMatoPoint(cv::Mat &image)
{
    std::vector<cv::Point> edgePts;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) == 255)
            {
                edgePts.push_back(cv::Point(j, i));
            }
        }
    }
    return edgePts;
}

cv::Mat changePointoMat(std::vector<cv::Point> src, int rows, int cols)
{
    cv::Mat temp = cv::Mat::zeros(cv::Size(rows, cols), CV_8UC1); 
    for (size_t i =0; i<src.size(); i++)
    {
        int x = src[i].x;
        int y = src[i].y; 
        temp.at<uchar>(y, x) = 255;
    }
    return temp;
}

cv::Mat extractMat2Mat(cv::Mat &image, uchar pixel_value)
{
    cv::Mat dst=cv::Mat::zeros(image.size(), CV_8UC1);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) == pixel_value)
            {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    return dst;
}

// 거리 계산 함수
double calculateDistance(const cv::Point &p1, const cv::Point &p2)
{
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

// 가까운 두 점 통합 함수 (평균값을 사용하여 통합)
cv::Point mergePoints(const cv::Point& p1, const cv::Point& p2) {
    return cv::Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}

// 선분의 중간점 계산 함수
std::pair<cv::Point, cv::Point> mergeSegments(const std::pair<cv::Point, cv::Point>& seg1,
                                            const std::pair<cv::Point, cv::Point>& seg2) {
    cv::Point mid1 = (seg1.first + seg2.first) / 2;
    cv::Point mid2 = (seg1.second + seg2.second) / 2;
    return {mid1, mid2};
}


// 점들을 순차적으로 정렬하여 실선 재구성
std::vector<cv::Point> sortPoints(const std::vector<cv::Point> &points)
{
    std::vector<cv::Point> sortedPoints;

    if (points.empty())
        return sortedPoints;

    std::vector<cv::Point> remainingPoints = points;
    cv::Point currentPoint = remainingPoints[0];

    sortedPoints.push_back(currentPoint);
    remainingPoints.erase(remainingPoints.begin());

    while (!remainingPoints.empty())
    {
        auto nearestIt = std::min_element(remainingPoints.begin(), remainingPoints.end(),
                                          [&currentPoint](const cv::Point &p1, const cv::Point &p2)
                                          {
                                              return calculateDistance(currentPoint, p1) < calculateDistance(currentPoint, p2);
                                          });
        currentPoint = *nearestIt;
        sortedPoints.push_back(currentPoint);
        remainingPoints.erase(nearestIt);
    }

    return sortedPoints;
}

// 중복되는 cv::Point 데이터를 제거하는 함수
void removeDuplicatePoints(std::vector<cv::Point> &points)
{
    // points를 정렬
    std::sort(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b)
              { return (a.x < b.x) || (a.x == b.x && a.y < b.y); });

    // 중복된 요소를 points의 끝으로 이동
    auto last = std::unique(points.begin(), points.end());

    // 중복된 요소를 제거
    points.erase(last, points.end());
}

// 거리 내의 점들을 병합하는 함수
void mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold)
{
    std::vector<cv::Point> mergedPoints;

    while (!points.empty())
    {
        cv::Point basePoint = points.back();
        points.pop_back();

        std::vector<cv::Point> closePoints;
        closePoints.push_back(basePoint);

        for (auto it = points.begin(); it != points.end();)
        {
            if (cv::norm(basePoint - *it) <= distanceThreshold)
            {
                closePoints.push_back(*it);
                it = points.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // 평균 위치를 계산하여 병합된 점을 추가
        cv::Point avgPoint(0, 0);
        for (const auto &pt : closePoints)
        {
            avgPoint += pt;
        }
        avgPoint.x /= closePoints.size();
        avgPoint.y /= closePoints.size();
        mergedPoints.push_back(avgPoint);
    }

    points = mergedPoints;
}

void drawingSetpRectangle(cv::Mat &image, cv::Point circlesCenters, int radius)
{
    int max_x = circlesCenters.x + radius;
    int max_y = circlesCenters.y + radius;
    int min_x = circlesCenters.x - radius;
    int min_y = circlesCenters.y - radius;
    cv::Rect rect(cv::Point(min_x, min_y), cv::Point(max_x, max_y));

    cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 1); // 파란색 사각형
}


// 주어진 중심과 반지름을 기반으로 원의 경계 점을 찾는 함수
std::vector<cv::Point> edgePointsInCircle(const cv::Point &center, int radius)
{
    std::vector<cv::Point> points;

    // 원의 경계에서 점들을 추출
    for (double angle = 0; angle < 2 * CV_PI; angle += 0.1)
    { // 각도를 0.1씩 증가시켜 점을 추출
        int x = static_cast<int>(center.x + radius * cos(angle));
        int y = static_cast<int>(center.y + radius * sin(angle));
        points.push_back(cv::Point(x, y));
    }

    return points;
}

void drawingOutLineCircule(const cv::Mat &image, cv::Point circlesCenters, int radius)
{
     // 원형 영역을 이미지에 그리기
    std::vector<cv::Point> edgePoints = edgePointsInCircle(circlesCenters, radius);
    
    // 원을 실선으로 그리기
    for (size_t i = 0; i < edgePoints.size(); i++)
    {
        cv::Point start = edgePoints[i];
        cv::Point end = edgePoints[(i + 1) % edgePoints.size()]; // 마지막 점과 첫 점을 연결
        cv::line(image, start, end, cv::Scalar(0, 255, 255), 1);       // 파란색으로 실선 그리기
    }
}

// 기울기를 계산하는 함수
double calculateSlope(const cv::Point& p1, const cv::Point& p2) {
    if (p1.x == p2.x) {
        return std::numeric_limits<double>::infinity();  // X 좌표가 같으면 기울기는 무한대
    }
    return static_cast<double>(p2.y - p1.y) / static_cast<double>(p2.x - p1.x);
}



// 근접한 선 성분을 찾아 평균값으로 갱신하는 함수
void mergeCloseSegments(std::vector<std::pair<cv::Point, cv::Point>>& segments, double threshold) {
    for (size_t i = 0; i < segments.size() - 1; ++i) {
        for (size_t j = i + 1; j < segments.size(); ++j) {
            
            double dist1 = calculateDistance(segments[i].first, segments[j].first);
            double dist2 = calculateDistance(segments[i].second, segments[j].second);

            std::cout << "dist1: " << dist1 << ", dist2: " << dist2 << std::endl;

            // 두 선분이 일정 거리 이하로 가까운 경우 통합
            if (dist1 <= threshold && dist2 <= threshold) {
                
                auto mergedSegment = mergeSegments(segments[i], segments[j]);
                
                segments[i] = mergedSegment;
                // j번째 선분을 삭제
                segments.erase(segments.begin() + j);
                --j;  // 삭제 후 인덱스 조정

                std::cout << "Merged segment " << i << " with " << j << ":\n";
                std::cout << "New segment: [(" << mergedSegment.first.x << ", " << mergedSegment.first.y << "), ("
                        << mergedSegment.second.x << ", " << mergedSegment.second.y << ")]\n";
            }
        }
    }
}

// 기울기를 90도 또는 180도로 조정하는 함수
std::pair<cv::Point, cv::Point> adjustLineSlope(std::pair<cv::Point, cv::Point> lines, bool* state)
{

    cv::Point p1 = lines.first;
    cv::Point p2 = lines.second;

    std::pair<cv::Point, cv::Point> agjustline;

    // 같은 점인지 확인
    if (p1 == p2)
    {
        std::cerr << "Error: 두 점이 동일합니다. 선 조정을 할 수 없습니다." << std::endl;      
        *state = 2;
    }
    else
    {
        // 좌표 값의 범위를 검사 (예시로 좌표 값의 범위를 0 ~ 10000으로 설정)

        double slope = calculateSlope(p1, p2);
        // 아크탄젠트를 이용해 라디안 값을 구한 후 각도로 변환 (라디안 -> 각도 변환: 180 / PI)
        double angle = atan(slope) * (180.0 / M_PI);

        if (abs(angle) >= 60.0)
        {
            // if (std::isinf(slope) || (abs(slope) <= 10 &&  abs(slope) > 0) ) {
            *state = 0;
            // 수직
            std::cout << *state << " slope: " << abs(slope) << ", angle: " << abs(angle) << std::endl;
            int mean_x = round((p1.x + p2.x) / 2);
            
            agjustline.first.x = mean_x;
            agjustline.first.y = p1.y;

            agjustline.second.x = mean_x;
            agjustline.second.y = p2.y;
        }
        else if (abs(angle) <=30.0 )
        {
            // 수평
            *state = 1;
            std::cout << *state << " slope: " << abs(slope) << ", angle: " << abs(angle) << std::endl;
            int mean_y = round((p1.y + p2.y) / 2);
            
            agjustline.first.x = p1.x;
            agjustline.first.y = mean_y;

            agjustline.second.x = p2.x;
            agjustline.second.y = mean_y;
        }
        else 
        {
            *state = 2;
        }
    }
    return agjustline;
}
