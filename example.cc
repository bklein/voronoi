#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>

#include <gflags/gflags.h>

#include "voronoi.hpp"

#define CHECK(cond) \
{ \
  if (!(cond)) { \
    std::cerr << "CHECK FAILED  " << __LINE__ << ": " << #cond << std::endl; \
  } else { \
    std::cout << "CHECK SUCCESS " << __LINE__ << ": " << #cond << std::endl; \
  } \
}

DEFINE_int32(width, 640, "image width");
DEFINE_int32(height, 480, "image height");
DEFINE_int32(point_scale, 10000, "points generated per image area");
DEFINE_string(out, "out.png", "output filename");
DEFINE_bool(viz, false, "visualize image");
DEFINE_bool(centers, false, "draw point centers");
DEFINE_bool(blur, true, "apply Gaussian blur");
DEFINE_int32(blur_kernel_size, 3, "kernel width of Gaussian blur");
DEFINE_int32(blur_iterations, 10, "iterations to run Gaussian blur");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const int width             = FLAGS_width;
  const int height            = FLAGS_height;
  const int point_area_scale  = FLAGS_point_scale;
  const int num_points        = (width * height) / point_area_scale;
  const bool viz              = FLAGS_viz;
  const bool blur             = FLAGS_blur;
  const bool draw_centers     = FLAGS_centers;

  std::cout << "dim: " << width << "x" << height << std::endl;
  std::cout << "points: " << num_points << std::endl;

  std::vector<cv::Point2d> points;
  generateRandomPoints(points, num_points, cv::Point2d(width, height));

  std::vector<cv::Vec3b> colors;
  generatePointColors(points, colors);

  CHECK(num_points > 0);
  CHECK(points.size() == static_cast<size_t>(num_points));
  CHECK(points.size() == colors.size());

  const auto distance_metric = DistanceMetric::NormL2;
  cv::Mat3b image = drawVoronoi(points,
                                colors,
                                cv::Size(width, height),
                                distance_metric);
  if (blur) {
    applyGaussianBlurIterations(image, FLAGS_blur_kernel_size, FLAGS_blur_iterations);
  }

  if (draw_centers) {
    drawVoronoiCenters(image, points, (width / 200) + 1);
  }

  cv::imwrite("out.png", image);

  if (viz) {
    cv::imshow("voronoi", image);
    cv::waitKey(0);
  }

  return 0;
}
