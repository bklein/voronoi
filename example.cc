#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <gflags/gflags.h>

#define CHECK(cond) \
{ \
  if (!(cond)) { \
    std::cerr << "CHECK FAILED  " << __LINE__ << ": " << #cond << std::endl; \
  } else { \
    std::cout << "CHECK SUCCESS " << __LINE__ << ": " << #cond << std::endl; \
  } \
}

class ScopedTimer {
 public:
  ScopedTimer(const std::string& label)
    : label_(label),
      start_time_(cv::getTickCount()) {
  }

  ~ScopedTimer() {
    const auto end_time = cv::getTickCount();
    std::cout << label_ << " took " << 1000.0 * static_cast<double>(end_time - start_time_) / cv::getTickFrequency() << " ms" << std::endl;
  }

 private:
  const std::string label_;
  const int64_t start_time_;
};

DEFINE_int32(width, 640, "image width");
DEFINE_int32(height, 480, "image height");
DEFINE_int32(point_scale, 10000, "points generated per image area");
DEFINE_string(out, "out.png", "output filename");
DEFINE_bool(viz, false, "visualize image");
DEFINE_bool(centers, false, "draw point centers");
DEFINE_bool(blur, true, "apply Gaussian blur");
DEFINE_int32(blur_kernel_size, 3, "kernel width of Gaussian blur");
DEFINE_int32(blur_iterations, 1000, "iterations to run Gaussian blur");

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
  std::vector<cv::Vec3b> colors;

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis_color(0, 255);

    std::uniform_real_distribution<double>  dis_x(0.0, width);
    std::uniform_real_distribution<double>  dis_y(0.0, height);
    //std::normal_distribution<double>  dis_x(0.0, width);
    //std::normal_distribution<double>  dis_y(0.0, height);

    for (int i=0; i<num_points; ++i) {
      points.emplace_back(dis_x(gen), dis_y(gen));
    }
  }
  {
    cv::Mat3b colormap;
    {
      cv::Mat1b tmp(1, 256);
      for (int i=0; i<tmp.cols; ++i)
        tmp(i) = static_cast<uint8_t>(i);
      cv::applyColorMap(tmp, colormap, cv::COLORMAP_JET);
    }

    for (int i=0; i<num_points; ++i) {
      const int colormap_idx = cv::saturate_cast<uint8_t>(((1.0 * i) / (1.0 * num_points)) * 255.0);
      colors.emplace_back(colormap(colormap_idx));
    }
  }

  CHECK(num_points > 0);
  CHECK(points.size() == static_cast<size_t>(num_points));
  CHECK(points.size() == colors.size());

  cv::Mat3b image = cv::Mat3b::zeros(height, width);

  {
    ScopedTimer t("voronoi");
    for (int row = 0; row<image.rows; ++row) {
      for (int col=0; col<image.cols; ++col) {
        size_t best_idx = std::numeric_limits<size_t>::max();
        double best_norm = std::numeric_limits<double>::max();
        for (size_t i=0; i<points.size(); ++i) {
          const double x = points[i].x;
          const double y = points[i].y;
          const double vector_u = static_cast<double>(col) - x;
          const double vector_v = static_cast<double>(row) - y;
          //const double norm = std::sqrt(vector_u * vector_u + vector_v * vector_v);
          //const double norm = vector_u * vector_u + vector_v * vector_v;
          const double norm = std::abs(vector_u) + std::abs(vector_v);
          //const double norm = std::max(std::abs(vector_u), std::abs(vector_v));
          if (norm < best_norm) {
            best_norm = norm;
            best_idx = i;
          }
        }
        image(row, col) = colors[best_idx];
      }
      if (viz & (row % 8 == 0)) {
        cv::imshow("voronoi", image);
        cv::waitKey(1);
      }
    }
  }

  if (blur) {
    ScopedTimer t("Gaussian blur");
    const int kernel_size = FLAGS_blur_kernel_size;
    const int blur_iterations = FLAGS_blur_iterations;
    for (int i=0; i<blur_iterations; ++i) {
      cv::GaussianBlur(image, image, cv::Size(kernel_size, kernel_size), 0);
    }
  }

  if (draw_centers) {
    for (const auto& pt : points) {
      cv::circle(image, pt, (width / 200) + 1, cv::Scalar::all(0), -1);
    }
  }

  cv::imwrite("out.png", image);

  if (viz) {
    cv::imshow("voronoi", image);
    cv::waitKey(0);
  }

  return 0;
}
