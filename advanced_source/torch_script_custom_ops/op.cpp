#include <opencv2/opencv.hpp>
#include <torch/script.h>

// warp_perspective 시작
torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
  // image_mat 시작
  cv::Mat image_mat(/*행=*/image.size(0),
                    /*열=*/image.size(1),
                    /*타입=*/CV_32FC1,
                    /*데이터=*/image.data_ptr<float>());
  // image_mat 완료

  // warp_mat 시작
  cv::Mat warp_mat(/*행=*/warp.size(0),
                   /*열=*/warp.size(1),
                   /*타입=*/CV_32FC1,
                   /*데이터=*/warp.data_ptr<float>());
  // warp_mat 완료

  // output_mat 시작
  cv::Mat output_mat;
  cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8, 8});
  // output_mat 완료

  // output_tensor 시작
  torch::Tensor output = torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8, 8});
  return output.clone();
  // output_tensor 완료
}
// warp_perspective 완료

// registry 시작
TORCH_LIBRARY(my_ops, m) {
  m.def("warp_perspective", warp_perspective);
}
// registry 완료
