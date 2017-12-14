#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main(void)
{
    VideoCapture left_cap(0);
    VideoCapture right_cap(1);

    if ( !left_cap.isOpened() )
    {
        cout << "left camera could not open." << endl;
        return -1;
    }

    if ( !right_cap.isOpened() )
    {
        cout << "right camera could not open." << endl;
        return -1;
    }

    namedWindow("left_camera", CV_WINDOW_AUTOSIZE);
    namedWindow("right_camera", CV_WINDOW_AUTOSIZE);
    int key_input;
    while (true)
    {

        //웹캠에서 캡처되는 속도 출력
        Mat leftImg_input, rightImg_input;

        //카메라로부터 이미지를 가져옴
        left_cap>>leftImg_input;
        right_cap>>rightImg_input;

        imshow("left_camera", leftImg_input);
        imshow("right_camera", rightImg_input);

        key_input = waitKey(1);

        if(key_input == 32){
            cout << "save image" << endl;
            imwrite("leftimage.jpg",leftImg_input);
            imwrite("rightimage.jpg",rightImg_input);
        }
        if(key_input == 27) break;
    }
    return 0;
}