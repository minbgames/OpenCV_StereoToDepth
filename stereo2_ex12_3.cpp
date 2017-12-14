#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include <iostream>

using namespace std;

static void StereoCalib(const char* imageList, int nx, int ny, int useUncalibrated)
{

    int _preFilterSize=41;
    int _preFilterCap=31;
    int _SADWindowSize=41;
    int _minDisparity=64;
    int _numberOfDisparities=128;
    int _textureThreshold=10;
    int _uniquenessRatio=15;

    cvNamedWindow("setting", CV_WINDOW_AUTOSIZE);

    cvCreateTrackbar("_preFilterSize", "setting", &_preFilterSize, 255);
    cvCreateTrackbar("_preFilterCap", "setting", &_preFilterCap, 63);
    cvCreateTrackbar("_SADWindowSize", "setting", &_SADWindowSize, 255);
    cvCreateTrackbar("_minDisparity", "setting", &_minDisparity, 255);
    cvCreateTrackbar("_numberOfDisparities", "setting", &_numberOfDisparities, 255);
    cvCreateTrackbar("_textureThreshold", "setting", &_textureThreshold, 2000);
    cvCreateTrackbar("_uniquenessRatio", "setting", &_uniquenessRatio, 179);

    CvCapture* l_capture = cvCaptureFromCAM(1);
    CvCapture* r_capture = cvCaptureFromCAM(2);
    //왼쪽 카메라 영상과 오른쪽 카메라 영상을 받는다.

    int displayCorners = 1; //코너찾는 과정 볼것인가?
    int showUndistorted = 1; //외곡보정된 모습 볼것인가?
    bool isVerticalStereo = false; //좌우 또는 상하로 배열된 스트레오 영상을 모두 지원한다
    const int maxScale = 1; //?????????
    const float squareSize = 0.037f; //실제 정사각형의 크기 (단위 m)
    double baseline=0.063;

    FILE* f = fopen(imageList, "rt");
    int i, j, lr, nframes, n = nx*ny, N = 0;
    vector<string> imageNames[2];
    vector<CvPoint3D32f> objectPoints;
    vector<CvPoint2D32f> points[2];
    vector<int> npoints;
    vector<uchar> active[2];
    vector<CvPoint2D32f> temp(n);
    CvSize imageSize = {0, 0};

    // 행렬과 벡터를 저장할 변수 생성
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
    CvMat _M1 = cvMat(3, 3, CV_64F, M1 );
    CvMat _M2 = cvMat(3, 3, CV_64F, M2 );
    CvMat _D1 = cvMat(1, 5, CV_64F, D1 );
    CvMat _D2 = cvMat(1, 5, CV_64F, D2 );
    CvMat _R  = cvMat(3, 3, CV_64F, R );
    CvMat _T  = cvMat(3, 1, CV_64F, T );
    CvMat _E  = cvMat(3, 3, CV_64F, E );
    CvMat _F  = cvMat(3, 3, CV_64F, F );

    if( displayCorners )
        cvNamedWindow( "corners", 1 );

    // 체스판의 영상의 목록을 읽는다.
    if( !f )
    {
        fprintf(stderr, "can not open file %s\n", imageList );
        return;
    }

    for(i=0;;i++)
    {
        char buf[1024];
        int count = 0, result=0;
        lr = i % 2;
        vector<CvPoint2D32f>& pts = points[lr];

        if( !fgets( buf, sizeof(buf)-3, f ))
            break;

        size_t len = strlen(buf);
        while( len > 0 && isspace(buf[len-1]))
            buf[--len] = '\0';
        if( buf[0] == '#')
            continue;

        IplImage* img = cvLoadImage( buf, 0 );
        if( !img )
            break;

        imageSize = cvGetSize(img);
        imageNames[lr].push_back(buf);

        // 체스판 내부의 코너점을 찾는다
        for( int s = 1; s <= maxScale ; s++ )
        {
            IplImage* timg = img;
            if( s > 1 )
            {
                timg = cvCreateImage(cvSize(img->width*s, img->height*s),
                                     img->depth, img->nChannels );
                cvResize( img, timg, CV_INTER_CUBIC );
            }
            result = cvFindChessboardCorners( timg, cvSize(nx, ny),
                                              &temp[0], &count,
                                              CV_CALIB_CB_ADAPTIVE_THRESH |
                                              CV_CALIB_CB_NORMALIZE_IMAGE);
            if( timg != img )
                cvReleaseImage( &timg );
            if( result || s == maxScale )
                for( j = 0; j < count; j++ )
                {
                    temp[j].x /= s;
                    temp[j].y /= s;
                }
            if( result )
                break;
        }

        // 찾은 코너점 보여주기
        if( displayCorners )
        {
            printf("%s\n", buf);
            IplImage* cimg = cvCreateImage( imageSize, 8, 3 );
            cvCvtColor( img, cimg, CV_GRAY2BGR );
            cvDrawChessboardCorners( cimg, cvSize(nx, ny), &temp[0],
                                     count, result );
            cvShowImage( "corners", cimg );
            cvReleaseImage( &cimg );
            if( cvWaitKey(0) == 27 ) // ESC Å°žŠ Ž©ž£žé ÁŸ·á.
                exit(-1);
        }
        else
            putchar('.');

        N = pts.size();
        pts.resize(N + n, cvPoint2D32f(0,0));
        active[lr].push_back((uchar)result);

        // assert( result != 0 );
        if( result )
        {
            // 서브픽셀 보간을 하지 않으면 보정이 정확하지 않다.
            cvFindCornerSubPix( img, &temp[0], count,
                                cvSize(11, 11), cvSize(-1,-1),
                                cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                                               30, 0.01) );
            copy( temp.begin(), temp.end(), pts.begin() + N );
        }
        cvReleaseImage( &img );
    }
    fclose(f);
    printf("\n");

    // 체스판의 3D객체점 목록
    nframes = active[0].size();
    objectPoints.resize(nframes*n);
    for( i = 0; i < ny; i++ )
        for( j = 0; j < nx; j++ )
            objectPoints[i*nx + j] =
                    cvPoint3D32f(i*squareSize, j*squareSize, 0);
    for( i = 1; i < nframes; i++ )
        copy( objectPoints.begin(), objectPoints.begin() + n,
              objectPoints.begin() + i*n );

    npoints.resize(nframes,n);
    N = nframes*n;


    /*------------------------calibration---------------------------*/

    CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
    CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
    cvSetIdentity(&_M1);
    cvSetIdentity(&_M2);
    cvZero(&_D1);
    cvZero(&_D2);

    // 스테레오 카메라 보정한다.
    printf("Running stereo calibration ...");
    fflush(stdout);
    cvStereoCalibrate( &_objectPoints, &_imagePoints1,
                       &_imagePoints2, &_npoints,
                       &_M1, &_D1, &_M2, &_D2,
                       imageSize, &_R, &_T, &_E, &_F,
                       CV_CALIB_FIX_ASPECT_RATIO +
                       CV_CALIB_ZERO_TANGENT_DIST +
                       CV_CALIB_SAME_FOCAL_LENGTH,
                       cvTermCriteria(CV_TERMCRIT_ITER+
                                      CV_TERMCRIT_EPS, 100, 1e-5));
    printf(" done\n");

    /*-------------------calibration complete------------------*/

    // 보정품질검사:
    // 출력 기본 행렬은 사실상 모든 출력 정보를 포함하기 때문에
    // 에피폴라 기하 제약을 이용하여 보정 품질을 검사할 수 있다.
    vector<CvPoint3D32f> lines[2];
    points[0].resize(N);
    points[1].resize(N);
    _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    lines[0].resize(N);
    lines[1].resize(N);
    CvMat _L1 = cvMat(1, N, CV_32FC3, &lines[0][0]);
    CvMat _L2 = cvMat(1, N, CV_32FC3, &lines[1][0]);

    //왜곡이 제거된 상태에서 작동된다.
    cvUndistortPoints( &_imagePoints1, &_imagePoints1,
                       &_M1, &_D1, 0, &_M1 );
    cvUndistortPoints( &_imagePoints2, &_imagePoints2,
                       &_M2, &_D2, 0, &_M2 );
    cvComputeCorrespondEpilines( &_imagePoints1, 1, &_F, &_L1 );
    cvComputeCorrespondEpilines( &_imagePoints2, 2, &_F, &_L2 );

    // 에러 확인을 위한 코드
    double avgErr = 0;
    for( i = 0; i < N; i++ )
    {
        double err = fabs(points[0][i].x*lines[1][i].x +
                          points[0][i].y*lines[1][i].y + lines[1][i].z)
                     + fabs(points[1][i].x*lines[0][i].x +
                            points[1][i].y*lines[0][i].y + lines[0][i].z);
        avgErr += err;
    }
    printf( "avg err = %g\n", avgErr/(nframes*n) );

    //----------------calibration 에러 확인--------------------

    int height = imageSize.height;
    int width  = imageSize.width;


    /*------------------------조정 수행 -----------------------*/
    if( showUndistorted )
    {
        CvMat* mx1   = cvCreateMat( height, width, CV_32F );
        CvMat* my1   = cvCreateMat( height, width, CV_32F );
        CvMat* mx2   = cvCreateMat( height, width, CV_32F );
        CvMat* my2   = cvCreateMat( height, width, CV_32F );
        CvMat* img1r = cvCreateMat( height, width, CV_8U  );
        CvMat* img2r = cvCreateMat( height, width, CV_8U  );
        CvMat* disp  = cvCreateMat( height, width, CV_16S );
        CvMat* vdisp = cvCreateMat( height, width, CV_8U  );
        CvMat* reproject = cvCreateMat( height, width, CV_32FC3  );
        CvMat* pair;
        double R1[3][3], R2[3][3], P1[3][4], P2[3][4], Q[4][4];
        CvMat _R1 = cvMat(3, 3, CV_64F, R1);
        CvMat _R2 = cvMat(3, 3, CV_64F, R2);
        CvMat _Q = cvMat(4, 4, CV_64F, Q);

        double multify1=0;
        double multify2=0;
        // BOUGUET 방법
        if( useUncalibrated == 0 )
        {
            CvMat _P1 = cvMat(3, 4, CV_64F, P1);
            CvMat _P2 = cvMat(3, 4, CV_64F, P2);
            cvStereoRectify( &_M1, &_M2, &_D1, &_D2, imageSize,
                             &_R, &_T,
                             &_R1, &_R2, &_P1, &_P2, &_Q,
                             0 /*CV_CALIB_ZERO_DISPARITY*/ );
            isVerticalStereo = fabs(P2[1][3]) > fabs(P2[0][3]);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if(i==3 && j==2){
                        multify1=cvmGet(&_Q, i, j);
                        cvmSet(&_Q, i, j,baseline);
                    }
                    if(i==3 && j==3){
                        multify2=cvmGet(&_Q, i, j);
                        cvmSet(&_Q, i, j,multify2*baseline/multify1);
                    }
                    cout << " " << cvmGet(&_Q, i, j) << " ";
                }

                cout << endl;
            }

            // cvRemap() ÇÔŒö¿¡Œ­ »ç¿ëÇÒ mapµéÀ» °è»ê
            cvInitUndistortRectifyMap(&_M1, &_D1, &_R1, &_P1, mx1, my1);
            cvInitUndistortRectifyMap(&_M2, &_D2, &_R2, &_P2, mx2, my2);
        }
            // HARTLEY 방법수행
        else if( useUncalibrated == 1 || useUncalibrated == 2 )
        {
            double H1[3][3], H2[3][3], iM[3][3];
            CvMat _H1 = cvMat(3, 3, CV_64F, H1);
            CvMat _H2 = cvMat(3, 3, CV_64F, H2);
            CvMat _iM = cvMat(3, 3, CV_64F, iM);

            if( useUncalibrated == 2 )
                cvFindFundamentalMat( &_imagePoints1,
                                      &_imagePoints2, &_F);
            cvStereoRectifyUncalibrated( &_imagePoints1,
                                         &_imagePoints2, &_F,
                                         imageSize,
                                         &_H1, &_H2, 3);
            cvInvert(&_M1, &_iM);
            cvMatMul(&_H1, &_M1, &_R1);
            cvMatMul(&_iM, &_R1, &_R1);
            cvInvert(&_M2, &_iM);
            cvMatMul(&_H2, &_M2, &_R2);
            cvMatMul(&_iM, &_R2, &_R2);

            // cvRemap() 함수에서 사용할 map을 계산
            cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_M1,mx1,my1);
            cvInitUndistortRectifyMap(&_M2,&_D1,&_R2,&_M2,mx2,my2);
        }
        else
            assert(0); //중지 시킴
        cvNamedWindow( "rectified", 1 );

        // 영상을 조정하고 시차지도를 구한다.
        if( !isVerticalStereo )
            pair = cvCreateMat( height, width*2, CV_8UC3 );
        else
            pair = cvCreateMat( height*2, width, CV_8UC3 );

        // 스테레오 대응을 위한 설정.
        CvStereoBMState *BMState = cvCreateStereoBMState();
        assert(BMState != 0);

        /*-------------------- 동영상 받기 ------------------------*/

        IplImage* img1;
        IplImage* img2;

        IplImage *img1_g = cvCreateImage(CvSize(width,height),IPL_DEPTH_8U,1);
        IplImage *img2_g = cvCreateImage(CvSize(width,height),IPL_DEPTH_8U,1);

        while(1)
        {
            if(_preFilterSize%2==0){
                _preFilterSize+=+1;
            }
            if(_preFilterSize<5) _preFilterSize=5;
            if(_preFilterCap==0) _preFilterCap=1;

            if(_SADWindowSize%2==0){
                _SADWindowSize+=+1;
            }
            if(_SADWindowSize<5) _SADWindowSize=5;

            if(_numberOfDisparities%16!=0){
                _numberOfDisparities-=(_numberOfDisparities%16);
                if(_numberOfDisparities<16){
                    _numberOfDisparities=16;
                }
            }

            BMState->preFilterSize=_preFilterSize;
            BMState->preFilterCap=_preFilterCap;
            BMState->SADWindowSize=_SADWindowSize;
            BMState->minDisparity=-_minDisparity;
            BMState->numberOfDisparities=_numberOfDisparities;
            BMState->textureThreshold=_textureThreshold;
            BMState->uniquenessRatio=_uniquenessRatio;

            cvGrabFrame( l_capture );
            img1 = cvRetrieveFrame( l_capture );
            cvGrabFrame( r_capture );
            img2 = cvRetrieveFrame( r_capture );

            cvCvtColor(img1,img1_g,CV_RGB2GRAY);
            cvCvtColor(img2,img2_g,CV_RGB2GRAY);

            if( img1_g && img2_g )
            {
                CvMat part;
                cvRemap( img1_g, img1r, mx1, my1 );
                cvRemap( img2_g, img2r, mx2, my2 );

                if( !isVerticalStereo || useUncalibrated != 0 ) //시차지도 이미지 출력
                {
                    cvFindStereoCorrespondenceBM( img1r, img2r, disp,
                                                  BMState);
                    cvNormalize( disp, vdisp, 0, 256, CV_MINMAX );

                    cvNamedWindow( "disparity" );
                    cvShowImage( "disparity", vdisp );

                    cvReprojectImageTo3D(vdisp,reproject,&_Q);
                    cvNamedWindow( "reproject" );
                    cvShowImage( "reproject", reproject );

                }
                if( !isVerticalStereo )
                {
                    cvGetCols( pair, &part, 0, width );
                    cvCvtColor( img1r, &part, CV_GRAY2BGR );
                    cvGetCols( pair, &part, width,
                               width*2 );
                    cvCvtColor( img2r, &part, CV_GRAY2BGR );
                    for( j = 0; j < height; j += 16 )
                        cvLine( pair, cvPoint(0,j),
                                cvPoint(width*2,j),
                                CV_RGB(0,255,0));
                }
                else
                {
                    cvGetRows( pair, &part, 0, height );
                    cvCvtColor( img1r, &part, CV_GRAY2BGR );
                    cvGetRows( pair, &part, height,
                               height*2 );
                    cvCvtColor( img2r, &part, CV_GRAY2BGR );
                    for( j = 0; j < width; j += 16 )
                        cvLine( pair, cvPoint(j,0),
                                cvPoint(j,height*2),
                                CV_RGB(0,255,0));
                }
                cvShowImage( "rectified", pair );
                if( cvWaitKey(1) == 27 )
                    break;
            }

        }
        cvReleaseImage( &img1 );
        cvReleaseImage( &img2 );
        cvReleaseImage( &img1_g );
        cvReleaseImage( &img2_g );
        cvReleaseStereoBMState( &BMState );
        cvReleaseMat( &mx1 );
        cvReleaseMat( &my1 );
        cvReleaseMat( &mx2 );
        cvReleaseMat( &my2 );
        cvReleaseMat( &img1r );
        cvReleaseMat( &img2r );
        cvReleaseMat( &disp  );
        cvReleaseMat( &vdisp );
        cvReleaseMat( &pair );
    }

    cvDestroyAllWindows();
}

int main(void)
{
    StereoCalib("/home/m/Desktop/PROGRAMMING/Clion_code/list.txt", 7, 4, 0);
    //list.txt는 체스판 영상의 목록을 담고 있는 텍스트파일
    //nx, ny 코너판의 개수를 뜻함
    //useUncalibrated가 0이면 Hartley방법으로 1이면 Bouquet방법을 이용

    return 0;
}