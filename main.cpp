#include <QCoreApplication>
#include <iostream>
#include <set>                  // заголовочный файл множеств и мультимножеств
#include <deque>                // Деки
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

using namespace std;
using namespace cv;

#define FRAME_WIDTH 2448
#define FRAME_HEIGHT 2048
#define BOARD_X 11
#define BOARD_Y 8

// --- GLOBAL VARIABLES -----------------------------------------------------// 

// --- FUNCTION -------------------------------------------------------------//


// --- MAIN -----------------------------------------------------------------//
int main()  //int argc, char *argv[]
{
//    Mat img = imread("ChArUcoBoard.png", IMREAD_COLOR);
//    resize(img, img, Size(640, 480));
//    imshow("Output", img);
    
// --- MAIN VARIABLES -------------------------------------------------------//
    vector < Mat > ImagesL, ImagesR;                         // FILES IMAGE    
    Size imageSize = Size(FRAME_WIDTH, FRAME_HEIGHT);
    
// --- STEP 1 --- Load left and right frames --------------------------------//
    string pos_dir = "/home/roman/imagesStereo";
    string img_name_L = "/FLZcmCameraBaslerJpegFrame*.png";
    string img_name_R = "/FRZcmCameraBaslerJpegFrame*.png";
    string files_name_L = pos_dir + img_name_L;
    string files_name_R = pos_dir + img_name_R;
    vector< String > files_L, files_R;
    glob( files_name_L, files_L );
    glob( files_name_R, files_R );
    
    set< string > files_set_L, files_set_R;
    vector< string > files_set_LR;
    for (size_t i = 0; i < files_L.size(); i++) 
    {
        string temp = files_L[i];
        files_L[i].erase(0, pos_dir.length() + 3);
        files_set_L.insert(files_L[i]);
        files_L[i] = temp;
    }
    for (size_t i = 0; i < files_R.size(); i++) 
    {
        string temp = files_R[i];
        files_R[i].erase(0, pos_dir.length() + 3);
        files_set_R.insert(files_R[i]);
        files_R[i] = temp;
    }
    set_intersection( files_set_L.begin(), files_set_L.end(), 
                      files_set_R.begin(), files_set_R.end(), 
                      back_inserter(files_set_LR));          // inserter(files_set_LR, files_set_LR.begin())
    
    Rect myROI(0, 0, FRAME_HEIGHT, FRAME_WIDTH);            // 2048 x 2448
    
    unsigned int nFrames = 0;
    for ( size_t i = 0; i < files_set_LR.size(); i += 2 )      // 1/12 files
    {
        Mat imgL = imread( pos_dir + "/FL" + files_set_LR[i] ); // load the image
        Mat imgR = imread( pos_dir + "/FR" + files_set_LR[i] );
        imgL(Rect(0, 0, 2448, 2048)).copyTo(imgL);
        imgR(Rect(0, 0, 2448, 2048)).copyTo(imgR);
        ImagesL.push_back( imgL );
        ImagesR.push_back( imgR );
        nFrames++;
    }
    
    
// --- STEP 2 --- Calibration left and right camera -----------------------------------//
    Mat frameL, frameR, frameLR;
    TermCriteria termcritSub( TermCriteria::EPS | TermCriteria::MAX_ITER, 100, 0.0001 );
    
        // ChArUco board variables
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(10));  // DICT_6X6_250 = 10
        // create charuco board object
    Ptr<aruco::CharucoBoard> charucoboard = aruco::CharucoBoard::create(BOARD_X, BOARD_Y, 10, 7, dictionary);
    
        // Collect data from each frame
    vector < vector< vector< Point2f > > > allCornersL, allCornersR;
    vector < vector< int > > allIdsL, allIdsR;
    //vector< Mat > allImgsL, allImgsR;
    vector < Mat > allCharucoCornersL, allCharucoCornersR; 
    vector < Mat > allCharucoIdsL, allCharucoIdsR;
    
    vector < unsigned int > nGoodboard;
    for (unsigned int i = 0; i < nFrames; i++)                        // Цикл для определенного числа калибровочных кадров
    {
        Mat imgGrayL, imgGrayR;
        cvtColor(ImagesL[i], imgGrayL, COLOR_BGR2GRAY);
        cvtColor(ImagesR[i], imgGrayR, COLOR_BGR2GRAY);
        vector< int > idsL, idsR;
        vector< vector< Point2f > > cornersL, cornersR;    
        
            // detect markers
        aruco::detectMarkers( imgGrayL, 
                              dictionary, 
                              cornersL, 
                              idsL, 
                              aruco::DetectorParameters::create());
        aruco::detectMarkers( imgGrayR, 
                              dictionary, 
                              cornersR, 
                              idsR, 
                              aruco::DetectorParameters::create());
        
        if ( (idsL.size() == BOARD_X * BOARD_Y / 2) && (idsR.size() == BOARD_X * BOARD_Y / 2) )    // Проверка удачно найденых углов == 44
        {
                // SUB PIXEL DETECTION
            for (size_t j = 0; j < cornersL.size(); j++)
            {
                cornerSubPix( imgGrayL, 
                              cornersL[j], 
                              Size(20, 20), 
                              Size(-1, -1), 
                              termcritSub);
            }
            for (size_t j = 0; j < cornersR.size(); j++)
            {
                cornerSubPix( imgGrayR, 
                              cornersR[j], 
                              Size(20, 20), 
                              Size(-1, -1), 
                              termcritSub);
            }
            
                // Interpolate charuco corners
            Mat charucoCornersL, charucoCornersR;
            Mat charucoIdsL, charucoIdsR;
            aruco::interpolateCornersCharuco( cornersL, 
                                              idsL, 
                                              imgGrayL, 
                                              charucoboard, 
                                              charucoCornersL,
                                              charucoIdsL);
            aruco::interpolateCornersCharuco( cornersR, 
                                              idsR, 
                                              imgGrayR, 
                                              charucoboard, 
                                              charucoCornersR,
                                              charucoIdsR);
            allCornersL.push_back(cornersL);
            allIdsL.push_back(idsL);
            //allImgsL.push_back(ImagesL[i]);
            allCharucoCornersL.push_back(charucoCornersL);
            allCharucoIdsL.push_back(charucoIdsL);
            
            allCornersR.push_back(cornersR);
            allIdsR.push_back(idsR);
            //allImgsR.push_back(ImagesR[i]);
            allCharucoCornersR.push_back(charucoCornersR);
            allCharucoIdsR.push_back(charucoIdsR);
            
                // Draw results
            /*aruco::drawDetectedMarkers( imgGrayL, cornersL ); // ImagesL[i]
            aruco::drawDetectedMarkers( imgGrayR, cornersR ); // ImagesR[i]
            if(charucoCornersL.rows > 0) aruco::drawDetectedCornersCharuco( imgGrayL, 
                                                                            charucoCornersL, 
                                                                            charucoIdsL);
            if(charucoCornersR.rows > 0) aruco::drawDetectedCornersCharuco( imgGrayR, 
                                                                            charucoCornersR, 
                                                                            charucoIdsR);
            
            putText( imgGrayL, "L", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);   // frameL
            putText( imgGrayR, "R", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);   // frameR
            
            frameLR = Mat::zeros(Size(2 * FRAME_WIDTH, FRAME_HEIGHT), CV_8UC1);
            Rect r1(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
            Rect r2(FRAME_WIDTH, 0, FRAME_WIDTH, FRAME_HEIGHT);
            imgGrayL.copyTo(frameLR( r1 ));
            imgGrayR.copyTo(frameLR( r2 ));
            //imshow("calibration", frameLR);
            imwrite( pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(nGoodboard) + ".png", frameLR);
            cout << pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(nGoodboard) + ".png" << endl;*/
            
            nGoodboard.push_back(i);
        }
    }
    cout << "nGoodboard = " << nGoodboard.size() << endl;
    
    /*for(unsigned int i = 0; i < nGoodboard; i++) 
    {
            // interpolate using camera parameters
        vector< Point2f > currentCharucoCornersL, currentCharucoCornersR;
        Mat currentCharucoIdsL, currentCharucoIdsR;
        aruco::interpolateCornersCharuco( allCornersL[i], 
                                          allIdsL[i], 
                                          allImgsL[i], 
                                          charucoboard,
                                          currentCharucoCornersL, 
                                          currentCharucoIdsL);
        aruco::interpolateCornersCharuco( allCornersR[i], 
                                          allIdsR[i], 
                                          allImgsR[i], 
                                          charucoboard,
                                          currentCharucoCornersR, 
                                          currentCharucoIdsR);

        allCharucoCornersL.push_back(currentCharucoCornersL);
        allCharucoIdsL.push_back(currentCharucoIdsL);
        allCharucoCornersR.push_back(currentCharucoCornersR);
        allCharucoIdsR.push_back(currentCharucoIdsR);
    }*/
    
        // Calibration left and right camera
    Matx33d cameraMatrixL, cameraMatrixR;
    Matx<double, 1, 5> distCoeffsL, distCoeffsR;
    vector<Mat> rvecsL, tvecsL, rvecsR, tvecsR;
    int calibrationFlags = CALIB_FIX_K1 | 
                           CALIB_FIX_K2 | 
                           CALIB_FIX_K3;      // Calibration flags    | CALIB_FIX_K6
    calibrateCameraCharuco( allCharucoCornersL,
                            allCharucoIdsL, 
                            charucoboard, 
                            imageSize,
                            cameraMatrixL,
                            distCoeffsL,
                            rvecsL,
                            tvecsL,
                            calibrationFlags);
    FileStorage fs;
    fs.open("../Stereo_calib_ChArUco.txt", FileStorage::WRITE);    // Write in file data calibration
    fs << "intrinsicL" << cameraMatrixL;
    fs << "distCoeffsL" << distCoeffsL;
    //fs << "rvecsL" << rvecsL;
    //fs << "tvecsL" << tvecsL;
    calibrateCameraCharuco( allCharucoCornersR,
                            allCharucoIdsR, 
                            charucoboard, 
                            imageSize,
                            cameraMatrixR,
                            distCoeffsR,
                            rvecsR,
                            tvecsR,
                            calibrationFlags);
    fs << "intrinsicR" << cameraMatrixR;
    fs << "distCoeffsR" << distCoeffsR;
    //fs << "rvecsR" << rvecsR;
    //fs << "tvecsR" << tvecsR;
    
    
// --- STEP 3 --- Stereo calibration ----------------------------------------//
    vector < vector < Point3f > > objectPoints;
    objectPoints.resize(nGoodboard.size());
    int squareSize = 1;
    for(unsigned int i = 0; i < nGoodboard.size(); i++ )
    {
        for( int j = 0; j < BOARD_X - 1; j++ )
            for( int k = 0; k < BOARD_Y - 1; k++ )
                objectPoints[i].push_back( Point3f( k * squareSize, j * squareSize, 0 ) );
    }
    Mat R, T, E, F;
    int Stereo_flag =  CALIB_FIX_INTRINSIC;
//                      CALIB_FIX_ASPECT_RATIO |
//                      CALIB_ZERO_TANGENT_DIST |
//                      CALIB_USE_INTRINSIC_GUESS |
//                      CALIB_SAME_FOCAL_LENGTH |
//                      CALIB_RATIONAL_MODEL |
//                      CALIB_FIX_K3 | 
//                      CALIB_FIX_K4 | 
//                      CALIB_FIX_K5;
    double rms = stereoCalibrate( objectPoints, 
                                  allCharucoCornersL, allCharucoCornersR, 
                                  cameraMatrixL, distCoeffsL,
                                  cameraMatrixR, distCoeffsR,
                                  imageSize, 
                                  R, T, E, F,
                                  Stereo_flag );
    cout << "done with RMS error=" << rms << endl;
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;
    fs << "RMS" << rms;
    
    
// --- STEP 4 --- Stereo rectify --------------------------------------------//
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    stereoRectify( cameraMatrixL, 
                   distCoeffsL, 
                   cameraMatrixR, 
                   distCoeffsR, 
                   imageSize, 
                   R, T, R1,R2, P1, P2, Q, 
                   CALIB_ZERO_DISPARITY, 
                   -1, 
                   imageSize, 
                   &validRoi[0], 
                   &validRoi[1]);
    fs << "R1" <<R1;
    fs << "P1" <<P1;
    fs << "R2" <<R2;
    fs << "P2" <<P2;
    fs << "Q" <<Q;
    
// --- STEP 5 --- Undistort Rectify Map -------------------------------------//
    Mat rmap[2][2];
    initUndistortRectifyMap( cameraMatrixL, 
                             distCoeffsL, 
                             R1, P1, 
                             imageSize, 
                             CV_32FC1, 
                             rmap[0][0], rmap[0][1] );
    initUndistortRectifyMap( cameraMatrixR, 
                             distCoeffsR, 
                             R2, P2, 
                             imageSize, 
                             CV_16SC2, 
                             rmap[1][0], rmap[1][1] );
    
    
// --- STEP 6 --- Undistort, Remap ------------------------------------------//
    for (size_t n = 0; n < nGoodboard.size(); n++)
    {
        Mat tempFL, tempFR;
            // Left
        aruco::drawDetectedMarkers( ImagesL[nGoodboard[n]], allCornersL[n] );
        aruco::drawDetectedCornersCharuco( ImagesL[nGoodboard[n]], 
                                           allCharucoCornersL[n], 
                                           allCharucoIdsL[n]);
        undistort( ImagesL[nGoodboard[n]], tempFL, cameraMatrixL, distCoeffsL);
        remap(tempFL, ImagesL[nGoodboard[n]], rmap[0][0], rmap[0][1], INTER_LINEAR);
        putText( ImagesL[nGoodboard[n]], "L", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
            // Right
        aruco::drawDetectedMarkers( ImagesR[nGoodboard[n]], allCornersR[n] );
        aruco::drawDetectedCornersCharuco( ImagesR[nGoodboard[n]], 
                                           allCharucoCornersR[n], 
                                           allCharucoIdsR[n]);
        undistort( ImagesR[nGoodboard[n]], tempFR, cameraMatrixR, distCoeffsR);
        remap(tempFR, ImagesR[nGoodboard[n]], rmap[1][0], rmap[1][1], INTER_LINEAR);
        putText( ImagesR[nGoodboard[n]], "R", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10); 
    
        
        frameLR = Mat::zeros(Size(2 * FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3);
        Rect r1(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        Rect r2(FRAME_WIDTH, 0, FRAME_WIDTH, FRAME_HEIGHT);
        ImagesL[nGoodboard[n]].copyTo(frameLR( r1 ));
        ImagesR[nGoodboard[n]].copyTo(frameLR( r2 ));
        for( int i = 0; i < frameLR.rows; i += 100 )
            for( int j = 0; j < frameLR.cols; j++ )
                frameLR.at< Vec3b >(i, j)[2] = 255;
        //imshow("calibration", frameLR);
        imwrite( pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(n) + ".png", frameLR);
        cout << pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(n) + ".png" << endl;
    }
    
    fs.release();
    cout << " --- Calibration data written into file: ../Stereo_calib_ChArUco.txt" << endl << endl;
    
    //cin.get();
    waitKey(0);
    return 0;   // a.exec();
}
