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
// --- MAIN VARIABLES -------------------------------------------------------//
    vector < Mat > ImagesL, ImagesR;                         // FILES IMAGE    
    
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
    for ( size_t i = 0; i < files_set_LR.size(); i += 1 )      // 1/12 files
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
    vector < vector < Point2f > > allcharucoCornersL, allcharucoCornersR, imagePoints[2]; 
    vector < Mat > allcharucoIdsL, allcharucoIdsR;
    
    unsigned int nGoodboard = 0;
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
            vector < Point2f > charucoCornersL, charucoCornersR;
            Mat charucoIdsL, charucoIdsR;
            aruco::interpolateCornersCharuco( cornersL, 
                                              idsL, 
                                              ImagesL[i], 
                                              charucoboard, 
                                              charucoCornersL,
                                              charucoIdsL);
            aruco::interpolateCornersCharuco( cornersR, 
                                              idsR, 
                                              ImagesR[i], 
                                              charucoboard, 
                                              charucoCornersR,
                                              charucoIdsR);
                // Draw results
            aruco::drawDetectedMarkers( ImagesL[i], cornersL );
            aruco::drawDetectedMarkers( ImagesR[i], cornersR );
            if(charucoCornersL.size() > 0) aruco::drawDetectedCornersCharuco( ImagesL[i], 
                                                                               charucoCornersL, 
                                                                               charucoIdsL);
            if(charucoCornersR.size() > 0) aruco::drawDetectedCornersCharuco( ImagesR[i], 
                                                                               charucoCornersR, 
                                                                               charucoIdsR);
            
            cout << "Frame captured" << endl;
            allCornersL.push_back(cornersL);
            allIdsL.push_back(idsL);
            //allImgsL.push_back(ImagesL[i]);
            //allcharucoCornersL.push_back(charucoCornersL);
            imagePoints[0].push_back(charucoCornersL);
            allcharucoIdsL.push_back(charucoIdsL);
            
            allCornersR.push_back(cornersR);
            allIdsR.push_back(idsR);
            //allImgsR.push_back(ImagesR[i]);
            //allcharucoCornersR.push_back(charucoCornersR);
            imagePoints[1].push_back(charucoCornersR);
            allcharucoIdsR.push_back(charucoIdsR);
            
            ImagesL[i].copyTo(frameL);
            ImagesR[i].copyTo(frameR);
            frameLR = Mat::zeros(Size(2 * FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3);
            
            putText( frameL, "L", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
            putText( frameR, "R", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
            
            Rect r1(0, 0, FRAME_WIDTH, FRAME_HEIGHT);                // Создаем фрагменты для склеивания зображения
            Rect r2(FRAME_WIDTH, 0, FRAME_WIDTH, FRAME_HEIGHT);
            frameL.copyTo(frameLR( r1 ));
            frameR.copyTo(frameLR( r2 ));
            //imshow("calibration", frameLR);      // Вывод последнего удачного калибровачного кадра и кадра потока
            imwrite( pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(nGoodboard) + ".png", frameLR);
            cout << pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(nGoodboard) + ".png" << endl;
            
            nGoodboard++;
        }
    }
    cout << "nGoodboard = " << nGoodboard << endl;
    
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
    calibrateCameraCharuco( imagePoints[0],     // allcharucoCornersL
                            allcharucoIdsL, 
                            charucoboard, 
                            Size(FRAME_WIDTH, FRAME_HEIGHT),
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
    calibrateCameraCharuco( imagePoints[1],     // allcharucoCornersR
                            allcharucoIdsR, 
                            charucoboard, 
                            Size(FRAME_WIDTH, FRAME_HEIGHT),
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
    imagePoints[0].resize(nGoodboard);
    imagePoints[1].resize(nGoodboard);
    objectPoints.resize(nGoodboard);
    int squareSize = 1;
    for(unsigned int i = 0; i < nGoodboard; i++ )
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
                                  imagePoints[0], imagePoints[1],   // allcharucoCornersL, allcharucoCornersR,
                                  cameraMatrixL, distCoeffsL,
                                  cameraMatrixR, distCoeffsR,
                                  Size(FRAME_WIDTH, FRAME_HEIGHT), 
                                  R, T, E, F,
                                  Stereo_flag );
    cout << "done with RMS error=" << rms << endl;
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;
    fs << "RMS" << rms;
    
//    Mat img = imread("ChArUcoBoard.png", IMREAD_COLOR);
//    resize(img, img, Size(640, 480));
//    imshow("Output", img);
    
    fs.release();
    cout << " --- Calibration data written into file: ../Stereo_calib_ChArUco.txt" << endl << endl;
    
    //cin.get();
    waitKey(0);
    return 0;   // a.exec();
}
