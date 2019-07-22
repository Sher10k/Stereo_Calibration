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

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
  
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
    set_intersection(files_set_L.begin(), files_set_L.end(), 
                     files_set_R.begin(), files_set_R.end(), 
                     back_inserter(files_set_LR));          // inserter(files_set_LR, files_set_LR.begin())
    
    vector< Mat > frame_list_L, frame_list_R;               // FILES IMAGE 
    Rect myROI(0, 0, 2048, 2448);
    
    for ( size_t i = 0; i < files_set_LR.size(); i += 12 )
    {
        Mat imgL = imread( pos_dir + "/FL" + files_set_LR[i] ); // load the image
        Mat imgR = imread( pos_dir + "/FR" + files_set_LR[i] );
        imgL(Rect(0, 0, 2448, 2048)).copyTo(imgL);
        imgR(Rect(0, 0, 2448, 2048)).copyTo(imgR);
        frame_list_L.push_back( imgL );
        frame_list_R.push_back( imgR );
    }
    
// --- STEP 2 --- Calibration left camera -----------------------------------//
    Mat frame, frame2, frame3;
    Mat frame4 = Mat::zeros(Size(2 * FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3);
    int calibrationFlags = CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3;      // Calibration flags    | CALIB_FIX_K6
    unsigned int nFrames = static_cast< unsigned int >(files_set_LR.size());
    
        // ChArUco board variables
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(10));  // DICT_6X6_250 = 10
        // create charuco board object
    Ptr<aruco::CharucoBoard> charucoboard = aruco::CharucoBoard::create( 11, 8, 10, 7, dictionary);
    
    vector< Mat > allCharucoCornersL, allCharucoCornersR;
    vector< Mat > allCharucoIdsL, allCharucoIdsR;
//    vector< Mat > filteredImages;
//    allCharucoCornersL.reserve(static_cast<unsigned int>(nFrames));
//    allCharucoIdsL.reserve(static_cast<unsigned int>(nFrames));
    Ptr<aruco::DetectorParameters> detectorParamsL = aruco::DetectorParameters::create();
    Ptr<aruco::DetectorParameters> detectorParamsR = aruco::DetectorParameters::create();
        // collect data from each frame
    vector< vector< vector< Point2f > > > allCornersL, allCornersR;
    vector< vector< int > > allIdsL, allIdsR;
    vector< Mat > allImgsL, allImgsR;
    
    unsigned int n = 0;
    while(1)                        // Цикл для определенного числа калибровочных кадров
    {
        vector< int > idsL, idsR;
        vector< vector< Point2f > > cornersL, cornersR;    
        
            // detect markers
        aruco::detectMarkers( frame_list_L[n], 
                              dictionary, 
                              cornersL, 
                              idsL, 
                              detectorParamsL);
        aruco::detectMarkers( frame_list_R[n], 
                              dictionary, 
                              cornersR, 
                              idsR, 
                              detectorParamsR);
        
        if ( (idsL.size() > 0) && (idsR.size() > 0) )    // Проверка удачно найденых углов
        {
                // interpolate charuco corners
            Mat currentCharucoCornersL, currentCharucoIdsL;
            Mat currentCharucoCornersR, currentCharucoIdsR;
            aruco::interpolateCornersCharuco( cornersL, 
                                              idsL, 
                                              frame_list_L[n], 
                                              charucoboard, 
                                              currentCharucoCornersL,
                                              currentCharucoIdsL);
            aruco::interpolateCornersCharuco( cornersR, 
                                              idsR, 
                                              frame_list_R[n], 
                                              charucoboard, 
                                              currentCharucoCornersR,
                                              currentCharucoIdsR);
                // draw results
            aruco::drawDetectedMarkers( frame_list_L[n], cornersL );
            aruco::drawDetectedMarkers( frame_list_R[n], cornersR );
            if(currentCharucoCornersL.total() > 0) aruco::drawDetectedCornersCharuco( frame_list_L[n], 
                                                                                      currentCharucoCornersL, 
                                                                                      currentCharucoIdsL);
            if(currentCharucoCornersR.total() > 0) aruco::drawDetectedCornersCharuco( frame_list_R[n], 
                                                                                      currentCharucoCornersR, 
                                                                                      currentCharucoIdsR);
            
            cout << "Frame captured" << endl;
            allCornersL.push_back(cornersL);
            allIdsL.push_back(idsL);
            allCornersR.push_back(cornersR);
            allIdsR.push_back(idsR);
            
        }
        CAP.read(frame3);
        putText( frame3, 
                 "Press 'c' to add current frame. 'ESC' to finish and calibrate ChArUco_board",
                 Point(5, 20), 
                 FONT_HERSHEY_SIMPLEX, 
                 0.5, 
                 Scalar(255, 0, 0), 
                 2);
        Rect r1(0, 0, frame3.cols, frame3.rows);                // Создаем фрагменты для склеивания зображения
        Rect r2(frame3.cols, 0, frame3.cols, frame3.rows);
        frame3.copyTo(frame4( r1 ));
        calib_frame.copyTo(frame4( r2 ));
        imshow("calibration", frame4);      // Вывод последнего удачного калибровачного кадра и кадра потока
        
        n++;
    }
    
    
// --- STEP 3 --- Calibration right camera ----------------------------------//
    
    
    
//    Mat img = imread("ChArUcoBoard.png", IMREAD_COLOR);
//    resize(img, img, Size(640, 480));
//    imshow("Output", img);
    
    //cin.get();
    waitKey(0);
    return 0;   // a.exec();
}
