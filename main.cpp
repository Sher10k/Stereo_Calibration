#include <QCoreApplication>
#include <iostream>
#include <fstream>
#include <set>                  // заголовочный файл множеств и мультимножеств
#include <deque>                // Деки
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

using namespace std;
using namespace cv;

#define FRAME_WIDTH 2064    //2448
#define FRAME_HEIGHT 1544   //2048
#define BOARD_X 11
#define BOARD_Y 8

// --- GLOBAL VARIABLES -----------------------------------------------------// 
static string pos_dir = "/home/roman/imagesStereo";
static Size imageSize = Size(FRAME_WIDTH, FRAME_HEIGHT);

    // ChArUco board variables
static Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary( aruco::DICT_6X6_250 );  // DICT_6X6_250 = 10
    // create charuco board object
static Ptr<aruco::CharucoBoard> charucoboard = aruco::CharucoBoard::create(BOARD_X, BOARD_Y, 0.03f, 0.02f, dictionary);

static TermCriteria termcritSub( TermCriteria::EPS | TermCriteria::MAX_ITER, 100, 0.0001 );

static int Calibration_flags = CALIB_FIX_K1 | 
                              CALIB_FIX_K2 | 
                              CALIB_FIX_K3;      // Calibration flags    | CALIB_FIX_K6

static int Stereo_flag =  CALIB_FIX_INTRINSIC;
//                          CALIB_FIX_ASPECT_RATIO |
//                          CALIB_ZERO_TANGENT_DIST |
//                          CALIB_USE_INTRINSIC_GUESS |
//                          CALIB_SAME_FOCAL_LENGTH |
//                          CALIB_RATIONAL_MODEL |
//                          CALIB_FIX_K3 | 
//                          CALIB_FIX_K4 | 
//                          CALIB_FIX_K5;

static int _numCornersHor = 4; 
static int _numCornersVer = 3;
static Size board_sz = Size( _numCornersHor, _numCornersVer ); 

// --- FUNCTION -------------------------------------------------------------//
void read_file_image( vector < string > * imgPath )
{
        // MENU
    cout << "Input name_dir fo calib image: ";
    cin >> pos_dir;
    if (pos_dir == "0") pos_dir = "/home/roman/stereoIMG_ChessBoard/"; //"/home/roman/imagesStereo/";
    size_t kn = 2;
    cout << "Which file to use: ";
    cin >> kn;
    if (kn <= 0) kn = 5;
    cout << "Start stereo calibration" << endl;
    
    string img_name_L = "FLZcmCameraBaslerJpegFrame*.png";
    string img_name_R = "FRZcmCameraBaslerJpegFrame*.png";
    string files_name_L = pos_dir + img_name_L;
    string files_name_R = pos_dir + img_name_R;
    vector < String > files_L, files_R;
    glob( files_name_L, files_L );
    glob( files_name_R, files_R );
    
    set < string > files_set_L, files_set_R;
    vector < string > files_set_LR;
    for ( size_t i = 0; i < files_L.size(); i++ ) 
    {
        string temp = files_L[i];
        files_L[i].erase(0, pos_dir.length() + 2);
        files_set_L.insert(files_L[i]);
        files_L[i] = temp;
    }
    for ( size_t i = 0; i < files_R.size(); i++ ) 
    {
        string temp = files_R[i];
        files_R[i].erase(0, pos_dir.length() + 2);
        files_set_R.insert(files_R[i]);
        files_R[i] = temp;
    }
    set_intersection( files_set_L.begin(), files_set_L.end(), 
                      files_set_R.begin(), files_set_R.end(), 
                      back_inserter(files_set_LR));          // inserter(files_set_LR, files_set_LR.begin())
    
    for ( size_t i = 0; i < files_set_LR.size(); i += kn )      // 1/12 files
    {
//        Rect myROI(0, 0, FRAME_HEIGHT, FRAME_WIDTH);            // 2048 x 2448
//        Mat imgL = imread( pos_dir + "FL" + files_set_LR[i] ); // load the image
//        Mat imgR = imread( pos_dir + "FR" + files_set_LR[i] );
        imgPath[0].push_back( pos_dir + "FL" + files_set_LR[i] );
        imgPath[1].push_back( pos_dir + "FR" + files_set_LR[i] );
//        imgL(Rect(0, 0, 2448, 2048)).copyTo(imgL);
//        imgR(Rect(0, 0, 2448, 2048)).copyTo(imgR);
//        rotate(imgL, imgL, ROTATE_180);
//        rotate(imgR, imgR, ROTATE_180);
    }
}

void read_CharucoBoard( vector < string > * imgPath,
                        vector < vector< vector< Point2f > > > * allCorners, 
                        vector < vector< int > > * allIds, 
                        vector < Mat > * allCharucoCorners, 
                        vector < Mat > * allCharucoIds, 
                        set < unsigned int > * nGoodboard )
{
    for (unsigned int i = 0; i < imgPath->size(); i++)                        // Цикл для определенного числа калибровочных кадров img->size()
    {
        //Mat imgi = img->at(i);
        Mat imgi = imread( imgPath->at(i) );
        imgi(Rect(0, 0, 2448, 2048)).copyTo(imgi);
        //rotate(imgi, imgi, ROTATE_180);
        Mat imgGray;
        cvtColor( imgi, imgGray, COLOR_BGR2GRAY );
        
        vector< vector< Point2f > > corners;
        vector< int > ids;
        
            // Detect markers
        aruco::detectMarkers( imgGray, 
                              dictionary, 
                              corners, 
                              ids, 
                              aruco::DetectorParameters::create());
        
        if ( ids.size() == (BOARD_X * BOARD_Y) / 2 )                          // Проверка удачно найденых углов == 44
        {
                // SUB PIXEL DETECTION
            for (size_t j = 0; j < corners.size(); j++)
            {
//                cornerSubPix( imgGray, 
//                              corners[j], 
//                              Size(20, 20), 
//                              Size(-1, -1), 
//                              termcritSub);
                cornerSubPix( imgGray, 
                              corners[j], 
                              Size(5, 5), 
                              Size(-1, -1), 
                              termcritSub);
            }
                // Interpolate charuco corners
            Mat charucoCorners, charucoIds;
            aruco::interpolateCornersCharuco( corners, 
                                              ids, 
                                              imgGray, 
                                              charucoboard, 
                                              charucoCorners,
                                              charucoIds);
            allCorners->push_back(corners);
            allIds->push_back(ids);
            allCharucoCorners->push_back(charucoCorners);
            allCharucoIds->push_back(charucoIds);
            
            //nGoodboard->push_back(i);
            nGoodboard->insert(i);
        }
    } 
}

void read_Chessboards( vector < string > * imgPath,
                       vector< vector< Point2f > > * allCorners,
                       set < unsigned int > * nGoodboard )
{
    for (unsigned int i = 0; i < imgPath->size(); i++)                        // Цикл для определенного числа калибровочных кадров img->size()
    {
        Mat imgi = imread( imgPath->at(i) );
        Mat imgGray;
        cvtColor( imgi, imgGray, COLOR_BGR2GRAY );
        
        vector< Point2f > corners;
        
            // Find cernels on chessboard
        bool found = findChessboardCorners( imgGray,
                                            board_sz,
                                            corners,
                                            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE ); //CALIB_CB_NORMALIZE_IMAGE, CV_CALIB_CB_FILTER_QUADS
        
        if (found)    // Проверка удачно найденых углов
        {
            cornerSubPix( imgGray,
                          corners,
                          Size(11,11),
                          Size(-1,-1),
                          TermCriteria( TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1 ) );   // Уточнение углов
            
            allCorners->push_back(corners);
            
            //nGoodboard->push_back(i);
            nGoodboard->insert(i);
        }
    } 
}

void output_stereo_img( Mat * img,
                        string * imgPath,
                        vector < vector < Point2f > > * Corners, 
                        Mat * CharucoCorners,
                        Mat * CharucoIds,
                        Matx33d * cameraMatrix, 
                        Matx < double, 1, 5 > * distCoeffs,
                        Mat rmapX,
                        Mat rmapY )
{
    Mat imgi = imread( *imgPath );
    imgi(Rect(0, 0, 2448, 2048)).copyTo( *img );
    //rotate(*img, *img, ROTATE_180);
    Mat tempF;
    
    aruco::drawDetectedMarkers( *img, 
                                *Corners );
    aruco::drawDetectedCornersCharuco( *img, 
                                       *CharucoCorners, 
                                       *CharucoIds);
    undistort( *img, tempF, 
               cameraMatrix[0], 
               distCoeffs[0] );
    remap( tempF, *img, 
           rmapX, rmapY, 
           INTER_LINEAR );
}

void output_stereo_img_chessboard( Mat * img,
                                   string * imgPath,
                                   vector < Point2f > * Corners,
                                   Matx33d * cameraMatrix, 
                                   Matx < double, 1, 5 > * distCoeffs,
                                   Mat rmapX,
                                   Mat rmapY )
{
    Mat imgi = imread( *imgPath );
    imgi.copyTo( *img );
    Mat tempF;
    
    drawChessboardCorners( *img,
                           board_sz,
                           *Corners, 
                           true ); 
    undistort( *img, tempF, 
               cameraMatrix[0], 
               distCoeffs[0] );
    remap( tempF, *img, 
           rmapX, rmapY, 
           INTER_LINEAR );
}

// --- MAIN -----------------------------------------------------------------//
int main()  //int argc, char *argv[]
{
//    Mat img = imread("ChArUcoBoard.png", IMREAD_COLOR);
//    resize(img, img, Size(640, 480));
//    imshow("Output", img);
    
// --- MAIN VARIABLES -------------------------------------------------------// 
    vector < string > imgPath[2];                                           // FILES IMAGE
    FileStorage fs;
//    fs.open( "Stereo_calib_ChArUco.txt", FileStorage::WRITE );    // Write in file data calibration
    fs.open( "Stereo_calib_ChessBoard.txt", FileStorage::WRITE );    // Write in file data calibration

// --- STEP 1 --- Load left and right frames --------------------------------//
    read_file_image( & imgPath[0] );
    
// --- STEP 2 --- Calibration left and right camera -----------------------------------//
    Mat frame[2];
    
        // ChessBoard
    vector< vector< Point2f > > allCornersL;
    vector< vector< Point2f > > allCornersR;
    set < unsigned int > nGoodboard[2];
    read_Chessboards( & imgPath[0],
                      & allCornersL,
                      & nGoodboard[0] );
    read_Chessboards( & imgPath[1],
                      & allCornersR,
                      & nGoodboard[1] );
    cout << "Элементы множества nGoodboard[0]: ";
    copy( nGoodboard[0].begin(), nGoodboard[0].end(), ostream_iterator<int>(cout, " "));
    cout << endl;
    cout << "Элементы множества nGoodboard[1]: ";
    copy( nGoodboard[1].begin(), nGoodboard[1].end(), ostream_iterator<int>(cout, " "));
    cout << endl;
    vector < unsigned int > nGoodboardv;
    set_intersection( nGoodboard[0].begin(), nGoodboard[0].end(), 
                      nGoodboard[1].begin(), nGoodboard[1].end(), 
                      back_inserter(nGoodboardv));
    cout << "Соответствующие пары nGoodboardv: ";
    for (size_t i = 0; i < nGoodboardv.size(); i++) cout << nGoodboardv[i] << " ";
    cout << endl << "nGoodboardv = " << nGoodboardv.size() << endl;
    
        // Sorting the relevant elements
    vector < unsigned int > nGL(nGoodboard[0].size());
    copy( nGoodboard[0].begin(), nGoodboard[0].end(), nGL.begin() );
    vector < unsigned int > nGR(nGoodboard[1].size());
    copy( nGoodboard[1].begin(), nGoodboard[1].end(), nGR.begin() );
    for ( unsigned int i = 0, j = 0, k = 0; k < nGoodboardv.size() ; )
    {
        if ( nGL.at(i) == nGR.at(j) )
        {
            k++;
            i++;
            j++;
        }
        else if ( nGL.at(i) != nGoodboardv.at(k))
        {
            nGL.erase(nGL.begin() + i);
            allCornersL.erase(allCornersL.begin() + i);
        }
        else if ( nGR.at(j) != nGoodboardv.at(k))
        {
            nGR.erase(nGR.begin() + j);
            allCornersR.erase(allCornersR.begin() + j);
        }
        else
        {
            nGL.erase(nGL.begin() + i);
            allCornersL.erase(allCornersL.begin() + i);
            
            nGR.erase(nGR.begin() + j);
            allCornersR.erase(allCornersR.begin() + j);
        }
    }
    nGL.resize(nGoodboardv.size());
    nGR.resize(nGoodboardv.size());
    allCornersL.resize(nGoodboardv.size());
    allCornersR.resize(nGoodboardv.size());
    
// --- Read camera internal settings
    cout << endl << " --- --- READ camera options" << endl;
    Matx < double, 3, 3 > cameraMatrix[2];
    Matx < double, 1, 5 > distCoeffs[2];
    string parametersDir = "/home/roman/Reconst_Stereo/calibration_parameters/initrics/22500062/2019.07.02/";
    
// --- Left camera options
    cout << " --- LEFT camera" << endl;
        // Camera matrix
    fstream file_params( parametersDir + "mtx.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "mtx.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++)
            file_params >> cameraMatrix[0](i, j);
    cout << "cameraMatrixL = " << endl << cameraMatrix[0] << endl;
    file_params.close();
        // Distortion matrix
    file_params.open( parametersDir + "dist.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "dist.csv" << endl;
        exit(0);
    }
    for ( int j = 0; j < 5; j++ )
        file_params >> distCoeffs[0](0, j);
    cout << "distCoeffsL = " << endl << distCoeffs[0] << endl;
    file_params.close();
    
// --- Right camera options
    cout << endl << " --- RIGHT camera" << endl;
        // Camera matrix
    file_params.open( parametersDir + "to_right/22500061/mtxR.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/mtxR.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++)
            file_params >> cameraMatrix[1](i, j);
    cout << "cameraMatrixR = " << endl << cameraMatrix[1] << endl;
    file_params.close();
        // Distortion matrix
    file_params.open( parametersDir + "to_right/22500061/distR.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/distR.csv" << endl;
        exit(0);
    }
    for ( int j = 0; j < 5; j++ )
        file_params >> distCoeffs[1](0, j);
    cout << "distCoeffsR = " << endl << distCoeffs[1] << endl;
    file_params.close();
    cout << " --- --- END READ camera options" << endl;
// --- END Read camera internal settings
    
    
        // Collect data from each frame
/*    vector < vector< vector< Point2f > > > allCorners[2];
    vector < vector< int > > allIds[2];
    vector < Mat > allCharucoCorners[2];
    vector < Mat > allCharucoIds[2];
    
    set < unsigned int > nGoodboard[2];
    read_CharucoBoard( & imgPath[0],
                       & allCorners[0], 
                       & allIds[0], 
                       & allCharucoCorners[0], 
                       & allCharucoIds[0], 
                       & nGoodboard[0] );
    read_CharucoBoard( & imgPath[1],
                       & allCorners[1], 
                       & allIds[1], 
                       & allCharucoCorners[1], 
                       & allCharucoIds[1], 
                       & nGoodboard[1] );
    cout << "Элементы множества nGoodboard[0]: ";
    copy( nGoodboard[0].begin(), nGoodboard[0].end(), ostream_iterator<int>(cout, " "));
    cout << endl;
    cout << "Элементы множества nGoodboard[1]: ";
    copy( nGoodboard[1].begin(), nGoodboard[1].end(), ostream_iterator<int>(cout, " "));
    cout << endl;
    vector < unsigned int > nGoodboardv;
    set_intersection( nGoodboard[0].begin(), nGoodboard[0].end(), 
                      nGoodboard[1].begin(), nGoodboard[1].end(), 
                      back_inserter(nGoodboardv));
    cout << "Соответствующие пары nGoodboardv: ";
    for (size_t i = 0; i < nGoodboardv.size(); i++) cout << nGoodboardv[i] << " ";
    cout << endl << "nGoodboardv = " << nGoodboardv.size() << endl;
    
        // Sorting the relevant elements
    vector < unsigned int > nGL(nGoodboard[0].size());
    copy( nGoodboard[0].begin(), nGoodboard[0].end(), nGL.begin() );
    vector < unsigned int > nGR(nGoodboard[1].size());
    copy( nGoodboard[1].begin(), nGoodboard[1].end(), nGR.begin() );
    for ( unsigned int i = 0, j = 0, k = 0; k < nGoodboardv.size() ; )
    {
        if ( nGL.at(i) == nGR.at(j) )
        {
            k++;
            i++;
            j++;
        }
        else if ( nGL.at(i) != nGoodboardv.at(k))
        {
            nGL.erase(nGL.begin() + i);
            allCorners[0].erase(allCorners[0].begin() + i);
            allIds[0].erase(allIds[0].begin() + i);
            allCharucoCorners[0].erase(allCharucoCorners[0].begin() + i);
            allCharucoIds[0].erase(allCharucoIds[0].begin() + i);
        }
        else if ( nGR.at(j) != nGoodboardv.at(k))
        {
            nGR.erase(nGR.begin() + j);
            allCorners[1].erase(allCorners[1].begin() + j);
            allIds[1].erase(allIds[1].begin() + j);
            allCharucoCorners[1].erase(allCharucoCorners[1].begin() + j);
            allCharucoIds[1].erase(allCharucoIds[1].begin() + j);
        }
        else
        {
            nGL.erase(nGL.begin() + i);
            allCorners[0].erase(allCorners[0].begin() + i);
            allIds[0].erase(allIds[0].begin() + i);
            allCharucoCorners[0].erase(allCharucoCorners[0].begin() + i);
            allCharucoIds[0].erase(allCharucoIds[0].begin() + i);
            
            nGR.erase(nGR.begin() + j);
            allCorners[1].erase(allCorners[1].begin() + j);
            allIds[1].erase(allIds[1].begin() + j);
            allCharucoCorners[1].erase(allCharucoCorners[1].begin() + j);
            allCharucoIds[1].erase(allCharucoIds[1].begin() + j);
        }
    }
    nGL.resize(nGoodboardv.size());
    nGR.resize(nGoodboardv.size());
    for (unsigned int n = 0; n <2; n++)
    {
        allCorners[n].resize(nGoodboardv.size());
        allIds[n].resize(nGoodboardv.size());
        allCharucoCorners[n].resize(nGoodboardv.size());
        allCharucoIds[n].resize(nGoodboardv.size());
    }
    
        // Calibration left and right camera
    Matx33d cameraMatrix[2];
    Matx< double, 1, 5 > distCoeffs[2];
    vector< Mat > rvecs[2], tvecs[2];
    
    calibrateCameraCharuco( allCharucoCorners[0],
                            allCharucoIds[0], 
                            charucoboard, 
                            imageSize,
                            cameraMatrix[0],
                            distCoeffs[0],
                            rvecs[0],
                            tvecs[0],
                            Calibration_flags);
    calibrateCameraCharuco( allCharucoCorners[1],
                            allCharucoIds[1], 
                            charucoboard, 
                            imageSize,
                            cameraMatrix[1],
                            distCoeffs[1],
                            rvecs[1],
                            tvecs[1],
                            Calibration_flags);
    
    fs << "intrinsicL" << cameraMatrix[0];
    fs << "distCoeffsL" << distCoeffs[0];
    fs << "rvecsL" << rvecs[0];
    fs << "tvecsL" << tvecs[0];
    fs << "intrinsicR" << cameraMatrix[1];
    fs << "distCoeffsR" << distCoeffs[1];
    fs << "rvecsR" << rvecs[1];
    fs << "tvecsR" << tvecs[1];*/
    
    fs << "intrinsicL" << cameraMatrix[0];
    fs << "distCoeffsL" << distCoeffs[0];
    fs << "intrinsicR" << cameraMatrix[1];
    fs << "distCoeffsR" << distCoeffs[1];
    
// --- STEP 3 --- Stereo calibration ----------------------------------------//
    vector < vector < Point3f > > objectPoints;
//    objectPoints.resize(nGoodboardv.size());
//    int squareSize = 1;
//    for(unsigned int i = 0; i < nGoodboardv.size(); i++ )
//    {
//        for( int j = 0; j < BOARD_X - 1; j++ )
//            for( int k = 0; k < BOARD_Y - 1; k++ )
//                objectPoints[i].push_back( Point3f( k * squareSize, j * squareSize, 0 ) );
//    }
    
    int numSquares = _numCornersHor * _numCornersVer; 
    vector< Point3f > obj;
    for(int j = 0; j < numSquares; j++)
        obj.push_back( Point3d( j / _numCornersHor, j % _numCornersHor, 0.0 ) );
    for(unsigned int i = 0; i < nGoodboardv.size(); i++ )
    {
        objectPoints.push_back( obj );
    }
    
    Mat R, T, E, F;

//    double rms = stereoCalibrate( objectPoints, 
//                                  allCharucoCorners[0], allCharucoCorners[1], 
//                                  cameraMatrix[0], distCoeffs[0],
//                                  cameraMatrix[1], distCoeffs[1],
//                                  imageSize, 
//                                  R, T, E, F,
//                                  Stereo_flag );
    double rms = stereoCalibrate( objectPoints, 
                                  allCornersL, allCornersR, 
                                  cameraMatrix[0], distCoeffs[0],
                                  cameraMatrix[1], distCoeffs[1],
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
    stereoRectify( cameraMatrix[0], 
                   distCoeffs[0], 
                   cameraMatrix[1], 
                   distCoeffs[1], 
                   imageSize, 
                   R, T, R1, R2, P1, P2, Q, 
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
    initUndistortRectifyMap( cameraMatrix[0], 
                             distCoeffs[0], 
                             R1, P1, 
                             imageSize, 
                             CV_32FC1, 
                             rmap[0][0], rmap[0][1] );
    initUndistortRectifyMap( cameraMatrix[1], 
                             distCoeffs[1], 
                             R2, P2, 
                             imageSize, 
                             CV_32FC1,
                             rmap[1][0], rmap[1][1] );
    
// --- STEP 6 --- Undistort, Remap ------------------------------------------//
    for (size_t n = 0; n < nGoodboardv.size(); n++)
    {
        Mat imgL, imgR;
            // Left
//        output_stereo_img( & imgL,
//                           & imgPath[0][nGoodboardv[n]],
//                           & allCorners[0][n], 
//                           & allCharucoCorners[0][n],
//                           & allCharucoIds[0][n], 
//                           & cameraMatrix[0], 
//                           & distCoeffs[0], 
//                           rmap[0][0],
//                           rmap[0][1] );
        output_stereo_img_chessboard( & imgL,
                                      & imgPath[0][nGoodboardv[n]],
                                      & allCornersL[n],
                                      & cameraMatrix[0],
                                      & distCoeffs[0],
                                      rmap[0][0],
                                      rmap[0][1] );
            // Right
//        output_stereo_img( & imgR, 
//                           & imgPath[1][nGoodboardv[n]],
//                           & allCorners[1][n], 
//                           & allCharucoCorners[1][n],
//                           & allCharucoIds[1][n], 
//                           & cameraMatrix[1], 
//                           & distCoeffs[1], 
//                           rmap[1][0],
//                           rmap[1][1] );
        output_stereo_img_chessboard( & imgR,
                                      & imgPath[1][nGoodboardv[n]],
                                      & allCornersR[n],
                                      & cameraMatrix[1],
                                      & distCoeffs[1],
                                      rmap[1][0],
                                      rmap[1][1] );
        
        Mat frameLR = Mat::zeros(Size(2 * FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3);
        Rect r1(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        Rect r2(FRAME_WIDTH, 0, FRAME_WIDTH, FRAME_HEIGHT);
        putText( imgL, "L", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
        putText( imgR, "R", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
        imgL.copyTo(frameLR( r1 ));
        imgR.copyTo(frameLR( r2 ));
        for( int i = 0; i < frameLR.rows; i += 100 )
            for( int j = 0; j < frameLR.cols; j++ )
                frameLR.at< Vec3b >(i, j)[2] = 255;
        //imshow("calibration", frameLR);
//        imwrite( pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(n) + ".png", frameLR);
//        cout << pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(n) + ".png" << endl;
        imwrite( "/home/roman/stereoIMG_ChessBoard/ImgCalibStereo_ChessBoard/Сali_pair_of_images_000" + to_string(n) + ".png", frameLR );
        cout << "/home/roman/stereoIMG_ChessBoard/ImgCalibStereo_ChessBoard/Сali_pair_of_images_000" + to_string(n) + ".png" << endl;
    }
        
    fs.release();
    cout << " --- Calibration data written into file: Stereo_calib_ChArUco.txt" << endl << endl;
    
    //cin.get();
    waitKey(0);
    return 0;   // a.exec();
}
